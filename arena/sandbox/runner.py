"""Sandbox runner: one interface, two backends.

`SandboxRunner.run(job, out_dir)` starts a fresh, throwaway sandbox, feeds it one JSON job (no secrets), streams
its JSON-line events back, collects /out into out_dir, and destroys the sandbox. The app never parses a web page
or decodes a downloaded file itself; it only reads the sandbox's manifest, re-encoded clips and stills, and it
validates those before use (see validate_output).

Backends (chosen by SANDBOX_BACKEND):
  docker  a local container: read-only root, unprivileged user, all capabilities dropped, no-new-privileges,
          CPU / memory / pids caps, tmpfs scratch, a wall-clock kill, `--rm`
  vultr   a throwaway Vultr VM per box (see VultrRunner): the same container, driven over SSH, on a VM that only
          admits the app's IP and is deleted when the box ends or is killed
"""

from __future__ import annotations

import base64
import contextvars
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

IMAGE = os.environ.get("SANDBOX_IMAGE", "arena-scraper:0.1")
CAPS = {
    "cpus": os.environ.get("SANDBOX_CPUS", "1.0"),
    "memory": os.environ.get("SANDBOX_MEMORY", "512m"),
    "pids": int(os.environ.get("SANDBOX_PIDS", "128")),
    "seconds": int(os.environ.get("SANDBOX_SECONDS", "300")),
}
MAX_OUT_BYTES = 400_000_000
CLIP_ID = re.compile(r"^[0-9a-f]{10}$")


class BoxKilled(RuntimeError):
    """The operator pressed Kill: the box's sandbox is gone and the run stops at its next checkpoint."""


class SandboxRunner:
    name = "base"

    def run(self, job: dict, out_dir: Path, on_event: Callable[[dict], None]) -> dict:
        raise NotImplementedError

    def kill(self) -> None:
        """Stop whatever is running now. Called from another thread."""

    def close(self) -> None:
        """Release everything this runner holds. Always called when its box ends."""


class Box:
    """One robot box is one run. Every sandbox job in the run goes through the same runner (so on Vultr the box owns
    one VM for its lifetime), and the server can kill the box from another thread."""

    def __init__(self, rid: str):
        self.rid = rid
        self.runner = _backend()
        self.killed = False

    def kill(self) -> None:
        self.killed = True
        self.runner.kill()


_box: contextvars.ContextVar[Box | None] = contextvars.ContextVar("box", default=None)


@contextmanager
def box(rid: str) -> Iterator[Box]:
    b = Box(rid)
    tok = _box.set(b)
    try:
        yield b
    finally:
        _box.reset(tok)
        b.runner.close()


def check_killed() -> None:
    b = _box.get()
    if b is not None and b.killed:
        raise BoxKilled(f"box {b.rid} was killed by the operator")


class DockerRunner(SandboxRunner):
    """A hardened throwaway container. With `ssh` set, the same `docker run` executes on a remote host (a Vultr VM)
    and /out comes back as a tar stream that is unpacked with the data filter before validate_output sees it."""

    name = "docker"

    def __init__(self, ssh: list[str] | None = None, host: str = "local"):
        self.ssh = ssh
        self.host = host
        self._cname: str | None = None
        self._proc: subprocess.Popen | None = None

    def _cmd(self, args: list[str]) -> list[str]:
        return [*self.ssh, shlex.join(args)] if self.ssh else args

    def kill(self):
        if self._cname:
            subprocess.run(
                self._cmd(["docker", "kill", self._cname]),
                capture_output=True,
                timeout=30,
                check=False,
            )
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def run(self, job, out_dir, on_event):
        check_killed()
        out_dir.mkdir(parents=True, exist_ok=True)
        cname = f"arena-sbx-{uuid.uuid4().hex[:8]}"
        seconds = min(int(job.get("sandbox_seconds", CAPS["seconds"])), 900)
        scratch = Path(tempfile.mkdtemp(prefix="arena-out-"))
        os.chmod(scratch, 0o777)
        mount = f"/tmp/{cname}" if self.ssh else str(scratch)
        if self.ssh:
            subprocess.run(
                self._cmd(["mkdir", "-m", "777", mount]),
                check=True,
                capture_output=True,
                timeout=60,
            )
        cmd = self._cmd(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                cname,
                "--read-only",
                "--tmpfs",
                "/tmp:rw,size=200m",
                "--tmpfs",
                "/home/scraper:rw,size=16m",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--cpus",
                CAPS["cpus"],
                "--memory",
                CAPS["memory"],
                "--memory-swap",
                CAPS["memory"],
                "--pids-limit",
                str(CAPS["pids"]),
                "--add-host",
                "host.docker.internal:host-gateway",
                "-v",
                f"{mount}:/out:rw",
                IMAGE,
            ]
        )
        job = {
            **job,
            "caps": {**CAPS, "seconds": seconds},
            "time_budget_s": min(job.get("time_budget_s", 240), seconds - 20),
        }
        t0 = time.time()
        on_event(
            {
                "type": "sandbox_start",
                "backend": self.name if not self.ssh else "vultr",
                "host": self.host,
                "container": cname,
                "caps": {**CAPS, "seconds": seconds},
            }
        )
        self._cname = cname
        p = self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        killer = threading.Timer(seconds, self.kill)
        killer.start()
        try:
            p.stdin.write(json.dumps(job))
            p.stdin.close()
            for ev in _json_lines(p.stdout):
                on_event(ev)
            p.wait()
        finally:
            killer.cancel()
            self._cname = None
            subprocess.run(
                self._cmd(["docker", "rm", "-f", cname]),
                capture_output=True,
                timeout=60,
                check=False,
            )
        if self.ssh:
            _fetch_tar(
                self._cmd(["sh", "-c", f"tar -C {mount} -cf - . && rm -rf {mount}"]),
                scratch,
            )
        manifest = validate_output(scratch, out_dir)
        shutil.rmtree(scratch, ignore_errors=True)
        on_event(
            {
                "type": "sandbox_destroyed",
                "host": self.host,
                "container": cname,
                "seconds": round(time.time() - t0, 1),
                "exit": p.returncode,
                "stderr_tail": (p.stderr.read() or "")[-300:],
            }
        )
        check_killed()
        return manifest


def _fetch_tar(cmd: list[str], dest: Path) -> None:
    """Stream a tar from an untrusted host into dest: regular files and dirs only, no escapes, total size capped."""
    total = 0

    def keep(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
        nonlocal total
        member = tarfile.data_filter(member, path)
        if not (member.isfile() or member.isdir()):
            return None
        total += member.size
        return member if total <= MAX_OUT_BYTES else None

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
            tf.extractall(dest, filter=keep)
    except tarfile.TarError:
        pass  # an empty or broken tar means no output; validate_output then finds no manifest
    finally:
        p.stdout.close()
        p.wait(timeout=60)


VULTR_API = "https://api.vultr.com/v2"
CLOUD_INIT = """#cloud-config
package_update: true
packages: [docker.io]
runcmd:
  - systemctl enable --now docker
"""
# Runs on the VM once it is reachable: unpack the scraper build context and the planted hostile pages, build the
# sandbox image unless the snapshot already has it, and serve the hostile pages on the docker bridge only.
REMOTE_SETUP = (
    "cloud-init status --wait >/dev/null 2>&1; "
    "rm -rf /opt/arena && mkdir -p /opt/arena && tar -C /opt/arena -xf - && "
    "(docker image inspect {image} >/dev/null 2>&1 || docker build -q -t {image} /opt/arena/sandbox >/dev/null) && "
    "(ufw allow in on docker0 to any port 8765 proto tcp >/dev/null 2>&1 || true) && "
    "(setsid nohup python3 /opt/arena/hostile/server.py 8765 172.17.0.1 </dev/null >/dev/null 2>&1 &) && "
    "docker version --format '{{{{.Server.Version}}}}'"
)
ROOT = Path(__file__).resolve().parents[2]


class VultrRunner(SandboxRunner):
    """One throwaway Vultr VM per box.

    The app holds VULTR_API_KEY; nothing on the VM ever sees it. Per box: a fresh SSH key, a firewall group that
    only admits port 22 from the app's own IP, and a small CPU instance (Ubuntu + Docker via cloud-init, or
    VULTR_SNAPSHOT_ID with Docker and the image baked in). Every sandbox job runs as the same hardened `docker run`
    as the local backend, driven over SSH, so the VM is a second wall around the container. The instance, its
    firewall group and its key are deleted when the box ends, is killed, or crashes. scripts/vultr.py sweeps any
    leftovers tagged arena-sbx.
    """

    name = "vultr"

    def __init__(self):
        key = os.environ.get("VULTR_API_KEY")
        if not key:
            raise RuntimeError(
                "SANDBOX_BACKEND=vultr needs VULTR_API_KEY in the app's environment"
            )
        self.api = httpx.Client(
            base_url=VULTR_API, headers={"Authorization": f"Bearer {key}"}, timeout=30
        )
        self.region = os.environ.get("VULTR_REGION", "sjc")
        self.plan = os.environ.get("VULTR_PLAN", "vc2-1c-2gb")
        self.snapshot = os.environ.get("VULTR_SNAPSHOT_ID")
        self.label = f"arena-sbx-{uuid.uuid4().hex[:8]}"
        self.tmp = Path(tempfile.mkdtemp(prefix="arena-vm-"))
        self.instance: str | None = None
        self.firewall: str | None = None
        self.ssh_key: str | None = None
        self.docker: DockerRunner | None = None
        self.hourly: float | None = None
        self._emit: Callable[[dict], None] = lambda e: None
        self._t0 = 0.0
        self._lock = threading.Lock()
        self._closed = False

    def run(self, job, out_dir, on_event):
        check_killed()
        self._emit = on_event
        if self.docker is None:
            self._provision()
        return self.docker.run(job, out_dir, on_event)

    def kill(self):
        if self.docker:
            self.docker.kill()
        self.close()

    def _provision(self) -> None:
        self._t0 = time.time()
        kp = self.tmp / "id_ed25519"
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                self.label,
                "-f",
                str(kp),
            ],
            check=True,
            capture_output=True,
        )
        self.ssh_key = self._call(
            "POST",
            "/ssh-keys",
            {"name": self.label, "ssh_key": kp.with_suffix(".pub").read_text().strip()},
        )["ssh_key"]["id"]
        cidr = ipaddress.ip_network(
            os.environ.get("VULTR_ALLOW_CIDR") or _public_ip() + "/32"
        )
        self.firewall = self._call("POST", "/firewalls", {"description": self.label})[
            "firewall_group"
        ]["id"]
        self._call(
            "POST",
            f"/firewalls/{self.firewall}/rules",
            {
                "ip_type": "v4",
                "protocol": "tcp",
                "subnet": str(cidr.network_address),
                "subnet_size": cidr.prefixlen,
                "port": "22",
                "notes": "the app only",
            },
        )
        body = {
            "region": self.region,
            "plan": self.plan,
            "label": self.label,
            "hostname": self.label,
            "sshkey_id": [self.ssh_key],
            "firewall_group_id": self.firewall,
            "backups": "disabled",
            "tags": ["arena-sbx"],
        }
        if self.snapshot:
            body["snapshot_id"] = self.snapshot
        else:
            body["os_id"] = self._os_id()
            body["user_data"] = base64.b64encode(CLOUD_INIT.encode()).decode()
        with self._lock:
            if self._closed:
                raise BoxKilled("box closed before its VM was created")
            self.instance = self._call("POST", "/instances", body)["instance"]["id"]
        self.hourly = self._hourly()
        self._emit(
            {
                "type": "vm_create",
                "backend": "vultr",
                "instance": self.instance,
                "region": self.region,
                "plan": self.plan,
                "hourly_usd": self.hourly,
                "firewall": f"inbound 22/tcp from {cidr} only",
            }
        )
        ip = self._wait_active()
        ssh = [
            "ssh",
            "-i",
            str(kp),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.tmp / 'known_hosts'}",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "LogLevel=ERROR",
            f"root@{ip}",
        ]
        self._wait_ssh(ssh)
        self._emit(
            {
                "type": "vm_booting",
                "instance": self.instance,
                "ip": ip,
                "step": "installing Docker, building the sandbox image",
            }
        )
        ctx = subprocess.run(
            [
                "tar",
                "-C",
                str(ROOT),
                "--exclude",
                "__pycache__",
                "-cf",
                "-",
                "sandbox",
                "hostile/server.py",
            ],
            check=True,
            capture_output=True,
        ).stdout
        r = subprocess.run(
            [*ssh, REMOTE_SETUP.format(image=shlex.quote(IMAGE))],
            input=ctx,
            capture_output=True,
            timeout=900,
            check=False,
        )
        check_killed()
        if r.returncode != 0:
            raise RuntimeError(
                f"VM setup failed: {r.stderr.decode(errors='replace')[-300:]}"
            )
        self.docker = DockerRunner(
            ssh=ssh, host=f"vultr {self.instance[:8]} · {self.region}"
        )
        self._emit(
            {
                "type": "vm_ready",
                "instance": self.instance,
                "ip": ip,
                "docker": r.stdout.decode(errors="replace").strip()[-40:],
                "seconds": round(time.time() - self._t0, 1),
            }
        )

    def _wait_active(self) -> str:
        deadline = time.time() + int(os.environ.get("VULTR_BOOT_SECONDS", "600"))
        while time.time() < deadline:
            check_killed()
            i = self._call("GET", f"/instances/{self.instance}")["instance"]
            if i["status"] == "active" and i["main_ip"] not in ("", "0.0.0.0"):
                return i["main_ip"]
            time.sleep(5)
        raise TimeoutError("Vultr instance did not become active in time")

    def _wait_ssh(self, ssh: list[str]) -> None:
        deadline = time.time() + 300
        while time.time() < deadline:
            check_killed()
            if (
                subprocess.run(
                    [*ssh, "true"], capture_output=True, timeout=30, check=False
                ).returncode
                == 0
            ):
                return
            time.sleep(5)
        raise TimeoutError("SSH to the Vultr instance never came up")

    def _os_id(self) -> int:
        if os.environ.get("VULTR_OS_ID"):
            return int(os.environ["VULTR_OS_ID"])
        oses = self._call("GET", "/os", params={"per_page": 500})["os"]
        ubuntu = [
            o
            for o in oses
            if o["family"] == "ubuntu" and o["arch"] == "x64" and "LTS" in o["name"]
        ]
        return max(ubuntu, key=lambda o: o["name"])["id"]

    def _hourly(self) -> float | None:
        try:
            plans = self._call("GET", "/plans", params={"per_page": 500})["plans"]
            monthly = next(p["monthly_cost"] for p in plans if p["id"] == self.plan)
            return round(
                monthly / 672, 4
            )  # Vultr bills hourly, capped at 672 hours a month
        except (StopIteration, httpx.HTTPError, KeyError):
            return None

    def _call(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        r = self.api.request(method, path, json=body, params=params)
        if r.status_code >= 400:
            raise RuntimeError(
                f"Vultr {method} {path} -> {r.status_code}: {r.text[:200]}"
            )
        return r.json() if r.content else {}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self.instance:
            try:
                self._call("DELETE", f"/instances/{self.instance}")
            except (RuntimeError, httpx.HTTPError) as ex:
                self._emit(
                    {
                        "type": "warn",
                        "message": f"could not delete VM {self.instance}: {ex}"[:200],
                    }
                )
            else:
                alive = time.time() - self._t0
                self._emit(
                    {
                        "type": "vm_destroyed",
                        "instance": self.instance,
                        "seconds": round(alive, 1),
                        "cost_usd": round(max(alive / 3600, 1) * self.hourly, 4)
                        if self.hourly
                        else None,
                    }
                )
        if (
            self.firewall
        ):  # stays attached until the instance is gone, so retry for a bit
            for _ in range(20):
                try:
                    self._call("DELETE", f"/firewalls/{self.firewall}")
                    break
                except (RuntimeError, httpx.HTTPError):
                    time.sleep(3)
        if self.ssh_key:
            try:
                self._call("DELETE", f"/ssh-keys/{self.ssh_key}")
            except (RuntimeError, httpx.HTTPError):
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)


def _public_ip() -> str:
    return str(
        ipaddress.ip_address(
            httpx.get("https://api.ipify.org", timeout=10).text.strip()
        )
    )


def _backend() -> SandboxRunner:
    return {"docker": DockerRunner, "vultr": VultrRunner}[
        os.environ.get("SANDBOX_BACKEND", "docker")
    ]()


def get_runner() -> SandboxRunner:
    """The current box's runner inside a run, or a fresh one for scripts that run a single job."""
    b = _box.get()
    return b.runner if b is not None else _backend()


def _json_lines(stream) -> Iterator[dict]:
    for line in stream:
        line = line.strip()
        if not line.startswith("{") or len(line) > 20_000:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and isinstance(ev.get("type"), str):
            yield {
                k: v
                for k, v in ev.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }


def validate_output(scratch: Path, out_dir: Path) -> dict:
    """Trust nothing from the sandbox: check the manifest's shape, copy only expected files, cap total size."""
    mf = scratch / "manifest.json"
    if not mf.exists():
        return {"clips": []}
    raw = json.loads(mf.read_text()[:2_000_000])
    clips, total = [], 0
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    for c in raw.get("clips", [])[:32]:
        cid = str(c.get("id", ""))
        src = scratch / "clips" / f"{cid}.mp4"
        if not CLIP_ID.match(cid) or not src.is_file():
            continue
        total += src.stat().st_size
        if total > MAX_OUT_BYTES:
            break
        shutil.copyfile(src, out_dir / "clips" / f"{cid}.mp4")
        frames = []
        for fr in c.get("frames", [])[:12]:
            name = str(fr.get("file", ""))
            if (
                re.fullmatch(rf"{cid}_\d{{1,2}}\.jpg", name)
                and (scratch / "frames" / name).is_file()
            ):
                shutil.copyfile(scratch / "frames" / name, out_dir / "frames" / name)
                frames.append({"file": name, "t": float(fr.get("t", 0))})
        clean = {
            k: str(c.get(k, ""))[:300]
            for k in ("title", "source", "licence", "author", "page")
        }
        clips.append(
            {
                "id": cid,
                **clean,
                "seconds": float(c.get("seconds", 0)),
                "frames": frames,
            }
        )
    manifest = {"clips": clips}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest

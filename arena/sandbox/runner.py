"""Sandbox runner: one interface, two backends.

`SandboxRunner.run(job, out_dir)` starts a fresh, throwaway sandbox, feeds it one JSON job (no secrets), streams
its JSON-line events back, collects /out into out_dir, and destroys the sandbox. The app never parses a web page
or decodes a downloaded file itself; it only reads the sandbox's manifest, re-encoded clips and stills, and it
validates those before use (see validate_output).

Backends (chosen by SANDBOX_BACKEND):
  docker  a local container: read-only root, unprivileged user, all capabilities dropped, no-new-privileges,
          CPU / memory / pids caps, tmpfs scratch, a wall-clock kill, `--rm`
  vultr   a throwaway Vultr instance per job (see VultrRunner); config only, same job and event format
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

IMAGE = os.environ.get("SANDBOX_IMAGE", "arena-scraper:0.1")
CAPS = {
    "cpus": os.environ.get("SANDBOX_CPUS", "1.0"),
    "memory": os.environ.get("SANDBOX_MEMORY", "512m"),
    "pids": int(os.environ.get("SANDBOX_PIDS", "128")),
    "seconds": int(os.environ.get("SANDBOX_SECONDS", "300")),
}
MAX_OUT_BYTES = 400_000_000
CLIP_ID = re.compile(r"^[0-9a-f]{10}$")


class SandboxRunner:
    name = "base"

    def run(self, job: dict, out_dir: Path, on_event: Callable[[dict], None]) -> dict:
        raise NotImplementedError


class DockerRunner(SandboxRunner):
    name = "docker"

    def run(self, job, out_dir, on_event):
        out_dir.mkdir(parents=True, exist_ok=True)
        cname = f"arena-sbx-{uuid.uuid4().hex[:8]}"
        scratch = Path(tempfile.mkdtemp(prefix="arena-out-"))
        os.chmod(scratch, 0o777)
        cmd = [
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
            f"{scratch}:/out:rw",
            IMAGE,
        ]
        job = {
            **job,
            "caps": CAPS,
            "time_budget_s": min(job.get("time_budget_s", 240), CAPS["seconds"] - 20),
        }
        t0 = time.time()
        on_event(
            {
                "type": "sandbox_start",
                "backend": self.name,
                "container": cname,
                "caps": CAPS,
            }
        )
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        killer = threading.Timer(
            CAPS["seconds"],
            lambda: subprocess.run(["docker", "kill", cname], capture_output=True),
        )
        killer.start()
        try:
            p.stdin.write(json.dumps(job))
            p.stdin.close()
            for ev in _json_lines(p.stdout):
                on_event(ev)
            p.wait()
        finally:
            killer.cancel()
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        manifest = validate_output(scratch, out_dir)
        shutil.rmtree(scratch, ignore_errors=True)
        on_event(
            {
                "type": "sandbox_destroyed",
                "container": cname,
                "seconds": round(time.time() - t0, 1),
                "exit": p.returncode,
                "stderr_tail": (p.stderr.read() or "")[-300:],
            }
        )
        return manifest


class VultrRunner(SandboxRunner):
    """Throwaway Vultr instance per job. Not wired to an account in this prototype; this is the exact recipe.

    1. POST https://api.vultr.com/v2/instances  (Authorization: Bearer $VULTR_API_KEY, held by the app, never sent
       to the instance) with {"region": $VULTR_REGION, "plan": $VULTR_PLAN (a small CPU plan such as vc2-1c-1gb),
       "os_id": <Ubuntu LTS>, "label": "arena-sbx-<id>", "user_data": base64(cloud-init)}. The cloud-init installs
       Docker, pulls arena-scraper:<tag> and runs it with the SAME flags as DockerRunner, reading the job from a
       one-time URL and writing /out to a tarball.
    2. Poll GET /v2/instances/{id} until "active"; stream events from the instance over the tailnet (NetBird) or
       an SSH tunnel; fetch the tarball.
    3. DELETE /v2/instances/{id} in a finally block, whatever happened. Firewall group: egress only, no inbound
       except the app's NetBird peer.
    The returned manifest goes through the same validate_output as the Docker backend.
    """

    name = "vultr"

    def run(self, job, out_dir, on_event):
        raise NotImplementedError(
            "set SANDBOX_BACKEND=docker locally; the Vultr backend is built at the event"
        )


def get_runner() -> SandboxRunner:
    return {"docker": DockerRunner, "vultr": VultrRunner}[
        os.environ.get("SANDBOX_BACKEND", "docker")
    ]()


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

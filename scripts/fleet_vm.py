"""The big Vultr machine for the fleet: 192 vCPUs that segment and simulate every run at once.

  VULTR_API_KEY=... python scripts/fleet_vm.py up   [--plan vx1-g-192c-768g --region atl]   -> prints HEAVY_SSH json
  VULTR_API_KEY=... python scripts/fleet_vm.py down                                            -> deletes everything

State (instance id, firewall, key id, a private SSH key made for this machine) lives in STATE_DIR, never in the repo.
The API key is only read from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.sandbox import runner as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.environ.get("FLEET_STATE", "/tmp/replay-fleet"))
SETUP = (
    "export DEBIAN_FRONTEND=noninteractive; cloud-init status --wait >/dev/null 2>&1; "
    "sed -i 's/^#\\?MaxSessions.*/MaxSessions 300/; s/^#\\?MaxStartups.*/MaxStartups 300:30:600/' /etc/ssh/sshd_config && "
    "(systemctl reload ssh || systemctl reload sshd) && "
    "apt-get -qq update && apt-get -qq install -y python3-venv libosmesa6 libgl1 libglib2.0-0 ffmpeg >/dev/null && "
    "python3 -m venv /opt/venv && /opt/venv/bin/pip -q install 'mediapipe==0.10.14' 'numpy<2' mujoco==3.14.0 pillow "
    "chess==1.11.2 && mkdir -p /opt/jobs"
)


def up(plan: str, region: str) -> None:
    os.environ["VULTR_PLAN"], os.environ["VULTR_REGION"] = plan, region
    os.environ.setdefault("VULTR_ALLOW_CIDR", "0.0.0.0/0")
    v = R.VultrRunner()
    v._emit = lambda e: print(json.dumps(e), flush=True)
    v._provision()
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = STATE_DIR / "id_ed25519"
    shutil.copyfile(v.tmp / "id_ed25519", key)
    key.chmod(0o600)
    ssh = [a if a != str(v.tmp / "id_ed25519") else str(key) for a in v.docker.ssh]
    ssh = [
        a.replace(str(v.tmp / "known_hosts"), str(STATE_DIR / "known_hosts"))
        for a in ssh
    ]
    shutil.copyfile(v.tmp / "known_hosts", STATE_DIR / "known_hosts")
    mux = [
        "-o",
        "ControlMaster=auto",
        "-o",
        f"ControlPath={STATE_DIR}/cm-%C",
        "-o",
        "ControlPersist=900",
    ]
    ssh = [ssh[0], *mux, *ssh[1:]]
    print("setup: sshd limits, python, MediaPipe, MuJoCo, OSMesa", flush=True)
    subprocess.run([*ssh, SETUP], check=True)
    ctx = subprocess.run(
        [
            "tar",
            "-C",
            str(ROOT),
            "--exclude",
            "__pycache__",
            "-cf",
            "-",
            "arena",
            "vendor/so101",
        ],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(
        [*ssh, "mkdir -p /opt/replay && tar -C /opt/replay -xf -"],
        input=ctx,
        check=True,
    )
    check = subprocess.run(
        [
            *ssh,
            "nproc && /opt/venv/bin/python -c 'import mediapipe, mujoco; print(mediapipe.__version__, mujoco.__version__)'",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    state = {
        "instance": v.instance,
        "firewall": v.firewall,
        "ssh_key": v.ssh_key,
        "plan": plan,
        "region": region,
        "hourly_usd": v.hourly,
        "ssh": ssh,
    }
    (STATE_DIR / "state.json").write_text(json.dumps(state, indent=1))
    v._closed = True  # keep the VM: `down` deletes it
    print(
        f"ready: {check[0]} cpus, mediapipe {check[1]}, mujoco {check[2]}", flush=True
    )
    print("HEAVY_SSH=" + json.dumps(ssh))


def down() -> None:
    state = json.loads((STATE_DIR / "state.json").read_text())
    v = R.VultrRunner()
    v.instance, v.firewall, v.ssh_key = (
        state["instance"],
        state["firewall"],
        state["ssh_key"],
    )
    v._emit = lambda e: print(json.dumps(e), flush=True)
    v._t0 = (STATE_DIR / "state.json").stat().st_mtime
    v.hourly = state.get("hourly_usd")
    v.close()
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    print("deleted", state["instance"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "down"])
    ap.add_argument("--plan", default="vx1-g-192c-768g")
    ap.add_argument("--region", default="atl")
    a = ap.parse_args()
    up(a.plan, a.region) if a.cmd == "up" else down()

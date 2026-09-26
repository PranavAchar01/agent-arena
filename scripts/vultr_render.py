"""Rehearse and film a whole chess game on a throwaway Vultr VM, then bring back only the video and move log.

  VULTR_API_KEY=... python scripts/vultr_render.py [--plan vc2-4c-8gb] [-- chess_render.py args]

The VM gets the same per-box SSH key and firewall as the sandbox backend and is deleted in `finally`. MuJoCo renders
headless with OSMesa (CPU), so no GPU plan is needed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.sandbox import runner as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETUP = (
    "export DEBIAN_FRONTEND=noninteractive; cloud-init status --wait >/dev/null 2>&1; "
    "apt-get -qq update && apt-get -qq install -y python3-venv libosmesa6 libgl1 ffmpeg >/dev/null && "
    "python3 -m venv /opt/venv && /opt/venv/bin/pip -q install mujoco==3.14.0 chess==1.11.2 numpy pillow"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="vc2-4c-8gb")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/chess/deepblue-g6")
    ap.add_argument("render_args", nargs="*")
    a = ap.parse_args()
    os.environ["VULTR_PLAN"] = a.plan
    os.environ.setdefault("VULTR_ALLOW_CIDR", "0.0.0.0/0")
    v = R.VultrRunner()
    try:
        v._emit = lambda e: print(e, flush=True)
        v._provision()
        ssh = v.docker.ssh
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
                "vendor/chess_set/mujoco",
                "scripts/chess_render.py",
            ],
            check=True,
            capture_output=True,
        ).stdout
        print("setup: python, MuJoCo, OSMesa", flush=True)
        subprocess.run([*ssh, SETUP], check=True)
        subprocess.run(
            [*ssh, "mkdir -p /opt/game && tar -C /opt/game -xf -"],
            input=ctx,
            check=True,
        )
        cmd = (
            "cd /opt/game && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa /opt/venv/bin/python scripts/chess_render.py "
            f"--out /opt/out {' '.join(shlex.quote(x) for x in a.render_args)}"
        )
        rc = subprocess.run([*ssh, cmd]).returncode
        a.out.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory() as tmp
        ):  # fetch whatever exists, even after a failure
            R._fetch_tar([*ssh, "tar -C /opt/out -cf - ."], Path(tmp))
            for name in ("game.mp4", "moves.json", "poster.jpg"):
                src = Path(tmp) / name
                if src.is_file():
                    src.replace(a.out / name)
                    print("fetched", a.out / name)
        if rc:
            raise SystemExit(f"render exited {rc}")
    finally:
        v.close()


if __name__ == "__main__":
    main()

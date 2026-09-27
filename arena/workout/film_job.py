"""Run and film one robot copying one person. Used in-process, or on the big Vultr VM over SSH:

  python arena/workout/film_job.py IN.json OUT.mp4 RESULT.json

IN.json: {"t": [...], "elbow": [...human elbow angle, degrees...], "shoulder": [...elevation, degrees...] or null}
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena.workout import robot  # noqa: E402

W, H, FPS = 640, 360, 25


def film(
    elbow: np.ndarray, t: np.ndarray, out: Path, shoulder: np.ndarray | None = None
) -> dict:
    frames: list[np.ndarray] = []
    holder = {}
    orig = robot.CurlScene.__init__

    def init(self):
        orig(self)
        holder["sc"] = self
        holder["r"] = mujoco.Renderer(self.model, H, W)

    robot.CurlScene.__init__ = init

    def frame():  # called every 2 control steps (25 fps)
        holder["r"].update_scene(holder["sc"].data, camera="side")
        frames.append(holder["r"].render())

    try:
        res = robot.perform(elbow, t, frame, shoulder_deg=shoulder)
    finally:
        robot.CurlScene.__init__ = orig
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
    return res


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text())
    sh = np.array(job["shoulder"], float) if job.get("shoulder") is not None else None
    r = film(
        np.array(job["elbow"], float), np.array(job["t"], float), Path(sys.argv[2]), sh
    )
    Path(sys.argv[3]).write_text(json.dumps(r))

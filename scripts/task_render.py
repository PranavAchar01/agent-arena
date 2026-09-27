"""Rehearse and film one of the new tabletop tasks (arena/tasks/catalog.py).

python scripts/task_render.py hanoi|cups [--speed 3]   -> runs/task-<name>/{robot.mp4, moves.json, run.json, poster.jpg}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.tasks.catalog import TASKS  # noqa: E402
from arena.tasks.engine import TaskMover, TaskScene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--speed", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / "runs" / f"task-{a.task}"
    out.mkdir(parents=True, exist_ok=True)
    objs, statics, moves = TASKS[a.task]()
    sc = TaskScene(objs, statics)
    mv = TaskMover(sc)
    for _ in range(80):
        sc.step(sc.data.ctrl.copy())
    r = mujoco.Renderer(sc.model, H, W)
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
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / "robot.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    n = [0]
    last = [None]

    def grab():
        r.update_scene(sc.data, camera="judge")
        last[0] = r.render()
        ff.stdin.write(last[0].tobytes())

    def frame():  # called every 2 control steps, i.e. 25 times per simulated second
        n[0] += 1
        if n[0] % a.speed == 0:
            grab()

    for _ in range(FPS):
        grab()
    log, t0 = [], time.time()
    for m in moves:
        start = n[0] / a.speed / FPS
        res = mv.execute(m, frame=frame)
        if not res["ok"]:
            ff.stdin.close()
            ff.wait()
            raise SystemExit(f"{m.label}: failed every rehearsal")
        log.append({**{k: v for k, v in res.items()}, "start_s": round(start, 2)})
        print(m.label, res.get("err_mm"), "mm", flush=True)
    for _ in range(int(FPS * 1.5)):
        ff.stdin.write(last[0].tobytes())
    ff.stdin.close()
    ff.wait()
    from PIL import Image

    Image.fromarray(last[0]).save(out / "poster.jpg", quality=90)
    (out / "moves.json").write_text(json.dumps(log, indent=1, default=float))
    (ROOT / "web" / "tasks").mkdir(exist_ok=True)
    (ROOT / "web" / "tasks" / f"task-{a.task}.json").write_text((out / "moves.json").read_text())  # the local page reads it statically
    (out / "run.json").write_text(
        json.dumps(
            {
                "task": a.task,
                "moves": len(log),
                "speed": a.speed,
                "sim_s": round(n[0] / FPS, 1),
                "compute_s": round(time.time() - t0, 1),
            },
            indent=1,
        )
    )
    print("done", out / "robot.mp4")


if __name__ == "__main__":
    main()

"""From MediaPipe hand numbers to a normalised MotionShape (see arena/sim/retarget.py for what that holds and why).

Segmentation uses the hand's own kinematics: the main move is the fastest stretch of the pinch point (midpoint of
thumb tip and index tip); it starts and ends where the speed falls below a quarter of its peak. Everything is in
units of the hand's own size (wrist to middle knuckle), so the camera's distance does not matter.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .sim.retarget import MotionShape

ROOT = Path(__file__).resolve().parents[1]
POSE_PYTHON = os.environ.get(
    "POSE_PYTHON", str(Path.home() / "helloworld/understudy/.venv-pose/bin/python")
)
MODEL = ROOT / "models" / "hand_landmarker.task"


def track_hands(clip: Path, window, out: Path) -> dict:
    subprocess.run(
        [
            "nice",
            "-n",
            "19",
            POSE_PYTHON,
            str(ROOT / "arena" / "pose_worker.py"),
            str(clip),
            str(window[0]),
            str(window[1]),
            str(out),
            str(MODEL),
        ],
        check=True,
        capture_output=True,
        timeout=600,
    )
    return json.loads(out.read_text())


def _smooth(x, k=5):
    if len(x) < k:
        return x
    ker = np.ones(k) / k
    pad = np.pad(x, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(pad[:, j], ker, mode="valid") for j in range(x.shape[1])], 1
    )


def _tracks(pose: dict):
    """Greedy nearest-neighbour hand tracks (pinch point and hand size in pixels)."""
    w, h = pose["width"], pose["height"]
    tracks: list[list] = []
    for f in pose["frames"]:
        for hd in f["hands"]:
            pinch = (np.array(hd["thumb"]) + np.array(hd["index"])) / 2 * [w, h]
            size = np.linalg.norm((np.array(hd["wrist"]) - np.array(hd["mcp"])) * [w, h])
            best, bd = None, 1e9
            for tr in tracks:
                t_last, p_last, _ = tr[-1]
                d = np.linalg.norm(p_last - pinch)
                if f["t"] - t_last < 0.4 and t_last != f["t"] and d < max(3 * size, 40) and d < bd:
                    best, bd = tr, d
            (best.append if best is not None else lambda x: tracks.append([x]))((f["t"], pinch, size))
    return [tr for tr in tracks if len(tr) >= 10]


def _one_move(t, p, a, b, family, source):
    chord = p[b] - p[a]
    L = float(np.linalg.norm(chord))
    dur = float(t[b] - t[a])
    q = {"t0": round(float(t[a]), 2), "move_hand_units": round(L, 2), "move_seconds": round(dur, 2)}
    if L < 1.0:
        return None, {**q, "reason": f"move is only {L:.1f} hand-lengths long"}
    if not 0.25 <= dur <= 5.0:
        return None, {**q, "reason": f"move lasts {dur:.2f} s (need 0.25 to 5 s)"}
    if b - a < 5:
        return None, {**q, "reason": "move spans too few tracked frames"}
    seg = p[a:b + 1] - p[a]
    e = chord / L
    n = np.array([-e[1], e[0]])
    if n[1] > 0:  # image y points down; lift is toward the top of the frame
        n = -n
    u = np.maximum.accumulate(np.clip(seg @ e / L, 0, None))
    if u[-1] < 0.8:
        return None, {**q, "reason": "hand does not travel steadily from start to end"}
    u = np.clip(u / u[-1], 0, 1)
    perp = seg @ n  # hand units
    if perp.min() < -1.0:
        return None, {**q, "reason": "path dips far below its own start and end"}
    tau = (t[a:b + 1] - t[a]) / dur
    lift = perp if family != "push" else np.zeros_like(perp)
    side = np.clip(perp, -0.7, 0.7) if family == "push" else np.zeros_like(perp)
    q.update(max_lift=round(float(perp.max()), 2))
    k = np.linspace(0, 1, 50)
    shape = MotionShape(family, k, np.interp(k, tau, u), np.interp(k, tau, lift), np.interp(k, tau, side), dur,
                        f"{source}@{t[a]:.1f}s")
    return shape, q


def shapes_from_pose(pose: dict, family: str, source: str, max_moves: int = 6):
    """Every clean hand move in the clip. Returns (list of (shape, quality), list of rejected-move reasons)."""
    frames = len(pose["frames"])
    det = sum(1 for f in pose["frames"] if f["hands"]) / max(1, frames)
    tracks = _tracks(pose)
    if not tracks:
        return [], [{"reason": f"no steady hand track (hand seen in {det:.0%} of frames)"}]
    good, bad = [], []
    for tr in tracks:
        t = np.array([x[0] for x in tr])
        p = _smooth(np.array([x[1] for x in tr]) / float(np.median([x[2] for x in tr])))
        speed = np.linalg.norm(np.gradient(p, axis=0), axis=1) / np.maximum(np.gradient(t), 1e-3)
        used = np.zeros(len(t), bool)
        for pk in np.argsort(-speed):
            if used[pk] or speed[pk] < 1.5:  # below 1.5 hand-lengths per second is not a deliberate move
                continue
            thr = 0.25 * speed[pk]
            a = pk
            while a > 0 and speed[a] > thr and not used[a - 1]:
                a -= 1
            b = pk
            while b < len(t) - 1 and speed[b] > thr and not used[b + 1]:
                b += 1
            used[a:b + 1] = True
            shape, q = _one_move(t, p, a, b, family, source)
            (good.append((shape, q)) if shape is not None else bad.append(q))
    good.sort(key=lambda sq: -sq[1]["move_hand_units"])
    return good[:max_moves], bad

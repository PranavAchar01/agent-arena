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


def best_track(pose: dict):
    """Follow the hand that moves the most. Returns times, pinch points (px), hand size (px), detection rate."""
    w, h = pose["width"], pose["height"]
    frames = pose["frames"]
    tracks: list[list] = []
    for f in frames:
        for hd in f["hands"]:
            pinch = (np.array(hd["thumb"]) + np.array(hd["index"])) / 2 * [w, h]
            size = np.linalg.norm(
                (np.array(hd["wrist"]) - np.array(hd["mcp"])) * [w, h]
            )
            best, bd = None, 1e9
            for tr in tracks:
                t_last, p_last, _ = tr[-1]
                d = np.linalg.norm(p_last - pinch)
                if f["t"] - t_last < 0.5 and d < max(3 * size, 40) and d < bd:
                    best, bd = tr, d
            if best is not None and best[-1][0] != f["t"]:
                best.append((f["t"], pinch, size))
            elif best is None:
                tracks.append([(f["t"], pinch, size)])
    if not tracks:
        return None
    tr = max(
        tracks,
        key=lambda tr: len(tr) * np.ptp(np.array([p for _, p, _ in tr]), 0).sum(),
    )
    t = np.array([a for a, _, _ in tr])
    p = np.array([b for _, b, _ in tr])
    s = float(np.median([c for _, _, c in tr]))
    return t, p, s, len(tr) / max(1, len(frames))


def shape_from_pose(pose: dict, family: str, source: str):
    """Returns (MotionShape or None, quality dict with the reason when None)."""
    got = best_track(pose)
    if got is None:
        return None, {"reason": "no hand found by MediaPipe"}
    t, p, size, det = got
    q = {"detected": round(det, 2), "frames": len(t)}
    if len(t) < 8 or det < 0.5:
        return None, {**q, "reason": f"hand tracked in only {det:.0%} of frames"}
    p = _smooth(p / size)  # hand-size units
    dt = np.gradient(t)
    speed = np.linalg.norm(np.gradient(p, axis=0), axis=1) / np.maximum(dt, 1e-3)
    pk = int(np.argmax(speed))
    thr = 0.25 * speed[pk]
    a = pk
    while a > 0 and speed[a] > thr:
        a -= 1
    b = pk
    while b < len(t) - 1 and speed[b] > thr:
        b += 1
    chord = p[b] - p[a]
    L = float(np.linalg.norm(chord))
    dur = float(t[b] - t[a])
    q.update(
        move_hand_units=round(L, 2),
        move_seconds=round(dur, 2),
        peak_speed=round(float(speed[pk]), 2),
    )
    if L < 1.0:
        return None, {**q, "reason": f"main move is only {L:.1f} hand-lengths long"}
    if not 0.25 <= dur <= 5.0 or b - a < 5:
        return None, {**q, "reason": f"main move lasts {dur:.2f} s (need 0.25 to 5 s)"}
    seg = p[a : b + 1] - p[a]
    e = chord / L
    n = np.array([-e[1], e[0]])
    if n[1] > 0:  # image y points down; lift is toward the top of the frame
        n = -n
    u = np.maximum.accumulate(np.clip(seg @ e / L, 0, None))
    if u[-1] < 0.8:
        return None, {**q, "reason": "hand does not travel steadily from start to end"}
    u = np.clip(u / u[-1], 0, 1)
    perp = seg @ n  # hand units
    tau = (t[a : b + 1] - t[a]) / dur
    lift = perp if family != "push" else np.zeros_like(perp)
    side = np.clip(perp, -0.7, 0.7) if family == "push" else np.zeros_like(perp)
    q.update(
        max_lift=round(float(perp.max()), 2),
        jitter=round(float(np.std(np.diff(seg, 2, axis=0))), 3),
    )
    k = np.linspace(0, 1, 50)
    return MotionShape(
        family,
        k,
        np.interp(k, tau, u),
        np.interp(k, tau, lift),
        np.interp(k, tau, side),
        dur,
        source,
    ), q

"""From MediaPipe body landmarks to a clean elbow-angle trace and counted curl reps.

A curl is one joint: the elbow. Per frame the angle at the elbow between the upper arm (shoulder->elbow) and the
forearm (elbow->wrist) is measured on whichever arm is more visible, gaps are filled, the trace is smoothed, and a
rep is a full cycle from extended (large angle) to flexed (small angle) and back, with at least MIN_RANGE degrees of
travel. Everything is in degrees of the human elbow; the robot plays the same degrees on its own elbow (1:1).
"""

from __future__ import annotations

import numpy as np

# indices into the 13 kept joints (see pose_body.py): 0 nose, 1/2 shoulders, 3/4 elbows, 5/6 wrists, 7/8 hips ...
ARMS = {"left": (1, 3, 5), "right": (2, 4, 6)}
MIN_VIS = 0.5
MIN_RANGE = 45.0  # degrees between extended and flexed for a movement to count as a rep


def elbow_angles(
    frames: list[dict], aspect: float
) -> tuple[np.ndarray, np.ndarray, str]:
    """Returns (t, angle_deg with NaN gaps, side). `aspect` = width / height, so angles are measured in pixels."""
    t = np.array([f["t"] for f in frames], float)
    best, side = None, "right"
    for name, (s, e, w) in ARMS.items():
        ang = np.full(len(frames), np.nan)
        for k, f in enumerate(frames):
            lm = f.get("lm")
            if not lm or min(lm[s][2], lm[e][2], lm[w][2]) < MIN_VIS:
                continue
            p = [np.array([lm[j][0] * aspect, lm[j][1]]) for j in (s, e, w)]
            u, v = p[0] - p[1], p[2] - p[1]
            c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
            ang[k] = np.degrees(np.arccos(np.clip(c, -1, 1)))
        score = np.isfinite(ang).sum() + (
            np.nanmax(ang) - np.nanmin(ang) if np.isfinite(ang).any() else 0
        )
        if best is None or score > best[0]:
            best, side = (score, ang), name
    return t, best[1], side


def smooth(ang: np.ndarray, k: int = 5) -> np.ndarray:
    a = ang.copy()
    idx = np.arange(len(a))
    ok = np.isfinite(a)
    if ok.sum() < 4:
        return a
    a[~ok] = np.interp(idx[~ok], idx[ok], a[ok])
    pad = np.pad(a, (k, k), mode="edge")
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]


def reps(t: np.ndarray, a: np.ndarray) -> list[dict]:
    """Extended -> flexed -> extended cycles with enough range, found with a hysteresis threshold."""
    if len(a) < 10 or not np.isfinite(a).all():
        return []
    hi, lo = np.percentile(a, 90), np.percentile(a, 10)
    if hi - lo < MIN_RANGE:
        return []
    up, down = lo + 0.35 * (hi - lo), lo + 0.65 * (hi - lo)
    out, state, start, bottom = [], "ext", None, None
    for k in range(len(a)):
        if state == "ext" and a[k] < up:
            state, bottom = "flex", k
            start = start if start is not None else max(0, k - 1)
        elif state == "flex":
            if a[k] < a[bottom]:
                bottom = k
            if a[k] > down:
                j0 = start
                out.append(
                    {
                        "start": float(t[j0]),
                        "bottom": float(t[bottom]),
                        "end": float(t[k]),
                        "min_deg": round(float(a[bottom]), 1),
                        "max_deg": round(float(max(a[j0], a[k])), 1),
                    }
                )
                state, start = "ext", k
    return [
        r
        for r in out
        if r["max_deg"] - r["min_deg"] >= MIN_RANGE
        and 0.6 <= r["end"] - r["start"] <= 8
    ]


def analyse(frames: list[dict], aspect: float) -> dict:
    t, raw, side = elbow_angles(frames, aspect)
    seen = float(np.isfinite(raw).mean()) if len(raw) else 0.0
    a = smooth(raw)
    rs = reps(t, a) if seen > 0.4 else []
    return {
        "side": side,
        "tracked": round(seen, 2),
        "t": t.round(3).tolist(),
        "angle": np.round(a, 1).tolist(),
        "reps": rs,
        "ok": len(rs) >= 2,
    }

"""Where the block and the target can be on the desk (the SO-101's comfortable workspace, jaws pointing down)."""

from __future__ import annotations

import numpy as np

SEP = {"push": (0.08, 0.16), "place": (0.10, 0.18), "stack": (0.08, 0.17)}


def sample(task: str, rng: np.random.Generator):
    lo, hi = SEP[task]
    while True:
        b = np.array([rng.uniform(0.16, 0.24), rng.uniform(-0.10, 0.10)])
        t = np.array([rng.uniform(0.18, 0.27), rng.uniform(-0.10, 0.10)])
        if lo <= np.linalg.norm(t - b) <= hi:
            return tuple(b.round(4)), tuple(t.round(4)), float(rng.uniform(-0.5, 0.5))


PROBES = {
    task: [sample(task, np.random.default_rng(7_000 + i)) for i in range(3)]
    for task in SEP
}

"""Plumbing check only (synthetic shapes, never reported as a result): can the policy learn from ~60 episodes?"""

import sys, time
import numpy as np
from pathlib import Path
from arena.sim.scene import Scene
from arena.sim.retarget import synthetic, replay
from arena.sim.ik import IK
from arena.layouts import sample
from arena.policy import train, evaluate

task = sys.argv[1]
n = int(sys.argv[2])
sc = Scene.make(task)
ik = IK(sc.model)
rng = np.random.default_rng(1)
eps = []
t0 = time.time()
tried = 0
while len(eps) < n:
    b, t, y = sample(task, rng)
    tried += 1
    shp = synthetic(task, lift=rng.uniform(0.9, 1.6), duration=rng.uniform(0.8, 1.6))
    ep = replay(sc, shp, b, t, y, ik=ik)
    if ep.success:
        eps.append((ep.obs, ep.act))
print(f"{len(eps)}/{tried} episodes in {time.time() - t0:.0f}s")
Path("runs/sanity").mkdir(parents=True, exist_ok=True)
info = train(
    eps,
    Path(f"runs/sanity/{task}.pt"),
    iters=int(sys.argv[3]) if len(sys.argv) > 3 else 3000,
)
print(info)
t0 = time.time()
res, _ = evaluate(task, Path(f"runs/sanity/{task}.pt"))
print("eval", sum(res), "/", len(res), f"{time.time() - t0:.0f}s")

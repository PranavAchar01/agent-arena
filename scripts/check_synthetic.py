"""Robot-side check: does a textbook shape succeed on each task? (tests only, never in a dataset)"""

import sys, time
import numpy as np, mujoco
from PIL import Image
from arena.sim.scene import Scene
from arena.sim.retarget import synthetic, replay
from arena.sim.ik import IK

tasks = sys.argv[1:] or ["push", "place", "stack"]
cases = [
    ((0.17, 0.10), (0.24, -0.06)),
    ((0.20, -0.08), (0.22, 0.08)),
    ((0.14, 0.02), (0.26, 0.0)),
]
for task in tasks:
    sc = Scene.make(task)
    ik = IK(sc.model)
    r = mujoco.Renderer(sc.model, 240, 320)

    def rend(s):
        r.update_scene(s.data, camera="front")
        return r.render()

    for i, (bx, tx) in enumerate(cases):
        t0 = time.time()
        ep = replay(
            sc,
            synthetic(task),
            bx,
            tx,
            block_yaw=0.2,
            ik=ik,
            render=rend if i == 0 else None,
        )
        print(
            task,
            i,
            ep.success,
            ep.gates,
            "block",
            sc.block().round(3),
            "target",
            sc.target().round(3),
            f"{time.time() - t0:.1f}s",
        )
        if ep.frames:
            idx = np.linspace(0, len(ep.frames) - 1, 6).astype(int)
            Image.fromarray(np.concatenate([ep.frames[j] for j in idx], 1)).save(
                f"data/check_{task}.png"
            )

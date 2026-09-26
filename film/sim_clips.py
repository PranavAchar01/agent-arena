"""Render the trained SO-101 policy doing A (push), B (place), C (stack) on an unseen evaluation layout.

For each task: the first evaluation layout (same fixed unseen stream as the reported x/20) on which the policy
succeeds is rendered at 1280x720, 25 fps. The success rate itself is shown on screen, so a shown success never
stands in for the rate.
"""

from pathlib import Path

import mujoco

from arena.pipeline import write_mp4
from arena.policy import Runner, eval_layouts
from arena.sim.scene import Scene

OUT = Path("film/work")
OUT.mkdir(parents=True, exist_ok=True)
POLICIES = {
    "push": "runs/push-override/policy.pt",
    "place": "runs/place-seeded/policy.pt",
    "stack": "runs/stack-override/policy.pt",
}

for task, path in POLICIES.items():
    sc = Scene.make(task)
    r = mujoco.Renderer(sc.model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.0, 0.04]
    cam.distance, cam.azimuth, cam.elevation = 0.62, 215, -28

    def render(s):
        r.update_scene(s.data, camera=cam)
        return r.render()

    runner = Runner(Path(path))
    for i, (b, t, y) in enumerate(eval_layouts(task)):
        ok, _ = runner.rollout(sc, b, t, y)
        if ok:
            ok2, frames = runner.rollout(sc, b, t, y, render=render)
            assert ok2
            write_mp4(frames, OUT / f"sim_{task}.mp4", fps=25)
            print(task, "layout", i, "frames", len(frames), flush=True)
            break

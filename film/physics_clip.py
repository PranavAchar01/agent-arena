"""The physics shot: the tuned tower policy on an unseen layout it solves, rendered straight from MuJoCo with contact
points and contact forces drawn in, camera slowly orbiting. Real time (25 fps = every second 50 Hz control step)."""

import json
from pathlib import Path

import mujoco

from arena.pipeline import write_mp4
from arena.policy import Runner, eval_layouts
from arena.sim.scene import Scene

res = json.load(open("runs/box-tower/run.json"))["policy"]["eval"]
i = res.index(True)
sc = Scene.make("tower")
m = sc.model
m.vis.scale.contactwidth, m.vis.scale.contactheight = 0.03, 0.006
m.vis.scale.forcewidth, m.vis.map.force = 0.005, 0.012
r = mujoco.Renderer(m, 720, 1280)
opt = mujoco.MjvOption()
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.19, 0.0, 0.07]
cam.distance, cam.elevation = 0.36, -20
k = [0]


def rend(s):
    cam.azimuth = 185 + 0.12 * k[0]  # slow orbit
    k[0] += 1
    r.update_scene(s.data, camera=cam, scene_option=opt)
    return r.render()


ok, frames = Runner(Path("runs/box-tower/policy.pt")).rollout(
    sc, *eval_layouts("tower")[i], render=rend
)
assert ok
write_mp4(frames, Path("film/work/physics_live.mp4"))
print("layout", i, "frames", len(frames))

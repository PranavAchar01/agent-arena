"""Site background: the real SO-101 scene rendered in MuJoCo, then ordered (Bayer) dithered to a 4-tone palette."""
import mujoco, numpy as np
from PIL import Image
from arena.sim.scene import build, HALF
from arena.sim.ik import IK

m = build("stack"); d = mujoco.MjData(m)
ik = IK(m)
q, _, _ = ik.solve(np.array([0.2, -0.02, 0.09]), np.array([0, -1.2, 1.2, 1.2, 0]), (1, 0))
d.qpos[[m.joint(j).qposadr[0] for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]] = np.append(q, 0.5)
a = m.joint("block_free").qposadr[0]; d.qpos[a:a+3] = [0.2, -0.02, 0.055]
b = m.joint("target_free").qposadr[0]; d.qpos[b:b+3] = [0.24, 0.08, 0.022]
mujoco.mj_forward(m, d)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.2, 0.0, 0.08]; cam.distance = 0.62; cam.azimuth = 205; cam.elevation = -14
W, H = 1280, 720
r = mujoco.Renderer(m, H, W); r.update_scene(d, camera=cam); img = r.render().astype(float) / 255
lum = img @ [0.3, 0.59, 0.11]
lum = np.clip((lum - 0.05) / 0.9, 0, 1) ** 1.3
bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16
S = 2  # dither cell = 2x2 px
small = lum[::S, ::S]
th = np.tile(bayer, (small.shape[0] // 4 + 1, small.shape[1] // 4 + 1))[:small.shape[0], :small.shape[1]]
levels = 4
idx = np.clip(np.floor(small * (levels - 1) + th), 0, levels - 1).astype(int)
yellow = (img[::S, ::S, 0] > 0.55) & (img[::S, ::S, 2] < 0.35)
pal = np.array([[7, 9, 12], [22, 25, 30], [52, 56, 62], [120, 124, 128]])
out = pal[idx]
out[yellow & (idx >= 2)] = [196, 150, 62]
out[yellow & (idx == 1)] = [98, 74, 34]
red = (img[::S, ::S, 0] > 0.5) & (img[::S, ::S, 1] < 0.3) & (img[::S, ::S, 2] < 0.3)
out[red & (idx >= 1)] = [186, 72, 58]
blue = (img[::S, ::S, 2] > 0.5) & (img[::S, ::S, 0] < 0.35)
out[blue & (idx >= 1)] = [82, 120, 196]
im = Image.fromarray(out.astype(np.uint8)).resize((W * 2 // S * S // 2 * 2, H * 2 // S * S // 2 * 2), Image.NEAREST)
im.save("web/media/desk-dither.png", optimize=True)
print(im.size)

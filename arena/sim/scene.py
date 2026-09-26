"""MuJoCo scene: the SO-101 on a desk, with the three desk-tidying tasks.

  push   slide the red block onto the blue taped square
  place  pick the red block up and drop it in the bowl
  stack  pick the red block up and set it on top of the blue block

The arm is the public SO-101 MJCF (TheRobotStudio, Apache-2.0, vendor/so101) loaded with MjSpec; everything else is
added here. Two physics settings matter and were learned the hard way on this model:

* The inline actuator kp (998) saturates torque every step. We use the STS3215 gains from joints_properties.xml
  (kp 17.8) plus gravity compensation on every arm body, so position targets are actually reached.
* The jaw collision meshes are convex hulls that bulge into the gap between the fingers, so a block cannot seat
  between them. For grasping, the jaw meshes stop touching the block and two thin box pads on the inner finger
  faces do the contact. Pad coordinates are in the gripperframe site frame (+x from the housing to the jaw tips,
  +z from the fixed finger to the moving one) and were measured on the public SO-101 meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MJCF = ROOT / "vendor" / "so101" / "so101_new_calib.xml"

TASKS = ("push", "place", "stack")
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
HOME = np.array([0.0, -1.2, 1.2, 1.2, 0.0])
GRIP_CLOSED, GRIP_OPEN = -0.1745, 1.7453

TIMESTEP = 1 / 300
STEPS_PER_CONTROL = 6  # 50 Hz control
STEPS_PER_FRAME = 10  # 30 fps video
DT = TIMESTEP * STEPS_PER_CONTROL
KP = 17.8

HALF = 0.015  # red block: 3 cm cube, 30 g
BASE_HALF = 0.022  # blue block for stacking: 4.4 cm cube, 150 g
TAPE_HALF = 0.035  # 7 cm taped square
BOWL_R, BOWL_WALL_H, BOWL_WALL_T, BOWL_SEG = 0.05, 0.022, 0.006, 20

PAD_X = (-0.042, -0.002)
PAD_HALF_Y = 0.009
FIXED_PAD_Z = (-0.010, -0.003)
MOVING_PAD_Z = (0.017, 0.024)
GRASP_SOLREF = [0.007, 1.0]
GRASP_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
MESH_BIT, BLOCK_BIT = 1, 2

W, H = 640, 480


def _look_at(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    return np.concatenate([r, np.cross(r, f)])


def _base_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MJCF))
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    spec.visual.quality.shadowsize = 2048
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    for a in spec.actuators:
        a.gainprm[0], a.biasprm[1], a.biasprm[2] = KP, -KP, 0.0
    return spec


def _pad_frames():
    """Pad pose in the gripper and moving-jaw body frames, from the gripperframe site at gripper angle 0."""
    m = _base_spec().compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    s = m.site("gripperframe").id
    ps, rs = d.site_xpos[s], d.site_xmat[s].reshape(3, 3)
    out = {}
    for body, zr in (("gripper", FIXED_PAD_Z), ("moving_jaw_so101_v1", MOVING_PAD_Z)):
        b = m.body(body).id
        pb, rb = d.xpos[b], d.xmat[b].reshape(3, 3)
        pos = rb.T @ (ps + rs @ np.array([np.mean(PAD_X), 0.0, np.mean(zr)]) - pb)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rb.T @ rs).flatten())
        out[body] = (
            pos,
            quat,
            [(PAD_X[1] - PAD_X[0]) / 2, PAD_HALF_Y, (zr[1] - zr[0]) / 2],
        )
    return out


def build(task: str) -> mujoco.MjModel:
    assert task in TASKS, task
    pads = _pad_frames()
    spec = _base_spec()
    w = spec.worldbody

    spec.add_texture(
        name="desk_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=[0.93, 0.91, 0.86],
        rgb2=[0.93, 0.91, 0.86],
        mark=mujoco.mjtMark.mjMARK_RANDOM,
        markrgb=[0.85, 0.83, 0.78],
        random=0.02,
        width=512,
        height=512,
    )
    spec.add_material(
        name="desk", textures=["", "desk_tex"], texrepeat=[3, 3], reflectance=0.05
    )
    spec.add_material(name="floor", rgba=[0.16, 0.17, 0.2, 1])
    spec.add_material(name="red", rgba=[0.9, 0.2, 0.15, 1])
    spec.add_material(name="blue", rgba=[0.2, 0.45, 0.95, 1])
    spec.add_material(name="bowl", rgba=[0.96, 0.96, 0.93, 1])

    w.add_light(
        name="key",
        pos=[0.3, -0.5, 1.3],
        dir=[-0.2, 0.4, -1],
        diffuse=[0.75] * 3,
        specular=[0.15] * 3,
        castshadow=True,
    )
    w.add_light(
        name="fill",
        pos=[-0.3, 0.5, 1.1],
        dir=[0.3, -0.4, -1],
        diffuse=[0.35] * 3,
        castshadow=False,
    )
    w.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.1],
        pos=[0, 0, -0.75],
        material="floor",
    )
    w.add_geom(
        name="desk",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.4, 0.5, 0.02],
        pos=[0.2, 0, -0.02],
        material="desk",
        friction=[0.6, 0.005, 0.0001],
    )

    if task == "push":
        t, s, h = TAPE_HALF, 0.005, 0.0005
        target = w.add_body(name="target", mocap=True, pos=[0.25, 0, 0])
        for i, (px, py, sx, sy) in enumerate(
            [(t, 0, s, t + s), (-t, 0, s, t + s), (0, t, t + s, s), (0, -t, t + s, s)]
        ):
            target.add_geom(
                name=f"tape{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, h],
                pos=[px, py, h],
                material="blue",
                contype=0,
                conaffinity=0,
            )
    elif task == "place":
        target = w.add_body(name="target", mocap=True, pos=[0.25, 0, 0])
        target.add_geom(
            name="bowl_base",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[BOWL_R + BOWL_WALL_T, 0.002, 0],
            pos=[0, 0, 0.002],
            material="bowl",
        )
        seg = (BOWL_R + BOWL_WALL_T / 2) * np.tan(np.pi / BOWL_SEG) * 1.05
        for i in range(BOWL_SEG):
            a = 2 * np.pi * i / BOWL_SEG
            r = BOWL_R + BOWL_WALL_T / 2
            target.add_geom(
                name=f"bowl_wall{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[BOWL_WALL_T / 2, seg, BOWL_WALL_H / 2],
                pos=[r * np.cos(a), r * np.sin(a), BOWL_WALL_H / 2],
                quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                material="bowl",
            )
    else:
        target = w.add_body(name="target", pos=[0.25, 0, BASE_HALF])
        target.add_freejoint(name="target_free")
        target.add_geom(
            name="base_block",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[BASE_HALF] * 3,
            mass=0.15,
            material="blue",
            friction=[0.8, 0.005, 0.0001],
            condim=4,
        )

    if task in ("place", "stack"):
        for g in spec.geoms:
            if not g.contype:
                continue
            if g.parent.name in ("gripper", "moving_jaw_so101_v1"):
                g.contype, g.conaffinity = MESH_BIT, MESH_BIT
            else:
                g.conaffinity = MESH_BIT | BLOCK_BIT
        for body, (pos, quat, size) in pads.items():
            b = next(bb for bb in spec.bodies if bb.name == body)
            b.add_geom(
                name=f"pad_{body}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=pos,
                quat=quat,
                friction=[1.0, 0.005, 0.0001],
                condim=4,
                contype=BLOCK_BIT,
                conaffinity=BLOCK_BIT,
                solref=GRASP_SOLREF,
                solimp=GRASP_SOLIMP,
                group=3,
            )

    block = w.add_body(name="block", pos=[0.18, 0.08, HALF])
    block.add_freejoint(name="block_free")
    grasp = task in ("place", "stack")
    block.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[HALF] * 3,
        mass=0.03,
        material="red",
        friction=[0.6, 0.005, 0.0001],
        condim=4,
        **(
            {
                "contype": BLOCK_BIT,
                "conaffinity": BLOCK_BIT,
                "solref": GRASP_SOLREF,
                "solimp": GRASP_SOLIMP,
            }
            if grasp
            else {}
        ),
    )

    cam = [0.58, 0.26, 0.36]
    w.add_camera(
        name="front", pos=cam, xyaxes=_look_at(cam, [0.19, 0.0, 0.05]), fovy=48
    )
    top = [0.2, 0.0, 0.75]
    w.add_camera(name="top", pos=top, xyaxes=[0, -1, 0, 1, 0, 0], fovy=45)
    return spec.compile()


@dataclass
class Scene:
    task: str
    model: mujoco.MjModel
    data: mujoco.MjData

    @classmethod
    def make(cls, task: str) -> "Scene":
        m = build(task)
        return cls(task, m, mujoco.MjData(m))

    @property
    def qadr(self):
        return np.array([self.model.joint(j).qposadr[0] for j in JOINTS])

    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def block(self) -> np.ndarray:
        return self.data.xpos[self.model.body("block").id].copy()

    def target(self) -> np.ndarray:
        return self.data.xpos[self.model.body("target").id].copy()

    def site(self) -> np.ndarray:
        return self.data.site_xpos[self.model.site("gripperframe").id].copy()

    def _set_free(self, joint: str, xyz, yaw: float):
        a = self.model.joint(joint).qposadr[0]
        self.data.qpos[a : a + 3] = xyz
        self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        v = self.model.joint(joint).dofadr[0]
        self.data.qvel[v : v + 6] = 0

    def reset(
        self, block_xy, target_xy, block_yaw: float = 0.0, target_yaw: float = 0.0
    ):
        mujoco.mj_resetData(self.model, self.data)
        q = np.append(HOME, GRIP_CLOSED)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[:] = q
        self._set_free("block_free", [block_xy[0], block_xy[1], HALF], block_yaw)
        if self.task == "stack":
            self._set_free(
                "target_free", [target_xy[0], target_xy[1], BASE_HALF], target_yaw
            )
        else:
            self.model.body_pos[self.model.body("target").id][:2] = target_xy
            self.data.mocap_pos[0][:2] = target_xy
        mujoco.mj_forward(self.model, self.data)

    def obs(self) -> np.ndarray:
        """Policy observation: 6 joint positions, block xyz, target xyz."""
        return np.concatenate([self.q(), self.block(), self.target()])

    def step(self, ctrl: np.ndarray, n: int = STEPS_PER_CONTROL):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, n)

    def success(self) -> bool:
        b, t = self.block(), self.target()
        if self.task == "push":
            return bool(
                np.all(np.abs(b[:2] - t[:2]) < TAPE_HALF) and b[2] < HALF + 0.005
            )
        if self.task == "place":
            return bool(
                np.linalg.norm(b[:2] - t[:2]) < BOWL_R and b[2] < BOWL_WALL_H + HALF
            )
        on_top = abs(b[2] - (t[2] + BASE_HALF + HALF)) < 0.006
        return bool(
            on_top
            and np.linalg.norm(b[:2] - t[:2]) < BASE_HALF
            and t[2] < BASE_HALF + 0.003
        )

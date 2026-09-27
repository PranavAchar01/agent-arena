"""The SO-101 does the human's curl, 1:1, with a dumbbell, in MuJoCo physics.

1. Pick up: the dumbbell lies on the desk; closed-loop IK brings the open jaws round its 12 mm handle, closes and lifts
   (same controller as the chess moves).
2. Curl: joint space. The human's elbow angle (degrees, from MediaPipe) drives the robot's elbow joint one to one:
   when the person's elbow closes by 60 degrees, the robot's elbow closes by 60 degrees. The shoulder and wrist hold
   still, as they should in a strict curl.
3. Set down and let go.

Checks (all must pass): the dumbbell never leaves the grip during the curl (handle within 12 mm of the pads' centre),
the robot's elbow tracks the human angle within 8 degrees RMS, the dumbbell ends back on the desk upright.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from ..chess.scene import _look_at
from ..sim.ik import IK
from ..sim.scene import (
    BLOCK_BIT,
    DT,
    GRASP_SOLIMP,
    GRASP_SOLREF,
    GRIP_CLOSED,
    HOME,
    JOINTS,
    MESH_BIT,
    STEPS_PER_CONTROL,
)
from ..sim.scene import _base_spec, _pad_frames

HANDLE_R, HANDLE_HALF, PLATE_R, PLATE_HALF = 0.006, 0.028, 0.017, 0.005
SPOT = np.array([0.20, 0.0])
GRIP_OPEN = -0.02
REACH, DESCEND, MAX_DQ = 0.15, 0.03, 0.05
CURL_DQ = 0.09  # rad per 20 ms (4.5 rad/s) for joint-space curls: within the STS3215 no-load speed (~4.7 rad/s)
CURL_START = np.radians(
    40.0
)  # robot elbow at the human's "arm straight" (flexion 0); flexion adds up to ~135 deg


def build() -> mujoco.MjModel:
    pads = _pad_frames()
    spec = _base_spec()
    w = spec.worldbody
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.22, 0.24, 0.28], rgb2=[0.03, 0.03, 0.04], width=512, height=512)
    spec.add_material(name="desk", rgba=[0.16, 0.15, 0.15, 1], reflectance=0.12)
    spec.add_material(name="floor", rgba=[0.1, 0.1, 0.12, 1])
    spec.add_material(
        name="iron", rgba=[0.09, 0.09, 0.1, 1], specular=0.6, shininess=0.6
    )
    spec.add_material(
        name="chrome",
        rgba=[0.75, 0.76, 0.78, 1],
        specular=0.9,
        shininess=0.9,
        reflectance=0.2,
    )
    w.add_light(
        name="key",
        pos=[0.35, -0.45, 1.2],
        dir=[-0.2, 0.4, -1],
        diffuse=[0.8] * 3,
        specular=[0.2] * 3,
        castshadow=True,
    )
    w.add_light(
        name="fill",
        pos=[-0.2, 0.5, 1.0],
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
        friction=[0.8, 0.005, 0.0001],
    )
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
            friction=[1.5, 0.005, 0.0001],
            condim=4,
            contype=BLOCK_BIT,
            conaffinity=BLOCK_BIT,
            solref=GRASP_SOLREF,
            solimp=GRASP_SOLIMP,
            group=3,
        )
    contact = {
        "contype": BLOCK_BIT,
        "conaffinity": BLOCK_BIT,
        "solref": GRASP_SOLREF,
        "solimp": GRASP_SOLIMP,
        "condim": 4,
    }
    db = w.add_body(name="dumbbell", pos=[SPOT[0], SPOT[1], PLATE_R])
    db.add_freejoint(name="dumbbell_free")
    along_y = [0.7071068, 0.7071068, 0, 0]  # cylinder axis (local z) onto world y
    db.add_geom(
        name="handle",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[HANDLE_R, HANDLE_HALF, 0],
        quat=along_y,
        mass=0.02,
        material="chrome",
        friction=[1.2, 0.005, 0.0001],
        **contact,
    )
    for k, y in enumerate((-HANDLE_HALF - PLATE_HALF, HANDLE_HALF + PLATE_HALF)):
        db.add_geom(
            name=f"plate{k}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[PLATE_R, PLATE_HALF, 0],
            pos=[0, y, 0],
            quat=along_y,
            mass=0.03,
            material="iron",
            friction=[0.8, 0.005, 0.0001],
            **contact,
        )
        db.add_geom(
            name=f"collar{k}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.009, 0.002, 0],
            pos=[0, y - np.sign(y) * (PLATE_HALF + 0.002), 0],
            quat=along_y,
            mass=0,
            material="chrome",
            contype=0,
            conaffinity=0,
        )
    # front three-quarter view, close: the dumbbell lies across the frame, so both plates read clearly
    cam = [0.50, -0.36, 0.24]
    w.add_camera(name="judge", pos=cam, xyaxes=_look_at(cam, [0.10, 0.0, 0.18]), fovy=55)
    w.add_camera(name="side", pos=cam, xyaxes=_look_at(cam, [0.10, 0.0, 0.18]), fovy=55)
    return spec.compile()


class CurlScene:
    def __init__(self):
        self.model = build()
        self.data = mujoco.MjData(self.model)
        self.ik = IK(self.model)
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.site_id = self.model.site("gripperframe").id
        self.db = self.model.body("dumbbell").id
        mujoco.mj_resetData(self.model, self.data)
        q = np.append(HOME, GRIP_CLOSED)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.q = HOME.copy()
        self._centre = {}

    def site(self):
        return self.data.site_xpos[self.site_id].copy()

    def handle(self):
        return self.data.xpos[self.db].copy()

    def step(self, ctrl):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, STEPS_PER_CONTROL)

    def centre(self, grip: float) -> float:
        if grip not in self._centre:
            m = self.model
            d = mujoco.MjData(m)
            d.qpos[m.joint("gripper").qposadr[0]] = grip
            mujoco.mj_kinematics(m, d)
            R = d.site_xmat[self.site_id].reshape(3, 3)
            a = d.geom_xpos[m.geom("pad_gripper").id]
            b = d.geom_xpos[m.geom("pad_moving_jaw_so101_v1").id]
            self._centre[grip] = float(
                ((a + b) / 2 - d.site_xpos[self.site_id]) @ R[:, 2]
            )
        return self._centre[grip]

    def pad_mid(self):
        m, d = self.model, self.data
        return (
            d.geom_xpos[m.geom("pad_gripper").id]
            + d.geom_xpos[m.geom("pad_moving_jaw_so101_v1").id]
        ) / 2


def _seg(a, b, speed):
    n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / speed / DT))
    s = np.linspace(0, 1, n)
    s = 3 * s**2 - 2 * s**3
    return [np.asarray(a) + (np.asarray(b) - np.asarray(a)) * x for x in s]


def _cartesian(sc: CurlScene, pts, frame):
    """Closed-loop Cartesian control (see arena/chess/move.py): IK plus a leaky integral of the measured error."""
    q, corr = sc.q.copy(), np.zeros(3)
    for k, (p, close, grip) in enumerate(pts):
        qn, ep, _ = sc.ik.solve(p + corr, q, close, iters=25)
        if ep > 0.003:
            corr[:] = 0
            qn, ep, _ = sc.ik.solve(p, q, close, iters=40)
        q = q + np.clip(qn - q, -MAX_DQ, MAX_DQ)
        sc.step(np.append(q, grip))
        corr = np.clip(0.85 * corr + 0.35 * (p - sc.site()), -0.012, 0.012)
        if frame and k % 2 == 0:
            frame()
    sc.q = q


def _joints(sc: CurlScene, targets: np.ndarray, grip: float, frame, log=None):
    """Joint-space playback at the control rate; returns the measured elbow and the handle-to-pads distance."""
    adr = [sc.model.joint(j).qposadr[0] for j in ("shoulder_lift", "elbow_flex")]
    measured, slip = [], []
    for k, q in enumerate(targets):
        q = sc.q + np.clip(q - sc.q, -CURL_DQ, CURL_DQ)
        sc.q = q
        sc.step(np.append(q, grip))
        measured.append(sc.data.qpos[adr].copy())
        slip.append(float(np.linalg.norm(sc.handle() - sc.pad_mid())))
        if frame and k % 2 == 0:
            frame()
    return np.array(measured), np.array(slip)


def perform(
    angle_deg: np.ndarray, t: np.ndarray, frame: Callable | None = None, shoulder_deg: np.ndarray | None = None
) -> dict:
    """Pick up, copy the human arm (elbow, and shoulder elevation when given) 1:1, put down. Returns checks and the
    robot joint trajectory."""
    sc = CurlScene()
    for _ in range(60):
        sc.step(sc.data.ctrl.copy())
    close = np.array(
        [1.0, 0.0]
    )  # jaws close along x, across the handle (which lies along y)
    q0, _, _ = sc.ik.solve(np.array([*SPOT, 0.08]), sc.q, close)
    _, R = sc.ik.fk(q0)
    ez = np.array([*close, 0.0]) * float(
        np.sign(R[:, 2] @ np.array([*close, 0.0])) or 1.0
    )
    down = np.array([0, 0, -1.0])

    def site(z, grip):
        return np.array([SPOT[0], SPOT[1], z]) + 0.003 * down - sc.centre(grip) * ez

    hz, gz = (
        PLATE_R + 0.06,
        0.009,
    )  # gz: the pads (2-42 mm above the tips) straddle the 12 mm handle centred 17 mm up
    pts = [(p, close, GRIP_OPEN) for p in _seg(sc.site(), site(hz, GRIP_OPEN), REACH)]
    pts += [(site(hz, GRIP_OPEN), close, GRIP_OPEN)] * int(0.4 / DT)
    pts += [
        (p, close, GRIP_OPEN)
        for p in _seg(site(hz, GRIP_OPEN), site(gz, GRIP_OPEN), DESCEND)
    ]
    for k in range(int(0.5 / DT)):
        a = min(1, k * DT / 0.3)
        pts.append(
            (
                site(gz, GRIP_OPEN) * (1 - a) + site(gz, GRIP_CLOSED) * a,
                close,
                GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * a,
            )
        )
    lift = site(0.11, GRIP_CLOSED)
    pts += [
        (p, close, GRIP_CLOSED) for p in _seg(site(gz, GRIP_CLOSED), lift, REACH / 2)
    ]
    _cartesian(sc, pts, frame)
    picked = sc.handle()[2] > 0.06

    # the human arm drives the robot arm 1:1, joint by joint, from each joint's starting position:
    # elbow flexion (180 - elbow angle) closes the robot elbow; shoulder elevation raises the robot upper arm
    el, sh = JOINTS.index("elbow_flex"), JOINTS.index("shoulder_lift")
    start = sc.q.copy()
    ready = start.copy()
    ready[el] = CURL_START
    targets = [start + (ready - start) * s for s in np.linspace(0, 1, int(1.2 / DT))]
    tc = np.arange(t[0], t[-1], DT)
    flex = np.radians(np.clip(180.0 - np.interp(tc, t, angle_deg), 0, 135))
    flex -= flex.min()
    lift = np.zeros_like(flex)
    if shoulder_deg is not None:
        lift = np.radians(np.clip(np.interp(tc, t, shoulder_deg), 0, 180))
        lift = np.clip(lift - lift.min(), 0, np.radians(88))  # the SO-101 shoulder has ~90 degrees above this pose
    curl = np.repeat(ready[None], len(tc), 0)
    curl[:, el] = CURL_START - flex
    curl[:, sh] = ready[sh] - lift
    measured, slip = _joints(sc, np.vstack([targets, curl]), GRIP_CLOSED, frame)
    m_curl = measured[len(targets) :]
    s_curl = slip[len(targets) :]
    err = np.degrees(m_curl - curl[:, [sh, el]])
    track_rms = float(np.sqrt(np.mean(err**2)))
    held = bool(picked and s_curl.max() < 0.012)

    # back to the lift pose and set it down
    back = [curl[-1] + (start - curl[-1]) * s for s in np.linspace(0, 1, int(1.2 / DT))]
    _joints(sc, np.array(back), GRIP_CLOSED, frame)
    pts = [
        (p, close, GRIP_CLOSED)
        for p in _seg(sc.site(), site(gz + 0.002, GRIP_CLOSED), DESCEND * 2)
    ]
    for k in range(int(0.5 / DT)):
        a = min(1, k * DT / 0.3)
        pts.append(
            (
                site(gz + 0.002, GRIP_CLOSED),
                close,
                GRIP_CLOSED + (GRIP_OPEN - GRIP_CLOSED) * a,
            )
        )
    pts += [
        (p, close, GRIP_OPEN)
        for p in _seg(site(gz + 0.002, GRIP_OPEN), site(hz, GRIP_OPEN), REACH / 2)
    ]
    _cartesian(sc, pts, frame)
    for _ in range(int(0.5 / DT)):
        sc.step(np.append(sc.q, GRIP_OPEN))
    up = sc.data.xmat[sc.db].reshape(3, 3)[:, 1]  # handle axis
    set_down = bool(sc.handle()[2] < PLATE_R + 0.004 and abs(up[2]) < 0.2)
    return {
        "ok": bool(held and track_rms < 8 and set_down),
        "picked_up": bool(picked),
        "held_through_curl": held,
        "max_slip_mm": round(float(s_curl.max()) * 1000, 1),
        "elbow_tracking_rms_deg": round(track_rms, 2),  # RMS over the copied joints (elbow, and shoulder if used)
        "shoulder_range_deg": round(float(np.degrees(lift.max())), 1),
        "set_down": set_down,
        "curl_seconds": round(len(tc) * DT, 1),
        "flex_range_deg": round(float(np.degrees(flex.max())), 1),
        "trajectory": {
            "dt": DT,
            "joints": JOINTS,
            "curl_targets": curl.round(4).tolist(),
        },
    }

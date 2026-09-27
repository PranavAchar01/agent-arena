"""A tabletop for any pick-and-place task, with the same closed-loop controller and physics rehearsal as the chess
moves (arena/chess/move.py): IK plus a leaky integral of the measured gripper error, a measured reach ceiling that
also rejects self-collision, a joint rate limit, and grasp variants rehearsed until one passes the physics check.

What is new here: every move carries its own grasp height, place height, gripper opening and carry height, so
stacks (Tower of Hanoi discs, a cup pyramid, ...) work, not only pieces standing on one board.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..chess.move import (
    DEPTH,
    DESCEND,
    MAX_DQ,
    REACH,
    SETTLE,
    _ceiling_grid,
    _hold,
    _seg,
)
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

MM_PER_RAD, GAP0, ANG0 = (
    52.0,
    17.9,
    -0.04,
)  # measured: pad gap (mm) versus gripper angle, linear over the range used


def opening(width_mm: float, margin_mm: float = 9.0) -> float:
    """Gripper angle that opens the pads to width + margin."""
    return ANG0 + (width_mm + margin_mm - GAP0) / MM_PER_RAD


@dataclass
class Obj:
    name: str
    radius: float  # every object grips as a cylinder of this radius (visual extras are drawn on top)
    height: float
    rgba: tuple
    xyz: tuple  # initial centre
    mass: float = 0.02
    extras: list = field(
        default_factory=list
    )  # (type, size, pos, rgba) visual-only geoms
    friction: float = 1.2


@dataclass
class Move:
    obj: int
    src: np.ndarray  # xy
    grasp_z: float  # height of the grasp centre on the object at pick-up
    dst: np.ndarray  # xy
    place_z: float  # height of the object's centre once placed
    open: float  # gripper angle while approaching and releasing
    carry_z: (
        float  # grasp-centre height while carrying (clears everything on the table)
    )
    tol: float = 0.005
    label: str = ""
    knock: float = 0.003  # how far any other object may be nudged
    hold: float | None = (
        None  # gripper angle once the pads touch the object (aim with this; command full close)
    )


class TaskScene:
    def __init__(
        self,
        objs: list[Obj],
        statics: Callable | None = None,
        camera=((0.40, -0.25, 0.20), (0.20, 0.0, 0.025), 42),
    ):
        self.objs = objs
        spec = _base_spec()
        w = spec.worldbody
        spec.add_texture(
            name="sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.22, 0.24, 0.28],
            rgb2=[0.03, 0.03, 0.04],
            width=512,
            height=512,
        )
        spec.add_material(name="desk", rgba=[0.16, 0.15, 0.15, 1], reflectance=0.12)
        spec.add_material(name="floor", rgba=[0.1, 0.1, 0.12, 1])
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
        pads = _pad_frames()
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
                friction=[1.4, 0.005, 0.0001],
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
        for o in objs:
            b = w.add_body(name=o.name, pos=list(o.xyz))
            b.add_freejoint(name=f"{o.name}_free")
            b.add_geom(
                name=f"{o.name}_body",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[o.radius, o.height / 2, 0],
                rgba=list(o.rgba),
                mass=o.mass,
                friction=[o.friction, 0.01, 0.0002],
                **contact,
            )
            for k, (typ, size, pos, rgba) in enumerate(o.extras):
                b.add_geom(
                    name=f"{o.name}_x{k}",
                    type=typ,
                    size=size,
                    pos=pos,
                    rgba=rgba,
                    contype=0,
                    conaffinity=0,
                    mass=0,
                )
        if statics:
            statics(w)
        pos, tgt, fov = camera
        w.add_camera(name="judge", pos=list(pos), xyaxes=_look_at(pos, tgt), fovy=fov)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.site_id = self.model.site("gripperframe").id
        self.pieces = [type("P", (), {"body": o.name})() for o in objs]
        mujoco.mj_resetData(self.model, self.data)
        q = np.append(HOME, GRIP_CLOSED)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def site(self):
        return self.data.site_xpos[self.site_id].copy()

    def piece_pos(self, i: int):
        return self.data.xpos[self.model.body(self.objs[i].name).id].copy()

    def step(self, ctrl):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, STEPS_PER_CONTROL)

    def snapshot(self):
        return self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy()

    def restore(self, snap):
        self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:] = snap
        mujoco.mj_forward(self.model, self.data)


VARIANTS = [
    {"axis": "x", "slow": 1.0, "extra": 0.0},
    {"axis": "y", "slow": 1.0, "extra": 0.0},
    {"axis": "x", "slow": 0.6, "extra": 0.1},
    {"axis": "y", "slow": 0.6, "extra": 0.1},
    {"axis": "x", "slow": 0.45, "extra": -0.05},
    {"axis": "y", "slow": 0.45, "extra": -0.05},
]


class TaskMover:
    def __init__(self, sc: TaskScene):
        self.sc = sc
        self.ik = IK(sc.model)
        self.q = HOME.copy()
        self._centre: dict[float, float] = {}
        self._ceiling = _ceiling_grid(self.ik)

    def ceiling(self, xy) -> float:
        xs, ys, zs = self._ceiling
        i = np.clip(np.searchsorted(xs, xy[0]) - 1, 0, len(xs) - 2)
        j = np.clip(np.searchsorted(ys, xy[1]) - 1, 0, len(ys) - 2)
        fx = np.clip((xy[0] - xs[i]) / (xs[i + 1] - xs[i]), 0, 1)
        fy = np.clip((xy[1] - ys[j]) / (ys[j + 1] - ys[j]), 0, 1)
        return float(
            (zs[i, j] * (1 - fx) + zs[i + 1, j] * fx) * (1 - fy)
            + (zs[i, j + 1] * (1 - fx) + zs[i + 1, j + 1] * fx) * fy
        )

    def centre(self, grip: float) -> float:
        if grip not in self._centre:
            m = self.sc.model
            d = mujoco.MjData(m)
            d.qpos[m.joint("gripper").qposadr[0]] = grip
            mujoco.mj_kinematics(m, d)
            R = d.site_xmat[m.site("gripperframe").id].reshape(3, 3)
            a = d.geom_xpos[m.geom("pad_gripper").id]
            b = d.geom_xpos[m.geom("pad_moving_jaw_so101_v1").id]
            self._centre[grip] = float(
                ((a + b) / 2 - d.site_xpos[m.site("gripperframe").id]) @ R[:, 2]
            )
        return self._centre[grip]

    def _waypoints(self, mv: Move, close, ez_sign, op, slow):
        down = np.array([0, 0, -1.0])
        ez = np.array([*close, 0.0]) * ez_sign

        def site(xy, z, grip):
            p = np.array([xy[0], xy[1], z]) + DEPTH * down - self.centre(grip) * ez
            p[2] = min(p[2], self.ceiling(p[:2]))
            return p

        pts = []
        hover = site(mv.src, mv.carry_z, op)
        pts += [(p, close, op) for p in _seg(self.sc.site(), hover, REACH * slow)]
        pts += [(p, close, op) for p in _hold(hover, 0.5)]
        pts += [
            (p, close, op)
            for p in _seg(hover, site(mv.src, mv.grasp_z, op), DESCEND * slow)
        ]
        held = (
            mv.hold if mv.hold is not None else GRIP_CLOSED
        )  # where the jaws stop on the object
        g0 = site(mv.src, mv.grasp_z, held)
        for k in range(int(0.5 / DT)):
            a = min(1, k * DT / 0.3)
            pts.append(
                (
                    site(mv.src, mv.grasp_z, op) * (1 - a) + g0 * a,
                    close,
                    op + (GRIP_CLOSED - op) * a,
                )
            )
        lifted = site(mv.src, mv.carry_z, held)
        pts += [(p, close, GRIP_CLOSED) for p in _seg(g0, lifted, REACH * slow / 2)]
        above = site(mv.dst, mv.carry_z, held)
        pts += [(p, close, GRIP_CLOSED) for p in _seg(lifted, above, REACH * slow)]
        pts += [(p, close, GRIP_CLOSED) for p in _hold(above, 0.4)]
        low = site(mv.dst, mv.place_z + 0.0015, held)
        pts += [(p, close, GRIP_CLOSED) for p in _seg(above, low, DESCEND * slow)]
        for k in range(int(0.45 / DT)):
            a = min(1, k * DT / 0.3)
            pts.append(
                (
                    low * (1 - a) + site(mv.dst, mv.place_z + 0.0015, op) * a,
                    close,
                    GRIP_CLOSED + (op - GRIP_CLOSED) * a,
                )
            )
        r = site(mv.dst, mv.place_z, op)
        pts += [
            (p, close, op)
            for p in _seg(r, site(mv.dst, mv.carry_z + 0.01, op), REACH * slow / 2)
        ]
        return pts

    def _plan(self, mv: Move, v: dict):
        close = np.array([1.0, 0.0]) if v["axis"] == "x" else np.array([0.0, 1.0])
        q, _, _ = self.ik.solve(np.array([*mv.src, mv.carry_z]), self.q, close)
        _, R = self.ik.fk(q)
        sign = float(np.sign(R[:, 2] @ np.array([*close, 0.0])) or 1.0)
        return self._waypoints(mv, close, sign, mv.open + v["extra"], v["slow"])

    def _run(self, pts, frame):
        q, corr, worst = self.q.copy(), np.zeros(3), 0.0
        c = np.append(q, GRIP_CLOSED)
        for k, (p, close, grip) in enumerate(pts):
            q_new, ep, _ = self.ik.solve(p + corr, q, close, iters=25)
            if ep > 0.003:
                corr[:] = 0
                q_new, ep, _ = self.ik.solve(p, q, close, iters=40)
            q = q + np.clip(q_new - q, -MAX_DQ, MAX_DQ)
            worst = max(worst, ep)
            c = np.append(q, grip)
            self.sc.step(c)
            corr = np.clip(0.85 * corr + 0.35 * (p - self.sc.site()), -0.012, 0.012)
            if frame is not None and k % 2 == 0:
                frame()
        for _ in range(int(SETTLE / DT)):
            self.sc.step(c)
        self.q = q.copy()
        return worst

    def _check(self, mv: Move, before) -> dict:
        sc = self.sc
        pos = sc.piece_pos(mv.obj)
        up = sc.data.xmat[sc.model.body(sc.objs[mv.obj].name).id].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1))))
        err = float(np.linalg.norm(pos[:2] - mv.dst))
        dz = abs(pos[2] - mv.place_z)
        knocked = max(
            (
                float(np.linalg.norm(sc.piece_pos(i) - p))
                for i, p in before.items()
                if i != mv.obj
            ),
            default=0.0,
        )
        ok = err < mv.tol and tilt < 10 and knocked < mv.knock and dz < 0.004
        return {
            "ok": bool(ok),
            "err_mm": round(err * 1000, 1),
            "tilt_deg": round(tilt, 1),
            "knocked_mm": round(knocked * 1000, 1),
            "dz_mm": round(dz * 1000, 1),
        }

    def execute(self, mv: Move, frame=None, log=None) -> dict:
        """Rehearse the move headless with each variant until one passes, then play it (with frames)."""
        snap, q0 = self.sc.snapshot(), self.q.copy()
        before = {i: self.sc.piece_pos(i) for i in range(len(self.sc.objs))}
        for v in VARIANTS:
            pts = self._plan(mv, v)
            ik = self._run(pts, None)
            chk = {**self._check(mv, before), **v, "ik_mm": round(ik * 1000, 1)}
            if log:
                log(chk)
            self.sc.restore(snap)
            self.q = q0.copy()
            if chk["ok"]:
                self._run(pts, frame)
                return {"ok": True, "label": mv.label, **chk}
        return {"ok": False, "label": mv.label}

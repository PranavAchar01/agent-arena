"""Retarget a human motion shape onto the SO-101 and replay it with physics.

What a monocular internet clip can honestly give us is the SHAPE of the human's hand path, not metric positions:
how progress along the move unfolds over time (the speed profile), how high the hand arcs relative to the
distance it travels (the lift profile), how far it strays sideways, and how long the move took. That is what
`MotionShape` holds, normalised. Retargeting anchors the shape to the robot's scene: progress 0 is the block,
progress 1 is the target, lift and sideways deviation are scaled by the robot's own travel distance, and the time
is stretched by TIME_SCALE because an STS3215 servo arm is slower than a hand.

Declared robot-side additions (the same for every clip): a reach from home to above the block, the grasp
(descend, close with a dwell), and after the move the release (open with a dwell), a retreat and the return home.
Everything between grasp and release is the human's shape.

The replay is a full physics rollout: the block sits on a free joint, the jaws hold it by friction, nothing is
teleported. An episode enters the dataset only if the task succeeds in that rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ik import IK
from .scene import BASE_HALF, DT, GRIP_CLOSED, HALF, HOME, Scene

TIME_SCALE = 2.0
GRIP_OPEN_CMD = 0.7  # ~5 cm between the pads
GRIP_DEPTH = 0.012  # block centre this far up the fingers from the tips
CENTRE_OPEN, CENTRE_GRASP = (
    0.024,
    0.0135,
)  # block centre along the closing axis, open / fixed pad touching
HOVER = 0.075
PUSH_Z = 0.008  # jaw tips above the desk while pushing
PUSH_BACKOFF = (
    HALF + 0.022
)  # site distance behind the block centre when the push starts
REACH_SPEED = 0.12  # m/s for the declared reach / retreat segments
# Lift and sideways deviation are measured in human hand lengths (wrist to middle knuckle, ~8.5 cm on an adult).
# The SO-101 is roughly half a human arm's scale, so one hand length of human lift becomes 4.25 cm on the robot.
M_PER_HAND = 0.085 * 0.5


@dataclass
class MotionShape:
    """Normalised hand path of one human demonstration (see module doc)."""

    task: str
    tau: np.ndarray  # normalised time 0..1, K samples
    u: np.ndarray  # progress along the move 0..1 at each tau
    lift: np.ndarray  # height above the start-end chord, in hand lengths
    side: np.ndarray  # sideways deviation from the chord, in hand lengths
    duration: float  # seconds the human took for the move
    source: str = ""

    def to_json(self):
        return {
            "task": self.task,
            "tau": self.tau.round(4).tolist(),
            "u": self.u.round(4).tolist(),
            "lift": self.lift.round(4).tolist(),
            "side": self.side.round(4).tolist(),
            "duration": round(self.duration, 3),
            "source": self.source,
        }

    @classmethod
    def from_json(cls, d):
        return cls(
            d["task"],
            *(np.array(d[k], float) for k in ("tau", "u", "lift", "side")),
            d["duration"],
            d.get("source", ""),
        )


def synthetic(task: str, lift: float = 1.2, duration: float = 1.2) -> MotionShape:
    """A textbook minimum-jerk shape. Used only by tests to check the robot side; never written to a dataset."""
    tau = np.linspace(0, 1, 50)
    u = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    h = (0.0 if task == "push" else lift) * np.sin(np.pi * u)
    return MotionShape(task, tau, u, h, np.zeros_like(u), duration, "synthetic")


@dataclass
class Episode:
    obs: np.ndarray
    act: np.ndarray
    success: bool
    gates: dict = field(default_factory=dict)
    frames: list | None = None


def _seg(a, b, speed):
    n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / speed / DT))
    s = np.linspace(0, 1, n)
    s = 3 * s**2 - 2 * s**3
    return [np.asarray(a) + (np.asarray(b) - np.asarray(a)) * x for x in s]


def _hold(p, seconds):
    return [np.asarray(p)] * max(1, int(seconds / DT))


def plan(sc: Scene, shape: MotionShape):
    """Cartesian waypoints (site position, closing direction, gripper command) at the control rate."""
    b, t = sc.block(), sc.target()
    chord = t[:2] - b[:2]
    dist = float(np.linalg.norm(chord))
    fwd = chord / dist
    left = np.array([-fwd[1], fwd[0]])
    T = max(0.8, min(4.0, shape.duration * TIME_SCALE))
    n = int(T / DT)
    tau = np.linspace(0, 1, n)
    u = np.interp(tau, shape.tau, shape.u)
    lift = (
        np.interp(u, shape.u, shape.lift)
        if np.all(np.diff(shape.u) > 0)
        else np.interp(tau, shape.tau, shape.lift)
    )
    side = np.interp(tau, shape.tau, shape.side)
    pts: list = []

    if sc.task == "push":
        start = b[:2] - fwd * PUSH_BACKOFF
        # stop so the block centre ends on the target centre: pusher stays PUSH_BACKOFF behind it
        end = t[:2] - fwd * PUSH_BACKOFF
        close = left
        grip = GRIP_CLOSED
        home_site = sc.site()
        above = np.array([*start, 0.045])
        low = np.array([*start, PUSH_Z])
        for p in _seg(home_site, above, REACH_SPEED) + _seg(
            above, low, REACH_SPEED / 2
        ):
            pts.append((p, close, grip))
        L = float(np.linalg.norm(end - start))
        for ui, si in zip(u, side):
            xy = start + (end - start) * ui + left * si * M_PER_HAND
            pts.append((np.array([*xy, PUSH_Z]), close, grip))
        last = pts[-1][0]
        for p in _hold(last, 0.3) + _seg(last, last + [0, 0, 0.03], REACH_SPEED):
            pts.append((p, close, grip))
        return pts

    # place / stack: grasp across the two block faces nearest the radial direction from the base
    yaw = _yaw(sc)
    radial = b[:2] / np.linalg.norm(b[:2])
    faces = [
        np.array([np.cos(yaw + k * np.pi / 2), np.sin(yaw + k * np.pi / 2)])
        for k in range(4)
    ]
    close = max(faces, key=lambda f: abs(f @ radial))
    close3 = np.array([*close, 0.0])
    down = np.array([0, 0, -1.0])
    # site = block - R @ [-GRIP_DEPTH, 0, c] with R[:,0] = down, R[:,2] = +-close; sign fixed after the first IK
    ez = close3 * sc._ez_sign if hasattr(sc, "_ez_sign") else close3

    def site_for(center, c):
        return center + GRIP_DEPTH * down - c * ez

    release_z = (HALF + 0.012) if sc.task == "place" else (2 * BASE_HALF + HALF + 0.004)
    b_c = b.copy()
    t_c = np.array([*t[:2], release_z])
    home_site = sc.site()
    hover = site_for(b_c, CENTRE_OPEN) + [0, 0, HOVER]
    for p in _seg(home_site, hover, REACH_SPEED):
        pts.append((p, close, GRIP_CLOSED))
    for p in _hold(hover, 0.3):
        pts.append((p, close, GRIP_OPEN_CMD))
    for p in _seg(hover, site_for(b_c, CENTRE_OPEN), REACH_SPEED / 2):
        pts.append((p, close, GRIP_OPEN_CMD))
    for p in _seg(site_for(b_c, CENTRE_OPEN), site_for(b_c, CENTRE_GRASP), 0.05):
        pts.append((p, close, GRIP_OPEN_CMD))
    g0 = site_for(b_c, CENTRE_GRASP)
    for k in range(int(0.5 / DT)):
        pts.append(
            (
                g0,
                close,
                GRIP_OPEN_CMD + (GRIP_CLOSED - GRIP_OPEN_CMD) * min(1, k * DT / 0.35),
            )
        )
    # the human's move: chord from the grasped block to the release point, arcing by the human's lift profile.
    # A pick is lifted clear before it travels; the human's own lift profile decides how high.
    a3 = np.array([*b_c[:2], b_c[2]])
    e3 = t_c
    Lxy = float(np.linalg.norm(e3[:2] - a3[:2]))
    for ui, li, si in zip(u, lift, side):
        c = a3 + (e3 - a3) * ui
        c[:2] += left * si * M_PER_HAND
        c[2] += li * M_PER_HAND
        pts.append((site_for(c, CENTRE_GRASP), close, GRIP_CLOSED))
    r = pts[-1][0]
    for _ in range(int(0.3 / DT)):
        pts.append((r, close, GRIP_CLOSED))
    for k in range(int(0.5 / DT)):
        pts.append(
            (
                r,
                close,
                GRIP_CLOSED + (GRIP_OPEN_CMD - GRIP_CLOSED) * min(1, k * DT / 0.35),
            )
        )
    for p in _seg(r, r + [0, 0, 0.025], REACH_SPEED / 2):
        pts.append((p, close, GRIP_OPEN_CMD))
    return pts


def _yaw(sc: Scene) -> float:
    a = sc.model.joint("block_free").qposadr[0]
    w, x, y, z = sc.data.qpos[a + 3 : a + 7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def to_joints(sc: Scene, pts, ik: IK):
    q = HOME.copy()
    out, worst_pos, worst_down = [], 0.0, 0.0
    for p, close, grip in pts:
        q, ep, ed = ik.solve(p, q, close, iters=40)
        worst_pos, worst_down = max(worst_pos, ep), max(worst_down, ed)
        out.append(np.append(q, grip))
    return np.array(out), worst_pos, worst_down


def replay(
    sc: Scene,
    shape: MotionShape,
    block_xy,
    target_xy,
    block_yaw=0.0,
    target_yaw=0.0,
    ik: IK | None = None,
    render=None,
    settle_s: float = 0.6,
) -> Episode:
    """Plan, solve IK and run the physics rollout from a fresh reset. Returns the episode and its gates."""
    ik = ik or IK(sc.model)
    sc.reset(block_xy, target_xy, block_yaw, target_yaw)
    # pick the closing-axis sign the IK naturally lands on, so the site offset is on the correct side
    if sc.task != "push":
        sc._ez_sign = 1.0
        pts = plan(sc, shape)
        q, _, _ = ik.solve(pts[len(pts) // 3][0], HOME, pts[0][1])
        _, R = ik.fk(q)
        sc._ez_sign = float(np.sign(R[:, 2] @ np.array([*pts[0][1], 0.0])) or 1.0)
    pts = plan(sc, shape)
    qs, e_pos, e_down = to_joints(sc, pts, ik)
    obs, act, frames = [], [], []
    for k, target in enumerate(qs):
        obs.append(sc.obs())
        act.append(target)
        sc.step(target)
        if render is not None and k % 2 == 0:
            frames.append(render(sc))
    for _ in range(int(settle_s / DT)):
        sc.step(qs[-1])
    ok = sc.success()
    gates = {
        "ik_pos_err_mm": round(e_pos * 1000, 2),
        "ik_down_err_deg": round(e_down, 1),
        "success": ok,
        "steps": len(qs),
    }
    ok = ok and e_pos < 0.01
    return Episode(np.array(obs), np.array(act), ok, gates, frames or None)

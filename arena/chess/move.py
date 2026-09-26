"""Turn a chess move into SO-101 motion and run it through MuJoCo physics.

A move becomes one or more pick-and-place transfers (a capture first clears the victim to the tray; castling moves
the king then the rook; en passant clears the passed pawn; a promotion swaps the pawn for the spare queen). Each
transfer: hover above the piece with the jaws open just enough to clear its neighbours, descend, close, lift the
piece clear of every other piece, carry it along the human move shape, set it down, open, rise.

Every move is rehearsed headless first and checked: the moved piece must end upright within 4 mm of its square
and no other piece may have been knocked more than 3 mm. Only a move that passes is played to the viewer; a failed
rehearsal is retried with the other grasp axis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess
import mujoco
import numpy as np

from ..sim.ik import IK
from ..sim.retarget import MotionShape, synthetic
from ..sim.scene import DT, GRIP_CLOSED, HOME
from .scene import BASE_H, MAX_H, ChessScene, square_xy, tray_xy

GRIP_OPEN = (
    -0.05
)  # about 17 mm between the pads: clears a 13 mm base with 1.5 mm to each neighbour
GRASP_Z = 0.010  # grasp height on the piece's 16 mm base: jaw tips 7 mm above the desk
DEPTH = 0.003  # the jaw tips sit this far below the grasp centre
CARRY = MAX_H + 0.008  # a carried piece's base clears the tallest piece by this much
REACH = 0.18  # m/s for free moves
SETTLE = 0.25
DESCEND = 0.03  # m/s for the last few centimetres onto a square
GRIP_RELEASE = (
    -0.095
)  # the pads compress around a held piece, so letting go needs the full opening
# grasp variants rehearsed in order until one passes: closing axis, jaw opening before the grasp, speed factor
VARIANTS = [
    {"axis": "file", "open": GRIP_OPEN, "slow": 1.0},
    {"axis": "rank", "open": GRIP_OPEN, "slow": 1.0},
    {"axis": "file", "open": -0.075, "slow": 0.6},
    {"axis": "rank", "open": -0.075, "slow": 0.6},
    {"axis": "file", "open": -0.11, "slow": 0.5},
    {"axis": "rank", "open": -0.11, "slow": 0.5},
]
MAX_DQ = (
    0.05  # rad per 20 ms control step (2.5 rad/s), below the STS3215's no-load speed
)


@dataclass
class Transfer:
    piece: int
    src: np.ndarray  # xy
    dst: np.ndarray  # xy
    tol: float = 0.005  # a 12 mm piece 5 mm off-centre is still well inside its 22 mm square; the tray is looser


def _seg(a, b, speed):
    n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / speed / DT))
    s = np.linspace(0, 1, n)
    s = 3 * s**2 - 2 * s**3
    return [np.asarray(a) + (np.asarray(b) - np.asarray(a)) * x for x in s]


def _hold(p, seconds):
    return [np.asarray(p)] * max(1, int(seconds / DT))


_CEILING = None
ARM = {
    "base",
    "shoulder",
    "upper_arm",
    "lower_arm",
    "wrist",
    "gripper",
    "moving_jaw_so101_v1",
}


def _self_collides(ik: IK, q5) -> bool:
    """True if the arm touches itself (with 4 mm margin) at joint angles q5; pieces and the desk are ignored."""
    m, d = ik.m, ik.d
    d.qpos[ik.qadr] = q5
    mujoco.mj_forward(m, d)
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = m.body(m.geom(c.geom1).bodyid).name, m.body(m.geom(c.geom2).bodyid).name
        if (
            a in ARM
            and b in ARM
            and {a, b} not in ({"gripper", "moving_jaw_so101_v1"},)
            and c.dist < 0.004
        ):
            return True
    return False


def _ceiling_grid(ik: IK):
    """Max reachable site height with the jaws down, on a grid over the board and trays (cached per process)."""
    global _CEILING
    cache = Path(__file__).with_name("ceiling.npz")
    if _CEILING is None and cache.exists():
        z = np.load(cache)
        _CEILING = (z["xs"], z["ys"], z["zs"])
    if _CEILING is None:
        xs, ys = np.linspace(0.08, 0.31, 13), np.linspace(-0.16, 0.16, 17)
        zs = np.zeros((len(xs), len(ys)))
        for a, x in enumerate(xs):
            for b, y in enumerate(ys):
                lo, hi = 0.0, 0.12
                for _ in range(9):
                    mid = (lo + hi) / 2
                    q, err, ang = ik.solve(np.array([x, y, mid]), HOME, None, iters=80)
                    ok = err < 0.002 and ang < 8 and not _self_collides(ik, q)
                    lo, hi = (mid, hi) if ok else (lo, mid)
                zs[a, b] = lo
        _CEILING = (xs, ys, zs)
        np.savez(cache, xs=xs, ys=ys, zs=zs)
    return _CEILING


class Mover:
    def __init__(self, sc: ChessScene, shape: MotionShape | None = None):
        self.sc = sc
        self.ik = IK(sc.model)
        self.shape = shape or synthetic("place", lift=0.6, duration=1.0)
        self.q = HOME.copy()
        self._centre: dict[float, float] = {}
        self._ceiling = _ceiling_grid(self.ik)

    def ceiling(self, xy) -> float:
        """Highest jaws-down site height the arm reaches at xy (bilinear on a measured grid), minus a margin."""
        xs, ys, zs = self._ceiling
        i = np.clip(np.searchsorted(xs, xy[0]) - 1, 0, len(xs) - 2)
        j = np.clip(np.searchsorted(ys, xy[1]) - 1, 0, len(ys) - 2)
        fx = np.clip((xy[0] - xs[i]) / (xs[i + 1] - xs[i]), 0, 1)
        fy = np.clip((xy[1] - ys[j]) / (ys[j + 1] - ys[j]), 0, 1)
        z = (zs[i, j] * (1 - fx) + zs[i + 1, j] * fx) * (1 - fy) + (
            zs[i, j + 1] * (1 - fx) + zs[i + 1, j + 1] * fx
        ) * fy
        return float(z)

    def centre(self, grip: float) -> float:
        if grip not in self._centre:
            self._centre[grip] = self._pad_centre(grip)
        return self._centre[grip]

    def _pad_centre(self, grip: float) -> float:
        """Midpoint between the two pads along the closing axis, relative to the gripperframe site."""
        m = self.sc.model
        d = mujoco.MjData(m)
        d.qpos[m.joint("gripper").qposadr[0]] = grip
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[m.site("gripperframe").id].reshape(3, 3)
        a = d.geom_xpos[m.geom("pad_gripper").id]
        b = d.geom_xpos[m.geom("pad_moving_jaw_so101_v1").id]
        return float(((a + b) / 2 - d.site_xpos[m.site("gripperframe").id]) @ R[:, 2])

    # ---------- planning ----------
    def transfers(self, board: chess.Board, mv: chess.Move) -> list[Transfer]:
        """Physical transfers for `mv` played on `board` (the position before the move)."""
        sc = self.sc
        out: list[Transfer] = []
        mover_color = board.turn
        if board.is_en_passant(mv):
            victim_sq = chess.square(
                chess.square_file(mv.to_square), chess.square_rank(mv.from_square)
            )
        elif board.is_capture(mv):
            victim_sq = mv.to_square
        else:
            victim_sq = None
        if victim_sq is not None:
            v = sc.at[victim_sq]
            c = sc.pieces[v].color
            out.append(
                Transfer(v, square_xy(victim_sq), tray_xy(c, sc.captured[c]), 0.012)
            )
        p = sc.at[mv.from_square]
        if mv.promotion:
            spare = next(
                i
                for i, info in enumerate(sc.pieces)
                if info.body.endswith("q_spare")
                and info.color == mover_color
                and i not in sc.at.values()
            )
            out.append(
                Transfer(
                    p,
                    square_xy(mv.from_square),
                    tray_xy(mover_color, sc.captured[mover_color]),
                    0.012,
                )
            )
            # the spare queen is set out in the tray first (it waits under the table), then played like any piece
            sc._place(spare, tray_xy(mover_color, sc.captured[mover_color] + 1))
            mujoco.mj_forward(sc.model, sc.data)
            out.append(
                Transfer(spare, sc.piece_pos(spare)[:2], square_xy(mv.to_square))
            )
        else:
            out.append(Transfer(p, square_xy(mv.from_square), square_xy(mv.to_square)))
        if board.is_castling(mv):
            rank = chess.square_rank(mv.from_square)
            kingside = chess.square_file(mv.to_square) == 6
            r_from = chess.square(7 if kingside else 0, rank)
            r_to = chess.square(5 if kingside else 3, rank)
            out.append(Transfer(sc.at[r_from], square_xy(r_from), square_xy(r_to)))
        return out

    def _waypoints(
        self, t: Transfer, close: np.ndarray, ez_sign: float, op: float, slow: float
    ):
        down = np.array([0, 0, -1.0])
        ez = np.array([*close, 0.0]) * ez_sign

        def site(xy, z, grip):
            p = np.array([xy[0], xy[1], z]) + DEPTH * down - self.centre(grip) * ez
            p[2] = min(p[2], self.ceiling(p[:2]))
            return p

        pts = []
        start = self.sc.site()
        hover = site(t.src, GRASP_Z + CARRY, op)
        for p in _seg(start, hover, REACH * slow):
            pts.append((p, close, op))
        # the servos trail a moving target by ~5 mm; settle above the piece, then descend slowly
        for p in _hold(hover, 0.5):
            pts.append((p, close, op))
        for p in _seg(hover, site(t.src, GRASP_Z, op), DESCEND * slow):
            pts.append((p, close, op))
        g0 = site(t.src, GRASP_Z, GRIP_CLOSED)
        for k in range(int(0.45 / DT)):
            a = min(1, k * DT / 0.3)
            pts.append(
                (
                    site(t.src, GRASP_Z, op) * (1 - a) + g0 * a,
                    close,
                    op + (GRIP_CLOSED - op) * a,
                )
            )
        # lift straight up, then carry along the human's move shape (timing and arc), then lower straight down
        lifted = site(t.src, GRASP_Z + CARRY, GRIP_CLOSED)
        for p in _seg(g0, lifted, REACH * slow / 2):
            pts.append((p, close, GRIP_CLOSED))
        dist = float(np.linalg.norm(t.dst - t.src))
        T = float(np.clip(self.shape.duration * dist / 0.12, 0.6, 2.0))
        tau = np.linspace(0, 1, max(2, int(T / DT)))
        u = np.interp(tau, self.shape.tau, self.shape.u)
        arc = np.interp(tau, self.shape.tau, self.shape.lift)
        arc = (
            arc / max(1e-6, arc.max()) * min(0.02, 0.15 * dist)
            if arc.max() > 0
            else arc
        )
        for ui, ai in zip(u, arc, strict=True):
            xy = t.src + (t.dst - t.src) * ui
            pts.append(
                (site(xy, GRASP_Z + CARRY + ai, GRIP_CLOSED), close, GRIP_CLOSED)
            )
        above_dst = site(t.dst, GRASP_Z + CARRY, GRIP_CLOSED)
        down_dst = site(t.dst, GRASP_Z + 0.0015, GRIP_CLOSED)
        for p in _hold(above_dst, 0.4):
            pts.append((p, close, GRIP_CLOSED))
        for p in _seg(above_dst, down_dst, DESCEND * slow):
            pts.append((p, close, GRIP_CLOSED))
        for k in range(int(0.4 / DT)):
            a = min(1, k * DT / 0.25)
            pts.append(
                (
                    down_dst * (1 - a)
                    + site(t.dst, GRASP_Z + 0.0015, GRIP_RELEASE) * a,
                    close,
                    GRIP_CLOSED + (GRIP_RELEASE - GRIP_CLOSED) * a,
                )
            )
        r = site(t.dst, GRASP_Z, GRIP_RELEASE)
        for p in _seg(r, r + [0, 0, CARRY + 0.01], REACH * slow / 2):
            pts.append((p, close, GRIP_RELEASE))
        return pts

    def _plan(self, t: Transfer, v: dict):
        close = np.array([1.0, 0.0]) if v["axis"] == "rank" else np.array([0.0, 1.0])
        # which way the closing axis points once the IK settles decides which side of the site the pads are on
        q, _, _ = self.ik.solve(np.array([*t.src, GRASP_Z + CARRY]), self.q, close)
        _, R = self.ik.fk(q)
        sign = float(np.sign(R[:, 2] @ np.array([*close, 0.0])) or 1.0)
        return self._waypoints(t, close, sign, v["open"], v["slow"])

    # ---------- execution ----------
    def _run(self, pts, frame: Callable | None) -> float:
        """Closed loop at the control rate: IK toward the waypoint plus a leaky integral of the measured site error,
        which cancels the soft servos' lag. Returns the worst IK residual."""
        q, corr, worst = self.q.copy(), np.zeros(3), 0.0
        for k, (p, close, grip) in enumerate(pts):
            q_new, ep, _ = self.ik.solve(p + corr, q, close, iters=25)
            if (
                ep > 0.003
            ):  # the correction pushed past the reach envelope: aim at the waypoint itself
                corr[:] = 0
                q_new, ep, _ = self.ik.solve(p, q, close, iters=40)
            q = q + np.clip(
                q_new - q, -MAX_DQ, MAX_DQ
            )  # a joint never jumps, even if the solver switches branch
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

    def _check(self, t: Transfer, before: dict[int, np.ndarray]) -> dict:
        sc = self.sc
        pos = sc.piece_pos(t.piece)
        up = sc.data.xmat[sc.model.body(sc.pieces[t.piece].body).id].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        err = float(np.linalg.norm(pos[:2] - t.dst))
        knocked = max(
            (
                float(np.linalg.norm(sc.piece_pos(i)[:2] - p[:2]))
                for i, p in before.items()
                if i != t.piece
            ),
            default=0.0,
        )
        ok = err < t.tol and tilt < 10 and knocked < 0.003 and pos[2] < 0.003
        return {
            "ok": bool(ok),
            "err_mm": round(err * 1000, 1),
            "tilt_deg": round(tilt, 1),
            "knocked_mm": round(knocked * 1000, 1),
        }

    def execute(
        self,
        board: chess.Board,
        mv: chess.Move,
        frame: Callable | None = None,
        log: Callable | None = None,
    ) -> dict:
        """Rehearse the whole move headless, then play the passing plan (with frames). Updates the square map."""
        log = log or (lambda e: None)
        results = []
        for t in self.transfers(board, mv):
            snap, q0 = self.sc.snapshot(), self.q.copy()
            before = {i: self.sc.piece_pos(i) for i in range(len(self.sc.pieces))}
            chosen = None
            for v in VARIANTS:
                pts = self._plan(t, v)
                ik_err = self._run(pts, None)
                chk = {**self._check(t, before), **v, "ik_mm": round(ik_err * 1000, 1)}
                log({"type": "rehearsal", "piece": self.sc.pieces[t.piece].body, **chk})
                self.sc.restore(snap)
                self.q = q0.copy()
                if chk["ok"]:
                    chosen = (pts, chk)
                    break
            if chosen is None:
                results.append({"ok": False, "piece": self.sc.pieces[t.piece].body})
                return {"ok": False, "transfers": results}
            self._run(chosen[0], frame)
            results.append(
                {"ok": True, "piece": self.sc.pieces[t.piece].body, **chosen[1]}
            )
        self._apply(board, mv)
        return {"ok": True, "transfers": results}

    def _apply(self, board: chess.Board, mv: chess.Move):
        """Update square -> piece bookkeeping to match what was physically done."""
        sc = self.sc
        if board.is_en_passant(mv):
            vsq = chess.square(
                chess.square_file(mv.to_square), chess.square_rank(mv.from_square)
            )
        elif board.is_capture(mv):
            vsq = mv.to_square
        else:
            vsq = None
        if vsq is not None:
            c = sc.pieces[sc.at.pop(vsq)].color
            sc.captured[c] += 1
        p = sc.at.pop(mv.from_square)
        if mv.promotion:
            c = sc.pieces[p].color
            sc.captured[c] += 1
            spare = next(
                i
                for i, info in enumerate(sc.pieces)
                if info.body.endswith("q_spare") and info.color == c
            )
            sc.at[mv.to_square] = spare
        else:
            sc.at[mv.to_square] = p
        if board.is_castling(mv):
            rank = chess.square_rank(mv.from_square)
            kingside = chess.square_file(mv.to_square) == 6
            sc.at[chess.square(5 if kingside else 3, rank)] = sc.at.pop(
                chess.square(7 if kingside else 0, rank)
            )


__all__ = ["BASE_H", "Mover"]

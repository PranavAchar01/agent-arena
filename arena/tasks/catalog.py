"""New tabletop tasks for the SO-101: each is a scene (objects plus static props) and a list of moves.

  hanoi  Tower of Hanoi with three discs: the optimal 7 moves (2^n - 1), largest disc never on a smaller one
  cups   a 3-2-1 cup pyramid, speed-stacking style: six cups from a row into a pyramid, carried just high enough

Plans are computed (the Hanoi recursion, the pyramid layout); every move is rehearsed and checked in physics.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .engine import Move, Obj, opening

CYL, BOX = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX


# ---------- Tower of Hanoi ----------
PEGS = {
    "A": np.array([0.20, 0.075]),
    "B": np.array([0.20, 0.0]),
    "C": np.array([0.20, -0.075]),
}
DISC_R = [0.014, 0.012, 0.010]  # bottom to top: widths the pivoting jaw still holds squarely
DISC_T = 0.016
DISC_RGBA = [(0.86, 0.26, 0.2, 1), (0.93, 0.66, 0.2, 1), (0.2, 0.62, 0.66, 1)]


def hanoi_solution(n: int, a="A", b="B", c="C") -> list[tuple[str, str]]:
    return (
        []
        if n == 0
        else hanoi_solution(n - 1, a, c, b) + [(a, c)] + hanoi_solution(n - 1, b, a, c)
    )


def hanoi():
    base = PEGS["A"]
    objs = [
        Obj(
            f"disc{k}",
            r,
            DISC_T,
            DISC_RGBA[k],
            (base[0], base[1], DISC_T * (k + 0.5)),
            0.006,
            extras=[
                (
                    CYL,
                    [r * 0.35, DISC_T / 2 + 0.0005, 0],
                    [0, 0, 0],
                    (0.12, 0.1, 0.09, 1),
                )
            ],
        )  # the hole, drawn
        for k, r in enumerate(DISC_R)
    ]

    def statics(w):
        w.add_geom(
            name="hanoi_base",
            type=BOX,
            size=[0.03, 0.115, 0.002],
            pos=[0.20, 0.0, -0.001],
            rgba=[0.55, 0.38, 0.24, 1],
            contype=0,
            conaffinity=0,
        )
        for (
            name,
            xy,
        ) in PEGS.items():  # drawn only: a real peg would need a hole in every disc
            w.add_geom(
                name=f"peg_{name}",
                type=CYL,
                size=[0.0035, 0.024, 0],
                pos=[xy[0], xy[1], 0.024],
                rgba=[0.62, 0.45, 0.3, 1],
                contype=0,
                conaffinity=0,
            )

    stacks = {"A": [0, 1, 2], "B": [], "C": []}
    moves = []
    for a, c in hanoi_solution(3):
        k = stacks[a].pop()
        grasp = DISC_T * (len(stacks[a]) + 0.5)
        place = DISC_T * (len(stacks[c]) + 0.5)
        stacks[c].append(k)
        tallest = max(len(s) for s in stacks.values()) * DISC_T
        moves.append(
            Move(
                k,
                PEGS[a],
                grasp,
                PEGS[c],
                place,
                opening(DISC_R[k] * 2000),
                carry_z=tallest + DISC_T / 2 + 0.009,
                hold=opening(DISC_R[k] * 2000, -4.0),  # the soft pads close ~4 mm past first contact
                tol=0.008, knock=0.005,
                label=f"disc {3 - k} from {a} to {c}",
            )
        )
    return objs, statics, moves


# ---------- cup pyramid ----------
CUP_R, CUP_H = 0.0105, 0.022
PYR = np.array([0.215, 0.0])


def cups():
    supply = [np.array([0.13, y]) for y in np.linspace(-0.10, 0.10, 6)]
    objs = [
        Obj(
            f"cup{k}",
            CUP_R,
            CUP_H,
            (0.84, 0.18, 0.2, 1),
            (p[0], p[1], CUP_H / 2),
            0.008,
            extras=[
                (
                    CYL,
                    [CUP_R * 1.12, 0.0012, 0],
                    [0, 0, -CUP_H / 2 + 0.0012],
                    (0.95, 0.9, 0.88, 1),
                ),
                (
                    CYL,
                    [CUP_R * 0.82, 0.0006, 0],
                    [0, 0, CUP_H / 2 + 0.0004],
                    (0.62, 0.1, 0.12, 1),
                ),
            ],
        )
        for k, p in enumerate(supply)
    ]
    step = 2 * CUP_R + 0.0025
    layout = [
        (PYR + [0, -step], 0),
        (PYR + [0, 0], 0),
        (PYR + [0, step], 0),  # bottom row
        (PYR + [0, -step / 2], 1),
        (PYR + [0, step / 2], 1),
        (PYR, 2),
    ]  # middle, top
    moves = []
    for k, (xy, level) in enumerate(layout):
        place = CUP_H * (level + 0.5)
        built = CUP_H * (
            level if level else 1
        )  # the tallest thing already standing on the pyramid
        moves.append(
            Move(
                k,
                supply[k],
                CUP_H / 2 + 0.002,
                xy,
                place,
                opening(CUP_R * 2000),
                carry_z=built + CUP_H + 0.012,
                hold=opening(CUP_R * 2000, -4.0),
                label=f"cup {k + 1} to row {level + 1}",
            )
        )

    def statics(w):
        w.add_geom(
            name="mat",
            type=BOX,
            size=[0.05, 0.06, 0.0008],
            pos=[PYR[0], PYR[1], -0.0008],
            rgba=[0.2, 0.34, 0.6, 1],
            contype=0,
            conaffinity=0,
        )

    return objs, statics, moves


TASKS = {"hanoi": hanoi, "cups": cups}

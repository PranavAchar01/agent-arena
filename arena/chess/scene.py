"""MuJoCo chess scene: the SO-101 at the side of a small board, 32 code-built pieces. It plays both colours.

Geometry is set by the arm's reach with the jaws pointing straight down (measured: at 2 cm height it reaches
x <= 0.28 m for |y| <= 0.10 m). Squares are 2.4 cm, the board centre is 0.19 m in front of the base, so the whole
board sits inside that envelope. Pieces are 1.3 cm wide at the base, which leaves room for the finger pads between
neighbours when the jaws close along a file.

Coordinates: the arm looks along +x. Rank r (0 = rank 1, white) runs along +x; file f (0 = a) runs from +y to -y,
so a1 is on the arm's left, as it is for a white player sitting where the arm sits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chess
import mujoco
import numpy as np

from ..sim.scene import (
    BLOCK_BIT,
    GRASP_SOLIMP,
    GRASP_SOLREF,
    GRIP_CLOSED,
    HOME,
    JOINTS,
    MESH_BIT,
    STEPS_PER_CONTROL,
    _base_spec,
    _look_at,
    _pad_frames,
)

SQ = 0.022
CENTER = np.array([0.195, 0.0])
BASE_R = 0.006
BOARD_TOP = 0.0
TRAY = {
    chess.WHITE: (0.21, 0.135),
    chess.BLACK: (0.21, -0.135),
}  # captured pieces line up beside the board
BASE_H = 0.016  # the grippable column every piece stands on (the pads start 2 mm above the jaw tips, and the tips must stay clear of the desk)
PIECE_H = {
    chess.PAWN: 0.025,
    chess.KNIGHT: 0.029,
    chess.BISHOP: 0.031,
    chess.ROOK: 0.027,
    chess.QUEEN: 0.033,
    chess.KING: 0.035,
}
MAX_H = max(PIECE_H.values())


def square_xy(sq: int) -> np.ndarray:
    """The arm sits beside the a-file: files run away from it along +x, White's ranks on its right (-y)."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return CENTER + np.array([(f - 3.5) * SQ, (r - 3.5) * SQ])


def tray_xy(color: bool, k: int) -> np.ndarray:
    """k-th captured piece of `color`: two rows of eight beyond its own back rank."""
    side = -1 if color == chess.WHITE else 1
    return np.array([0.125 + (k % 8) * 0.02, side * (0.112 + (k // 8) * 0.02)])


def _cyl(body, name, r, z0, z1, mat, **kw):
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[r, (z1 - z0) / 2, 0],
        pos=[0, 0, (z0 + z1) / 2],
        material=mat,
        **kw,
    )


LOOK = (
    Path(__file__).resolve().parents[2] / "vendor" / "chess_set" / "mujoco"
)  # scripts/build_chess_assets.py (Poly Haven "Chess Set", CC0)
MARBLE = (LOOK / "board.png").is_file()
MESH = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _piece_geoms(body, name: str, kind: int, mat: str, contact: dict):
    """The grip cylinder is the only part that collides. It is drawn as a scanned Staunton mesh when the marble assets
    are built, else as a low-poly silhouette from primitives."""
    vis = {"contype": 0, "conaffinity": 0, "mass": 0}
    if MARBLE:
        _cyl(
            body,
            f"{name}_base",
            BASE_R,
            0,
            BASE_H,
            mat,
            mass=0.012,
            friction=[0.9, 0.005, 0.0001],
            condim=4,
            group=3,
            rgba=[0, 0, 0, 0],
            **contact,
        )
        body.add_geom(
            name=f"{name}_look",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"mesh_{MESH[kind]}",
            material=mat,
            **vis,
        )
        return
    _cyl(
        body,
        f"{name}_base",
        BASE_R,
        0,
        BASE_H,
        mat,
        mass=0.012,
        friction=[0.9, 0.005, 0.0001],
        condim=4,
        **contact,
    )
    h = PIECE_H[kind]
    if kind == chess.PAWN:
        _cyl(body, f"{name}_neck", 0.0038, 0.0120, 0.0190, mat, **vis)
        body.add_geom(
            name=f"{name}_head",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.0045, 0, 0],
            pos=[0, 0, 0.0245],
            material=mat,
            **vis,
        )
    elif kind == chess.ROOK:
        _cyl(body, f"{name}_tower", 0.0052, 0.012, h - 0.004, mat, **vis)
        _cyl(body, f"{name}_top", 0.006, h - 0.004, h, mat, **vis)
    elif kind == chess.KNIGHT:
        _cyl(body, f"{name}_neck", 0.0045, 0.0120, 0.0190, mat, **vis)
        body.add_geom(
            name=f"{name}_head",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.0065, 0.0035, 0.0045],
            pos=[-0.002, 0, 0.0275],
            quat=[0.966, 0, 0.259, 0],
            material=mat,
            **vis,
        )
    elif kind == chess.BISHOP:
        _cyl(body, f"{name}_neck", 0.004, 0.0120, 0.0220, mat, **vis)
        body.add_geom(
            name=f"{name}_mitre",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=[0.0045, 0.0045, 0.0055],
            pos=[0, 0, 0.0285],
            material=mat,
            **vis,
        )
    elif kind == chess.QUEEN:
        _cyl(body, f"{name}_neck", 0.0042, 0.0120, 0.0250, mat, **vis)
        _cyl(body, f"{name}_crown", 0.0058, 0.0250, 0.0290, mat, **vis)
        body.add_geom(
            name=f"{name}_orb",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.0025, 0, 0],
            pos=[0, 0, 0.0345],
            material=mat,
            **vis,
        )
    else:  # king
        _cyl(body, f"{name}_neck", 0.0045, 0.0120, 0.0270, mat, **vis)
        _cyl(body, f"{name}_crown", 0.0058, 0.0270, 0.0290, mat, **vis)
        body.add_geom(
            name=f"{name}_cross_v",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.0012, 0.0012, 0.0028],
            pos=[0, 0, 0.0352],
            material=mat,
            **vis,
        )
        body.add_geom(
            name=f"{name}_cross_h",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.0012, 0.0028, 0.0012],
            pos=[0, 0, 0.0362],
            material=mat,
            **vis,
        )


@dataclass
class PieceInfo:
    body: str
    joint: str
    color: bool
    kind: int


@dataclass
class ChessScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    pieces: list[PieceInfo]
    at: dict[int, int] = field(default_factory=dict)  # square -> piece index
    captured: dict[bool, int] = field(
        default_factory=lambda: {chess.WHITE: 0, chess.BLACK: 0}
    )

    @classmethod
    def make(cls, board: chess.Board | None = None) -> ChessScene:
        board = board or chess.Board()
        model, pieces = build()
        sc = cls(model, mujoco.MjData(model), pieces)
        sc.reset(board)
        return sc

    @property
    def qadr(self):
        return np.array([self.model.joint(j).qposadr[0] for j in JOINTS])

    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def site(self) -> np.ndarray:
        return self.data.site_xpos[self.model.site("gripperframe").id].copy()

    def piece_pos(self, i: int) -> np.ndarray:
        return self.data.xpos[self.model.body(self.pieces[i].body).id].copy()

    def reset(self, board: chess.Board):
        mujoco.mj_resetData(self.model, self.data)
        q = np.append(HOME, GRIP_CLOSED)
        self.data.qpos[self.qadr] = q
        self.data.ctrl[:] = q
        self.at = {}
        self.captured = {chess.WHITE: 0, chess.BLACK: 0}
        free = list(range(len(self.pieces)))
        for sq, p in board.piece_map().items():
            i = next(
                k
                for k in free
                if self.pieces[k].color == p.color
                and self.pieces[k].kind == p.piece_type
            )
            free.remove(i)
            self._place(i, square_xy(sq))
            self.at[sq] = i
        for i in free:
            c = self.pieces[i].color
            if self.pieces[i].body.endswith(
                "q_spare"
            ):  # promotion queens wait under the table until needed
                self._place(i, np.array([0.2, 0.3 if c == chess.WHITE else -0.3]))
                a = self.model.joint(self.pieces[i].joint).qposadr[0]
                self.data.qpos[a + 2] = -0.74
                continue
            self._place(
                i, tray_xy(c, self.captured[c])
            )  # any other piece missing from the position starts in its tray
            self.captured[c] += 1
        mujoco.mj_forward(self.model, self.data)

    def _place(self, i: int, xy):
        a = self.model.joint(self.pieces[i].joint).qposadr[0]
        # the knight's head is modelled facing -x; turn it toward the opponent (White looks along +y)
        yaw = -np.pi / 2 if self.pieces[i].color == chess.WHITE else np.pi / 2
        self.data.qpos[a : a + 3] = [xy[0], xy[1], BOARD_TOP]
        self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        v = self.model.joint(self.pieces[i].joint).dofadr[0]
        self.data.qvel[v : v + 6] = 0

    def step(self, ctrl: np.ndarray, n: int = STEPS_PER_CONTROL):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, n)

    def snapshot(self):
        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy(),
            dict(self.at),
            dict(self.captured),
        )

    def restore(self, snap):
        qpos, qvel, ctrl, at, captured = snap
        self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:] = qpos, qvel, ctrl
        self.at, self.captured = dict(at), dict(captured)
        mujoco.mj_forward(self.model, self.data)


def build() -> tuple[mujoco.MjModel, list[PieceInfo]]:
    pads = _pad_frames()
    spec = _base_spec()
    w = spec.worldbody
    spec.add_material(
        name="desk",
        rgba=[0.16, 0.15, 0.15, 1] if MARBLE else [0.83, 0.8, 0.74, 1],
        reflectance=0.12 if MARBLE else 0.04,
    )
    spec.add_material(name="floor", rgba=[0.14, 0.15, 0.18, 1])
    if MARBLE:
        for k in MESH.values():
            spec.add_mesh(name=f"mesh_{k}", file=str(LOOK / f"{k}.obj"))
        for tex in ("board", "pieces_white", "pieces_black"):
            spec.add_texture(
                name=f"tex_{tex}",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=str(LOOK / f"{tex}.png"),
            )
        spec.add_material(
            name="marble_board",
            textures=["", "tex_board"],
            specular=0.35,
            shininess=0.6,
            reflectance=0.06,
        )
    spec.add_material(name="sq_light", rgba=[0.9, 0.86, 0.76, 1])
    spec.add_material(name="sq_dark", rgba=[0.45, 0.33, 0.24, 1])
    spec.add_material(name="frame", rgba=[0.25, 0.18, 0.13, 1])
    if MARBLE:
        spec.add_material(
            name="white_piece",
            textures=["", "tex_pieces_white"],
            specular=0.5,
            shininess=0.7,
        )
        spec.add_material(
            name="black_piece",
            textures=["", "tex_pieces_black"],
            specular=0.5,
            shininess=0.7,
        )
    else:
        spec.add_material(
            name="white_piece", rgba=[0.95, 0.93, 0.87, 1], specular=0.3, shininess=0.4
        )
        spec.add_material(
            name="black_piece", rgba=[0.12, 0.12, 0.13, 1], specular=0.4, shininess=0.5
        )
    w.add_light(
        name="key",
        pos=[0.35, -0.45, 1.2],
        dir=[-0.2, 0.4, -1],
        diffuse=[0.75] * 3,
        specular=[0.15] * 3,
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
        friction=[0.7, 0.005, 0.0001],
    )
    half = 4 * SQ
    if (
        MARBLE
    ):  # one marble board: 8x8 real tiles plus a frame, laid out to this exact grid
        edge = half + SQ * 56 / 160
        w.add_geom(
            name="board",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[edge, edge, 0.002],
            pos=[CENTER[0], CENTER[1], -0.0015],
            quat=[0.7071068, 0, 0, 0.7071068],
            material="marble_board",
            contype=0,
            conaffinity=0,
        )
    else:
        w.add_geom(
            name="frame",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half + 0.008, half + 0.008, 0.0008],
            pos=[CENTER[0], CENTER[1], -0.0008],
            material="frame",
            contype=0,
            conaffinity=0,
        )
        for sq in chess.SQUARES:
            xy = square_xy(sq)
            dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
            w.add_geom(
                name=f"sq_{chess.square_name(sq)}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[SQ / 2, SQ / 2, 0.0004],
                pos=[xy[0], xy[1], -0.0004 + 0.00005],
                material="sq_dark" if dark else "sq_light",
                contype=0,
                conaffinity=0,
            )

    # the same grasp contact scheme as the desk tasks: jaw meshes never touch pieces, thin finger pads do
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
            friction=[1.2, 0.005, 0.0001],
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
    }
    pieces: list[PieceInfo] = []
    start = chess.Board()
    for color in (chess.WHITE, chess.BLACK):
        for kind in (
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ):
            n = len(start.pieces(kind, color))
            for k in range(n):
                name = f"{'w' if color else 'b'}{chess.piece_symbol(kind)}{k}"
                body = w.add_body(name=name, pos=[0, 0, 0])
                body.add_freejoint(name=f"{name}_free")
                _piece_geoms(
                    body, name, kind, "white_piece" if color else "black_piece", contact
                )
                pieces.append(PieceInfo(name, f"{name}_free", color, kind))
    # promotions need a spare queen per side
    for color in (chess.WHITE, chess.BLACK):
        name = f"{'w' if color else 'b'}q_spare"
        body = w.add_body(name=name, pos=[0, 0, 0])
        body.add_freejoint(name=f"{name}_free")
        _piece_geoms(
            body, name, chess.QUEEN, "white_piece" if color else "black_piece", contact
        )
        pieces.append(PieceInfo(name, f"{name}_free", color, chess.QUEEN))

    tv = [
        0.40,
        -0.26,
        0.22,
    ]  # three-quarter view from White's side: the arm reaches over the whole board
    w.add_camera(name="judge", pos=tv, xyaxes=_look_at(tv, [0.18, 0.0, 0.01]), fovy=38)
    side = [0.2, -0.36, 0.22]
    w.add_camera(
        name="side", pos=side, xyaxes=_look_at(side, [0.18, 0.0, 0.02]), fovy=48
    )
    top = [0.195, 0.0, 0.6]
    w.add_camera(name="top", pos=top, xyaxes=[0, -1, 0, 1, 0, 0], fovy=36)
    return spec.compile(), pieces

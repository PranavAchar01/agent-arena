"""Build the chess look from Poly Haven's CC0 "Chess Set" (vendor/chess_set): six piece meshes and a marble board.

  python scripts/build_chess_assets.py      -> vendor/chess_set/mujoco/{pawn,...,king}.obj, board.png

Pieces: one instance of each type is taken from the glTF (Y-up, metres, 58 mm squares), turned Z-up, centred on its
base and scaled to the simulated board (base about 12 mm, heights in the real set's proportions). They are visual
only; the physics keeps the 16 mm grip cylinder under each piece.
Board: real marble tiles are cut from the set's texture atlas and laid out 8x8 with a dark marble frame, so every
square lines up exactly with the simulated board.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "vendor" / "chess_set"
OUT = SRC / "mujoco"
sys.path.insert(0, str(ROOT))
from arena.chess.scene import SQ  # noqa: E402

NODES = {  # glTF node prefix -> piece type (the black instance; white shares the geometry and UV layout)
    "piece_pawn_black_01": "pawn",
    "piece_rook_black_01": "rook",
    "piece_knight_black_01": "knight",
    "piece_bishop_black_01": "bishop",
    "piece_queen_black": "queen",
    "piece_king_black": "king",
}
HEIGHT_SCALE = 0.33  # real king 95 mm -> 31 mm, pawn 54 mm -> 18 mm
BASE_D = 0.0125  # widest base after scaling, just over the 12 mm grip cylinder
TILE, PX, BORDER = 150, 160, 56  # atlas tile pitch, output px per square, frame px


def pieces():
    sc = trimesh.load(SRC / "chess_set_2k.gltf", force="scene")
    OUT.mkdir(exist_ok=True)
    y_to_z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    for prefix, kind in NODES.items():
        parts = []
        for node in sc.graph.nodes_geometry:
            if node.startswith(prefix):
                T, g = sc.graph[node]
                m = sc.geometry[g].copy()
                m.apply_transform(T)
                parts.append(m)
        m = trimesh.util.concatenate(parts)
        m.apply_transform(y_to_z)
        lo, hi = m.bounds
        m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        base = max(hi[0] - lo[0], hi[1] - lo[1])
        s_h = min(BASE_D / base, 0.42)
        m.apply_scale([s_h, s_h, HEIGHT_SCALE])
        m.export(OUT / f"{kind}.obj", include_normals=True, include_texture=True)
        lo, hi = m.bounds
        print(
            f"{kind:7s} base {1000 * (hi[0] - lo[0]):.1f} x {1000 * (hi[1] - lo[1]):.1f} mm, height {1000 * hi[2]:.1f} mm"
        )
    for leftover in OUT.glob("material*"):
        leftover.unlink()


def board():
    atlas = Image.open(SRC / "textures" / "chess_set_board_diff_2k.jpg").convert("RGB")
    a = np.asarray(atlas.convert("L"), float)
    light, dark = [], []
    for r in range(4):
        for c in range(7):
            x0, y0 = 141 + TILE * c + 18, 11 + TILE * r + 18
            box = (x0, y0, x0 + TILE - 36, y0 + TILE - 36)
            (
                light if a[box[1] : box[3], box[0] : box[2]].mean() > 140 else dark
            ).append(atlas.crop(box))
    rng = np.random.default_rng(1997)
    n = 8 * PX + 2 * BORDER
    out = Image.new("RGB", (n, n))
    frame = atlas.crop((180, 800, 1380, 1900)).resize((n, n))
    out.paste(frame, (0, 0))
    for r in range(8):
        for c in range(8):
            pool = (
                dark if (r + c) % 2 == 0 else light
            )  # MuJoCo maps image row 0 to rank 1 (checked with a red corner); a1 is dark
            t = (
                pool[rng.integers(len(pool))]
                .rotate(90 * int(rng.integers(4)))
                .resize((PX, PX), Image.LANCZOS)
            )
            out.paste(t, (BORDER + c * PX, BORDER + r * PX))
    out.save(OUT / "board.png")
    print(
        f"board {n}px, {len(light)} light + {len(dark)} dark tiles, frame {BORDER / PX * SQ * 1000:.1f} mm"
    )


if __name__ == "__main__":
    pieces()
    board()

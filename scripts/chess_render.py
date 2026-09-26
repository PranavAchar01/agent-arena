"""Render a whole game played by the simulated SO-101: every move rehearsed in physics, then played and filmed.

  python scripts/chess_render.py [--pgn FILE] [--stride 18] [--out runs/chess/deepblue-g6]

Frames are taken every `stride` control steps (50 Hz) and written at 30 fps, so the video runs at 30*stride/50 x
real time (stride 18 = 10.8x). Writes game.mp4, moves.json (SAN, ply, start/end time in the video, physics check
per transfer) and poster.jpg.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.chess.move import Mover  # noqa: E402
from arena.chess.scene import ChessScene  # noqa: E402

DEEP_BLUE_G6 = """[Event "IBM Man-Machine, New York USA"]
[Site "New York, NY USA"]
[Date "1997.05.11"]
[Round "6"]
[White "Deep Blue (Computer)"]
[Black "Garry Kasparov"]
[Result "1-0"]

1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Nd7 5.Ng5 Ngf6 6.Bd3 e6 7.N1f3 h6 8.Nxe6 Qe7 9.O-O fxe6 10.Bg6+ Kd8
11.Bf4 b5 12.a4 Bb7 13.Re1 Nd5 14.Bg3 Kc8 15.axb5 cxb5 16.Qd3 Bc6 17.Bf5 exf5 18.Rxe7 Bxe7 19.c4 1-0
"""
W, H, FPS = 1280, 720, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", type=Path)
    ap.add_argument("--stride", type=int, default=18)
    ap.add_argument("--camera", default="judge")
    ap.add_argument("--out", type=Path, default=Path("runs/chess/deepblue-g6"))
    a = ap.parse_args()
    game = chess.pgn.read_game(
        io.StringIO(a.pgn.read_text() if a.pgn else DEEP_BLUE_G6)
    )
    a.out.mkdir(parents=True, exist_ok=True)
    board = game.board()
    sc = ChessScene.make(board)
    mv = Mover(sc)
    for _ in range(60):
        sc.step(sc.data.ctrl.copy())
    r = mujoco.Renderer(sc.model, H, W)
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(a.out / "game.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    n = [0, 0]  # control steps seen, frames written
    last = [None]

    def frame():
        n[0] += 2  # Mover calls frame() every 2 control steps
        if n[0] % a.stride:
            return
        r.update_scene(sc.data, camera=a.camera)
        img = r.render()
        last[0] = img
        ff.stdin.write(img.tobytes())
        n[1] += 1

    def hold(seconds):
        for _ in range(int(seconds * FPS)):
            ff.stdin.write(last[0].tobytes())
            n[1] += 1

    frame_first = n[0]
    r.update_scene(sc.data, camera=a.camera)
    last[0] = r.render()
    hold(1.0)  # the starting position
    moves, t0 = [], time.time()
    for ply, m in enumerate(game.mainline_moves()):
        san = board.san(m)
        start = n[1] / FPS
        res = mv.execute(board, m, frame=frame)
        if not res["ok"]:
            ff.stdin.close()
            ff.wait()
            raise SystemExit(f"move {ply + 1} {san} failed every rehearsal: {res}")
        board.push(m)
        moves.append(
            {
                "ply": ply + 1,
                "san": san,
                "uci": m.uci(),
                "fen": board.fen(),
                "start_s": round(start, 2),
                "end_s": round(n[1] / FPS, 2),
                "transfers": res["transfers"],
            }
        )
        print(f"{ply + 1:2d} {san:7s} {moves[-1]['start_s']:6.1f}s", flush=True)
    hold(1.5)  # the final position
    ff.stdin.close()
    ff.wait()
    from PIL import Image

    Image.fromarray(last[0]).save(a.out / "poster.jpg", quality=90)
    speed = FPS * a.stride / 50
    headers = dict(game.headers)
    (a.out / "moves.json").write_text(
        json.dumps(
            {
                "headers": headers,
                "speed": speed,
                "fps": FPS,
                "stride": a.stride,
                "frames": n[1],
                "video_s": round(n[1] / FPS, 2),
                "sim_s": round((n[0] - frame_first) / 50, 1),
                "compute_s": round(time.time() - t0, 1),
                "moves": moves,
            },
            indent=1,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )
    print(
        f"video {n[1] / FPS:.1f}s at {speed:.1f}x · sim {(n[0] - frame_first) / 50:.0f}s · {a.out / 'game.mp4'}"
    )


if __name__ == "__main__":
    main()

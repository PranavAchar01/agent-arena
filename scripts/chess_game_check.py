"""Play Deep Blue vs Kasparov 1997 game 6 through the physics: every move rehearsed and checked (python scripts/chess_game_check.py)."""

import io, time, chess, chess.pgn
from arena.chess.scene import ChessScene
from arena.chess.move import Mover

PGN = "1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Nd7 5.Ng5 Ngf6 6.Bd3 e6 7.N1f3 h6 8.Nxe6 Qe7 9.O-O fxe6 10.Bg6+ Kd8 11.Bf4 b5 12.a4 Bb7 13.Re1 Nd5 14.Bg3 Kc8 15.axb5 cxb5 16.Qd3 Bc6 17.Bf5 exf5 18.Rxe7 Bxe7 19.c4 1-0"
game = chess.pgn.read_game(io.StringIO(PGN))
b = game.board()
sc = ChessScene.make(b)
mv = Mover(sc)
for _ in range(60):
    sc.step(sc.data.ctrl.copy())
t_all = time.time()
fails = 0
for i, m in enumerate(game.mainline_moves()):
    san = b.san(m)
    t0 = time.time()
    r = mv.execute(b, m)
    b.push(m)
    tr = r["transfers"]
    print(
        f"{i + 1:2d} {san:7s} {'OK ' if r['ok'] else 'FAIL'} {[(x['piece'], x.get('err_mm'), x.get('axis'), x.get('open')) for x in tr]} {time.time() - t0:.1f}s",
        flush=True,
    )
    if not r["ok"]:
        fails += 1
        break
print("total", round(time.time() - t_all, 1), "s")

# Chess pivot (his call, Sat 2026-09-26 ~14:00 PT)

"One use case, use Scrapling, change the stack. The live demo is something really ambitious: a chess match where a
robot moves chess pieces." Live demo slot is 3 minutes. The real SO-101 is ready ~Monday, so the event demo is the
simulated SO-101 in MuJoCo.

## Build order
1. `arena/chess/scene.py`: SO-101 + 8x8 board, 2.4 cm squares, board centre x=0.19 (x 0.094..0.286, y +-0.096),
   32 code-built low-poly pieces (free bodies). Reach map (jaws down): z=0.02 reaches x<=0.28 for |y|<=0.10.
2. `arena/chess/move.py`: square-to-square pick and place (captures go to a tray first), jaws aligned with the file
   axis, opening about 2 cm, lift profile from human chess-move shapes. Every move is simulated headless first and
   verified (piece on target square, nothing else disturbed); only a verified move is played to the viewer.
3. `arena/chess/game.py`: python-chess; Stockfish if installed, else a small minimax.
4. Server `/api/chess/*` + WebSocket JPEG stream of the robot executing its move; `web/chess.html`.
5. Scrapling replaces httpx fetch/parse in the sandbox scraper; the agent finds openly licensed chess-move videos,
   MediaPipe gives the human move shape (lift height, timing) the robot uses.

## Honesty lines
- Simulation until the real arm is calibrated. Say "simulated SO-101".
- The move shape comes from people; the square-to-square path is IK. Don't call it an end-to-end learned policy.

# Chess pivot (his call, Sat 2026-09-26 ~14:00 PT)

"One use case, use Scrapling, change the stack. The live demo is something really ambitious: a chess match where a
robot moves chess pieces." Live demo slot is 3 minutes. The real SO-101 is ready ~Monday, so the event demo is the
simulated SO-101 in MuJoCo.

## Build order
1. DONE `arena/chess/scene.py`: SO-101 at the SIDE of the board (his call: it plays both colours without moving),
   2.2 cm squares, board centre x=0.195, 12 mm pieces on a 16 mm grippable base, trays beyond each back rank.
2. DONE `arena/chess/move.py`: closed-loop pick and place (the soft STS3215 servos lag ~5 mm), carry height capped by
   a measured reach ceiling that also rejects self-collision, joint rate limit, 6 rehearsal variants per transfer.
   Deep Blue vs Kasparov 1997 G6: 37/37 moves pass (<= 5 mm off centre, nothing knocked > 3 mm).
3. DONE `arena/chess/agent.py` + `pipeline.py`: prompt -> search (Scrapling in sandbox) -> pick article by number ->
   extract all move lists -> pick game by number -> strict python-chess parse -> cached physics replay by move hash.
4. DONE server mode "chess", /chess/{key}/game.mp4, web/chess.html (?demo=1 types the prompt). Replay rendered on a
   Vultr VM (scripts/vultr_render.py, OSMesa) and cached under runs/chess/<key>/. Vultr warm pool: one fresh VM
   pre-booted so a live box starts instantly.
5. DONE sandbox/scraper/pgn.py (Scrapling). Human video motion shapes are NOT used in chess yet (synthetic timing).

## Honesty lines
- Simulation until the real arm is calibrated. Say "simulated SO-101".
- The square-to-square path is IK + closed-loop control, not a learned policy. Don't call it learned.
- The replay is precomputed physics (about 2 min of compute for 37 moves), shown sped up; say so.

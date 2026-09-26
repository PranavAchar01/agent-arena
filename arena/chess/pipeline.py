"""Prompt to robot: "Replay Deep Blue vs. Kasparov, 1997, Game 6".

1. the agent turns the request into a search (model output is small JSON, validated)
2. a throwaway sandbox searches Wikipedia with Scrapling and lists article titles
3. the agent picks articles by number; a fresh sandbox fetches them and extracts every move list on the page
4. the agent picks which list is the requested game by number; python-chess re-parses it move by move
5. the robot's physics replay for that exact move sequence is looked up (every move rehearsed and checked in
   MuJoCo ahead of time, see scripts/chess_render.py / scripts/vultr_render.py) and streamed to the page
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..sandbox.runner import get_runner
from . import agent

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "runs" / "chess"
UA = "HandoffArena/0.1 (hackathon prototype; one request per page)"


def replay_info(k: str) -> dict | None:
    f = CACHE / k / "moves.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    tr = [t for m in d["moves"] for t in m["transfers"]]
    return {
        "key": k,
        "video": f"/chess/{k}/game.mp4",
        "poster": f"/chess/{k}/poster.jpg",
        "speed": d["speed"],
        "video_s": d["video_s"],
        "sim_s": d["sim_s"],
        "compute_s": d["compute_s"],
        "plies": len(d["moves"]),
        "captures": sum(1 for m in d["moves"] if "x" in m["san"]),
        "transfers": len(tr),
        "max_err_mm": max((t.get("err_mm") or 0) for t in tr),
        "retries": sum(
            1
            for t in tr
            if t.get("open") not in (None, -0.05) or t.get("axis") == "rank"
        ),
        "moves": [
            {k2: m[k2] for k2 in ("ply", "san", "uci", "fen", "start_s", "end_s")}
            for m in d["moves"]
        ],
    }


def run(text: str, run_dir: Path, emit) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    want = agent.resolve(text)
    emit({"type": "plan", **want})
    events: list[dict] = []

    def fwd(e):
        events.append(e)
        emit(e)

    get_runner().run(
        {"mode": "pgn", "query": want["query"], "time_budget_s": 60, "user_agent": UA},
        run_dir / "search",
        fwd,
    )
    cands = [e for e in events if e["type"] == "candidate"]
    picks = agent.pick_pages([c["title"] for c in cands], want)
    emit(
        {
            "type": "picked",
            "n": len(picks),
            "of": len(cands),
            "titles": [cands[i]["title"] for i in picks],
        }
    )
    if not picks:
        emit({"type": "error", "message": "no article looked like it holds this game"})
        return {}
    man = get_runner().run(
        {
            "mode": "pgn",
            "pages": [cands[i]["page"] for i in picks],
            "time_budget_s": 90,
            "user_agent": UA,
        },
        run_dir,
        fwd,
    )
    games = man.get("games", [])
    order = agent.pick_game(games, want) if games else []
    game = None
    for i in order:
        g, why = agent.parse_moves(games[i]["moves"])
        emit(
            {
                "type": "validate",
                "index": i,
                "ok": g is not None,
                "reason": why,
                "context": games[i]["context"][-80:],
                "plies": g.end().board().ply() if g else 0,
            }
        )
        if g is not None:
            game = g
            src = games[i]
            break
    if game is None:
        emit(
            {
                "type": "error",
                "message": "no scraped move list passed the python-chess check",
            }
        )
        return {}
    k = agent.key(game)
    sans, b = [], game.board()
    for m in game.mainline_moves():
        sans.append(b.san(m))
        b.push(m)
    emit(
        {
            "type": "game",
            "key": k,
            "plies": len(sans),
            "sans": sans,
            "source": src["page"],
            "title": src["title"],
            "context": src["context"][-100:],
            "white": want["white"],
            "black": want["black"],
            "year": want["year"],
        }
    )
    info = replay_info(k)
    if info:
        emit({"type": "replay", **info})
    else:
        emit(
            {
                "type": "not_rehearsed",
                "key": k,
                "message": "this game has not been rehearsed in physics yet; "
                "run scripts/vultr_render.py for it (about 8 minutes on a Vultr VM)",
            }
        )
    summary = {
        "text": text,
        "want": want,
        "key": k,
        "source": src["page"],
        "seconds": round(time.time() - t0, 1),
    }
    (run_dir / "run.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "game.pgn").write_text(agent.pgn_text(game))
    emit({"type": "done", "timings_s": {"total": summary["seconds"]}})
    return summary

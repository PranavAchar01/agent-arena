"""Seeded run: same pipeline, but the clips to download are titles a human picked from the search results
(recorded as seeded=True). The VLM verifier, motion gates and physics gates are unchanged."""

import json, sys, time
from pathlib import Path
from arena.pipeline import run
from arena.agent import plan as plan_search

SEEDS = {
    "push": (
        ["blitz chess", "chess blitz", "chess"],
        ["Joaquin Perkins blitz chess vs USCF expert.webm", "Chess Blitz Game 1"],
    ),
    "place": (
        ["blitz chess", "chess blitz", "chess"],
        ["Joaquin Perkins blitz chess vs USCF expert.webm", "Chess Blitz Game 1"],
    ),
    "stack": (
        ["lego building", "building lego", "unboxing building lego"],
        [
            "Building the Lego Saturn V.webm",
            "Building Lego Mining Truck 4202",
            "Unboxing and Building Lego City Tow Truck 7638",
            "Building and Unboxing Lego City Garage 4207",
        ],
    ),
}
TEXT = {
    "push": "push a block onto a target",
    "place": "put a block in a bowl",
    "stack": "stack a block on top of another block",
}
task = sys.argv[1]
d = Path("runs") / f"{task}-seeded"
d.mkdir(parents=True, exist_ok=True)
log = open(d / "events.jsonl", "w")


def emit(e):
    e = {**e, "at": round(time.time(), 2)}
    log.write(json.dumps(e, default=str) + "\n")
    log.flush()
    if e["type"] in (
        "picked",
        "verdict",
        "moves",
        "episodes",
        "dataset",
        "trained",
        "evaluated",
        "error",
    ):
        print(json.dumps(e, default=str)[:240], flush=True)


text = f"I want to train a robot to {TEXT[task]}"
p = plan_search(text)
p["family"] = task
p["queries"] = SEEDS[task][0]
p["include"] = []
p["exclude"] = []
run(text, d, emit, plan=p, pick_titles=SEEDS[task][1], clip_seconds=90)

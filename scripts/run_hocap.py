"""A -> B -> C on HO-Cap (CC BY 4.0): real people picking up and setting down objects at a table."""

import json, re, sys, time
from pathlib import Path
from arena.pipeline import run

Y = Path("runs/hocap/hocap_recordings.yaml").read_text()
url = lambda k: re.search(rf"^{k}: *(\S+)", Y, re.M).group(1)
TEXT = {
    "place": "put the block in the bowl",
    "stack": "stack the block on another block",
    "tower": "stack one more block to make a tower",
}
task, subjects, per = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
d = Path("runs") / (sys.argv[5] if len(sys.argv) > 5 else f"hocap-{task}")
d.mkdir(parents=True, exist_ok=True)
log = open(d / "events.jsonl", "w")


def emit(e):
    e = {**e, "at": round(time.time(), 2)}
    log.write(json.dumps(e, default=str) + "\n")
    log.flush()
    if e["type"] in (
        "plan",
        "clip",
        "verdict",
        "moves",
        "episodes",
        "dataset",
        "trained",
        "evaluated",
        "error",
        "done",
    ):
        print(json.dumps(e, default=str)[:220], flush=True)


ds = {
    "dataset": "hocap",
    "subject_urls": {s: url(s) for s in subjects},
    "camera": "043422252387",
    "stride": 3,
    "sandbox_seconds": 900,
    "time_budget_s": 860,
    "per_subject": per,
    "max_clips": 8,
    "clip_seconds": 40,
}
import os

os.environ["AUG_PER_SHAPE"] = sys.argv[4] if len(sys.argv) > 4 else "24"
import arena.pipeline as P

P.AUG_PER_SHAPE = int(os.environ["AUG_PER_SHAPE"])
run(f"I want to train a robot to {TEXT[task]}", d, emit, family=task, dataset=ds)

"""CLI: run one task end to end and log every event.  usage: run_task.py NAME "I want to train a robot to ..." """
import json, sys, time
from pathlib import Path
from arena.pipeline import run
name, text = sys.argv[1], sys.argv[2]
d = Path("runs") / name
d.mkdir(parents=True, exist_ok=True)
log = open(d / "events.jsonl", "w")
def emit(e):
    log.write(json.dumps(e, default=str) + "\n"); log.flush()
    if e["type"] not in ("candidate", "skip", "train"):
        print(json.dumps(e, default=str)[:260], flush=True)
run(text, d, emit)

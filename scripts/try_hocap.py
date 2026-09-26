import json, re, sys, time
from pathlib import Path
from arena.sandbox.runner import get_runner

Y = Path("runs/hocap/hocap_recordings.yaml").read_text()
url = lambda k: re.search(rf"^{k}: *(\S+)", Y, re.M).group(1)
subs = sys.argv[2].split(",")
job = {
    "dataset": "hocap",
    "subject_urls": {s: url(s) for s in subs},
    "camera": sys.argv[4] if len(sys.argv) > 4 else "105322251225",
    "stride": int(sys.argv[5]) if len(sys.argv) > 5 else 2,
    "per_subject": int(sys.argv[3]),
    "max_clips": 8,
    "clip_seconds": 40,
    "time_budget_s": 280,
    "user_agent": "AgentArenaProto/0.1 (https://github.com/PranavAchar01; hackathon prototype) python-httpx/0.28",
}
t0 = time.time()
m = get_runner().run(
    job, Path(sys.argv[1]), lambda e: print(json.dumps(e)[:200], flush=True)
)
print("kept", len(m["clips"]), f"{time.time() - t0:.0f}s")

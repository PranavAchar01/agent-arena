import json, sys, time
from pathlib import Path
from arena.sandbox.runner import get_runner

H = "http://host.docker.internal:8765"
job = {
    "queries": sys.argv[2:] or ["stacking blocks"],
    "max_clips": int(sys.argv[1]),
    "per_query": 10,
    "extra_pages": [
        f"{H}/loop",
        f"{H}/huge.html",
        f"{H}/inject.html",
        f"{H}/script.html",
        f"{H}/fake.html",
    ],
}
t0 = time.time()
m = get_runner().run(
    job, Path("runs/try"), lambda e: print(json.dumps(e)[:230], flush=True)
)
print("kept", len(m["clips"]), f"{time.time() - t0:.0f}s")

"""Plan with the agent, then run the sandbox in listing mode (no downloads) to see what exists."""
import json, sys, time
from pathlib import Path
from arena.agent import plan
from arena.sandbox.runner import get_runner
UA = "AgentArenaProto/0.1 (https://github.com/PranavAchar01; hackathon prototype) python-httpx/0.28"
text = sys.argv[1]
t0 = time.time(); p = plan(text); print("PLAN", f"{time.time()-t0:.0f}s", json.dumps(p))
cands = []
def on(e):
    if e["type"] == "candidate": cands.append(e)
    elif e["type"] == "search": print("search", e["source"], e["query"], e["hits"])
get_runner().run({"queries": p["queries"], "include": p["include"], "exclude": p["exclude"], "per_query": 20,
                  "max_clips": 0, "user_agent": UA}, Path("runs/discover"), on)
ok = [c for c in cands if c["licence"]]
print(len(cands), "candidates")
for c in sorted(cands, key=lambda c: -c["score"])[:25]:
    print(c["score"], c["source"][:4], c["licence"][:30], "|", c["title"][:90])

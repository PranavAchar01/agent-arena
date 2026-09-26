import json, sys
from pathlib import Path
from arena.sandbox.runner import get_runner
UA = "AgentArenaProto/0.1 (https://github.com/PranavAchar01; hackathon prototype) python-httpx/0.28"
qs = sys.argv[1].split(",")
cands, hits = [], []
def on(e):
    if e["type"] == "candidate": cands.append(e)
    elif e["type"] == "search": hits.append((e["source"][:4], e["query"], e["hits"]))
    elif e["type"] == "skip" and "licence" in e.get("reason",""): cands.append({**e, "score": -9, "source": "", "licence": e["reason"][-30:]})
get_runner().run({"queries": qs, "include": [], "exclude": [], "per_query": 25, "max_clips": 0, "user_agent": UA}, Path("runs/probe"), on)
print(hits)
for c in cands: print(c.get("score"), c["source"][:4], (c.get("licence") or "")[:28], "|", c["title"][:95])

import sys, json

for l in open(f"runs/{sys.argv[1]}/events.jsonl"):
    e = json.loads(l)
    t = e["type"]
    if t == "verdict":
        print("V", e["accept"], e["title"][:40], "|", e["reason"][:120])
    elif t == "moves":
        print("M", e["id"], e["found"], e["rejected"], e["reasons"])
    elif t == "motion":
        print(
            "  mv",
            e["move"],
            e["ok"],
            e["probe"],
            e.get("max_lift"),
            e.get("move_hand_units"),
        )
    elif t in ("episodes", "dataset", "trained", "evaluated", "error", "picked"):
        print(t, {k: v for k, v in e.items() if k not in ("type", "at")})
    elif t == "done" and "timings_s" in e:
        print("timings", e["timings_s"])

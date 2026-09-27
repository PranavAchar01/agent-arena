"""Queue the library's prompts on a fleet server and wait for them all.

  python scripts/fleet.py [--server http://127.0.0.1:8803]

The fleet server is the normal app (arena.server) started with SANDBOX_BACKEND=docker, FLEET_UNIQUE=1 (no video is
used twice across the library), BOX_WORKERS=N and, when the big Vultr VM is up, HEAVY_SSH. Writes runs/library.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "Find a really easy workout an SO-101 can do",
    "Teach my robot arm to do bicep curls",
    "Make my SO-101 curl a dumbbell like a gym beginner",
    "Show my arm how to do alternating dumbbell curls",
    "Teach my arm hammer curls",
    "Hammer curls with a light dumbbell",
    "Neutral-grip curls for my robot",
    "Seated concentration curls",
    "Teach my SO-101 a strict concentration curl",
    "Overhead triceps extensions for my robot arm",
    "Teach my arm a dumbbell triceps extension",
    "Seated overhead triceps extension",
    "Triceps kickbacks",
    "Teach my robot a dumbbell kickback",
    "Teach my arm front raises",
    "Dumbbell front raise for the shoulders",
    "A slow front raise, like a physio exercise",
    "Lateral raises",
    "Side raises with a light dumbbell",
    "Teach my arm a shoulder raise for beginners",
    "Dumbbell shoulder press",
    "Teach my robot to press a dumbbell overhead",
    "Seated shoulder press",
    "Upright rows",
    "Teach my arm a dumbbell upright row",
    "A physiotherapy arm exercise my SO-101 can copy",
    "A warm-up arm exercise for my robot",
    "What's a good arm workout for a robot?",
    "An exercise that trains the shoulder",
    "An exercise that trains the triceps",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8803")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.server, timeout=60)
    ids = []
    for text in PROMPTS:
        rid = c.post("/api/runs", json={"text": text, "mode": "workout"}).json()["id"]
        ids.append((rid, text))
        print("queued", rid, text, flush=True)
    status: dict[str, str] = {}
    t0 = time.time()
    while len(status) < len(ids) and time.time() - t0 < 6 * 3600:
        for rid, text in ids:
            if rid in status:
                continue
            try:
                ev = c.get(f"/api/runs/{rid}").json()["events"]
            except (httpx.HTTPError, ValueError):
                continue
            last = ev[-1]["type"] if ev else ""
            if last in ("error", "killed") or (
                last == "done" and "timings_s" in ev[-1]
            ):
                status[rid] = (
                    "ok"
                    if last == "done"
                    else f"{last}: {ev[-1].get('message', '')[:80]}"
                )
                print(
                    f"[{len(status)}/{len(ids)}] {rid} {status[rid]} | {text}",
                    flush=True,
                )
        time.sleep(15)
    lib = [
        {"id": rid, "text": text, "status": status.get(rid, "unfinished")}
        for rid, text in ids
    ]
    (ROOT / "runs" / "library.json").write_text(json.dumps(lib, indent=1))
    print("done:", sum(1 for x in lib if x["status"] == "ok"), "of", len(lib))


if __name__ == "__main__":
    main()

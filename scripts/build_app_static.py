"""Static build of the Replay app (web/index.html) for Vercel: the six library robots with their recorded runs.

python scripts/build_app_static.py [--out app-site]

The live pipeline (sandboxes, MuJoCo, MediaPipe) needs the local server; the hosted page serves each run's recorded
events, videos and downloads as files, and a typed prompt replays a recorded run (site.js says so in the box).
Run scripts/build_library.py first: the downloads are its skill.zip files.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CHESS_KEY = "91342743086ddfc4"
# run id on the local server -> id in the library build (for its skill.zip)
SHOWN = {
    "workout-030427-b43c": "workout-030427-b43c",
    "box-place": "box-place",
    "box-tower": "box-tower",
    "task-cups": "task-cups",
    "task-hanoi": "task-hanoi",
    "chess-140740-0d00": "chess-deep-blue-g6",
}
MEDIA = ("robot.mp4", "showcase.mp4", "policy.mp4")
SECRETISH = re.compile(r"\b[A-Z0-9]{36}\b|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "app-site")
    out = ap.parse_args().out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out, ignore=shutil.ignore_patterns("library.html"))
    (out / "data" / "run").mkdir(parents=True)
    (out / "data" / "dl").mkdir()
    listing = []
    for rid, lib_id in SHOWN.items():
        src = RUNS / rid
        ev = src / "events.jsonl"
        events = (
            [json.loads(x) for x in ev.read_text().splitlines() if x.strip()]
            if ev.is_file()
            else []
        )
        blob = json.dumps(events)
        if SECRETISH.search(blob):
            raise SystemExit(
                f"{rid}: something key-shaped in the events, not publishing"
            )
        summary = (
            json.loads((src / "run.json").read_text())
            if (src / "run.json").is_file()
            else None
        )
        moves = (
            json.loads((src / "moves.json").read_text())
            if (src / "moves.json").is_file()
            else None
        )
        (out / "data" / "run" / f"{rid}.json").write_text(
            json.dumps(
                {"id": rid, "events": events, "summary": summary, "moves": moves},
                default=str,
            )
        )
        d = out / "runs" / rid
        d.mkdir(parents=True)
        for name in MEDIA:
            if (src / name).is_file():
                shutil.copy2(src / name, d / name)
        for kind in ("clips", "frames"):
            if (src / kind).is_dir():
                shutil.copytree(src / kind, d / kind)
        z = ROOT / "site" / "runs" / lib_id / "skill.zip"
        if z.is_file():
            shutil.copy2(z, out / "data" / "dl" / f"{rid}.zip")
        listing.append(
            {
                "id": rid,
                "finished": round((src / "run.json").stat().st_mtime) if summary else 0,
                "text": (summary or {}).get("text"),
            }
        )
    chess = out / "chess" / CHESS_KEY
    chess.mkdir(parents=True)
    for name in ("game.mp4", "poster.jpg"):
        if (RUNS / "chess" / CHESS_KEY / name).is_file():
            shutil.copy2(RUNS / "chess" / CHESS_KEY / name, chess / name)
    (out / "data" / "runs.json").write_text(json.dumps({"runs": listing, "live": []}))
    (out / "vercel.json").write_text(
        json.dumps(
            {
                "cleanUrls": False,
                "rewrites": [
                    {"source": "/api/runs", "destination": "/data/runs.json"},
                    {
                        "source": "/api/runs/:id/download",
                        "destination": "/data/dl/:id.zip",
                    },
                    {"source": "/api/runs/:id", "destination": "/data/run/:id.json"},
                ],
                "headers": [
                    {
                        "source": "/data/dl/(.*)",
                        "headers": [
                            {"key": "Content-Disposition", "value": "attachment"}
                        ],
                    }
                ],
            },
            indent=1,
        )
    )
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{len(listing)} robots, {size / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()

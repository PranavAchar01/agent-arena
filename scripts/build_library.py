"""Build the public Replay library (static site) from finished runs, ready for `vercel deploy`.

  python scripts/build_library.py [--out site]

Takes every finished workout run (runs/workout-*/run.json) plus the rehearsed chess game, compresses the robot videos,
writes a poster of each person the robots learned from (their clearest MediaPipe frame), repackages each skill as a
small zip (trajectories, pose, sources, README; no video) and writes site/index.html + site/data.json.
Every source is openly licensed; the page credits each author and links the original.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CHESS_KEY = "91342743086ddfc4"
CHESS_RUN = "chess-140740-0d00"


def enc(src: Path, dst: Path, w: int, crf: int) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-an",
            "-vf",
            f"scale={w}:-2",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        capture_output=True,
        timeout=300,
    )
    return r.returncode == 0


def poster(clip: Path, t: float, dst: Path) -> bool:
    if not clip.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, t):.2f}",
            "-i",
            str(clip),
            "-frames:v",
            "1",
            "-vf",
            "scale=320:-2,format=yuvj420p",
            "-q:v",
            "5",
            str(dst),
        ],
        capture_output=True,
        timeout=60,
    )
    return r.returncode == 0 and dst.is_file()


def clearest(pose: dict, window=None) -> float:
    """Time of the frame where MediaPipe saw the arm most clearly (from the stored elbow trace)."""
    t, ang = pose.get("t", []), pose.get("angle", [])
    ok = [
        i
        for i, a in enumerate(ang)
        if a is not None and (not window or window[0] <= t[i] <= window[1])
    ]
    return float(t[ok[len(ok) // 2]]) if ok else 2.0


def workout(run: Path, out: Path) -> dict | None:
    s = json.loads((run / "run.json").read_text())
    if s.get("kind") != "workout" or not (run / "robot.mp4").is_file():
        return None
    rid = run.name
    d = out / "runs" / rid
    enc(run / "robot.mp4", d / "robot.mp4", 640, 30)
    sources = (
        json.loads((run / "sources.json").read_text())
        if (run / "sources.json").is_file()
        else []
    )
    pose = (
        json.loads((run / "human_pose.json").read_text())
        if (run / "human_pose.json").is_file()
        else {}
    )
    people = {p["id"]: p for p in s.get("people", [])} or (
        {s["best"]: {**s["robot"], "id": s["best"]}} if s.get("best") else {}
    )
    src_out = []
    for src in sources:
        cid = src.get("id")
        if not cid:
            continue
        item = {k: src.get(k) for k in ("title", "page", "author", "licence", "reps")}
        pr = people.get(cid)
        if pr:
            item["robot"] = (
                f"runs/{rid}/robots/{cid}.mp4"
                if enc(
                    run / "robots" / f"{cid}.mp4", d / "robots" / f"{cid}.mp4", 400, 32
                )
                else None
            )
            item["copied"] = {
                k: pr.get(k)
                for k in ("reps", "elbow_tracking_rms_deg", "max_slip_mm", "ok")
            }
            item["best"] = cid == s.get("best")
        if poster(
            run / "clips" / f"{cid}.mp4",
            clearest(pose.get(cid, {}), (pr or {}).get("window")),
            d / "people" / f"{cid}.jpg",
        ):
            item["poster"] = f"runs/{rid}/people/{cid}.jpg"
        src_out.append(item)
    src_out.sort(
        key=lambda x: (not x.get("best"), not x.get("robot"), -(x.get("reps") or 0))
    )
    z = d / "skill.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("README.md", "skill.csv", "human_pose.json", "sources.json"):
            if (run / name).is_file():
                zf.write(run / name, name)
        for f in (run / "skills").glob("*.csv") if (run / "skills").is_dir() else []:
            zf.write(f, f"skills/{f.name}")
    rb = s.get("robot", {})
    return {
        "id": rid,
        "kind": "workout",
        "prompt": s["text"],
        "exercise": s.get("exercise", "dumbbell biceps curl"),
        "video": f"runs/{rid}/robot.mp4",
        "download": f"runs/{rid}/skill.zip",
        "videos": s.get("clips"),
        "robots": s.get("robots", 1),
        "reps": s.get("reps"),
        "error_deg": rb.get("elbow_tracking_rms_deg"),
        "slip_mm": rb.get("max_slip_mm"),
        "ok": rb.get("ok"),
        "sources": src_out,
        "finished": round((run / "run.json").stat().st_mtime),
    }


def chess(out: Path) -> dict | None:
    rep = RUNS / "chess" / CHESS_KEY
    if not (rep / "game.mp4").is_file():
        return None
    d = out / "runs" / "chess-deep-blue-g6"
    enc(rep / "game.mp4", d / "robot.mp4", 640, 28)
    moves = json.loads((rep / "moves.json").read_text())
    tr = [t for m in moves["moves"] for t in m["transfers"]]
    with zipfile.ZipFile(d / "skill.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(rep / "moves.json", "moves.json")
        if (RUNS / CHESS_RUN / "game.pgn").is_file():
            zf.write(RUNS / CHESS_RUN / "game.pgn", "game.pgn")
        zf.writestr(
            "README.md",
            "# SO-101 plays Deep Blue vs. Kasparov, 1997, Game 6\n\nmoves.json: every move, the "
            "pick-and-places it took and the physics check of each (placement error, rehearsal variant).\n"
            "game.pgn: the game as scraped from Wikipedia and checked move by move with python-chess.\n",
        )
    return {
        "id": "chess-deep-blue-g6",
        "kind": "chess",
        "prompt": "Replay Deep Blue vs. Kasparov, 1997, Game 6",
        "exercise": "chess, both sides",
        "video": "runs/chess-deep-blue-g6/robot.mp4",
        "download": "runs/chess-deep-blue-g6/skill.zip",
        "moves": len(moves["moves"]),
        "robots": 1,
        "transfers": len(tr),
        "error_mm": max(t.get("err_mm") or 0 for t in tr),
        "ok": True,
        "sources": [
            {
                "title": "Deep Blue versus Garry Kasparov",
                "page": "https://en.wikipedia.org/wiki/Deep_Blue_versus_Garry_Kasparov",
                "author": "Wikipedia contributors",
                "licence": "CC BY-SA 4.0",
            }
        ],
        "finished": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    a = ap.parse_args()
    out = a.out
    if (out / "runs").exists():
        shutil.rmtree(out / "runs")
    out.mkdir(exist_ok=True)
    items = []
    for run in sorted(RUNS.glob("workout-*")):
        if (run / "run.json").is_file():
            try:
                it = workout(run, out)
            except (json.JSONDecodeError, KeyError):
                it = None
            if it:
                items.append(it)
                print("added", run.name, it["exercise"], flush=True)
    items.sort(key=lambda x: -x["finished"])
    c = chess(out)
    if c:
        items.insert(0, c)
    (out / "data.json").write_text(json.dumps({"items": items}, indent=1))
    shutil.copyfile(ROOT / "web" / "library.html", out / "index.html")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{len(items)} skills, {size / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()

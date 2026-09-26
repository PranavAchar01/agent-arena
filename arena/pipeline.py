"""One run, end to end: sentence -> search plan -> sandboxed scrape -> VLM verification -> hand motion -> SO-101
retarget + physics gate -> dataset -> policy -> evaluation on fixed unseen layouts -> rendered video.

Every step reports through emit(event_dict); the web app streams those events. Nothing here touches the open web:
all fetching and decoding of downloaded media happens in the sandbox.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from . import llm
from .agent import plan as plan_search
from .agent import rank
from .layouts import PROBES, sample
from .motion import shapes_from_pose, track_hands
from .policy import N_EVAL, evaluate, train
from .sandbox.runner import get_runner
from .sources import pexels
from .sim.ik import IK
from .sim.retarget import replay
from .sim.scene import Scene
from .verify import verify

ROOT = Path(__file__).resolve().parents[1]
UA = os.environ.get(
    "SCRAPER_UA",
    "AgentArenaProto/0.1 (https://github.com/PranavAchar01; hackathon prototype) "
    "python-httpx/0.28",
)
HOSTILE = os.environ.get("HOSTILE_BASE", "http://host.docker.internal:8765")
AUG_PER_SHAPE = int(os.environ.get("AUG_PER_SHAPE", "24"))
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", "8"))


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t0 = time.time()

            def __exit__(self, *a):
                timer.t[name] = round(time.time() - self.t0, 1)

        return _T()


def renderer(model, w=640, h=360, camera="front"):
    r = mujoco.Renderer(model, h, w)

    def f(sc):
        r.update_scene(sc.data, camera=camera)
        return r.render()

    return f


def write_mp4(frames, path: Path, fps=25):
    if not frames:
        return
    h, w, _ = frames[0].shape
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    for fr in frames:
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()


def run(
    text: str,
    run_dir: Path,
    emit,
    family: str | None = None,
    hostile: bool = True,
    plan: dict | None = None,
    pick_titles: list[str] | None = None,
    clip_seconds: int = 40,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    T = Timer()
    summary: dict = {
        "text": text,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": llm.describe(),
    }

    # 1. the agent plans the search
    with T("plan"):
        plan = plan or plan_search(text)
    if family:
        plan["family"] = family
    task = plan["family"]
    emit({"type": "plan", **plan})
    summary["plan"] = plan

    # 2. the sandbox scrapes (and absorbs the planted hostile pages)
    events = []

    def fwd(e):
        events.append(e)
        emit(e)

    # phase A: a sandbox searches and lists open-licence candidates (no downloads)
    search_job = {
        "queries": plan["queries"],
        "include": plan["include"],
        "exclude": plan["exclude"],
        "per_query": 20,
        "max_clips": 0,
        "user_agent": UA,
        "time_budget_s": 120,
        "extra_pages": [
            f"{HOSTILE}/{p}"
            for p in ("loop", "huge.html", "inject.html", "script.html", "fake.html")
        ]
        if hostile
        else [],
    }
    with T("search"):
        get_runner().run(search_job, run_dir / "search", fwd)
    found = [e for e in events if e["type"] == "candidate"]
    archive = [c for c in found if c.get("source") != "web"]
    keyed = pexels(plan["queries"])  # optional: only when PEXELS_API_KEY is set (the key stays in the app)
    if keyed:
        emit({"type": "search", "source": "Pexels API (app side)", "query": ", ".join(plan["queries"][:3]), "hits": len(keyed)})
        archive += keyed
    planted = [c for c in found if c.get("source") == "web"]
    # the agent reads the titles (as data) and picks what to download; it can only answer with line numbers
    with T("rank"):
        if pick_titles:  # seeded run: a human chose the titles; the VLM and every gate still apply
            chosen = [c for c in archive if c["title"] in pick_titles]
        else:
            picks = rank(archive, plan["summary"] or text, task, k=MAX_CLIPS) if archive else []
            chosen = [archive[i] for i in picks]
    emit(
        {
            "type": "picked",
            "n": len(chosen),
            "of": len(archive),
            "titles": [c["title"] for c in chosen],
        }
    )
    # phase B: a fresh sandbox downloads the picks (plus the planted test links) and re-encodes them
    fetch_job = {
        "queries": [],
        "fetch": chosen + planted,
        "max_clips": MAX_CLIPS + len(planted),
        "clip_seconds": clip_seconds,
        "user_agent": UA,
        "time_budget_s": 240,
    }
    with T("scrape"):
        manifest = get_runner().run(fetch_job, run_dir, fwd)
    clips = manifest["clips"]
    summary["scrape"] = {
        "searched": sum(e.get("hits", 0) for e in events if e["type"] == "search"),
        "candidates": sum(1 for e in events if e["type"] == "candidate"),
        "downloaded": len(clips),
        "blocked": [e for e in events if e["type"] == "blocked"],
    }

    # 3. the VLM verifies each clip
    verdicts = {}
    with T("verify"):
        for c in clips:
            emit({"type": "verifying", "id": c["id"], "title": c["title"]})
            v = verify(c, run_dir, plan["summary"] or text, task)
            verdicts[c["id"]] = v
            emit(
                {
                    "type": "verdict",
                    "id": c["id"],
                    "title": c["title"],
                    "licence": c["licence"],
                    "source": c["source"],
                    "author": c["author"],
                    "page": c["page"],
                    **v,
                }
            )
    summary["verify"] = verdicts

    # 4. hand motion -> shape -> robot physics gate on three probe layouts
    sc = Scene.make(task)
    ik = IK(sc.model)
    shapes = []
    with T("motion"):
        for c in clips:
            v = verdicts[c["id"]]
            if not v["accept"]:
                continue
            pose = track_hands(
                run_dir / "clips" / f"{c['id']}.mp4",
                v["window"],
                run_dir / f"pose_{c['id']}.json",
            )
            found, rejected = shapes_from_pose(pose, task, c["id"])
            emit({"type": "moves", "id": c["id"], "found": len(found), "rejected": len(rejected),
                  "reasons": sorted({r["reason"] for r in rejected})[:4]})
            for shape, q in found:
                probe = [replay(sc, shape, b, t, y, ik=ik).success for b, t, y in PROBES[task]]
                ok = sum(probe) >= 2
                emit({"type": "motion", "id": c["id"], "move": shape.source, "ok": ok, "probe": probe, **q,
                      "reason": "" if ok else f"robot replay failed the task on {3 - sum(probe)} of 3 probe layouts"})
                if ok:
                    shapes.append(shape)
    (run_dir / "shapes.json").write_text(json.dumps([s.to_json() for s in shapes]))
    summary["shapes"] = [s.source for s in shapes]

    # 5. dataset: every accepted human shape replayed on new layouts, each rollout re-gated by physics
    episodes, meta = [], []
    rng = np.random.default_rng(1)
    with T("dataset"):
        for s in shapes:
            kept = 0
            for _ in range(AUG_PER_SHAPE):
                b, t, y = sample(task, rng)
                ep = replay(sc, s, b, t, y, ik=ik)
                if ep.success:
                    episodes.append((ep.obs, ep.act))
                    meta.append(
                        {
                            "source": s.source,
                            "block": b,
                            "target": t,
                            "yaw": y,
                            "steps": len(ep.act),
                        }
                    )
                    kept += 1
            emit(
                {
                    "type": "episodes",
                    "source": s.source,
                    "kept": kept,
                    "tried": AUG_PER_SHAPE,
                }
            )
    if episodes:
        np.savez_compressed(
            run_dir / "dataset.npz",
            obs=np.concatenate([o for o, _ in episodes]),
            act=np.concatenate([a for _, a in episodes]),
            episode_index=np.concatenate(
                [np.full(len(a), i) for i, (_, a) in enumerate(episodes)]
            ),
        )
        (run_dir / "episodes.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
    summary["dataset"] = {
        "episodes": len(episodes),
        "frames": int(sum(len(a) for _, a in episodes)),
        "bytes": (run_dir / "dataset.npz").stat().st_size if episodes else 0,
    }
    emit({"type": "dataset", **summary["dataset"]})

    # 6. train + evaluate
    if len(episodes) >= 5:
        with T("train"):
            info = train(episodes, run_dir / "policy.pt", log=emit)
        emit({"type": "trained", **info})
        rend = renderer(sc.model)
        with T("evaluate"):
            res, vids = evaluate(
                task, run_dir / "policy.pt", render_first=4, render=rend
            )
        frames = [f for _, fr in vids for f in fr]
        write_mp4(frames, run_dir / "policy.mp4")
        summary["policy"] = {
            **info,
            "eval_success": int(sum(res)),
            "eval_n": N_EVAL,
            "eval": res,
            "state_based": True,
        }
        emit(
            {
                "type": "evaluated",
                "success": int(sum(res)),
                "n": N_EVAL,
                "video": "policy.mp4",
            }
        )
    else:
        summary["policy"] = None
        emit(
            {
                "type": "evaluated",
                "success": 0,
                "n": 0,
                "reason": "fewer than 5 gated episodes; nothing to train",
            }
        )
    summary["timings_s"] = T.t
    (run_dir / "run.json").write_text(json.dumps(summary, indent=1, default=str))
    emit({"type": "done", "timings_s": T.t})
    return summary

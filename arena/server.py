"""The VM-style backend the web app talks to. REST to start runs, a WebSocket to stream a run's events.

  POST /api/runs            {"text": "...", "family"?: "push|place|stack"}  -> {"id"}   (one heavy run at a time)
  GET  /api/runs            recorded and live runs
  GET  /api/runs/{id}       every event so far + run.json when finished
  WS   /api/runs/{id}/ws    events as they happen (replays history first)
  POST /api/runs/{id}/kill  kill switch: destroys the box's sandbox (its Vultr VM on that backend) and stops the run
  GET  /api/runs/{id}/audit hash-chained JSONL audit log (every fetch, block, model call, VM create/destroy)
  GET  /api/runs/{id}/audit/verify   recomputes the chain
  GET  /api/config          which LLM / VLM endpoint and sandbox backend are configured
  GET  /runs/{id}/...       stills, clips, policy video

Run: .venv/bin/python -m uvicorn arena.server:app --port 8800
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
import re
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .pipeline import run as run_pipeline
from .sandbox.runner import BoxKilled, box, check_killed, close_pool, start_pool

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
WEB = ROOT / "web"
RUN_ID = re.compile(r"^[a-z0-9-]{3,40}$")

app = FastAPI(title="agent-arena-proto")
_lock = threading.Lock()
_live: dict[str, list[dict]] = {}
_boxes: dict = {}
GENESIS = "0" * 64


class NewRun(BaseModel):
    text: str = Field(min_length=8, max_length=200)
    family: str | None = Field(default=None, pattern="^(place|stack|tower)$")
    mode: str = Field(default="robot", pattern="^(robot|chess|workout)$")
    robot: str = Field(
        default="so101", pattern="^(so101)$"
    )  # the only arm simulated in this prototype


def _events(rid: str) -> list[dict]:
    if rid in _live:
        return list(_live[rid])
    f = RUNS / rid / "events.jsonl"
    if not f.exists():
        raise HTTPException(404, "no such run")
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


@app.get("/api/config")
def config():
    return {
        "models": llm.describe(),
        "sandbox": os.environ.get("SANDBOX_BACKEND", "docker"),
        "workers": WORKERS,
        "queued": _queue.qsize(),
    }


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(RUNS.glob("*/run.json")):
        s = json.loads(d.read_text())
        pol = s.get("policy") or {}
        out.append(
            {
                "id": d.parent.name,
                "text": s.get("text"),
                "family": s.get("plan", {}).get("family"),
                "episodes": s.get("dataset", {}).get("episodes"),
                "success": pol.get("eval_success"),
                "n": pol.get("eval_n"),
            }
        )
    return {"runs": out, "live": list(_live)}


@app.get("/api/runs/{rid}")
def run_detail(rid: str):
    if not RUN_ID.match(rid):
        raise HTTPException(400, "bad id")
    summary = RUNS / rid / "run.json"
    return {
        "id": rid,
        "events": _events(rid),
        "summary": json.loads(summary.read_text()) if summary.exists() else None,
    }


# HO-Cap subjects per task family, so parallel boxes learn from different people
SUBJECTS = {
    "place": ["subject_1", "subject_2"],
    "stack": ["subject_5", "subject_6"],
    "tower": ["subject_3", "subject_4"],
}
_queue: "queue.Queue[tuple]" = queue.Queue()


def _hocap(family: str | None) -> dict:
    y = (ROOT / "runs" / "hocap" / "hocap_recordings.yaml").read_text()
    subs = SUBJECTS.get(family or "place", SUBJECTS["place"])
    urls = {k: re.search(rf"^{k}: *(\S+)", y, re.M).group(1) for k in subs}
    return {
        "dataset": "hocap",
        "subject_urls": urls,
        "camera": "043422252387",
        "stride": 3,
        "per_subject": 2,
        "max_clips": 8,
        "clip_seconds": 40,
        "sandbox_seconds": 900,
        "time_budget_s": 860,
    }


def _chain(prev: str, e: dict) -> str:
    return hashlib.sha256(
        (prev + json.dumps(e, sort_keys=True, default=str)).encode()
    ).hexdigest()


def _worker():
    while True:
        rid, req, d = _queue.get()
        log = open(d / "events.jsonl", "a")
        head = [_live[rid][0]["h"]]

        def emit(e):
            e = {k: v for k, v in e.items() if k != "h"}
            e["at"] = round(time.time(), 2)
            e["h"] = head[0] = _chain(head[0], e)
            _live[rid].append(e)
            log.write(json.dumps(e, default=str) + "\n")
            log.flush()
            if e["type"] not in (
                "killed",
                "error",
                "vm_destroyed",
                "sandbox_destroyed",
                "warn",
            ):
                check_killed()  # a killed box stops at its next step, wherever it is

        try:
            with box(rid) as b:
                _boxes[rid] = b
                llm.audit.set(emit)
                if req.mode == "chess":
                    from .chess.pipeline import run as run_chess

                    run_chess(req.text, d, emit)
                    continue
                if req.mode == "workout":
                    from .workout.pipeline import run as run_workout

                    run_workout(req.text, d, emit)
                    continue
                fam = req.family
                if (
                    fam is None
                ):  # let the planner choose, then fetch from the matching people
                    from .agent import plan as plan_search

                    p = plan_search(req.text)
                    fam = p["family"]
                else:
                    p = None
                run_pipeline(req.text, d, emit, family=fam, plan=p, dataset=_hocap(fam))
        except BoxKilled as ex:
            emit({"type": "killed", "message": str(ex)})
        except Exception as ex:  # noqa: BLE001 - surface any failure to the page instead of dying silently
            emit({"type": "error", "message": f"{type(ex).__name__}: {str(ex)[:200]}"})
        finally:
            log.close()
            _boxes.pop(rid, None)
            _live.pop(rid, None)


# local Docker shares this Mac's CPU, so one box at a time; on Vultr every box has its own VM
WORKERS = int(
    os.environ.get(
        "BOX_WORKERS", "3" if os.environ.get("SANDBOX_BACKEND") == "vultr" else "1"
    )
)
for _ in range(WORKERS):
    threading.Thread(target=_worker, daemon=True).start()
start_pool()
atexit.register(
    close_pool
)  # warm VMs are deleted when the server exits; scripts/vultr.py sweep covers a hard kill


@app.post("/api/runs/{rid}/kill")
def kill(rid: str):
    if not RUN_ID.match(rid):
        raise HTTPException(400, "bad id")
    b = _boxes.get(rid)
    if b is None:
        raise HTTPException(409, "box is not running")
    threading.Thread(
        target=b.kill, daemon=True
    ).start()  # VM deletion can take a few seconds
    return {"id": rid, "killing": True}


@app.get("/api/runs/{rid}/audit")
def audit(rid: str):
    f = RUNS / rid / "events.jsonl"
    if not RUN_ID.match(rid) or not f.is_file():
        raise HTTPException(404)
    return PlainTextResponse(
        f.read_text(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{rid}-audit.jsonl"'},
    )


@app.get("/api/runs/{rid}/audit/verify")
def audit_verify(rid: str):
    f = RUNS / rid / "events.jsonl"
    if not RUN_ID.match(rid) or not f.is_file():
        raise HTTPException(404)
    prev, n = GENESIS, 0
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if "h" not in e:
            return {
                "ok": None,
                "reason": "recorded before the audit chain existed",
                "events": n,
            }
        h = e.pop("h")
        if _chain(prev, e) != h:
            return {"ok": False, "broken_at": n, "events": n}
        prev, n = h, n + 1
    return {"ok": True, "events": n, "head": prev}


@app.post("/api/runs")
def start(req: NewRun):
    rid = f"{ {'chess': 'chess', 'workout': 'workout'}.get(req.mode, 'live') }-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
    d = RUNS / rid
    d.mkdir(parents=True)
    first = {
        "type": "queued",
        "position": _queue.qsize(),
        "robot": req.robot,
        "text": req.text,
        "at": round(time.time(), 2),
    }
    first["h"] = _chain(GENESIS, first)
    _live[rid] = [first]
    (d / "events.jsonl").write_text(json.dumps(_live[rid][0]) + "\n")
    _queue.put((rid, req, d))
    return {"id": rid, "queued": _queue.qsize()}


@app.websocket("/api/runs/{rid}/ws")
async def stream(ws: WebSocket, rid: str):
    await ws.accept()
    if not RUN_ID.match(rid):
        await ws.close()
        return
    sent = 0
    try:
        while True:
            try:
                evs = _events(rid)
            except HTTPException:
                await ws.send_json({"type": "error", "message": "no such run"})
                break
            for e in evs[sent:]:
                await ws.send_json(e)
            sent = len(evs)
            if (evs and evs[-1]["type"] in ("done", "error")) and rid not in _live:
                break
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return
    await ws.close()


@app.get("/runs/{rid}/{kind}/{name}")
def media(rid: str, kind: str, name: str):
    if (
        not RUN_ID.match(rid)
        or kind not in ("frames", "clips")
        or not re.fullmatch(r"[0-9a-f]{10}[_a-z0-9]*\.(jpg|mp4)", name)
    ):
        raise HTTPException(404)
    f = RUNS / rid / kind / name
    if not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/runs/{rid}/showcase.mp4")
def showcase_video(rid: str):
    f = RUNS / rid / "showcase.mp4"
    if not RUN_ID.match(rid) or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/runs/{rid}/robot.mp4")
def robot_video(rid: str):
    f = RUNS / rid / "robot.mp4"
    if not RUN_ID.match(rid) or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/runs/{rid}/policy.mp4")
def policy_video(rid: str):
    f = RUNS / rid / "policy.mp4"
    if not RUN_ID.match(rid) or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/runs/{rid}/download")
def download(rid: str):
    """Every finished robot can be downloaded: its skill (trajectory or policy), sources, video and README."""
    import io
    import zipfile

    d = RUNS / rid
    if not RUN_ID.match(rid) or not d.is_dir():
        raise HTTPException(404)
    files = {
        n: d / n
        for n in (
            "README.md",
            "skill.csv",
            "human_pose.json",
            "sources.json",
            "robot.mp4",
            "policy.pt",
            "policy.mp4",
            "showcase.mp4",
            "manifest.json",
            "run.json",
            "game.pgn",
        )
    }
    summary = (
        json.loads((d / "run.json").read_text()) if (d / "run.json").is_file() else {}
    )
    key = summary.get("key")
    if key and re.fullmatch(
        r"[0-9a-f]{16}", key
    ):  # a chess run: the rehearsed game lives in the replay cache
        files["moves.json"] = RUNS / "chess" / key / "moves.json"
        files["game.mp4"] = RUNS / "chess" / key / "game.mp4"
    files["audit.jsonl"] = d / "events.jsonl"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if not (d / "README.md").is_file():
            zf.writestr(
                "README.md",
                f"# Replay robot {rid}\n\nPrompt: {summary.get('text', '')}\n\n"
                "policy.pt: trained SO-101 policy (state based) · moves.json/game.pgn: the rehearsed chess game · "
                "audit.jsonl: every step of the run, hash-chained.\n",
            )
        for name, f in files.items():
            if f.is_file() and f.stat().st_size < 200_000_000:
                zf.write(f, name)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="replay-{rid}.zip"'},
    )


@app.get("/chess/{key}/{name}")
def chess_media(key: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{16}", key) or name not in (
        "game.mp4",
        "poster.jpg",
        "moves.json",
    ):
        raise HTTPException(404)
    f = RUNS / "chess" / key / name
    if not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/chess/{key}")
def chess_replay(key: str):
    from .chess.pipeline import replay_info

    info = replay_info(key) if re.fullmatch(r"[0-9a-f]{16}", key) else None
    if info is None:
        raise HTTPException(404)
    return info


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

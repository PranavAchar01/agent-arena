"""The VM-style backend the web app talks to. REST to start runs, a WebSocket to stream a run's events.

  POST /api/runs            {"text": "...", "family"?: "push|place|stack"}  -> {"id"}   (one heavy run at a time)
  GET  /api/runs            recorded and live runs
  GET  /api/runs/{id}       every event so far + run.json when finished
  WS   /api/runs/{id}/ws    events as they happen (replays history first)
  GET  /api/config          which LLM / VLM endpoint and sandbox backend are configured
  GET  /runs/{id}/...       stills, clips, policy video

Run: .venv/bin/python -m uvicorn arena.server:app --port 8800
"""

from __future__ import annotations

import asyncio
import json
import re
import queue
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm
from .pipeline import run as run_pipeline
from .sandbox.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
WEB = ROOT / "web"
RUN_ID = re.compile(r"^[a-z0-9-]{3,40}$")

app = FastAPI(title="agent-arena-proto")
_lock = threading.Lock()
_live: dict[str, list[dict]] = {}


class NewRun(BaseModel):
    text: str = Field(min_length=8, max_length=200)
    family: str | None = Field(default=None, pattern="^(place|stack|tower)$")
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
        "sandbox": get_runner().name,
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


def _worker():
    while True:
        rid, req, d = _queue.get()
        log = open(d / "events.jsonl", "a")

        def emit(e):
            e = {**e, "at": round(time.time(), 2)}
            _live[rid].append(e)
            log.write(json.dumps(e, default=str) + "\n")
            log.flush()

        try:
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
        except Exception as ex:  # noqa: BLE001 - surface any failure to the page instead of dying silently
            emit({"type": "error", "message": f"{type(ex).__name__}: {str(ex)[:200]}"})
        finally:
            log.close()
            _live.pop(rid, None)


threading.Thread(target=_worker, daemon=True).start()


@app.post("/api/runs")
def start(req: NewRun):
    rid = f"live-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
    d = RUNS / rid
    d.mkdir(parents=True)
    _live[rid] = [
        {
            "type": "queued",
            "position": _queue.qsize(),
            "robot": req.robot,
            "text": req.text,
            "at": round(time.time(), 2),
        }
    ]
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


@app.get("/runs/{rid}/policy.mp4")
def policy_video(rid: str):
    f = RUNS / rid / "policy.mp4"
    if not RUN_ID.match(rid) or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

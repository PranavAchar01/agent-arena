# Agent Arena prototype: status

Local only. Never push. Resume from here after an app restart.

## How to run
- Claude shim (local OpenAI-compatible VLM/LLM): `.venv/bin/python -m uvicorn arena.claude_shim:app --port 8790` (pid in runs/shim.pid)
- Hostile test pages: `.venv/bin/python hostile/server.py 8765` (pid in runs/hostile.pid)
- Sandbox image: `cd sandbox && docker build -t arena-scraper:0.1 .`
- Web + API: preview config "arena" in ~/helloworld/.claude/launch.json (port 8800); replay: /?run=<id>&speed=8
- One task from the CLI: `scripts/run_task.py NAME "I want to train a robot to ..."`; all three: `scripts/run_all.sh`
- .venv is thin: heavy packages (torch, mujoco, cv2) come from ~/helloworld/so101/.venv via a .pth (disk floor);
  mediapipe runs in ~/helloworld/understudy/.venv-pose (POSE_PYTHON)

## Done
- [x] SO-101 MuJoCo scene, 3 tasks (push / place in bowl / stack), IK, retarget + physics replay (arena/sim)
- [x] Sandbox (Docker, no secrets, caps, --rm) + 5 planted hostile pages, all 5 blocked with distinct reasons
- [x] Agent: LLM plan -> search sandbox -> LLM ranks titles (indices only) -> fetch sandbox
- [x] VLM verifier (contact sheet, strict JSON, retry), MediaPipe multi-move extraction, physics probe gate
- [x] Dataset + ChunkMLP policy + 20-layout eval + policy.mp4; FastAPI server (REST + WS); web UI (dithered MuJoCo bg)

## Findings so far
- Zero-key open archives (Wikimedia Commons, Internet Archive) have very little tabletop hand footage. First pass:
  push 0 / place 0 usable clips (VLM rejected curling, mechanism animations, title cards correctly).
- Analogs with real volume on Commons: chess (many CC0 tournament videos), Go stones, speed stacking, Lego building.
- Optional Pexels source (app-side key, PEXELS_API_KEY) written but UNTESTED (no key; he must create it).

## Next
- [ ] Rerun all three with analog-aware planner/ranker
- [ ] docs/SCRAPER.md, docs/STATS.md (+ VLM accuracy hand-check, LoRA CPU measurement)
- [ ] Report to Pranav, wait for confirmation before filming (Phase 2)

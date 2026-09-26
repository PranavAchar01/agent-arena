# Agent Arena prototype: status

Local only. Never push. Resume from here after an app restart.

## Done
- [x] SO-101 MuJoCo scene, 3 tasks (push / place in bowl / stack), IK, retarget + physics replay (arena/sim)
- [x] Synthetic-shape check: all 3 tasks succeed in physics when the block is in the workspace (scripts/check_synthetic.py)

## Next
- [ ] Sandboxed scraper (Docker) + hostile test pages
- [ ] VLM verifier (OpenAI-compatible; local = Claude CLI shim)
- [ ] MediaPipe motion shape extraction
- [ ] Dataset + policy training + eval + render
- [ ] FastAPI backend + web UI
- [ ] docs/SCRAPER.md, docs/STATS.md

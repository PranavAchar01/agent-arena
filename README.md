# Handoff (Agent Arena prototype)

Type "I want to train a robot to X". An agent plans a search for openly licensed videos of people doing X; a
throwaway sandbox does every fetch and absorbs junk and hostile pages; a VLM checks each clip; MediaPipe hand motion
is retargeted to an SO-101 and gated by MuJoCo physics; a small policy trains on the CPU and is tested on 20 unseen
layouts. Local prototype for the Vultr x Cerebral Valley Agent Arena (rebuilt from scratch at the event).

- Decisions and evidence: docs/SCRAPER.md
- Measured numbers, logistics, timed plan: docs/STATS.md
- Current state and how to run: STATUS.md

## Agent infrastructure

- **One throwaway Vultr VM per robot box** (`SANDBOX_BACKEND=vultr`, arena/sandbox/runner.py). Per box: a fresh SSH
  key, a firewall group that admits only 22/tcp from the app's IP, a small CPU instance. Every sandbox job runs as
  the same hardened `docker run` as locally (read-only root, no capabilities, uid 10001, CPU/memory/pids caps,
  wall-clock kill, `--rm`), driven over SSH, so the VM is a second wall. Output comes back as a tar stream unpacked
  with the data filter (no links, no path escapes, size cap). VM, firewall and key are deleted when the box ends,
  crashes or is killed. The Vultr API key never leaves the app.
- **Kill switch**: `POST /api/runs/{id}/kill` (a Kill button on every box) destroys the sandbox and its VM at once;
  the run stops at its next step.
- **Audit log**: every event (fetch, block, model call with sizes and timing but no prompt, VM create/destroy with
  cost) is SHA-256 hash-chained. `GET /api/runs/{id}/audit` downloads it, `/audit/verify` recomputes the chain.
- **Parallel boxes**: `BOX_WORKERS` (default 3 on Vultr, 1 locally).

At the event:

```bash
export VULTR_API_KEY=...            # from the Vultr dashboard, never written to a file
.venv/bin/python scripts/vultr.py check
.venv/bin/python scripts/vultr.py smoke    # one VM end to end, about 3 minutes
SANDBOX_BACKEND=vultr .venv/bin/python -m uvicorn arena.server:app --port 8800
.venv/bin/python scripts/vultr.py sweep    # after a crash: delete leftover arena-sbx VMs, firewalls, keys
```

Optional: `VULTR_REGION` (sjc), `VULTR_PLAN` (vc2-1c-2gb), `VULTR_SNAPSHOT_ID` (Docker and the image baked in, skips
the ~2 minute setup), `VULTR_ALLOW_CIDR` (defaults to this machine's public IP; on networks that send SSH out through a different NAT IP, such as Shack15, use `0.0.0.0/0`: SSH stays key-only with a fresh key per box).

Models: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (+ VLM_* overrides). For Vultr Serverless Inference set
LLM_BASE_URL=https://api.vultrinference.com/v1 and a model id from `scripts/vultr.py check` (with VULTR_INFERENCE_KEY).
Also PEXELS_API_KEY (optional), SCRAPER_UA, POSE_PYTHON.

Tests: `.venv/bin/python -m pytest tests/` (Vultr lifecycle against a fake API; the SSH path against local Docker).

Simulation only. The policy reads robot state and object positions, not camera pixels.

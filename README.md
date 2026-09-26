# Handoff (Agent Arena prototype)

Type "I want to train a robot to X". An agent plans a search for openly licensed videos of people doing X; a
throwaway sandbox does every fetch and absorbs junk and hostile pages; a VLM checks each clip; MediaPipe hand motion
is retargeted to an SO-101 and gated by MuJoCo physics; a small policy trains on the CPU and is tested on 20 unseen
layouts. Local prototype for the Vultr x Cerebral Valley Agent Arena (rebuilt from scratch at the event).

- Decisions and evidence: docs/SCRAPER.md
- Measured numbers, logistics, timed plan: docs/STATS.md
- Current state and how to run: STATUS.md

Config (Vultr = config only): LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (+ VLM_* overrides), SANDBOX_BACKEND=docker|vultr,
PEXELS_API_KEY (optional), SCRAPER_UA, POSE_PYTHON.

Simulation only. The policy reads robot state and object positions, not camera pixels.

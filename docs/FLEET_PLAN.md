# Fleet + public library (his call, Sun 2026-09-27 ~01:15 PT)

His words: "implement the 192 CPUs ... at least 30 runs, you choose the prompts ... a good scrollable library view ...
host the library on Vercel ... I open source all of the [skills] I created." Spend OK'd for the 192-core VM.

## Architecture
- Scraping stays local (Docker sandbox; YouTube blocks cloud IPs). LLM calls via the local shim.
- Heavy steps run on ONE Vultr vx1-g-192c-768g (192 vCPU, 768 GB, $6.26/h, atl/ewr/ams), created for the fleet run and
  deleted after: MediaPipe Pose per clip (streamed back over SSH), one robot per person (MuJoCo + render) in parallel.
- `HEAVY_SSH` (ssh prefix) switches arena/workout/pipeline.py segment() and film() to the VM; code + clips uploaded.
- Robot copies elbow AND shoulder 1:1 (2-joint), so the catalog covers curls, raises, presses, extensions.
- Fleet server: port 8803, SANDBOX_BACKEND=docker, BOX_WORKERS=4; scripts/fleet.py posts ~30 prompts.

## Library (Vercel, public)
- scripts/build_library.py -> site/ (static): grid of runs (prompt, robot video, exercise, people, stats), per-run
  detail (sources: each person's clip + the robot that copied them), download skill zip. CC attribution on every clip.
- `vercel --prod` from site/ (account phantom3452). Videos compressed (~1 MB each) to stay small.

## Honesty
- Label downloads "skills" (joint trajectories, small policies), not VLMs. Simulated SO-101 (MuJoCo). CC videos only.

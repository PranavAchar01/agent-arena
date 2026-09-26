# How closely Handoff fits the Agent Arena brief

Source: the event page, https://cerebralvalley.ai/e/vultr-the-agent-arena (read 2026-09-25). Paraphrased: build what
production agent infrastructure should look like. Agents now write code, drive browsers and run real operations,
and the infrastructure that contains and coordinates them has to get better. Vultr supplies VM backends, serverless
inference and the compute agents run on "when they stop talking and start acting". Judging criteria and tracks are
not published on the page.

## Fit, point by point

| Brief asks for | What Handoff has today | Fit |
|---|---|---|
| Agents running real operations | A scraping agent that plans searches, picks sources, fetches and decodes untrusted media, then hands verified data to a tuning job | Strong |
| **Contain** what agents do | Every box gets a throwaway sandbox: read-only root, no capabilities, uid 10001, 1 CPU / 512 MB / 128 pids, hard kill, `--rm`, zero secrets inside. 70 planted hostile pages across 14 runs, 0 reached the app. The model only ever sees titles as data and answers with line numbers | **Strongest point** |
| **Coordinate** many agents | New UI: every prompt spawns its own robot box with its own sandbox and its own tuning job, all visible at once | Good in the UI; the backend still runs boxes one after another on this Mac |
| Vultr VM backends | `SandboxRunner` interface; local Docker backend works; Vultr backend is a documented stub (create instance with cloud-init, stream events, delete in `finally`) | **Gap until built on the day** |
| Vultr serverless inference | All LLM/VLM calls go through one OpenAI-compatible client (`LLM_BASE_URL`, `VLM_MODEL`); today it points at a local Claude CLI shim | Config only, but which vision model Vultr offers is unverified |
| Compute agents run on (no GPUs at this event) | Whole pipeline is CPU: MediaPipe, MuJoCo, a 166k-parameter policy trained in about 20 s | Strong |
| "In production" | Missing: per-box kill switch, live resource meters, retries, persistent audit log, auth | Gap |

**Overall: a good fit if it is pitched as agent infrastructure, a weak one if it is pitched as a robotics demo.**
The judges are an infrastructure audience. The robot is the workload that makes containment visible; the product is
the fabric that lets many untrusted, web-touching agents run side by side and hand back only verified output.

## What to build on the day to close the gaps (in order)

1. **One Vultr instance per box.** `VultrRunner`: create a small CPU instance from a snapshot with Docker and the
   scraper image, run the same job, stream events back, destroy it in `finally`. Show the instance ID in the box
   header instead of the container name. This turns "coordinate" into real parallelism.
2. **Vultr Serverless Inference** for the planner, the title ranker and the VLM verifier. Confirm on arrival which
   vision model is available; if none, keep the verifier on its current endpoint and say so.
3. **Kill switch and audit log per box.** A "Kill" button in the technical overview that destroys the instance, and a
   downloadable JSONL audit log (every fetch, every block, every model call). These are the "production" words.
4. **NetBird** (Vultr Marketplace) so sandbox instances stream events to the app with no public inbound port.
   Bonus, only if 1 to 3 are done.

## Pitch line (true today, stronger after step 1)

"Handoff runs every agent that touches the open web in its own throwaway sandbox. Type what you want a robot to do:
each robot gets its own sandboxed scraper, a vision model checks everything it brings back, and a policy is tuned and
tested before it is marked ready to deploy. We planted 70 hostile pages; none got through."

## What not to say

- That it runs on Vultr today (it runs on local Docker until step 1 is built).
- That boxes train in parallel today (they queue on this Mac).
- That the robot is deployed on hardware (simulation only; policies read state, not pixels).
- That a VLM was fine-tuned (it was not; the verifier is used as is).

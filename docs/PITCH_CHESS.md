# Replay: 3-minute live demo

Before going on: server on :8801 (Vultr backend, warm VM ready), open `http://localhost:8801/`. Two finished robots
are preloaded in "Your robots": the newest workout run and the chess run. Backup for chess alone:
`http://localhost:8801/chess.html?replay=91342743086ddfc4`.

## Flow
1. **Intro (10 s).** Replay turns a sentence into a robot skill. VLMs made robots able to read the world; the
   bottleneck now is teaching them new tasks fast.
2. **Live.** Pick **SO-101**, click the first chip or type **Find a really easy workout an SO-101 can do**, press
   Train. A new box appears at the top: the agent picks the dumbbell biceps curl, Scrapling scrapes openly licensed
   curl videos in a throwaway sandbox, then the box turns into video tiles and MediaPipe draws the skeleton on each
   one live (about 5 to 15 s per video, two at a time).
3. **Earlier result.** "The whole run takes a few minutes; here is one I ran earlier": the workout box below it.
   Click it: prompt, the human (skeleton) next to the SO-101 doing the same reps with a dumbbell, elbow 1:1, then
   every scraped video with its reps. Download gives the robot joint trajectory (50 Hz, SO-101 joint names).
4. **Harder result.** Third box: prompt "Replay Deep Blue vs. Kasparov, 1997, Game 6". Click it: prompt card at the
   top, then the robot playing both sides, moves highlighted as it plays, stats, how it ran, full agent log.
5. **Close.** MuJoCo physics (maintained by Google DeepMind) with the official SO-101 model and its real servo
   gains; built to transfer to a real SO-101, hardware test next week.
6. Kill the live box when done (Kill button) so the laptop is not training during questions.

## Wording that stays true
- Say "a VLM checks the examples; the skill is tuned and tested in physics", not "a tuned VLM".
- Say "I ran this earlier today", not "overnight" (the logs carry real timestamps).
- Chess: moves come from the web via Scrapling and are checked by python-chess; the arm's motion is planned and
  rehearsed in physics, not learned. The replay is precomputed on Vultr and sped up.

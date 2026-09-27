# Replay: 3-minute live demo

**Before going on:**
- The server is running on :8801 (Vultr backend, warm VM ready).
- Open `http://localhost:8801/` and **hard-refresh (Cmd+Shift+R)**.
- "Your robots" should show 6 finished robots, newest on top:
  1. chess
  2. Tower of Hanoi
  3. cup pyramid
  4. block tower
  5. block in the bowl
  6. dumbbell curl
- Backups:
  - chess alone: `http://localhost:8801/chess.html?replay=91342743086ddfc4`
  - the public library: https://replay-library.vercel.app

## Flow
1. **Intro (10 s).** "Replay turns a sentence into a robot skill. VLMs let robots read the world. The bottleneck now
   is teaching them new tasks fast."
2. **Live (about 45 s).**
   - Pick **SO-101**. Click the first chip, or type **Find a really easy workout an SO-101 can do**. Press Train.
   - A new box appears on top. The agent picks the dumbbell biceps curl.
   - Scrapling finds openly licensed curl videos on YouTube, inside a throwaway Vultr sandbox.
   - The box turns into video tiles, and MediaPipe draws the skeleton on each one live.
3. **"Here is what it has already made" (about 60 s).** Scroll the grid, one line per robot, and don't open all of
   them:
   - **Curl:** the arm copies a person's rep 1:1 with a dumbbell. Its elbow stays within about 1 degree of the human's.
   - **Block in the bowl, block tower:** small policies trained on real people's hand motion. They solved 19 and 18 of
     20 layouts they had never seen.
   - **Tower of Hanoi:** the optimal 7 moves. The largest disc is never put on a smaller one.
   - **Cup pyramid:** six cups, 3-2-1.
   - Each move is rehearsed in physics and only played if it lands. Worst placement: about 7 mm.
4. **The big one (about 40 s).** Click the chess box.
   - Prompt: "Replay Deep Blue vs. Kasparov, 1997, Game 6".
   - The robot plays both sides, all 37 moves. Moves are highlighted as it plays.
   - The game was scraped with Scrapling and checked with python-chess.
5. **Close (15 s).**
   - MuJoCo physics (maintained by Google DeepMind), with the official SO-101 model and its real servo gains.
   - Every skill downloads.
   - The public library is at replay-library.vercel.app.
   - Real SO-101 hardware test next week.
6. **Kill the live box** when done (Kill button), so the laptop isn't working during questions.

## Wording that stays true
- **Say** "a VLM checks the examples; the skill is tuned and tested in physics". **Not** "a tuned VLM".
- **Say** "I ran these earlier". The logs carry real timestamps.
- **Not learned from video:** chess, Hanoi and cups.
  - The rules and moves come from the web or are computed.
  - The arm's motion is planned and rehearsed in physics.
  - The replays are precomputed and sped up.
- **Learned from people:**
  - The curl copies scraped YouTube Creative Commons videos.
  - The block skills are trained on HO-Cap (CC BY 4.0), real people at a table.
- **Hardware:** simulated only so far. Never say it has run on a real arm.

## Q&A, short and true
**Does it work on a real robot?**
- Not yet.
- It is the real SO-101 model and servo gains, and the outputs are the same six joint commands the arm takes.
- My arm is assembled; hardware test next.

**Where does the video come from?**
- YouTube, filtered to Creative Commons.
- The licence is checked on every video before download.
- Everything is scraped by Scrapling inside a throwaway sandbox (a Vultr VM per robot).

**What does Vultr do here?**
- One robot is one VM that exists only for that run and is deleted after.
- The whole pipeline is CPU. No GPU anywhere.

**How long does one robot take?**
- A few minutes for the curl, mostly downloading and reading video.
- The block policies train in about 21 seconds on a CPU.

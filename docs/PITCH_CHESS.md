# Handoff Chess: 3-minute live demo

Open before going on: `http://localhost:8801/chess.html` on the Vultr-backed server (warm VM ready; check the
header says "Sandboxes on Vultr"). Backup: `runs/chess/<key>/game.mp4` in a player.

## 0:00 to 0:20, the hook
"In 1997 Deep Blue beat Garry Kasparov in 19 moves. It was the first time a machine beat a world champion.
Deep Blue never touched a piece. A human moved them for it. Today one robot arm is going to play both sides."

## 0:20 to 1:10, the live part (type the prompt, or open `?demo=1`)
Type: **Replay Deep Blue vs. Kasparov, 1997, Game 6**. Narrate the terminal as it scrolls:
- "An agent turns my sentence into a search."
- "Every web request happens on a fresh Vultr VM, inside a locked-down container with no secrets in it.
  The scraper is Scrapling."
- "It found the match article, pulled all twelve games off the page, and the agent picked Game 6, the Caro-Kann."
- "The model never writes a single move. python-chess re-reads every move the scraper found: 37 legal moves."
- "The VM is deleted when the job ends."

## 1:10 to 2:20, the robot (about 20 seconds of video, then talk over the stats)
"This is a simulated SO-101, the open-source arm, playing both colours from the side of the board.
Every one of these 37 moves was rehearsed in physics before it was played: the piece has to land upright within
5 mm of its square and nothing else on the board may move more than 3 mm. If a grasp fails the rehearsal, it
tries another grasp. Captured pieces go to the tray."
Point at the stats row: worst placement error, pick-and-places, robot time vs video time.

## 2:20 to 3:00, why it matters
"Agents that act on the real world need two things: a contained way to read the messy web, and a way to prove
an action is safe before doing it. Here that's a throwaway Vultr VM per job, and a physics rehearsal per move.
My real SO-101 is built; Monday this runs on the physical arm."

## Say / don't say
- Say "simulated SO-101", "rehearsed in physics", "replay is sped up". The physics for the whole game takes about
  2 minutes of compute and was run on a Vultr VM ahead of time; the web and agent part is live.
- Don't say the arm is real today, that the motion is a learned policy (it is IK plus closed-loop control), or
  that the model found the moves (the scraper did; python-chess checked them).

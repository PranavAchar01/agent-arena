"""VLM verifier: does this clip really show one person's hand doing the task, with the object and the whole motion?

The sandbox hands back 8 evenly spaced stills per clip. They are tiled into one contact sheet with timestamps and
sent to the VLM (through the OpenAI-compatible client) with a fixed rubric. The reply is parsed strictly; anything
malformed counts as a rejection. The verifier also names the time window of one clean repetition, which the motion
extractor then uses.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .llm import chat, parse_json

RUBRIC = """You are the data-quality gate for a robot-learning pipeline. The image is a contact sheet of evenly spaced stills
(timestamps in seconds, top-left of each tile) from one short video. The robot must learn: "{task}"
(task family: {family}; {family_def}).

Accept ONLY if ALL hold:
1. person: real camera footage of a person at normal speed (not animation, not a diagram, not a robot, not a
   time-lapse or sped-up video: if a whole build or task completes implausibly fast across the stills, reject);
2. hand: a hand is doing the manipulation and is clearly visible in several stills;
3. object: small graspable objects (blocks, cups, pieces, bricks, toys) are visible;
4. activity: the person is doing this kind of motion during the video (fast motions fall between stills; judge
   the activity, e.g. a tower growing across stills means stacking is happening);
5. match: the activity is the same kind of motion as the task (a real hand picking up any small everyday object, carrying it and setting it down counts for place,
   stack and tower: the robot learns the hand's pick-carry-set-down motion). Judge the kind of hand motion only.
   Do NOT reject because the scene has no bowl, block or stack: the robot's own scene supplies the goal.

Reply with JSON only:
{{"accept": true|false, "checks": {{"person": bool, "hand": bool, "object": bool, "activity": bool, "match": bool}},
  "window": [start_seconds, end_seconds] where the hands are working (or null for the whole video),
  "reason": "<one short sentence>"}}"""

DEFS = {
    "unjar": "reach into a container, take a block out and set it down outside the container",
    "stack": "pick a block up and set it on top of another block",
    "tower": "pick a block up and set it on top of a stack of blocks",
    "push": "slide an object across a surface without lifting it",
    "place": "pick an object up and put it down somewhere else or into a container",
}


def contact_sheet(
    frames: list[dict], frame_dir: Path, out: Path, tile_w: int = 320
) -> Path:
    tiles = []
    for fr in frames[:12]:
        im = Image.open(frame_dir / fr["file"]).convert("RGB")
        im = im.resize((tile_w, int(im.height * tile_w / im.width)))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 64, 20], fill=(0, 0, 0))
        d.text((5, 4), f"{fr['t']:.1f}s", fill=(255, 255, 255))
        tiles.append(im)
    th = max(t.height for t in tiles)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * th), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tile_w, (i // cols) * th))
    sheet.save(out, quality=85)
    return out


def verify(clip: dict, run_dir: Path, task: str, family: str) -> dict:
    sheet = contact_sheet(
        clip["frames"],
        run_dir / "frames",
        run_dir / "frames" / f"{clip['id']}_sheet.jpg",
    )
    try:
        raw = chat(
            RUBRIC.format(task=task, family=family, family_def=DEFS[family]),
            images=[sheet],
            max_tokens=400,
        )
    except Exception as e:  # noqa: BLE001 - a failed call is a rejection, not a crash
        return {
            "accept": False,
            "reason": f"verifier call failed: {type(e).__name__}",
            "checks": {},
            "window": None,
        }
    v = parse_json(raw)
    if not isinstance(v, dict) or not isinstance(v.get("accept"), bool):
        # one retry: some replies wrap the JSON in prose; ask again for the object alone
        try:
            raw = chat(RUBRIC.format(task=task, family=family, family_def=DEFS[family]) +
                       "\n\nReturn ONLY the JSON object, nothing before or after it.", images=[sheet], max_tokens=400)
            v = parse_json(raw)
        except Exception:  # noqa: BLE001
            v = None
    if not isinstance(v, dict) or not isinstance(v.get("accept"), bool):
        return {
            "accept": False,
            "reason": "verifier reply was not valid JSON",
            "checks": {},
            "window": None,
        }
    checks = {
        k: bool(v.get("checks", {}).get(k))
        for k in ("person", "hand", "object", "activity", "match")
    }
    window = v.get("window")
    if not (
        isinstance(window, list)
        and len(window) == 2
        and all(isinstance(x, (int, float)) for x in window)
        and 0 <= window[0] < window[1] <= clip["seconds"] + 1
    ):
        window = [0.0, clip["seconds"]]
    accept = v["accept"] and all(checks.values())
    return {
        "accept": accept,
        "checks": checks,
        "window": window,
        "reason": str(v.get("reason", ""))[:200],
        "sheet": sheet.name,
    }

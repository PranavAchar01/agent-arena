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

RUBRIC = """You are the data-quality gate for a robot-learning pipeline. The image is a contact sheet of 8 stills
(timestamps in seconds, top-left of each tile) from one short video. The robot must learn: "{task}"
(task family: {family}; {family_def}).

Accept ONLY if ALL hold:
1. person: real footage of a person (not animation, not a diagram, not a robot);
2. hand: one hand is doing the manipulation and is clearly visible;
3. object: a small graspable object (block, cup, piece, toy) is visible;
4. motion: at least one COMPLETE {family} motion happens within the video (start and end both visible);
5. match: the motion is the same kind of motion as the task (analogues count: moving a chess piece is a
   place motion, stacking cups is a stack motion, sliding a cup across a table is a push motion).

Reply with JSON only:
{{"accept": true|false, "checks": {{"person": bool, "hand": bool, "object": bool, "motion": bool, "match": bool}},
  "window": [start_seconds, end_seconds] of one clean repetition (or null),
  "reason": "<one short sentence>"}}"""

DEFS = {
    "push": "slide an object across a surface without lifting it",
    "place": "pick an object up and put it down somewhere else or into a container",
    "stack": "pick an object up and set it on top of another object",
}


def contact_sheet(
    frames: list[dict], frame_dir: Path, out: Path, tile_w: int = 320
) -> Path:
    tiles = []
    for fr in frames[:8]:
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
        return {
            "accept": False,
            "reason": "verifier reply was not valid JSON",
            "checks": {},
            "window": None,
        }
    checks = {
        k: bool(v.get("checks", {}).get(k))
        for k in ("person", "hand", "object", "motion", "match")
    }
    window = v.get("window")
    if not (
        isinstance(window, list)
        and len(window) == 2
        and all(isinstance(x, (int, float)) for x in window)
        and 0 <= window[0] < window[1] <= clip["seconds"] + 1
    ):
        window = None
    accept = v["accept"] and all(checks.values()) and window is not None
    return {
        "accept": accept,
        "checks": checks,
        "window": window,
        "reason": str(v.get("reason", ""))[:200],
        "sheet": sheet.name,
    }

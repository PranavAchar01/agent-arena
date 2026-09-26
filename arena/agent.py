"""The scraping agent's brain. It runs in the app (it holds the model key); its hands run in the sandbox.

plan()   turns "I want to train a robot to X" into a search plan: which of the supported task families X is,
         search queries for openly licensed video, and include / exclude terms the sandbox uses to rank titles.
The sandbox never receives a key, and nothing the web says is fed back to this model as instructions: page text
stays inside the sandbox, and only the validated manifest (titles, licences, re-encoded clips, stills) comes out.
"""

from __future__ import annotations

from .llm import chat, parse_json

FAMILIES = {
    "push": "slide an object across the table to a spot (push, nudge, slide)",
    "place": "pick an object up and put it into a container (bowl, box, basket, cup)",
    "stack": "pick an object up and set it on top of another (stack blocks, cups, build a tower)",
}

PROMPT = """You plan a web search for openly licensed videos of a PERSON doing a tabletop task with their hand, to
teach a small robot arm (SO-101, one gripper). The user wrote: "{text}"

Supported task families:
{families}

Reply with JSON only:
{{"family": one of {keys}, "summary": "<the task in 3-6 words>",
  "queries": [8 SHORT queries of 1-3 words each (archive search engines match every word, so long queries find
              nothing). Include everyday activities that contain the same hand motion, e.g. board-game moves,
              cup stacking, building with toy bricks, cooking (adding things to a bowl), cup-and-ball tricks],
  "include": [10-16 lowercase single words or short phrases; a relevant title contains at least one],
  "exclude": [8-12 lowercase terms that signal an off-task video, e.g. microscopy, cell, protein, trailer, gameplay, animation]}}"""


def plan(text: str) -> dict:
    fam = "\n".join(f"- {k}: {v}" for k, v in FAMILIES.items())
    out = parse_json(
        chat(
            PROMPT.format(text=text[:300], families=fam, keys=list(FAMILIES)),
            max_tokens=700,
        )
    )
    if not isinstance(out, dict) or out.get("family") not in FAMILIES:
        raise ValueError("planner returned no usable plan")
    clean = lambda xs, n: [str(x)[:60] for x in (xs or []) if isinstance(x, str)][:n]  # noqa: E731
    return {
        "family": out["family"],
        "summary": str(out.get("summary", ""))[:80],
        "queries": clean(out.get("queries"), 8),
        "include": [s.lower() for s in clean(out.get("include"), 20)],
        "exclude": [s.lower() for s in clean(out.get("exclude"), 16)],
    }

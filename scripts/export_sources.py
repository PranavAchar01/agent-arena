"""Static source data for the detail sheets: hand tracks for the block runs, page previews for chess, Hanoi and cups.

python scripts/export_sources.py   -> web/sources/<rid>/<clip>.json, web/sources/pages/<name>.jpg

Served as plain files, so the running server needs no restart; the static app build copies web/ as is.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "web" / "sources"
BLOCKS = ("box-place", "box-tower")
PAGES = {
    "deep-blue": "https://en.wikipedia.org/wiki/Deep_Blue_versus_Garry_Kasparov",
    "deep-blue-g6": "https://en.wikipedia.org/wiki/Deep_Blue_versus_Kasparov,_1997,_Game_6",
    "hanoi": "https://en.wikipedia.org/wiki/Tower_of_Hanoi",
    "cups": "https://en.wikipedia.org/wiki/Cup_stacking",
}
PTS = ("wrist", "mcp", "index", "thumb")


def hands(rid: str) -> None:
    d = OUT / rid
    d.mkdir(parents=True, exist_ok=True)
    for f in (RUNS / rid).glob("pose_*.json"):
        pose = json.loads(f.read_text())
        frames = []
        for fr in pose["frames"]:
            h = fr["hands"][0] if fr["hands"] else None
            frames.append(
                [
                    round(fr["t"], 3),
                    [[round(v, 4) for v in h[k]] for k in PTS] if h else None,
                ]
            )
        (d / f"{f.stem[5:]}.json").write_text(
            json.dumps({"fps": pose["fps"], "frames": frames}, separators=(",", ":"))
        )


def pages() -> None:
    from playwright.sync_api import sync_playwright

    d = OUT / "pages"
    d.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1100, "height": 700})
        for name, url in PAGES.items():
            pg.goto(url, wait_until="domcontentloaded")
            pg.wait_for_timeout(1500)
            pg.screenshot(path=str(d / f"{name}.jpg"), quality=80, type="jpeg")
        b.close()


if __name__ == "__main__":
    for rid in BLOCKS:
        hands(rid)
    pages()
    print(sorted(str(x.relative_to(OUT)) for x in OUT.rglob("*.*")))

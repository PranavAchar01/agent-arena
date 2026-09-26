"""Frame-by-frame recorder (Playwright + fake clock), like the Player Two films: deterministic under load.

  record.py cards          -> film/work/card_*.png (1920x1080; captions with transparent background)
  record.py app SECONDS    -> film/work/app.mp4 (the web app replaying runs/place-seeded in film mode)

Run with a Python that has playwright (e.g. ~/helloworld/scout/.venv/bin/python). The app server must be up
on :8800. The browser and ffmpeg started here are closed at the end.
"""

import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "film" / "work"
WORK.mkdir(parents=True, exist_ok=True)
FPS = 25


def cards(p):
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
    for card in ("title", "end", "place", "stack", "tower"):
        page.goto(f"file://{ROOT}/film/cards.html?card={card}")
        page.wait_for_timeout(1500)  # fonts
        page.screenshot(
            path=str(WORK / f"card_{card}.png"),
            omit_background=card not in ("title", "end"),
        )
        print("card", card, flush=True)
    b.close()


def app(p, seconds: float):
    b = p.chromium.launch()
    ctx = b.new_context(
        viewport={"width": 1280, "height": 720}, device_scale_factor=1.5
    )
    page = ctx.new_page()
    page.clock.install()
    page.goto("http://127.0.0.1:8800/?run=hocap-place&speed=10&film=1&delay=700")
    page.clock.pause_at(page.evaluate("Date.now()") + 50)  # freeze page time; only run_for advances it
    page.wait_for_timeout(
        2500
    )  # fonts, background image, first fetch (real time; the fake clock is paused)
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            str(FPS),
            "-c:v",
            "png",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-r",
            str(FPS),
            str(WORK / "app.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    done_at = None
    n = int(seconds * FPS)
    for i in range(n):
        page.clock.run_for(1000 // FPS)
        page.wait_for_timeout(15)  # let fetches and image loads land in real time
        ff.stdin.write(page.screenshot(type="png"))
        if (
            done_at is None
            and page.evaluate("document.querySelector('#kicker').textContent")
            == "Run complete"
        ):
            done_at = i
        if done_at is not None and i - done_at > FPS * 2.5:
            break
        if i % 100 == 0:
            print("frame", i, flush=True)
    ff.stdin.close()
    ff.wait()
    b.close()
    print("app frames", i + 1, "complete at", done_at, flush=True)


with sync_playwright() as p:
    if sys.argv[1] == "cards":
        cards(p)
    else:
        app(p, float(sys.argv[2]))

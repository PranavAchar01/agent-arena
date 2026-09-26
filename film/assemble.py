"""Cut the 1-minute film: title -> app run (B, fully automatic) -> SO-101 doing A, B, C -> end card. Silent.

Segments are normalised to 1920x1080 @ 25 fps, then joined with 0.4 s crossfades. Output: film/handoff-demo.mp4
and a 720p copy.
"""

import subprocess
from pathlib import Path

W = Path(__file__).resolve().parent / "work"
OUT = Path(__file__).resolve().parent
FPS, XF = 25, 0.4
V = "scale=1920:1080:flags=lanczos,setsar=1,fps=25,format=yuv420p"


def run(args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def dur(p):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(p),
        ],
        capture_output=True,
        text=True,
    ).stdout
    return float(out)


segs = []
# title card with a slow push-in
run(
    [
        "-loop",
        "1",
        "-t",
        "4.5",
        "-i",
        W / "card_title.png",
        "-vf",
        f"scale=2112:1188,zoompan=z='1+0.0006*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=25,{V}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        W / "seg_title.mp4",
    ]
)
segs.append(W / "seg_title.mp4")
run(["-i", W / "app.mp4", "-vf", V, "-c:v", "libx264", "-crf", "18", W / "seg_app.mp4"])
segs.append(W / "seg_app.mp4")
for task in ("place", "stack", "tower"):
    src = W / f"sim_{task}.mp4"
    speed = max(1.0, dur(src) / 6.0)  # fit the whole episode into ~6 s
    run(
        [
            "-i",
            src,
            "-i",
            W / f"card_{task}.png",
            "-filter_complex",
            f"[0:v]setpts=PTS/{speed:.3f},{V}[v];[1:v]scale=1920:1080[c];[v][c]overlay=0:0,format=yuv420p",
            "-t",
            "6",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            W / f"seg_{task}.mp4",
        ]
    )
    segs.append(W / f"seg_{task}.mp4")
run(
    [
        "-loop",
        "1",
        "-t",
        "6",
        "-i",
        W / "card_end.png",
        "-vf",
        V,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        W / "seg_end.mp4",
    ]
)
segs.append(W / "seg_end.mp4")

# crossfade chain
lens = [dur(s) for s in segs]
args, chain, off = [], "", 0.0
for s in segs:
    args += ["-i", str(s)]
prev = "[0:v]"
for i in range(1, len(segs)):
    off += lens[i - 1] - XF
    tag = f"[x{i}]"
    chain += f"{prev}[{i}:v]xfade=transition=fade:duration={XF}:offset={off:.3f}{tag};"
    prev = tag
chain = chain.rstrip(";")
run(
    [
        *args,
        "-filter_complex",
        chain,
        "-map",
        prev,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        OUT / "handoff-demo.mp4",
    ]
)
run(
    [
        "-i",
        OUT / "handoff-demo.mp4",
        "-vf",
        "scale=1280:720",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-an",
        OUT / "handoff-demo-720p.mp4",
    ]
)
print(
    "segments",
    [round(x, 2) for x in lens],
    "total",
    round(dur(OUT / "handoff-demo.mp4"), 2),
)

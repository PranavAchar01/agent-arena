"""Calm, synthesized sound for the film (no samples, no licences): an ambient pad plus soft cues placed on the
exact frames the page marked (film/work/marks.json) and on the section cuts.

usage: sound.py APP_OFFSET APP_LEN PHYS_START CLOSE_START TOTAL OUT.wav
"""

import json
import sys
import wave

import numpy as np

SR = 48_000
off_app, app_len, phys, close, total = map(float, sys.argv[1:6])
out = sys.argv[6]
N = int(total * SR)
L = np.zeros(N)
R = np.zeros(N)
rng = np.random.default_rng(3)
t_all = np.arange(N) / SR


def db(x):
    return 10 ** (x / 20)


def put(sig, at, gain_db=0.0, pan=0.0):
    i = int(at * SR)
    if i >= N:
        return
    sig = sig[: N - i] * db(gain_db)
    L[i : i + len(sig)] += sig * np.sqrt(0.5 * (1 - pan))
    R[i : i + len(sig)] += sig * np.sqrt(0.5 * (1 + pan))


def env(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)


def lowpass(x, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X / np.sqrt(1 + (f / cutoff) ** 4), len(x))


def bell(freq, dur=2.4):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = sum(
        a * np.sin(2 * np.pi * freq * m * t) * np.exp(-t / (dur * d))
        for m, a, d in (
            (1, 1.0, 0.45),
            (2.01, 0.35, 0.25),
            (3.02, 0.12, 0.15),
            (0.5, 0.25, 0.6),
        )
    )
    return s * np.minimum(1, t / 0.004)


def tick(n_ms=9, cutoff=5000):
    n = int(n_ms / 1000 * SR)
    return lowpass(rng.standard_normal(n), cutoff) * env(n, 0.0005, n_ms / 4000)


def thock(f0=190, f1=110, dur=0.16):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 + (f1 - f0) * t / dur
    return (
        np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, dur / 3.5)
       
    )


def whoosh(dur=1.1, peak=0.55):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    # band-limited noise whose brightness follows the swell
    lo, hi = lowpass(x, 700), lowpass(x, 3200)
    mix = np.clip(t / (dur * peak), 0, 1)
    shape = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    return (lo * (1 - mix) + hi * mix) * shape


# ---- ambient pad: soft sine chords, slow breathing, three moods (app, physics, close) ----
def chord(freqs, start, end, gain_db):
    a, b = int(start * SR), int(min(end, total) * SR)
    t = np.arange(b - a) / SR
    sig = np.zeros(b - a)
    for k, f in enumerate(freqs):
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.07 + 0.013 * k) * t + k)
        sig += np.sin(2 * np.pi * f * t + k) * lfo / len(freqs)
        sig += (
            0.3 * np.sin(2 * np.pi * f * 1.003 * t) * lfo / len(freqs)
        )  # slight detune shimmer
    fade = np.minimum(1, np.minimum(t / 2.5, (t[-1] - t) / 2.5))
    sig *= fade * db(gain_db)
    L[a:b] += sig * 0.92
    R[a:b] += sig * 1.0


A = [110.0, 164.81, 220.0, 277.18, 329.63]  # A major add9-ish, warm and low
F = [
    87.31,
    130.81,
    174.61,
    220.0,
    261.63,
    392.0,
]  # F major 7 color for "what you are looking at"
E = [82.41, 123.47, 164.81, 207.65, 246.94, 329.63]  # E for the close, resolves home
chord(A, 0.0, phys + 1.2, -21)
chord(F, phys - 1.0, close + 1.2, -21)
chord(A + [440.0], close - 1.0, total, -21)

# ---- section cuts ----
for at in (off_app - 0.35, phys - 0.45, close - 0.45):
    put(whoosh(1.2), at, -26, pan=0.0)
put(bell(220, 3.5), 0.25, -24)  # a soft opening tone under the title
put(bell(440 * 0.75, 3.0), phys + 2.3, -25)  # the physics line lands
for k, at in enumerate((phys + 4.0, phys + 5.2, phys + 6.4)):
    put(
        tick(40, 1800) * 0.9, at, -26, pan=-0.3 + 0.3 * k
    )  # paper flick as each clipping pops
    put(bell(880 * (1 + 0.125 * k), 1.2), at + 0.02, -34, pan=-0.3 + 0.3 * k)
put(bell(220, 4.5), close + 0.2, -22)
put(bell(330, 4.0), close + 0.25, -28)

# ---- app cues from the page's own marks ----
m = json.load(open("film/work/marks.json"))
t0, marks = m["t0"], m["marks"]
last = {}
ready_k = 0
for kind, ms in marks:
    at = off_app + (ms - t0) / 1000
    if at > off_app + app_len:
        continue
    if kind in last and at - last[kind] < 0.06 and kind != "key":
        continue  # merge bursts (several blocks in one frame)
    last[kind] = at
    if kind == "key":
        put(tick(7, 6000), at, -33 + rng.uniform(-2, 2), pan=rng.uniform(-0.2, 0.2))
    elif kind == "submit":
        put(thock(), at, -22)
        put(bell(523.25, 1.6), at + 0.05, -32)
    elif kind == "glide":
        put(whoosh(1.1), at, -29)
    elif kind == "blocked":
        put(thock(120, 80, 0.09), at, -33, pan=rng.uniform(-0.3, 0.3))
    elif kind == "verified":
        put(bell(1318.5, 0.5), at, -38, pan=rng.uniform(-0.4, 0.4))
    elif kind == "ready":
        root = (659.25, 739.99, 830.61)[ready_k % 3]
        put(bell(root, 2.6), at, -24, pan=(-0.35, 0.0, 0.35)[ready_k % 3])
        put(bell(root * 1.5, 2.2), at + 0.09, -30, pan=(-0.35, 0.0, 0.35)[ready_k % 3])
        ready_k += 1
    elif kind == "sheet":
        put(whoosh(0.9, 0.4), at - 0.2, -28)

# ---- master: gentle fade in/out, soft limiter, -1 dBFS peak ----
fade = np.minimum(1, np.minimum(t_all / 0.8, (total - t_all) / 2.5))
st = np.stack([L, R]) * fade
st = np.tanh(st * 1.4) / 1.4
st *= db(-1) / max(1e-9, np.abs(st).max())
pcm = (st.T * 32767).astype(np.int16)
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", out, f"{total:.2f}s", "marks used", len(marks))

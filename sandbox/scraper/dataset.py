"""Dataset mode (runs inside the sandbox): take a few sequences of an openly licensed video dataset without
downloading the whole archive. HO-Cap (CC BY 4.0, https://irvlutd.github.io/HOCap/) ships each subject as one
multi-GB zip of JPEG frames; we read the zip's directory and only the frames of one camera, every STRIDE-th frame,
over HTTP Range requests, and encode them to a short clip here in the sandbox."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess

from remotezip import open_zip

LICENCE = "CC BY 4.0"
AUTHOR = "HO-Cap (UT Dallas IRVL and NVIDIA)"
PAGE = "https://irvlutd.github.io/HOCap/"


def hocap(job, emit, out, reencode, stills, deadline_ok):
    urls, cam, stride = job["subject_urls"], job["camera"], int(job.get("stride", 2))
    kept = []
    for subject, url in urls.items():
        if len(kept) >= job.get("max_clips", 6) or not deadline_ok():
            break
        z, f = open_zip(url, job["user_agent"], cap_bytes=job.get("cap_bytes", 250_000_000))
        names = z.namelist()
        seqs = sorted({n.split("/")[1] for n in names if n.endswith("meta.yaml")})
        emit("search", source=f"HO-Cap {subject} (remote zip, directory only)", query=cam, hits=len(seqs))
        for seq in seqs[: job.get("per_subject", 3)]:
            if len(kept) >= job.get("max_clips", 6) or not deadline_ok():
                break
            meta = z.read(f"{subject}/{seq}/meta.yaml").decode()
            task = re.search(r"task_id:\s*(\d+)", meta)
            frames = sorted(n for n in names if n.startswith(f"{subject}/{seq}/{cam}/color_"))[::stride]
            cid = hashlib.sha1(f"hocap/{subject}/{seq}/{cam}".encode()).hexdigest()[:10]
            title = f"HO-Cap {subject} {seq} (task {task.group(1) if task else '?'}, camera {cam})"
            emit("download", id=cid, title=title, source="ho-cap", licence=LICENCE)
            tmp = f"/tmp/{cid}"
            os.makedirs(tmp, exist_ok=True)
            for i, n in enumerate(frames):
                with open(f"{tmp}/f_{i:05d}.jpg", "wb") as fh:
                    fh.write(z.read(n))
            fps = 30 / stride
            src = f"{tmp}/seq.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", f"{tmp}/f_%05d.jpg",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", src], capture_output=True, timeout=300)
            dur = reencode(src, f"{out}/clips/{cid}.mp4", int(job.get("clip_seconds", 40)))
            shutil.rmtree(tmp, ignore_errors=True)
            if dur < 2:
                emit("blocked", url=PAGE, title=title, reason="frames did not decode into a clip; discarded")
                continue
            c = {"id": cid, "title": title, "source": "ho-cap", "licence": LICENCE, "author": AUTHOR,
                 "page": f"{PAGE}#{subject}/{seq}", "seconds": round(dur, 2),
                 "frames": stills(f"{out}/clips/{cid}.mp4", cid, dur, 8)}
            kept.append(c)
            with open(f"{out}/manifest.json", "w") as fh:  # written after every clip: a hard kill keeps what is done
                json.dump({"clips": kept}, fh, indent=1)
            emit("clip", **{k: c[k] for k in ("id", "title", "source", "licence", "author", "page", "seconds")})
        emit("warn", message=f"read {f.fetched / 1e6:.0f} MB of the {f.n / 1e9:.1f} GB {subject} archive")
    return kept

"""Optional keyed sources. The APP calls keyed JSON APIs (the key never enters the sandbox); the media files they
point to are still downloaded and decoded only inside the sandbox.

Pexels (https://www.pexels.com/license/): free to use and modify, attribution optional, no AI-training restriction
stated on the licence page (checked 2026-09-25). Needs a free API key: PEXELS_API_KEY. Pranav creates the key
himself; without it this source is simply skipped.
UNTESTED today (no key on this machine). The request and response shape follow https://www.pexels.com/api/documentation/.
"""

from __future__ import annotations

import os

import subprocess

import httpx


def _keychain(service: str) -> str | None:
    """Read a secret Pranav stored with: security add-generic-password -s pexels-api -a pranav -w"""
    r = subprocess.run(["security", "find-generic-password", "-s", service, "-w"], capture_output=True, text=True)
    return r.stdout.strip() or None


def pexels(queries: list[str], per_query: int = 6) -> list[dict]:
    key = os.environ.get("PEXELS_API_KEY") or _keychain("pexels-api")
    if not key:
        return []
    out, seen = [], set()
    with httpx.Client(headers={"Authorization": key}, timeout=15) as c:
        for q in queries:
            r = c.get(
                "https://api.pexels.com/videos/search",
                params={"query": q, "per_page": per_query, "size": "small"},
            )
            if r.status_code != 200:
                continue
            for v in r.json().get("videos", []):
                files = [
                    f
                    for f in v.get("video_files", [])
                    if f.get("file_type") == "video/mp4"
                    and (f.get("height") or 0) <= 720
                ]
                if not files or v["url"] in seen:
                    continue
                seen.add(v["url"])
                best = max(files, key=lambda f: f.get("height") or 0)
                slug = v["url"].rstrip("/").split("/")[-1]
                out.append(
                    {
                        "source": "pexels",
                        "title": slug.replace("-", " ")[:120],
                        "page": v["url"],
                        "media": best["link"],
                        "licence": "Pexels License",
                        "author": v.get("user", {}).get("name", ""),
                        "bytes": 0,
                        "duration": v.get("duration"),
                        "description": "",
                    }
                )
    return out

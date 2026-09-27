"""Scraper that runs INSIDE the throwaway sandbox. It is the only code that touches the open web.

Input: one JSON job on stdin (queries, caps, optional extra page URLs to visit). No keys, no tokens.
Output: JSON lines on stdout (events for the UI) and files under /out:
  /out/clips/<id>.mp4       re-encoded here: <= MAX_SECONDS, <= 360p, h264, no audio
  /out/frames/<id>_<k>.jpg  evenly spaced stills for the verifier
  /out/manifest.json        what was kept, with source URL, licence, author

Sources are chosen for licences and terms, not volume:
  Wikimedia Commons (MediaWiki API, every file carries a free licence in extmetadata)
  Internet Archive  (advancedsearch filtered to Creative Commons / public domain licenceurl, fetched with yt-dlp)
YouTube and Vimeo are deliberately not used: their robots.txt disallows search pages and their terms forbid
downloading. Arbitrary pages (extra_pages) go through the same guards as everything else: http(s) only, robots.txt
honoured, at most 5 redirects, byte caps, a wall-clock cap, no JavaScript ever executed, and page text treated as
data (instruction-like text is flagged and the page dropped).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.robotparser
from urllib.parse import quote, urljoin, urlparse

import httpx

UA = os.environ.get(
    "SCRAPER_UA", "AgentArenaProto/0.1 (open-licence video research prototype) httpx"
)
OUT = "/out"
MAX_SECONDS = 40
MAX_MEDIA_BYTES = 40_000_000
MAX_PAGE_BYTES = 2_000_000
MAX_REDIRECTS = 5
OPEN_LICENCES = re.compile(
    r"^(cc0|cc[- ]by(?:[- ]sa)?(?:[- ][0-9.]+)?|public domain|pd\b|pdm)", re.I
)
OPEN_LICENCE_URL = re.compile(
    r"creativecommons\.org/(licenses/by(-sa)?/|publicdomain/)", re.I
)
INJECTION = re.compile(
    r"(ignore (all |your |previous |prior )*instructions|you are now|system prompt|api[_ -]?key|"
    r"send (me|us) (your|the)|exfiltrat|\.ssh|/etc/passwd|curl .*\|\s*sh)",
    re.I,
)


def emit(kind, **kw):
    print(json.dumps({"type": kind, "t": round(time.time(), 3), **kw}), flush=True)


class Blocked(Exception):
    pass


_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def robots_ok(client: httpx.Client, url: str) -> bool:
    u = urlparse(url)
    root = f"{u.scheme}://{u.netloc}"
    if root not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = client.get(root + "/robots.txt", timeout=5)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except httpx.HTTPError:
            rp.parse([])
        _robots[root] = rp
    return _robots[root].can_fetch(UA, url)


def guarded_get(
    client: httpx.Client, url: str, cap: int, deadline: float, headers=None
) -> tuple[str, bytes, str]:
    """GET with manual redirects, scheme allowlist, robots.txt, byte cap and a deadline."""
    seen = []
    for _ in range(MAX_REDIRECTS + 1):
        if urlparse(url).scheme not in ("http", "https"):
            raise Blocked(f"non-web scheme refused: {urlparse(url).scheme}:")
        if not robots_ok(client, url):
            raise Blocked("robots.txt disallows this URL")
        seen.append(url)
        with client.stream("GET", url, timeout=10, headers=headers or {}) as r:
            if r.status_code == 429:
                raise Blocked(
                    "host asked us to slow down (HTTP 429); backed off politely"
                )
            if r.status_code >= 400:
                raise Blocked(f"HTTP {r.status_code}")
            if r.is_redirect:
                url = urljoin(url, r.headers.get("location", ""))
                continue
            buf = bytearray()
            for chunk in r.iter_bytes():
                buf += chunk
                if len(buf) > cap:
                    raise Blocked(
                        f"download passed the {cap // 1_000_000} MB cap and was cut off"
                    )
                if time.time() > deadline:
                    raise Blocked("wall-clock cap reached mid-download")
            return url, bytes(buf), r.headers.get("content-type", "")
    raise Blocked(
        f"redirect loop: more than {MAX_REDIRECTS} hops ({' -> '.join(urlparse(s).path or '/' for s in seen[:4])} ...)"
    )


def _scrapling_json(url: str) -> dict:
    """Search APIs are fetched with Scrapling (https://github.com/D4Vinci/Scrapling); JSON only, capped."""
    from scrapling.fetchers import Fetcher

    r = Fetcher.get(
        url, headers={"User-Agent": UA}, timeout=15, stealthy_headers=False, retries=1
    )
    if r.status >= 400:
        raise httpx.HTTPError(f"HTTP {r.status}")
    if len(r.body or b"") > MAX_PAGE_BYTES:
        raise Blocked("search response over the cap")
    return r.json()


def _smallest_derivative(ii: dict) -> str | None:
    """The smallest version Commons already serves that is still at least 240 px tall (else the original)."""
    ok = [
        d
        for d in ii.get("derivatives", [])
        if (d.get("height") or 0) >= 240 and "webm" in str(d.get("type", ""))
    ]
    best = min(
        ok, key=lambda d: (d.get("height") or 0) * (d.get("width") or 0), default=None
    )
    return (best or {}).get("src") or ii.get("url")


def _scrapling_media(url: str, dst: str) -> str | None:
    """Download media with Scrapling, asking for at most MAX_MEDIA_BYTES (HTTP Range) and refusing anything larger."""
    from scrapling.fetchers import Fetcher

    r = Fetcher.get(
        url,
        headers={"User-Agent": UA, "Range": f"bytes=0-{MAX_MEDIA_BYTES - 1}"},
        timeout=25,
        stealthy_headers=False,
        retries=1,
        max_redirects=3,
    )
    body = r.body or b""
    if r.status >= 400 or not body or len(body) > MAX_MEDIA_BYTES:
        return None
    with open(dst, "wb") as f:
        f.write(body)
    return dst


YT_CC = "EgIwAQ%253D%253D"  # YouTube's own search filter: Creative Commons licensed uploads only
YT_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def _mmss(s: str) -> int:
    parts = [int(x) for x in re.findall(r"\d+", s or "")]
    n = 0
    for x in parts:
        n = n * 60 + x
    return n


def search_youtube(client, q, limit):
    """YouTube search restricted to Creative Commons uploads, fetched with Scrapling; results parsed from the page's
    ytInitialData JSON (titles and ids only, as data)."""
    from scrapling.fetchers import Fetcher

    url = f"https://www.youtube.com/results?search_query={quote(q)}&sp={YT_CC}"
    r = Fetcher.get(url, headers={"User-Agent": YT_UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=20,
                    stealthy_headers=False, retries=1)
    body = r.body or b""
    if r.status >= 400 or len(body) > 4_000_000:
        raise httpx.HTTPError(f"HTTP {r.status}")
    m = re.search(rb"var ytInitialData = (\{.*?\});</script>", body, re.S)
    data = json.loads(m.group(1)) if m else {}
    out = []

    def walk(x):
        if isinstance(x, dict):
            v = x.get("videoRenderer")
            if isinstance(v, dict) and v.get("videoId"):
                title = "".join(t.get("text", "") for t in v.get("title", {}).get("runs", []))[:140]
                owner = (v.get("ownerText", {}).get("runs") or [{}])[0].get("text", "")[:80]
                out.append({"source": "youtube", "title": title, "page": f"https://www.youtube.com/watch?v={v['videoId']}",
                            "media": None, "licence": "Creative Commons (YouTube filter; checked before download)",
                            "author": owner, "bytes": 0, "duration": _mmss(v.get("lengthText", {}).get("simpleText", "")),
                            "description": ""})
            for y in x.values():
                walk(y)
        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(data)
    return out[:limit]


def fetch_youtube(c, tmp) -> str | None:
    """Only a short window, only if YouTube itself says the upload is Creative Commons."""
    import yt_dlp

    dur = int(c.get("duration") or 0)
    start = 3 if dur <= 90 else 10
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 15, "cachedir": False,
            "format": "bv*[height<=480][ext=mp4]/bv*[height<=480]/b[height<=480]/bv*/b", "outtmpl": tmp + ".%(ext)s",
            "max_filesize": MAX_MEDIA_BYTES,
            "http_headers": {"User-Agent": YT_UA}}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(c["page"], download=False)
        lic = str(info.get("license") or "")
        if "creative commons" not in lic.lower():
            raise Blocked(f"not Creative Commons on YouTube: {lic or 'standard licence'}")
        c["licence"] = "CC BY (YouTube: " + lic[:60] + ")"
        c["author"] = str(info.get("channel") or c.get("author") or "")[:80]
        y.download([c["page"]])
    files = [f for f in os.listdir("/tmp") if f.startswith(os.path.basename(tmp)) and not f.endswith(".part")]
    if not files:
        return None
    src, cut = os.path.join("/tmp", files[0]), tmp + "_cut.mp4"
    # keep only the window after the intro; reencode() then makes the clean small copy
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(MAX_SECONDS), "-i", src, "-an", "-c", "copy", cut],
                   capture_output=True, timeout=60)
    return cut if os.path.exists(cut) and os.path.getsize(cut) > 0 else src


def search_commons(client, q, limit):
    api = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=6"
        f"&gsrsearch={quote(q + ' filetype:video')}&gsrlimit={limit}&prop=videoinfo"
        "&viprop=url|size|mediatype|extmetadata|derivatives&viextmetadatafilter=LicenseShortName|Artist|ImageDescription"
    )
    out = []
    for p in (_scrapling_json(api).get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("videoinfo") or p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:80]
        out.append(
            {
                "source": "wikimedia-commons",
                "title": p.get("title", "")[5:],
                "page": ii.get("descriptionurl"),
                "media": _smallest_derivative(ii),
                "licence": lic,
                "author": artist.strip(),
                "bytes": ii.get("size", 0),
                "duration": ii.get("duration"),
                "description": re.sub(
                    r"<[^>]+>", " ", meta.get("ImageDescription", {}).get("value", "")
                )[:400],
            }
        )
    return out


def search_archive(client, q, limit):
    query = f"({q}) AND mediatype:(movies) AND licenseurl:(*creativecommons* OR *publicdomain*)"
    url = (
        "https://archive.org/advancedsearch.php?q="
        + quote(query)
        + f"&fl[]=identifier&fl[]=title&fl[]=licenseurl&fl[]=creator&rows={limit}&output=json"
    )
    out = []
    for d in _scrapling_json(url).get("response", {}).get("docs", []):
        creator = d.get("creator", "")
        out.append(
            {
                "source": "internet-archive",
                "title": str(d.get("title", ""))[:120],
                "page": f"https://archive.org/details/{d['identifier']}",
                "media": None,
                "licence": d.get("licenseurl", ""),
                "author": (creator[0] if isinstance(creator, list) else creator)[:80],
                "bytes": 0,
                "duration": None,
            }
        )
    return out


def licence_ok(c) -> bool:
    lic = c.get("licence") or ""
    if c.get("source") == "youtube":
        return True  # searched with YouTube's Creative Commons filter; the licence is re-checked before any download
    if c.get("source") == "pexels" and lic == "Pexels License":
        return True  # https://www.pexels.com/license/: free to use and modify
    if "nc" in lic.lower().replace("licenses/by-nc", "nc"):
        return False  # keep it simple and clean: no NonCommercial, no NoDerivatives
    if re.search(r"\bnd\b|-nd", lic.lower()):
        return False
    return bool(OPEN_LICENCES.search(lic.strip()) or OPEN_LICENCE_URL.search(lic))


def reencode(src: str, dst: str, seconds: int) -> float:
    """Decode the untrusted file and write a clean, small h264 copy. Returns its duration (0 if undecodable)."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        *(["-user_agent", UA] if src.startswith("http") else []),
        "-t",
        str(seconds),
        "-i",
        src,
        "-an",
        "-vf",
        "scale=-2:'min(360,ih)',fps=15",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        dst,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0 or not os.path.exists(dst):
        return 0.0
    probe = subprocess.run(["ffmpeg", "-i", dst], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe.stderr)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def stills(clip: str, cid: str, dur: float, n: int = 8):
    paths = []
    for k in range(n):
        t = dur * (k + 0.5) / n
        dst = f"{OUT}/frames/{cid}_{k}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{t:.2f}",
                "-i",
                clip,
                "-frames:v",
                "1",
                "-q:v",
                "4",
                dst,
            ],
            capture_output=True,
            timeout=60,
        )
        if os.path.exists(dst):
            paths.append({"file": os.path.basename(dst), "t": round(t, 2)})
    return paths


def fetch_media(client, c, deadline) -> str | None:
    tmp = f"/tmp/{hashlib.sha1(c['page'].encode()).hexdigest()[:10]}"
    if c["source"] == "youtube":
        return fetch_youtube(c, tmp)
    if c["source"] == "internet-archive":
        ident = c["page"].rstrip("/").split("/")[-1]
        meta = client.get(
            f"https://archive.org/metadata/{quote(ident)}", timeout=15
        ).json()
        vids = [
            f
            for f in meta.get("files", [])
            if str(f.get("name", "")).lower().endswith((".mp4", ".ogv", ".webm"))
            and f.get("size")
        ]
        if meta.get("metadata", {}).get("licenseurl"):
            c["licence"] = meta["metadata"]["licenseurl"]
        if vids:
            f = min(vids, key=lambda f: int(f["size"]))
            url = f"https://archive.org/download/{quote(ident)}/{quote(f['name'])}"
            rng = (
                {"Range": f"bytes=0-{MAX_MEDIA_BYTES - 1}"}
                if int(f["size"]) > MAX_MEDIA_BYTES
                else None
            )
            final, body, _ = guarded_get(client, url, MAX_MEDIA_BYTES, deadline, rng)
            with open(tmp, "wb") as fh:
                fh.write(body)
            return tmp
        import yt_dlp

        opts = {
            "quiet": True,
            "no_warnings": True,
            "outtmpl": tmp + ".%(ext)s",
            "max_filesize": MAX_MEDIA_BYTES,
            "format": "worst[height>=240]/worst",
            "noplaylist": True,
            "playlist_items": "1",
            "socket_timeout": 15,
            "http_headers": {"User-Agent": UA},
        }
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(c["page"], download=True)
            if info.get("_type") == "playlist":
                info = (info.get("entries") or [None])[0] or {}
            if info.get("license"):
                c["licence"] = info["license"]
            files = [
                f for f in os.listdir("/tmp") if f.startswith(os.path.basename(tmp))
            ]
            return os.path.join("/tmp", files[0]) if files else None
    if c["source"] == "wikimedia-commons" and c.get("media"):
        got = _scrapling_media(c["media"], tmp)
        if got:
            return got
    # large archive files: ask for only the first part (HTTP Range); a truncated webm/ogv still decodes from the start
    rng = (
        {"Range": f"bytes=0-{MAX_MEDIA_BYTES - 1}"}
        if c.get("bytes", 0) > MAX_MEDIA_BYTES
        else None
    )
    _, body, ctype = guarded_get(client, c["media"], MAX_MEDIA_BYTES, deadline, rng)
    with open(tmp, "wb") as f:
        f.write(body)
    return tmp


def visit_page(client, url, deadline):
    """An arbitrary web page: find video links in it, never run its scripts, treat its text as data."""
    final, body, ctype = guarded_get(client, url, MAX_PAGE_BYTES, deadline)
    if "html" not in ctype and "text" not in ctype:
        raise Blocked(f"expected a web page, got {ctype or 'unknown type'}")
    text = body.decode("utf-8", "replace")
    visible = re.sub(
        r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", text, flags=re.S | re.I
    )
    hit = INJECTION.search(visible)
    if hit:
        raise Blocked(
            f'page text tries to give the agent instructions ("{hit.group(0)[:40]}"); dropped as data'
        )
    scripts = len(re.findall(r"<script", text, re.I))
    links = re.findall(
        r"""(?:src|href)\s*=\s*["']([^"']+\.(?:mp4|webm|ogv)(?:\?[^"']*)?)["']""",
        text,
        re.I,
    )
    lic = re.search(
        r"creativecommons\.org/(?:licenses|publicdomain)/[a-z0-9./-]+", text, re.I
    )
    return [
        {
            "source": "web",
            "title": urlparse(final).path,
            "page": final,
            "media": urljoin(final, link),
            "licence": lic.group(0) if lic else "",
            "author": urlparse(final).netloc,
            "bytes": 0,
            "duration": None,
            "scripts_ignored": scripts,
        }
        for link in links[:3]
    ]


def main():
    global UA
    job = json.loads(sys.stdin.read())
    UA = job.get("user_agent") or UA
    deadline = time.time() + job.get("time_budget_s", 240)
    os.makedirs(f"{OUT}/clips", exist_ok=True)
    os.makedirs(f"{OUT}/frames", exist_ok=True)
    secrets = [
        k for k, v in os.environ.items() if v and re.search(r"KEY|TOKEN|SECRET|PASS", k)
    ]
    emit(
        "sandbox",
        message="sandbox up",
        user=os.getuid(),
        env_secrets=len(secrets),
        writable=["/out", "/tmp"],
        caps=job.get("caps", {}),
    )
    client = httpx.Client(
        headers={"User-Agent": job.get("user_agent") or UA}, follow_redirects=False
    )
    if job.get("mode") == "pgn":  # find a chess game's moves (Scrapling); see pgn.py
        from pgn import run as find_game

        find_game(job, emit, OUT, job.get("user_agent") or UA, deadline)
        return
    if job.get("dataset") == "hocap":
        from dataset import hocap

        kept = hocap(job, emit, OUT, reencode, stills, lambda: time.time() < deadline)
        with open(f"{OUT}/manifest.json", "w") as f:
            json.dump({"clips": kept}, f, indent=1)
        emit("done", kept=len(kept), candidates=len(kept))
        return
    cands, seen = [], set()
    if job.get("fetch"):
        for c in job["fetch"][:16]:
            if urlparse(str(c.get("page", ""))).scheme in ("http", "https"):
                cands.append(
                    {
                        k: c.get(k)
                        for k in (
                            "source",
                            "title",
                            "page",
                            "media",
                            "licence",
                            "author",
                            "bytes",
                            "duration",
                            "description",
                        )
                    }
                )
        job = {**job, "queries": [], "include": [], "exclude": []}
    for q in job["queries"]:
        all_sources = {"youtube": (search_youtube, "YouTube (Creative Commons)"), "commons": (search_commons, "Wikimedia Commons"),
                       "archive": (search_archive, "Internet Archive")}
        for fn, name in (all_sources[k] for k in job.get("sources", ["commons", "archive"]) if k in all_sources):
            try:
                found = fn(client, q, job.get("per_query", 12))
            except httpx.HTTPError as e:
                emit("warn", message=f"{name} search failed: {type(e).__name__}")
                continue
            emit("search", source=name, query=q, hits=len(found))
            for c in found:
                if c["page"] and c["page"] not in seen:
                    seen.add(c["page"])
                    cands.append(c)
    for url in job.get("extra_pages", []):
        try:
            got = visit_page(client, url, time.time() + 20)
            emit(
                "page",
                url=url,
                videos=len(got),
                scripts_ignored=sum(g["scripts_ignored"] for g in got),
            )
            cands.extend(got)
        except Exception as e:  # noqa: BLE001 - hostile input must never crash the run
            emit(
                "blocked",
                url=url,
                reason=str(e)
                if isinstance(e, Blocked)
                else f"network error: {type(e).__name__}",
            )

    inc = [t.lower() for t in job.get("include", [])]
    exc = [t.lower() for t in job.get("exclude", [])]

    def score(c):
        text = (
            f"{c['title']} {c.get('description', '')}".lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        if any(t in text for t in exc):
            return -1
        return sum(t in text for t in inc) if inc else 1

    ranked = []
    for c in cands:
        sc = score(c)
        if sc <= 0 and c["source"] != "web":
            emit(
                "skip",
                title=c["title"],
                reason="agent: title/description off-task"
                if sc == 0
                else "agent: excluded term",
            )
            continue
        if not licence_ok(c):
            emit(
                "skip",
                title=c["title"],
                reason=f"licence not open enough: {c['licence'] or 'none stated'}",
            )
            continue
        c["score"] = sc
        ranked.append(c)
        emit(
            "candidate",
            score=sc,
            **{
                k: c.get(k)
                for k in (
                    "source",
                    "title",
                    "page",
                    "media",
                    "licence",
                    "author",
                    "bytes",
                    "duration",
                    "description",
                )
            },
        )
    cands = sorted(ranked, key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if len(kept) >= job.get("max_clips", 8) or time.time() > deadline:
            break
        if not licence_ok(c):
            emit(
                "skip",
                title=c["title"],
                reason=f"licence not open enough: {c['licence'] or 'none stated'}",
            )
            continue
        cid = hashlib.sha1(c["page"].encode()).hexdigest()[:10]
        emit(
            "download",
            id=cid,
            title=c["title"],
            source=c["source"],
            licence=c["licence"],
        )
        try:
            src = fetch_media(client, c, min(deadline, time.time() + 60))
            if not src:
                raise Blocked("no downloadable file under the size cap")
            dur = reencode(
                src, f"{OUT}/clips/{cid}.mp4", int(job.get("clip_seconds", MAX_SECONDS))
            )
            if src.startswith("/tmp/"):
                os.remove(src)
            if dur < 2:
                raise Blocked("file is not a decodable video (or under 2 s); discarded")
        except Blocked as e:
            emit(
                "blocked",
                url=c.get("media") or c["page"],
                title=c["title"],
                reason=str(e),
            )
            continue
        except (
            Exception
        ) as e:  # untrusted input: any failure drops the candidate, never the run
            emit(
                "skip",
                title=c["title"],
                reason=f"fetch failed: {type(e).__name__}: {str(e)[:80]}",
            )
            continue
        c.update(
            id=cid,
            clip=f"clips/{cid}.mp4",
            seconds=round(dur, 2),
            frames=stills(f"{OUT}/clips/{cid}.mp4", cid, dur, 12 if dur > 50 else 8),
        )
        kept.append(c)
        emit(
            "clip",
            **{
                k: c[k]
                for k in (
                    "id",
                    "title",
                    "source",
                    "licence",
                    "author",
                    "page",
                    "seconds",
                )
            },
        )
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump({"clips": kept}, f, indent=1)
    emit("done", kept=len(kept), candidates=len(cands))


if __name__ == "__main__":
    main()

"""Find a famous chess game's moves on the open web, with Scrapling. Runs INSIDE the sandbox.

Two job phases, same as the video scraper:
  {"mode": "pgn", "query": "..."}              search: Wikipedia's search API, candidate article titles out
  {"mode": "pgn", "pages": ["https://..."]}    fetch: the pages the agent picked, candidate move lists out

Only a short allowlist of hosts is fetched. Page text is data: every move list goes back as a plain string, capped,
and the app re-parses it with python-chess (an illegal or made-up move ends the run there). Pages whose text tries to
instruct the agent are flagged and dropped.
"""

from __future__ import annotations

import json
import re
import time
from urllib.parse import quote, urlparse

from scrapling.fetchers import Fetcher

HOSTS = {"en.wikipedia.org", "www.chessgames.com", "lichess.org", "www.pgnmentor.com"}
MAX_PAGE_BYTES = 3_000_000
MAX_CANDIDATES = 16
MOVE_RUN = re.compile(
    r"(?<![0-9])1\.\s?[a-hNBRQKO0]\S{0,8}(?:\s+(?:\d{1,3}\.{1,3}\s?)?[a-hNBRQKO0]\S{0,8}){10,400}"
)
INJECTION = re.compile(
    r"(ignore (all |your |previous |prior )*instructions|you are now|system prompt|api[_ -]?key)",
    re.I,
)


def _get(url: str, ua: str, deadline: float):
    u = urlparse(url)
    if u.scheme != "https" or u.hostname not in HOSTS:
        raise PermissionError(f"host not on the allowlist: {u.hostname}")
    if time.time() > deadline:
        raise TimeoutError("wall-clock cap reached")
    r = Fetcher.get(
        url,
        headers={"User-Agent": ua},
        timeout=15,
        stealthy_headers=False,
        max_redirects=5,
        retries=1,
    )
    if r.status >= 400:
        raise PermissionError(f"HTTP {r.status}")
    if len(r.body or b"") > MAX_PAGE_BYTES:
        raise PermissionError("page over the 3 MB cap")
    return r


def _move_lists(text: str) -> list[dict]:
    """Runs of bare move text that start at move 1 (commentary breaks a run), trimmed at a result token, each with
    the 160 characters before it so the agent can tell which game it is."""
    out, seen = [], set()
    for m in MOVE_RUN.finditer(text):
        s = m.group(0)
        tail = text[m.end() : m.end() + 12]
        res = re.match(r"\s*(1-0|0-1|1/2-1/2|½–½|½-½)", tail)
        if res:
            s += " " + res.group(1)
        key = s[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {"moves": s[:6000], "context": text[max(0, m.start() - 160) : m.start()]}
        )
    return out[:MAX_CANDIDATES]


def run(job: dict, emit, out: str, ua: str, deadline: float) -> None:
    if job.get("query"):
        q = str(job["query"])[:200]
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=8&srsearch={quote(q)}"
        try:
            hits = _get(url, ua, deadline).json()["query"]["search"]
        except Exception as e:  # noqa: BLE001 - a failed search is an event, not a crash
            emit("warn", message=f"Wikipedia search failed: {type(e).__name__}")
            hits = []
        emit("search", source="Wikipedia (Scrapling)", query=q, hits=len(hits))
        for h in hits:
            title = str(h.get("title", ""))[:200]
            emit(
                "candidate",
                source="Wikipedia",
                title=title,
                page="https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
            )
    found = []
    for page in [str(p) for p in job.get("pages", [])][:3]:
        try:
            r = _get(page, ua, deadline)
        except Exception as e:  # noqa: BLE001 - hostile or broken pages must never crash the run
            emit("blocked", url=page, reason=str(e)[:160])
            continue
        text = re.sub(r"\s+", " ", " ".join(r.css("body ::text").getall()))
        if INJECTION.search(text):
            emit(
                "blocked",
                url=page,
                reason="page text tries to instruct the agent; dropped",
            )
            continue
        title = (r.css("title::text").get() or "").strip()[:200]
        lists = _move_lists(text)
        emit(
            "page",
            url=page,
            title=title,
            move_lists=len(lists),
            bytes=len(r.body or b""),
        )
        for c in lists:
            found.append({"page": page, "title": title, **c})
            emit(
                "pgn_candidate",
                page=page,
                context=c["context"][-70:],
                preview=c["moves"][:90],
            )
    with open(f"{out}/pgn.json", "w") as f:
        json.dump({"candidates": found}, f)
    emit("done", kept=len(found), candidates=len(found))

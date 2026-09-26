# Scraper decision

Checked 2026-09-25 with `gh api` (stars, licence, last commit, latest release) and by reading robots.txt / terms.

| Tool | Stars | Licence | Last commit | Latest release | Needs a browser | Reads video licence |
|---|---|---|---|---|---|---|
| yt-dlp/yt-dlp | 193,588 | Unlicense | 2026-09-16 | 2026.08.19 | no | yes, `license` field (YouTube, Vimeo, Internet Archive extractors) |
| unclecode/crawl4ai | 84,266 | Apache-2.0 | 2026-09-25 | v0.9.4 | yes (Playwright Chromium) | no |
| browser-use/browser-use | 116,286 | MIT | 2026-09-15 | 0.13.10 | yes (Playwright Chromium) | no |
| microsoft/playwright-python | 15,017 | Apache-2.0 | 2026-09-23 | v1.63.0 | is the browser driver | no |
| firecrawl/firecrawl (was mendableai) | n/a in check | AGPL-3.0 | 2026-09-25 | v2.11.0 | yes when self-hosted | no |
| searxng/searxng (behind local `scout`) | 37,638 | AGPL-3.0 | 2026-09-25 | rolling, no releases | no | no (search only) |

## Pick

**An LLM agent for planning and choosing, httpx for the archive APIs, yt-dlp for Internet Archive media, a static
ffmpeg for decoding, all inside a throwaway Docker sandbox.** No headless browser.

Why:

1. **The job is licence-first, not page-first.** Openly licensed video lives behind archive APIs that return the
   licence as data (Wikimedia Commons `extmetadata.LicenseShortName`, Internet Archive `licenseurl`). A browser
   agent would be reading pages to guess what an API states directly.
2. **yt-dlp is the most maintained media fetcher there is** (193k stars, commits this month, Unlicense) and it
   carries the `license` field for Internet Archive items. Its `max_filesize` caps downloads.
3. **Sandbox size and attack surface.** crawl4ai / browser-use / Playwright pull a Chromium (roughly 150 to 400 MB)
   and execute page JavaScript. The sandbox image here is 344 MB total (python:3.13-slim + yt-dlp + httpx +
   imageio-ffmpeg) and never runs a page's scripts. On a Vultr throwaway VM that is also a faster cold start.
4. **Firecrawl and SearXNG are AGPL-3.0.** Fine to run as separate services, but not needed: the archive APIs are
   the search. The local `scout` CLI (SearXNG + crawl4ai) is kept for research, not for the pipeline.
5. **Where a browser would help** is general-web discovery of pages that only link videos. That path exists
   (`extra_pages`, parsed as plain HTML with scripts ignored) and is exactly where the planted hostile pages go.

## Sources, and why these

| Source | Used | Why |
|---|---|---|
| Wikimedia Commons API | yes | every file has a free licence in metadata; API etiquette followed (descriptive User-Agent with a contact URL, serial requests, 429 means back off) |
| Internet Archive advancedsearch + yt-dlp | yes | filtered to `licenseurl` Creative Commons / public domain; robots.txt only disallows /control/ and /report/ |
| Pexels API | optional, off by default | licence page: free to use and modify, no attribution needed, no AI restriction stated. Needs a free API key (PEXELS_API_KEY) that Pranav must create; the app calls the API so the key never enters the sandbox. **Untested today.** |
| YouTube (even CC-BY) | no | robots.txt disallows /results and /api/; Terms of Service forbid downloading outside YouTube's player |
| Vimeo | no | robots.txt says scraping is prohibited and disallows /search/ and /download/ |
| Mixkit | no | terms forbid using scripts or bots to download items |
| EgoDex and similar research sets | no | CC BY-NC-ND (no derivatives) or a licence agreement to accept (that is Pranav's call, per item) |

Licences accepted: CC0, CC BY, CC BY-SA, public domain / PDM (and Pexels License when that source is on).
Rejected: anything NC or ND, anything with no stated licence.

## Guards (all inside the sandbox)

- http(s) only; `file:`, `javascript:` etc. refused
- robots.txt checked per host before any page or media fetch
- at most 5 redirects; 2 MB cap per page, 40 MB per media file (HTTP Range for large archive files), 60 s per fetch
- page text is scanned for instruction-like text; a page that tries to instruct the agent is dropped as data
- page scripts are counted and never executed
- every download is decoded and re-encoded by ffmpeg into a clean 360p h264 file; undecodable files are dropped
- container: read-only root, unprivileged uid 10001, all capabilities dropped, no-new-privileges, 1 CPU, 512 MB RAM,
  128 pids, tmpfs scratch, hard kill at 300 s, `--rm`; zero secrets in its environment (it reports the count)
- the host copies back only a validated manifest, `<10-hex-id>.mp4` clips and `<id>_<k>.jpg` stills, capped at 400 MB

## What we keep

Motion numbers only (normalised pinch-point path, lift, timing per move). Stills and re-encoded clips stay in the
local run folder for the verifier and the demo; no faces or personal data are extracted or stored beyond that, and
runs/ is gitignored.

## Measured yield (the real risk)

See docs/STATS.md. Short version: the zero-key open archives hold very little tabletop hand footage. Direct task
words ("pushing box", "block in bowl") find almost nothing; analogue activities (chess and Go moves, speed stacking,
Lego building) are where the usable clips are.

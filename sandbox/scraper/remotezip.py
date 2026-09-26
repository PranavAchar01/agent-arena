"""Read single members out of a large remote .zip over HTTP Range requests (the central directory, then only the
files we ask for). Lets the pipeline take ~30 MB of frames from an 8.5 GB dataset archive."""

from __future__ import annotations

import io
import zipfile

import httpx


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, ua: str, cap_bytes: int = 400_000_000):
        self.c = httpx.Client(headers={"User-Agent": ua}, follow_redirects=True, timeout=60)
        r = self.c.get(url, headers={"Range": "bytes=0-0"})
        self.url, self.n = str(r.url), int(r.headers["content-range"].split("/")[1])
        self.p, self.fetched, self.cap = 0, 0, cap_bytes

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.p

    def seek(self, off, whence=0):
        self.p = off if whence == 0 else self.p + off if whence == 1 else self.n + off
        return self.p

    def readinto(self, b):
        if self.p >= self.n:
            return 0
        end = min(self.n, self.p + len(b)) - 1
        d = self.c.get(self.url, headers={"Range": f"bytes={self.p}-{end}"}).content
        self.fetched += len(d)
        if self.fetched > self.cap:
            raise IOError("remote zip read passed its byte cap")
        b[: len(d)] = d
        self.p += len(d)
        return len(d)


def open_zip(url: str, ua: str, cap_bytes: int = 400_000_000) -> tuple[zipfile.ZipFile, RangeFile]:
    f = RangeFile(url, ua, cap_bytes)
    return zipfile.ZipFile(io.BufferedReader(f, 1 << 18)), f

"""Find and read GOES-19 ABI files from NOAA's open-data bucket on AWS (no account or key).

Files are read over HTTP with byte-range requests, so only the parts of each NetCDF/HDF5 file
we touch (metadata plus the chunks covering Bermuda) are downloaded.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone

import requests
from urllib3.util.retry import Retry

BUCKET = "https://noaa-goes19.s3.amazonaws.com"
_session = requests.Session()
_session.headers["User-Agent"] = "bermuda-weather-nowcast (+https://bermuda-weather.vercel.app)"
# S3 occasionally drops a connection; retry, and allow enough pooled connections for parallel downloads.
_adapter = requests.adapters.HTTPAdapter(
    pool_maxsize=16,
    max_retries=Retry(total=4, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504), allowed_methods=("GET", "HEAD")),
)
_session.mount("https://", _adapter)


def list_files(product: str, band: int | None, hour: datetime) -> list[tuple[datetime, str]]:
    """(scan start, key) for one hour of a product such as 'ABI-L2-CMIPC' (band None for
    products without bands, such as 'GLM-L2-LCFA')."""
    kind = f"{product}-M6C{band:02d}" if band is not None else product
    prefix = f"{product}/{hour:%Y/%j/%H}/OR_{kind}_G19_"
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = _session.get(BUCKET, params=params, timeout=30)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    out = []
    for k in keys:
        m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", k)
        if m:
            y, doy, hh, mm, ss = map(int, m.groups())
            t = datetime(y, 1, 1, hh, mm, ss, tzinfo=timezone.utc) + timedelta(days=doy - 1)
            out.append((t, k))
    return sorted(out)


def files_between(product: str, band: int | None, start: datetime, end: datetime) -> list[tuple[datetime, str]]:
    hour = start.replace(minute=0, second=0, microsecond=0)
    found = []
    while hour <= end:
        found += [(t, k) for t, k in list_files(product, band, hour) if start <= t <= end]
        hour += timedelta(hours=1)
    return found


class RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP byte ranges, with a small block cache."""

    def __init__(self, url: str, block: int = 256 * 1024):
        self.url, self.block, self.pos = url, block, 0
        self.cache: dict[int, bytes] = {}
        self.bytes_fetched = 0
        r = _session.head(url, timeout=30)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            lo = i * self.block
            hi = min(lo + self.block, self.size) - 1
            r = _session.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=60)
            r.raise_for_status()
            self.cache[i] = r.content
            self.bytes_fetched += len(r.content)
        return self.cache[i]

    def readinto(self, b) -> int:
        n = min(len(b), max(0, self.size - self.pos))
        if n == 0:
            return 0
        out, start, end = bytearray(), self.pos, self.pos + n
        for i in range(start // self.block, (end - 1) // self.block + 1):
            blk = self._block(i)
            lo = max(start - i * self.block, 0)
            hi = min(end - i * self.block, len(blk))
            out += blk[lo:hi]
        b[: len(out)] = out
        self.pos += len(out)
        return len(out)


def open_remote(key: str) -> RangeFile:
    return RangeFile(f"{BUCKET}/{key}")


def download(key: str) -> bytes:
    """A whole (small) file, for products like GLM where we need every record."""
    r = _session.get(f"{BUCKET}/{key}", timeout=60)
    r.raise_for_status()
    return r.content

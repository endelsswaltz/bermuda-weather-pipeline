"""Find GOES-19 scans by time and return them on the Bermuda grid, with an optional on-disk
cache so backtests and parameter sweeps do not download the same scan twice."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .fetch import files_between, open_remote
from .goes import Grid, read_cmip, resample

CONUS, FULL_DISK = "ABI-L2-CMIPC", "ABI-L2-CMIPF"
TOLERANCE = timedelta(seconds=150)


def nearest(available: list[tuple[datetime, str]], target: datetime, tolerance: timedelta = TOLERANCE):
    """The (time, key) closest to `target`, or None when nothing is within the tolerance."""
    if not available:
        return None
    t, k = min(available, key=lambda tk: abs(tk[0] - target))
    return (t, k) if abs(t - target) <= tolerance else None


def on_grid(key: str, grid: Grid, cache: Path | None = None, variable: str = "CMI") -> np.ndarray:
    """One scan resampled to `grid` (float32). Cached as .npy under `cache` when given."""
    path = cache / (Path(key).name.replace(".nc", f"_{len(grid.lats)}.npy")) if cache else None
    if path and path.exists():
        return np.load(path)
    window = (grid.lats[-1], grid.lats[0], grid.lons[0], grid.lons[-1])
    out = resample(read_cmip(open_remote(key), window, variable), grid).astype(np.float32)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
    return out


def listing(band: int | None, start: datetime, end: datetime, product: str = CONUS):
    return files_between(product, band, start, end)

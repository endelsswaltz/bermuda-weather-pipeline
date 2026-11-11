"""A rough rain-rate estimate from GOES-19 infrared, for the map's "rain estimate" layer.

This is NOT radar. It turns cloud-top temperature into a rain rate with the Auto-Estimator
power law (Vicente, Scofield and Menzel, 1998), fitted to deep convection:

    R = 1.1183e11 * exp(-3.6382e-2 * T ** 1.2)      R in mm/h, T = band 13 brightness temp. in K

with two screens: nothing warmer than -38 C counts, and thin cirrus (band 13 minus band 15
larger than a few kelvin) is removed. It still shows cold, thick, non-raining cloud as rain
and misses warm showers entirely.

NOAA's own product (ABI-L2-RRQPEF, 2 km, every 10 minutes, about 10 minutes after the scan) is
better calibrated and is used instead whenever a file exists for the scan time; the GOES-19 feed
of it on AWS stopped on 2 November 2026, so the fallback above is what normally runs.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .fetch import _session
from .goes import Grid
from .scenes import CONUS, FULL_DISK, listing, nearest, on_grid

KELVIN = 273.15
WARMEST_RAIN_K = KELVIN - 38.0   # 235 K, the classic cold-cloud rain threshold (GOES Precipitation Index)
CIRRUS_SPLIT_K = 2.0
MAX_RATE = 50.0
WINDOW_MIN, STEP_MIN = 180, 10

# Lower edge (mm/h) and colour of each class; below the first edge is transparent.
LEGEND = [
    (0.5, (130, 190, 255)), (1, (70, 130, 240)), (2, (60, 190, 110)), (5, (240, 220, 70)),
    (10, (245, 150, 50)), (20, (230, 60, 60)), (40, (200, 70, 200)),
]


def estimate(bt13: np.ndarray, bt15: np.ndarray | None = None) -> np.ndarray:
    """Rain rate (mm/h) from band 13 (and band 15 when available) brightness temperature in K."""
    t = np.where(np.isfinite(bt13), bt13, 320.0).astype(np.float64)
    rate = 1.1183e11 * np.exp(-3.6382e-2 * np.clip(t, 180.0, 320.0) ** 1.2)
    rate[t > WARMEST_RAIN_K] = 0.0
    if bt15 is not None:
        rate[np.isfinite(bt15) & (t - bt15 > CIRRUS_SPLIT_K)] = 0.0
    return np.where(np.isfinite(bt13), np.minimum(rate, MAX_RATE), np.nan)


def colorize(rate: np.ndarray, alpha: int = 205) -> Image.Image:
    edges = np.array([e for e, _ in LEGEND])
    colors = np.array([c for _, c in LEGEND], dtype=np.uint8)
    idx = np.searchsorted(edges, np.nan_to_num(rate, nan=0.0), side="right") - 1
    out = np.zeros((*rate.shape, 4), dtype=np.uint8)
    wet = idx >= 0
    out[wet, :3] = colors[idx[wet]]
    out[wet, 3] = alpha
    return Image.fromarray(out)


def ten_minute_scans(scans: list[tuple[datetime, str]]) -> list[tuple[datetime, str]]:
    """Every other 5-minute CONUS scan, on fixed clock marks (:01, :11, :21 ...), oldest first.

    Fixed marks, not "10 minutes back from the newest scan", so the same frames come up run
    after run and already-published ones can be reused.
    """
    return [(t, k) for t, k in sorted(scans) if (t.minute // 5) % 2 == 0]


def frame_name(t: datetime) -> str:
    return f"rain/{t:%Y%m%dT%H%M}.png"


def build_frames(out: Path, grid: Grid, latest: datetime, previous_base: str | None, to_mercator, now: datetime | None = None) -> dict:
    """Write rain/<time>.png for every 10 minutes of the last 3 hours and return the index.

    Frames already published (listed in the previous index) are copied, not recomputed, so a
    normal run only processes the one or two new scans.
    """
    now = now or datetime.now(timezone.utc)
    (out / "rain").mkdir(parents=True, exist_ok=True)
    previous: dict[str, dict] = {}
    if previous_base:
        try:
            for f in _session.get(f"{previous_base}rain/index.json", timeout=20).json()["frames"]:
                previous[f["time"]] = f
        except Exception:  # noqa: BLE001 - nothing published yet
            pass

    start = latest - timedelta(minutes=WINDOW_MIN + 5)
    b13 = listing(13, start, latest + timedelta(minutes=3), CONUS)
    b15 = listing(15, start, latest + timedelta(minutes=3), CONUS)
    noaa = listing(None, start, latest + timedelta(minutes=3), "ABI-L2-RRQPEF")
    def make(hit: tuple[datetime, str]):
        t, key = hit
        iso, name = t.isoformat(timespec="seconds"), frame_name(t)
        old = previous.get(iso)
        if old:
            try:
                r = _session.get(f"{previous_base}{old['url']}", timeout=20)
                r.raise_for_status()
                (out / name).write_bytes(r.content)
                return {**old, "url": name}
            except Exception:  # noqa: BLE001 - recompute below
                pass
        official = nearest(noaa, t, timedelta(minutes=5))
        if official:
            rate, source = on_grid(official[1], grid, variable="RRQPE"), "NOAA RRQPE"
        else:
            split = nearest(b15, t)
            rate = estimate(on_grid(key, grid), on_grid(split[1], grid) if split else None)
            source = "IR estimate"
        colorize(to_mercator(rate, grid)).save(out / name, optimize=True)
        near = rate[np.isfinite(rate)]
        return {"time": iso, "url": name, "source": source, "max_mm_h": round(float(near.max()), 1) if near.size else 0.0}

    # Oldest first; in parallel, because the first run has every frame to compute.
    with ThreadPoolExecutor(8) as pool:
        frames = [f for f in pool.map(make, ten_minute_scans(b13)) if f]
    index = {
        "generated": now.isoformat(timespec="seconds"), "bounds": grid.bounds, "frames": frames,
        "legend": [{"from": e, "rgb": list(c)} for e, c in LEGEND], "units": "mm/h",
        "note": "Satellite estimate from cloud-top temperature, not radar.",
    }
    (out / "rain" / "index.json").write_text(json.dumps(index))
    return index

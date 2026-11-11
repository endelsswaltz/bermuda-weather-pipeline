"""Sea-surface temperature breaks around Bermuda, for the fishing map.

A temperature break is a place where the sea temperature changes quickly over a short distance.
Bait and pelagic fish gather along them. We take NASA's daily MUR analysis (GHRSST L4, 0.01
degrees, served without a login by NOAA CoastWatch's ERDDAP), thin it to 0.04 degrees, and mark
cells where the temperature changes by at least `MIN_BREAK` degrees C per 10 km.

MUR is a smoothed, gap-filled analysis about a day behind; under persistent cloud it leans on
older data, so breaks can be out of place or missing. It refreshes once a day, and between
refreshes the job re-publishes the previous file.
"""
from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timezone

import numpy as np

from .fetch import _session

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv"
HALF_WIDTH, STRIDE = 4.0, 4     # degrees around Bermuda; every 4th 0.01-degree cell
MIN_BREAK = 0.5                 # degrees C per 10 km
MAX_CELLS = 2500
REFRESH_H = 12


def gradient_per_10km(sst: np.ndarray, lat0: float, step_deg: float) -> np.ndarray:
    """Size of the temperature change (C per 10 km) at each cell of a regular lat/lon grid."""
    gy, gx = np.gradient(sst)
    km_y = step_deg * 111.2
    km_x = step_deg * 111.2 * math.cos(math.radians(lat0))
    return np.hypot(gy / km_y, gx / km_x) * 10


def breaks(sst: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat0: float, min_break: float = MIN_BREAK, max_cells: int = MAX_CELLS):
    """[lat, lon, C per 10 km] for every cell at or over the threshold, strongest first."""
    step = abs(float(lats[1] - lats[0]))
    grad = gradient_per_10km(sst, lat0, step)
    rows, cols = np.where(np.nan_to_num(grad, nan=0.0) >= min_break)
    order = np.argsort(-grad[rows, cols])[:max_cells]
    return [[round(float(lats[rows[i]]), 2), round(float(lons[cols[i]]), 2), round(float(grad[rows[i], cols[i]]), 2)] for i in order]


def parse_grid(text: str):
    """ERDDAP griddap CSV (time,latitude,longitude,value) to (time, lats, lons, 2-D values)."""
    rows = list(csv.reader(io.StringIO(text)))[2:]
    rows = [r for r in rows if len(r) == 4]
    lats = np.array(sorted({float(r[1]) for r in rows}))
    lons = np.array(sorted({float(r[2]) for r in rows}))
    grid = np.full((len(lats), len(lons)), np.nan)
    li = {v: i for i, v in enumerate(lats)}
    lo = {v: i for i, v in enumerate(lons)}
    for _, la, ln, v in rows:
        if v != "NaN":
            grid[li[float(la)], lo[float(ln)]] = float(v)
    return rows[0][0], lats, lons, grid


def fetch(lat0: float, lon0: float, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    q = (f"analysed_sst%5B(last)%5D%5B({lat0 - HALF_WIDTH}):{STRIDE}:({lat0 + HALF_WIDTH})%5D"
         f"%5B({lon0 - HALF_WIDTH}):{STRIDE}:({lon0 + HALF_WIDTH})%5D")
    r = _session.get(f"{ERDDAP}?{q}", timeout=180)
    r.raise_for_status()
    when, lats, lons, sst = parse_grid(r.text)
    cells = breaks(sst, lats, lons, lat0)
    return {
        "generated": now.isoformat(timespec="seconds"), "analysis_time": when, "min_break": MIN_BREAK,
        "units": "C per 10 km", "step_deg": STRIDE * 0.01, "cells": cells,
        "sst_range": [round(float(np.nanmin(sst)), 1), round(float(np.nanmax(sst)), 1)],
        "source": "NASA JPL MUR SST (GHRSST L4) via NOAA CoastWatch ERDDAP",
    }


def fresh_or_fetch(previous_url: str, lat0: float, lon0: float, now: datetime | None = None) -> dict:
    """Reuse the published file while it is under REFRESH_H old; if a refresh fails, keep the old one."""
    now = now or datetime.now(timezone.utc)
    prev = None
    try:
        prev = _session.get(previous_url, timeout=20).json()
        if (now - datetime.fromisoformat(prev["generated"])).total_seconds() < REFRESH_H * 3600:
            return prev
    except Exception:  # noqa: BLE001 - nothing published yet
        pass
    try:
        return fetch(lat0, lon0, now)
    except Exception:
        if prev:
            return prev
        raise

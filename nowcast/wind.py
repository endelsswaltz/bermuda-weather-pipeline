"""Gridded surface wind around Bermuda for the map's particle layer.

Open-Meteo has no gridded download, so we ask for a 1° grid of points (121 locations) in one
request, about once an hour, and publish u/v components in knots. Between refreshes the job
re-publishes the previous file, because each GitHub Pages deploy replaces the whole site.
Open-Meteo data is CC BY 4.0, free for non-commercial use.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np

from .fetch import _session

URL = "https://api.open-meteo.com/v1/forecast"
STEP = 1.0
REFRESH_MIN = 55


def grid_points(lat0: float, lon0: float, half_width: float, step: float = STEP):
    """Row-major points, north row first, west to east."""
    n = int(round(2 * half_width / step)) + 1
    lats = [round(lat0 + half_width - i * step, 4) for i in range(n)]
    lons = [round(lon0 - half_width + j * step, 4) for j in range(n)]
    return lats, lons


def to_uv(speed_kt: np.ndarray, from_deg: np.ndarray):
    """Wind speed/direction (where it comes from) to u (east) and v (north) components, kt."""
    rad = np.radians(from_deg)
    return -speed_kt * np.sin(rad), -speed_kt * np.cos(rad)


def build(responses: list[dict], lats: list[float], lons: list[float], now: datetime) -> dict:
    """Pack Open-Meteo's per-location responses (same order as the points) into one grid file."""
    times = responses[0]["hourly"]["time"]
    ny, nx = len(lats), len(lons)
    spd = np.array([r["hourly"]["wind_speed_10m"] for r in responses], dtype=float)      # (points, times)
    deg = np.array([r["hourly"]["wind_direction_10m"] for r in responses], dtype=float)
    u, v = to_uv(spd, deg)
    shape = lambda a: np.nan_to_num(a.T.reshape(len(times), ny, nx)).round(1).tolist()  # noqa: E731
    return {
        "generated": now.isoformat(timespec="seconds"),
        "north": lats[0], "west": lons[0], "step": STEP, "nx": nx, "ny": ny,
        "times": times, "u": shape(u), "v": shape(v), "units": "kt",
        "source": "Open-Meteo (CC BY 4.0)",
    }


def fetch(lat0: float, lon0: float, half_width: float, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    lats, lons = grid_points(lat0, lon0, half_width)
    pts = [(la, lo) for la in lats for lo in lons]
    r = _session.get(URL, params={
        "latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
        "hourly": "wind_speed_10m,wind_direction_10m", "wind_speed_unit": "kn",
        "past_hours": 3, "forecast_hours": 7, "timeformat": "unixtime",
    }, timeout=60)
    r.raise_for_status()
    return build(r.json(), lats, lons, now)


def fresh_or_fetch(previous_url: str, lat0: float, lon0: float, half_width: float, now: datetime | None = None) -> dict:
    """Reuse the published file while it is under REFRESH_MIN old; otherwise fetch a new one."""
    now = now or datetime.now(timezone.utc)
    try:
        prev = _session.get(previous_url, timeout=20).json()
        age = (now - datetime.fromisoformat(prev["generated"])).total_seconds() / 60
        if age < REFRESH_MIN:
            return prev
    except Exception:  # noqa: BLE001 - nothing published yet, or unreadable
        pass
    return fetch(lat0, lon0, half_width, now)

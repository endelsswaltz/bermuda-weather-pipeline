"""Lightning from the GOES-19 Geostationary Lightning Mapper (GLM).

GLM L2 "LCFA" files cover 20 seconds each and list every flash the instrument saw: position,
time, radiant energy and footprint area. We keep good-quality flashes in the box around
Bermuda from the last hour, as a small GeoJSON for the map plus counts near the island.

GLM detects total lightning (in-cloud and cloud-to-ground) from orbit; detection efficiency
over the western Atlantic is roughly 70-90%, so a few flashes from weak storms are missed,
and positions are good to about 5-10 km.
"""
from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np

from .fetch import _session, download, files_between  # noqa: F401 (_session is used by run.py)

PRODUCT = "GLM-L2-LCFA"
J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


@dataclass
class Flashes:
    t: np.ndarray        # epoch seconds (float)
    lat: np.ndarray
    lon: np.ndarray
    energy_fj: np.ndarray
    area_km2: np.ndarray

    def __len__(self):
        return len(self.t)

    def select(self, keep: np.ndarray) -> "Flashes":
        return Flashes(self.t[keep], self.lat[keep], self.lon[keep], self.energy_fj[keep], self.area_km2[keep])

    @classmethod
    def empty(cls) -> "Flashes":
        z = np.zeros(0)
        return cls(z, z, z, z, z)

    @classmethod
    def concat(cls, parts: list["Flashes"]) -> "Flashes":
        parts = [p for p in parts if len(p)] or [cls.empty()]
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("t", "lat", "lon", "energy_fj", "area_km2")))


def _unpack(var: h5py.Dataset) -> np.ndarray:
    """Apply _Unsigned, _FillValue, scale_factor and add_offset (GLM packs most fields as int16)."""
    raw = var[()]
    unsigned = var.attrs.get("_Unsigned")
    if unsigned is not None and np.asarray(unsigned).item() in (b"true", "true") and raw.dtype.kind == "i":
        raw = raw.view(raw.dtype.str.replace("i", "u"))
    data = raw.astype(np.float64)
    fill = var.attrs.get("_FillValue")
    if fill is not None:
        f = np.asarray(fill).astype(raw.dtype).item()
        data[raw == f] = np.nan
    scale, offset = var.attrs.get("scale_factor"), var.attrs.get("add_offset")
    if scale is not None:
        data *= np.asarray(scale).item()
    if offset is not None:
        data += np.asarray(offset).item()
    return data


def read_flashes(source) -> Flashes:
    """Good-quality flashes from one GLM LCFA file (path, file object or bytes)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with h5py.File(source, "r") as f:
        if f["flash_lat"].shape[0] == 0:
            return Flashes.empty()
        units = f["flash_time_offset_of_first_event"].attrs["units"]
        units = units.decode() if isinstance(units, bytes) else str(units)
        base = datetime.fromisoformat(units.replace("seconds since ", "").strip()).replace(tzinfo=timezone.utc)
        out = Flashes(
            t=base.timestamp() + _unpack(f["flash_time_offset_of_first_event"]),
            lat=f["flash_lat"][()].astype(np.float64),
            lon=f["flash_lon"][()].astype(np.float64),
            energy_fj=_unpack(f["flash_energy"]) * 1e15,
            area_km2=_unpack(f["flash_area"]) / 1e6,
        )
        quality = _unpack(f["flash_quality_flag"])
    return out.select(quality == 0)


def in_box(fl: Flashes, lat0: float, lon0: float, half_width: float) -> Flashes:
    return fl.select((np.abs(fl.lat - lat0) <= half_width) & (np.abs(fl.lon - lon0) <= half_width))


def load_flashes(lat0: float, lon0: float, half_width: float, minutes: int = 60, end: datetime | None = None, workers: int = 12) -> tuple[Flashes, int]:
    """Flashes in the box from the last `minutes`; also returns how many 20-second files were read."""
    end = end or datetime.now(timezone.utc)
    keys = [k for _, k in files_between(PRODUCT, None, end - timedelta(minutes=minutes), end)]
    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(lambda k: in_box(read_flashes(download(k)), lat0, lon0, half_width), keys))
    return Flashes.concat(parts), len(keys)


def from_geojson(gj: dict) -> Flashes:
    f = gj.get("features", [])
    col = lambda fn: np.array([fn(x) for x in f], dtype=np.float64)  # noqa: E731
    if not f:
        return Flashes.empty()
    return Flashes(col(lambda x: x["properties"]["t"]), col(lambda x: x["geometry"]["coordinates"][1]),
                   col(lambda x: x["geometry"]["coordinates"][0]), col(lambda x: x["properties"].get("e", 0)),
                   col(lambda x: x["properties"].get("a", 0)))


def load_incremental(previous: dict | None, lat0: float, lon0: float, half_width: float, now: datetime,
                     keep_min: int = 180, first_min: int = 60) -> tuple[Flashes, int, float]:
    """Flashes from the last `keep_min` minutes: those already published plus the files since.

    Returns (flashes, files read, covered_to epoch). With nothing usable published it reads the
    last `first_min` minutes, and the window grows from there run by run.
    """
    covered = float(previous.get("covered_to", 0)) if previous else 0.0
    cutoff = now.timestamp() - keep_min * 60
    if previous and covered > cutoff:
        old = from_geojson(previous)
        old = old.select(old.t > cutoff)
        since = datetime.fromtimestamp(covered, timezone.utc)
        new, n = load_flashes(lat0, lon0, half_width, max(1, int((now - since).total_seconds() // 60) + 1), now)
        new = new.select(new.t > covered)
        return Flashes.concat([old, new]), n, now.timestamp()
    fl, n = load_flashes(lat0, lon0, half_width, first_min, now)
    return fl, n, now.timestamp()


def distance_nm(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    dx = (lon - lon0) * 60 * math.cos(math.radians(lat0))
    dy = (lat - lat0) * 60
    return np.hypot(dx, dy)


COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def summary(fl: Flashes, lat0: float, lon0: float, now: datetime, window_min: int = 60, recent_min: int = 30) -> dict:
    """Counts near the island, for the nowcast JSON and the storms-nearby flag."""
    d = distance_nm(lat0, lon0, fl.lat, fl.lon)
    age = (now.timestamp() - fl.t) / 60
    recent = age <= recent_min
    out = {
        "window_min": window_min, "recent_min": recent_min, "count_box": int(len(fl)),
        "count_25nm": int((d <= 25).sum()), "count_50nm": int((d <= 50).sum()),
        "recent_25nm": int((recent & (d <= 25)).sum()), "recent_50nm": int((recent & (d <= 50)).sum()),
        "nearest_nm": None, "nearest_dir": None, "latest": None,
    }
    if len(fl):
        i = int(np.argmin(np.where(recent, d, np.inf))) if recent.any() else int(np.argmin(d))
        out["nearest_nm"] = round(float(d[i]))
        bearing = (math.degrees(math.atan2((fl.lon[i] - lon0) * math.cos(math.radians(lat0)), fl.lat[i] - lat0)) + 360) % 360
        out["nearest_dir"] = COMPASS[int(round(bearing / 22.5)) % 16]
        out["latest"] = datetime.fromtimestamp(float(fl.t.max()), timezone.utc).isoformat(timespec="seconds")
    return out


def to_geojson(fl: Flashes, max_features: int = 8000) -> dict:
    """Newest flashes first, capped so the file stays small in a big outbreak."""
    order = np.argsort(-fl.t)[:max_features]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(float(fl.lon[i]), 3), round(float(fl.lat[i]), 3)]},
                "properties": {"t": int(fl.t[i]), "e": round(float(fl.energy_fj[i]), 1), "a": round(float(fl.area_km2[i]))},
            }
            for i in order
        ],
    }

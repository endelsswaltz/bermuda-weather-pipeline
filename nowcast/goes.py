"""Read GOES-R ABI Cloud and Moisture Imagery (CMIP) files and resample them to a regular
latitude/longitude grid around a point.

Projection maths follows the GOES-R Product User Guide (PUG), Volume 5, section 4.2.8:
the ABI fixed grid gives each pixel as a pair of scan angles (x, y) in radians as seen from
the satellite, which we convert to and from geodetic latitude/longitude.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import h5py
import numpy as np


@dataclass(frozen=True)
class Projection:
    lon0: float   # longitude of projection origin (degrees)
    h: float      # satellite distance from Earth's centre (m)
    r_eq: float   # semi-major axis (m)
    r_pol: float  # semi-minor axis (m)


@dataclass
class Scene:
    """A CMIP image: brightness temperature (K) on the ABI fixed grid."""
    start: datetime
    x: np.ndarray        # scan angles east-west (rad), 1-D
    y: np.ndarray        # scan angles north-south (rad), 1-D
    bt: np.ndarray       # brightness temperature (K), shape (len(y), len(x)); NaN where missing
    proj: Projection


def _scaled(var: h5py.Dataset, sel=()) -> np.ndarray:
    raw = var[sel] if sel != () else var[()]
    data = raw.astype(np.float64)
    fill = var.attrs.get("_FillValue")
    if fill is not None:
        data[raw == np.asarray(fill).item()] = np.nan
    scale = var.attrs.get("scale_factor")
    offset = var.attrs.get("add_offset")
    if scale is not None:
        data = data * np.asarray(scale).item()
    if offset is not None:
        data = data + np.asarray(offset).item()
    return data


def read_cmip(source, window: tuple[float, float, float, float] | None = None) -> Scene:
    """Read a GOES-R ABI L2 CMIP file (any sector, one band) from a path or file object.

    With `window` = (south, north, west, east) in degrees, only the rows and columns covering
    that box are read, so a remote file fetches just those chunks.
    """
    with h5py.File(source, "r") as f:
        p = f["goes_imager_projection"].attrs
        proj = Projection(
            lon0=float(np.asarray(p["longitude_of_projection_origin"]).item()),
            h=float(np.asarray(p["perspective_point_height"]).item() + np.asarray(p["semi_major_axis"]).item()),
            r_eq=float(np.asarray(p["semi_major_axis"]).item()),
            r_pol=float(np.asarray(p["semi_minor_axis"]).item()),
        )
        start = f.attrs["time_coverage_start"]
        start = start.decode() if isinstance(start, bytes) else str(start)
        x, y = _scaled(f["x"]), _scaled(f["y"])
        rows, cols = slice(None), slice(None)
        if window is not None:
            s_, n_, w_, e_ = window
            lat = np.array([s_, s_, n_, n_, (s_ + n_) / 2, (s_ + n_) / 2])
            lon = np.array([w_, e_, w_, e_, w_, e_])
            sx, sy = latlon_to_scan(lat, lon, proj)
            dx, dy = x[1] - x[0], y[1] - y[0]
            c = (np.array([np.nanmin(sx), np.nanmax(sx)]) - x[0]) / dx
            r = (np.array([np.nanmin(sy), np.nanmax(sy)]) - y[0]) / dy
            c0, c1 = int(max(np.floor(c.min()) - 2, 0)), int(min(np.ceil(c.max()) + 3, len(x)))
            r0, r1 = int(max(np.floor(r.min()) - 2, 0)), int(min(np.ceil(r.max()) + 3, len(y)))
            rows, cols = slice(r0, r1), slice(c0, c1)
        return Scene(
            start=datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(timezone.utc),
            x=x[cols], y=y[rows], bt=_scaled(f["CMI"], (rows, cols)), proj=proj,
        )


def latlon_to_scan(lat: np.ndarray, lon: np.ndarray, proj: Projection):
    """Geodetic lat/lon (degrees) to ABI scan angles (radians). NaN where not visible."""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lon0 = np.radians(proj.lon0)
    e2 = (proj.r_eq**2 - proj.r_pol**2) / proj.r_eq**2
    phi_c = np.arctan((proj.r_pol**2 / proj.r_eq**2) * np.tan(lat))
    r_c = proj.r_pol / np.sqrt(1 - e2 * np.cos(phi_c) ** 2)
    sx = proj.h - r_c * np.cos(phi_c) * np.cos(lon - lon0)
    sy = -r_c * np.cos(phi_c) * np.sin(lon - lon0)
    sz = r_c * np.sin(phi_c)
    visible = proj.h * (proj.h - sx) >= sy**2 + (proj.r_eq**2 / proj.r_pol**2) * sz**2
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    y = np.arctan(sz / sx)
    return np.where(visible, x, np.nan), np.where(visible, y, np.nan)


def scan_to_latlon(x: np.ndarray, y: np.ndarray, proj: Projection):
    """ABI scan angles (radians) to geodetic lat/lon (degrees). NaN off the Earth's disk."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon0 = np.radians(proj.lon0)
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (proj.r_eq**2 / proj.r_pol**2) * np.sin(y) ** 2)
    b = -2 * proj.h * np.cos(x) * np.cos(y)
    c = proj.h**2 - proj.r_eq**2
    disc = b**2 - 4 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2 * a)
    sx = rs * np.cos(x) * np.cos(y)
    sy = -rs * np.sin(x)
    sz = rs * np.cos(x) * np.sin(y)
    lat = np.degrees(np.arctan((proj.r_eq**2 / proj.r_pol**2) * sz / np.sqrt((proj.h - sx) ** 2 + sy**2)))
    lon = np.degrees(lon0 - np.arctan(sy / (proj.h - sx)))
    off = disc < 0
    return np.where(off, np.nan, lat), np.where(off, np.nan, lon)


@dataclass
class Grid:
    """A regular lat/lon grid; row 0 is the northern edge."""
    lats: np.ndarray   # 1-D, descending
    lons: np.ndarray   # 1-D, ascending
    step: float        # degrees

    @classmethod
    def around(cls, lat: float, lon: float, half_width: float, step: float) -> "Grid":
        n = int(round(2 * half_width / step)) + 1
        return cls(lats=lat + half_width - step * np.arange(n), lons=lon - half_width + step * np.arange(n), step=step)

    @property
    def bounds(self):
        """[[south, west], [north, east]] of the pixel edges, for map overlays."""
        h = self.step / 2
        return [[float(self.lats[-1] - h), float(self.lons[0] - h)], [float(self.lats[0] + h), float(self.lons[-1] + h)]]


def resample(scene: Scene, grid: Grid) -> np.ndarray:
    """Nearest-neighbour brightness temperature (K) on `grid`; NaN outside the scene."""
    lon2d, lat2d = np.meshgrid(grid.lons, grid.lats)
    sx, sy = latlon_to_scan(lat2d, lon2d, scene.proj)
    # x ascends; y descends (north first) in ABI files.
    dx = scene.x[1] - scene.x[0]
    dy = scene.y[1] - scene.y[0]
    with np.errstate(invalid="ignore"):
        col = np.rint((sx - scene.x[0]) / dx)
        row = np.rint((sy - scene.y[0]) / dy)
    ok = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < len(scene.x)) & (row >= 0) & (row < len(scene.y))
    out = np.full(lat2d.shape, np.nan)
    out[ok] = scene.bt[row[ok].astype(int), col[ok].astype(int)]
    return out

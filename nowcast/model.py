"""Rain-arrival nowcast for Bermuda from GOES-19 clean longwave infrared (band 13).

Method, in short:
1. Brightness temperature (BT) on a regular 0.04° grid around Bermuda (goes.resample).
2. Cloud motion from dense optical flow (OpenCV Farneback) between consecutive scans.
3. "Rain-bearing" cloud = tops colder than a threshold (default -40 °C), a common proxy for
   deep convection; it is a proxy only (thick cirrus can be cold without rain, and shallow
   warm showers are missed).
4. Extrapolate that mask forward along the motion field (semi-Lagrangian, steady motion) in
   10-minute steps up to 3 hours and report the first time it reaches Bermuda.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .goes import Grid

KELVIN = 273.15


@dataclass
class Config:
    rain_top_c: float = -40.0       # cloud tops colder than this count as rain-bearing
    storm_top_c: float = -52.0      # coldest top within 50 nm below this sets the storm flag
    cloud_margin_k: float = 6.0     # colder than the clear-ocean reference by this = cloud
    arrival_radius_nm: float = 8.0  # "reaches Bermuda" when rain cloud is within this distance
    min_cells_px: int = 4           # ignore specks smaller than this many pixels
    horizon_min: int = 180
    step_min: int = 10


@dataclass
class Nowcast:
    issued: str                    # ISO time of the latest scan used
    status: str                    # 'rain_now' | 'rain_expected' | 'no_rain' | 'unknown'
    text: str
    eta_min: float | None = None
    eta: str | None = None         # ISO time
    from_bearing: float | None = None   # degrees, direction the rain comes from
    from_dir: str | None = None
    distance_nm: float | None = None
    speed_kt: float | None = None
    moving_toward: float | None = None
    confidence: str = "low"
    cloud_cover_25nm: float | None = None   # percent
    cloud_cover_50nm: float | None = None
    coldest_top_50nm_c: float | None = None
    storms_nearby: bool = False
    nearest_rain_nm: float | None = None
    nearest_rain_dir: str | None = None
    config: dict = field(default_factory=dict)


COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
compass = lambda deg: COMPASS[int(round(deg / 22.5)) % 16]  # noqa: E731


# ---------- grid geometry ----------
def pixel_nm(grid: Grid, lat: float) -> tuple[float, float]:
    """Size of one grid pixel in nautical miles (east-west, north-south) at `lat`."""
    return grid.step * 60 * math.cos(math.radians(lat)), grid.step * 60


def distance_nm_map(grid: Grid, lat0: float, lon0: float) -> np.ndarray:
    lon2d, lat2d = np.meshgrid(grid.lons, grid.lats)
    dx = (lon2d - lon0) * 60 * np.cos(np.radians(lat0))
    dy = (lat2d - lat0) * 60
    return np.hypot(dx, dy)


def bearing_deg(dx_east: float, dy_north: float) -> float:
    return (math.degrees(math.atan2(dx_east, dy_north)) + 360) % 360


# ---------- motion ----------
def _to_u8(bt_k: np.ndarray) -> np.ndarray:
    """Scale BT to 8-bit with cold (bright cloud) high, filling gaps with the median."""
    filled = np.where(np.isfinite(bt_k), bt_k, np.nanmedian(bt_k))
    scaled = np.clip((310.0 - filled) / (310.0 - 190.0), 0, 1) * 255
    return scaled.astype(np.uint8)


def optical_flow(prev_k: np.ndarray, next_k: np.ndarray) -> np.ndarray:
    """Dense flow (pixels per frame interval) from prev to next; shape (H, W, 2) as (dx, dy)."""
    return cv2.calcOpticalFlowFarneback(
        _to_u8(prev_k), _to_u8(next_k), None,
        pyr_scale=0.5, levels=4, winsize=31, iterations=4, poly_n=7, poly_sigma=1.5, flags=0,
    )


def steering_flow(flow: np.ndarray, weight: np.ndarray, sigma_px: float = 25) -> np.ndarray:
    """A smooth motion field that also covers clear gaps.

    Optical flow only measures motion where there is cloud to track, so clear sky between a
    shower and the island reads as "not moving". Normalised convolution spreads the motion of
    nearby cloud across those gaps; where there is little cloud at all, fall back to the mean
    motion of all cloud in the box.
    """
    w = weight.astype(np.float32)
    k = int(sigma_px * 3) | 1
    den = cv2.GaussianBlur(w, (k, k), sigma_px)
    out = np.empty_like(flow, dtype=np.float32)
    total = float(w.sum())
    for i in range(2):
        num = cv2.GaussianBlur((flow[..., i] * w).astype(np.float32), (k, k), sigma_px)
        mean = float((flow[..., i] * w).sum() / total) if total > 0 else 0.0
        # Blend toward the box-wide mean where nearby cloud is sparse.
        local = num / np.maximum(den, 1e-6)
        trust = np.clip(den / 0.05, 0, 1)
        out[..., i] = trust * local + (1 - trust) * mean
    return out


def advect(field_: np.ndarray, flow_per_step: np.ndarray, steps: float) -> np.ndarray:
    """Field after `steps` frame intervals, assuming steady motion (semi-Lagrangian backtrack)."""
    h, w = field_.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx - flow_per_step[..., 0].astype(np.float32) * steps
    map_y = gy - flow_per_step[..., 1].astype(np.float32) * steps
    return cv2.remap(field_.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def drop_specks(mask: np.ndarray, min_px: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


# ---------- the nowcast ----------
def run_nowcast(frames_k: list[np.ndarray], times, grid: Grid, lat0: float, lon0: float, cfg: Config | None = None) -> tuple[Nowcast, np.ndarray]:
    """frames_k: BT (K) on `grid`, oldest first, equally spaced (10 min); times: their datetimes.

    Returns the nowcast and the smoothed motion field (pixels per 10 min) for map overlays.
    """
    cfg = cfg or Config()
    latest = frames_k[-1]
    issued = times[-1]
    dist = distance_nm_map(grid, lat0, lon0)
    centre = np.unravel_index(np.argmin(dist), dist.shape)
    px_ew, px_ns = pixel_nm(grid, lat0)
    interval_min = (times[-1] - times[-2]).total_seconds() / 60

    # Cloud: colder than the clear-ocean reference (the warm end of the scene).
    finite = np.isfinite(latest)
    clear_ref = float(np.nanpercentile(latest, 95))
    cloud = finite & (latest < clear_ref - cfg.cloud_margin_k)

    # Motion: flow between each consecutive pair, measured on cloud and spread across gaps;
    # the two estimates' disagreement at Bermuda feeds the confidence.
    flows = [steering_flow(optical_flow(a, b), cloud) for a, b in zip(frames_k[:-1], frames_k[1:])]
    flow = np.mean(flows, axis=0) * (cfg.step_min / interval_min)  # pixels per 10-minute step
    spread = float(np.linalg.norm(flows[-1][centre] - flows[0][centre])) if len(flows) > 1 else 0.0
    def cover(radius):
        m = (dist <= radius) & finite
        return round(100 * float(cloud[m].mean()), 1) if m.any() else None
    near50 = (dist <= 50) & finite
    coldest = float(np.nanmin(latest[near50])) - KELVIN if near50.any() else None

    # Rain-bearing cloud, now and extrapolated.
    rain = drop_specks(finite & (latest < cfg.rain_top_c + KELVIN), cfg.min_cells_px)
    near_island = dist <= cfg.arrival_radius_nm
    nc = Nowcast(
        issued=issued.isoformat(), status="no_rain", text="", config=asdict(cfg),
        cloud_cover_25nm=cover(25), cloud_cover_50nm=cover(50),
        coldest_top_50nm_c=round(coldest, 1) if coldest is not None else None,
        storms_nearby=coldest is not None and coldest < cfg.storm_top_c,
    )
    if rain.any():
        d = np.where(rain, dist, np.inf)
        r, c = np.unravel_index(np.argmin(d), d.shape)
        nc.nearest_rain_nm = round(float(d[r, c]), 0)
        nc.nearest_rain_dir = compass(bearing_deg((c - centre[1]) * px_ew, (centre[0] - r) * px_ns))

    vx, vy = flow[centre]  # pixels per step at Bermuda (x east, y south)
    speed_kt = math.hypot(vx * px_ew, vy * px_ns) * (60 / cfg.step_min)
    toward = bearing_deg(vx * px_ew, -vy * px_ns)

    if (rain & near_island).any():
        nc.status, nc.eta_min = "rain_now", 0
    else:
        for step in range(1, cfg.horizon_min // cfg.step_min + 1):
            future = advect(rain.astype(np.float32), flow, step) > 0.5
            if (future & near_island).any():
                nc.status, nc.eta_min = "rain_expected", step * cfg.step_min
                # Where that cloud is now: trace back from Bermuda along the motion.
                src_x = -vx * step * px_ew
                src_y = vy * step * px_ns  # pixel y grows southward
                nc.distance_nm = round(math.hypot(src_x, src_y), 0)
                nc.from_bearing = round(bearing_deg(src_x, src_y), 0)
                nc.from_dir = compass(nc.from_bearing)
                break

    if nc.eta_min is not None:
        from datetime import timedelta
        nc.eta = (issued + timedelta(minutes=nc.eta_min)).isoformat()
    nc.speed_kt = round(speed_kt, 0)
    nc.moving_toward = round(toward, 0)

    # Confidence: agreement between motion estimates and how far ahead we are looking.
    if nc.status == "rain_now":
        nc.confidence = "high"
    elif nc.status == "rain_expected":
        nc.confidence = "high" if nc.eta_min <= 60 and spread < 0.6 else "medium" if nc.eta_min <= 120 and spread < 1.2 else "low"
    else:
        nc.confidence = "medium" if spread < 1.2 else "low"
    nc.text = describe(nc)
    return nc, flow


def describe(nc: Nowcast) -> str:
    if nc.status == "rain_now":
        return "Showers over or right next to Bermuda now."
    if nc.status == "rain_expected":
        when = f"in about {nc.eta_min:.0f} min" if nc.eta_min < 60 else f"in about {nc.eta_min / 60:.1f} h".replace(".0 h", " h")
        moving = f", moving {compass(nc.moving_toward)} at ~{nc.speed_kt:.0f} kt" if nc.speed_kt and nc.speed_kt >= 3 else ""
        return f"Showers {nc.distance_nm:.0f} nm {nc.from_dir}{moving}; could reach Bermuda {when}."
    if nc.nearest_rain_nm is not None and nc.nearest_rain_nm <= 100:
        return f"No rain expected in the next 3 h. Nearest shower cloud {nc.nearest_rain_nm:.0f} nm {nc.nearest_rain_dir}, not heading this way."
    return "No rain expected in the next 3 h."

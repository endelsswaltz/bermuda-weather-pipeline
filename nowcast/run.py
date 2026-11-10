"""Run one nowcast: fetch the latest GOES-19 band 13 scans, estimate rain arrival at Bermuda,
and write nowcast.json, ir.png and motion.json to an output folder.

    python -m nowcast.run --out site
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from .fetch import files_between, open_remote
from . import glm, wind
from .goes import Grid, read_cmip, resample
from .model import KELVIN, Config, pixel_nm, run_nowcast

LAT, LON = 32.3, -64.78
HALF_WIDTH, STEP = 5.0, 0.04
PRODUCT, BAND = "ABI-L2-CMIPC", 13      # CONUS sector, 5-minute scans, covers ±5° around Bermuda
FALLBACK = "ABI-L2-CMIPF"               # full disk, 10-minute scans
FRAME_GAP = timedelta(minutes=10)
N_FRAMES = 3
LIGHTNING_MIN = 60          # flashes kept for the map
LIGHTNING_RECENT_MIN = 30   # flashes this recent within 50 nm set the storms-nearby flag
PAGES = "https://endelsswaltz.github.io/bermuda-weather-pipeline/"


def pick_frames(available: list[tuple[datetime, str]], latest: datetime | None = None) -> list[tuple[datetime, str]]:
    """The newest scan and the ones ~10 and ~20 minutes before it (oldest first)."""
    if not available:
        return []
    latest = latest or available[-1][0]
    chosen = []
    for i in range(N_FRAMES - 1, -1, -1):
        target = latest - i * FRAME_GAP
        t, k = min(available, key=lambda tk: abs((tk[0] - target).total_seconds()))
        if abs((t - target).total_seconds()) <= 150:
            chosen.append((t, k))
    return chosen if len(chosen) == N_FRAMES else []


def load_frames(grid: Grid, at: datetime | None = None):
    end = at or datetime.now(timezone.utc)
    window = (grid.lats[-1], grid.lats[0], grid.lons[0], grid.lons[-1])
    for product in (PRODUCT, FALLBACK):
        frames = pick_frames(files_between(product, BAND, end - timedelta(minutes=45), end))
        if frames:
            bts = [resample(read_cmip(open_remote(k), window), grid) for _, k in frames]
            if np.isfinite(bts[-1]).mean() > 0.9:
                return [t for t, _ in frames], bts, product
    raise RuntimeError("no usable GOES-19 band 13 scans in the last 45 minutes")


def to_mercator_rows(field_: np.ndarray, grid: Grid) -> np.ndarray:
    """Re-space the rows evenly in Web Mercator, so the image lines up on a Leaflet map.

    The grid is evenly spaced in latitude; stretched over 10° of latitude that would put the
    middle of the image a few nautical miles off on a Mercator map.
    """
    merc = lambda lat: np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))  # noqa: E731
    north, south = grid.bounds[1][0], grid.bounds[0][0]
    n = field_.shape[0]
    ys = merc(north) + (merc(south) - merc(north)) * (np.arange(n) + 0.5) / n
    lats = np.degrees(2 * np.arctan(np.exp(ys)) - np.pi / 2)
    rows = np.clip(np.rint((grid.lats[0] - lats) / grid.step).astype(int), 0, n - 1)
    return field_[rows]


def colorize(bt_k: np.ndarray) -> Image.Image:
    """RGBA image of BT using NASA GIBS's colour table for this band (transparent where missing)."""
    table = json.loads(resources.files("nowcast").joinpath("ir_colormap.json").read_text())["entries"]
    lo = np.array([e[0] for e in table])
    rgb = np.array([e[2:5] for e in table], dtype=np.uint8)
    c = bt_k - KELVIN
    idx = np.clip(np.searchsorted(lo, np.nan_to_num(c, nan=-999), side="right") - 1, 0, len(table) - 1)
    out = np.zeros((*bt_k.shape, 4), dtype=np.uint8)
    out[..., :3] = rgb[idx]
    out[..., 3] = np.where(np.isfinite(bt_k), 255, 0)
    return Image.fromarray(out)


def motion_vectors(flow: np.ndarray, grid: Grid, every_deg: float = 0.5):
    """Sparse motion arrows (knots) for the map overlay."""
    step = int(round(every_deg / grid.step))
    px_ew, px_ns = pixel_nm(grid, LAT)
    out = []
    for r in range(step // 2, len(grid.lats), step):
        for c in range(step // 2, len(grid.lons), step):
            u = float(flow[r, c, 0]) * px_ew * 6     # east, kt (flow is pixels per 10 min)
            v = -float(flow[r, c, 1]) * px_ns * 6    # north, kt
            out.append([round(float(grid.lats[r]), 2), round(float(grid.lons[c]), 2), round(u, 1), round(v, 1)])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="site", help="output folder")
    ap.add_argument("--rain-top-c", type=float, default=Config.rain_top_c)
    ap.add_argument("--storm-top-c", type=float, default=Config.storm_top_c)
    args = ap.parse_args(argv)

    cfg = Config(rain_top_c=args.rain_top_c, storm_top_c=args.storm_top_c)
    grid = Grid.around(LAT, LON, HALF_WIDTH, STEP)
    times, bts, product = load_frames(grid)
    nc, flow = run_nowcast(bts, times, grid, LAT, LON, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    try:
        grid_wind = wind.fresh_or_fetch(f"{PAGES}wind.json", LAT, LON, HALF_WIDTH, now)
        (out / "wind.json").write_text(json.dumps(grid_wind, separators=(",", ":")))
        wind_info = {"url": "wind.json", "generated": grid_wind["generated"]}
    except Exception as e:  # noqa: BLE001
        print(f"wind unavailable: {e}")
        wind_info = None

    # Lightning is a separate source; if it fails, publish the nowcast without it.
    try:
        flashes, n_files = glm.load_flashes(LAT, LON, HALF_WIDTH, LIGHTNING_MIN, now)
        lightning = {**glm.summary(flashes, LAT, LON, now, LIGHTNING_MIN, LIGHTNING_RECENT_MIN), "files": n_files,
                     "generated": now.isoformat(timespec="seconds"), "source": "NOAA GOES-19 GLM via AWS"}
        (out / "lightning.geojson").write_text(json.dumps({**glm.to_geojson(flashes), "generated": lightning["generated"]}))
        if lightning["recent_50nm"] > 0:
            nc.storms_nearby = True
    except Exception as e:  # noqa: BLE001
        print(f"lightning unavailable: {e}")
        lightning = None
    colorize(to_mercator_rows(bts[-1], grid)).save(out / "ir.png", optimize=True)
    payload = {
        **nc.__dict__,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scans": [t.isoformat() for t in times],
        "product": f"GOES-19 ABI {product} band 13",
        "image": {"url": "ir.png", "bounds": grid.bounds, "time": times[-1].isoformat()},
        "lightning": lightning,
        "wind": wind_info,
        "source": "NOAA GOES-19 via the NOAA Open Data Dissemination program on AWS",
    }
    (out / "nowcast.json").write_text(json.dumps(payload, indent=1, default=str))
    (out / "motion.json").write_text(json.dumps({"time": times[-1].isoformat(), "vectors": motion_vectors(flow, grid)}))
    print(f"{times[-1]:%Y-%m-%d %H:%M} UTC ({product}): {nc.text} [cloud 25 nm {nc.cloud_cover_25nm}%, "
          f"coldest 50 nm {nc.coldest_top_50nm_c} °C, confidence {nc.confidence}]")
    if lightning:
        print(f"lightning: {lightning['count_box']} flashes in {LIGHTNING_MIN} min, {lightning['recent_50nm']} within 50 nm "
              f"in {LIGHTNING_RECENT_MIN} min, nearest {lightning['nearest_nm']} nm {lightning['nearest_dir']}")


if __name__ == "__main__":
    main()

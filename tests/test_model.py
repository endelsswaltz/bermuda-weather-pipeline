from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nowcast.goes import Grid, Projection, latlon_to_scan, scan_to_latlon
from nowcast.model import Config, run_nowcast

LAT, LON = 32.3, -64.78
GOES19 = Projection(lon0=-75.0, h=35786023.0 + 6378137.0, r_eq=6378137.0, r_pol=6356752.31414)
CLEAR_K = 295.0  # warm ocean
COLD_K = 273.15 - 55  # a deep convective top


def scene_with_blob(grid: Grid, centre_lat: float, centre_lon: float, radius_px: int = 6, k: float = COLD_K):
    bt = np.full((len(grid.lats), len(grid.lons)), CLEAR_K)
    r = int(np.argmin(np.abs(grid.lats - centre_lat)))
    c = int(np.argmin(np.abs(grid.lons - centre_lon)))
    yy, xx = np.ogrid[: bt.shape[0], : bt.shape[1]]
    bt[(yy - r) ** 2 + (xx - c) ** 2 <= radius_px**2] = k
    # Some texture so optical flow has something to track besides the blob edge.
    rng = np.random.default_rng(0)
    return bt + rng.normal(0, 0.3, bt.shape)


def moving_frames(grid, start_lon, deg_per_frame, n=3, lat=LAT):
    t0 = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
    frames = [scene_with_blob(grid, lat, start_lon + i * deg_per_frame) for i in range(n)]
    times = [t0 + timedelta(minutes=10 * i) for i in range(n)]
    return frames, times


def test_projection_round_trip():
    for lat, lon in [(LAT, LON), (27.3, -69.8), (37.3, -59.8), (0.0, -75.0)]:
        x, y = latlon_to_scan(lat, lon, GOES19)
        lat2, lon2 = scan_to_latlon(x, y, GOES19)
        assert float(lat2) == pytest.approx(lat, abs=1e-6)
        assert float(lon2) == pytest.approx(lon, abs=1e-6)


def test_far_side_of_earth_is_not_visible():
    x, y = latlon_to_scan(0.0, 105.0, GOES19)
    assert np.isnan(x) and np.isnan(y)


def test_shower_moving_east_arrives_on_time_from_the_west():
    grid = Grid.around(LAT, LON, 3, 0.04)
    # 0.06° of longitude per 10 min at 32.3°N ≈ 3.0 nm per 10 min ≈ 18 kt.
    # Blob edge starts ~30 nm west of the 8 nm arrival radius → expect ~70–90 minutes.
    start_lon = LON - (30 + 8 + 6 * 0.04 * 60 * np.cos(np.radians(LAT))) / (60 * np.cos(np.radians(LAT)))
    frames, times = moving_frames(grid, start_lon - 2 * 0.06, 0.06)
    nc, _ = run_nowcast(frames, times, grid, LAT, LON)
    assert nc.status == "rain_expected"
    assert 60 <= nc.eta_min <= 110
    assert nc.from_dir in {"W", "WNW", "WSW"}
    assert 12 <= nc.speed_kt <= 24
    assert nc.text.startswith("Showers")


def test_shower_moving_away_is_not_expected():
    grid = Grid.around(LAT, LON, 3, 0.04)
    frames, times = moving_frames(grid, LON - 1.2, -0.06)  # west of Bermuda, heading further west
    nc, _ = run_nowcast(frames, times, grid, LAT, LON)
    assert nc.status == "no_rain"
    assert nc.nearest_rain_dir in {"W", "WNW", "WSW"}


def test_rain_overhead_now():
    grid = Grid.around(LAT, LON, 3, 0.04)
    frames, times = moving_frames(grid, LON, 0.0)
    nc, _ = run_nowcast(frames, times, grid, LAT, LON)
    assert nc.status == "rain_now" and nc.eta_min == 0


def test_cloud_cover_and_storm_flag():
    grid = Grid.around(LAT, LON, 3, 0.04)
    frames, times = moving_frames(grid, LON, 0.0)
    nc, _ = run_nowcast(frames, times, grid, LAT, LON)
    # The blob (radius 6 px ≈ 14 nm) over Bermuda covers part of the 25 nm circle, less of the 50 nm one.
    assert 0 < nc.cloud_cover_50nm < nc.cloud_cover_25nm < 100
    assert nc.storms_nearby and nc.coldest_top_50nm_c < -50


def test_thresholds_are_configurable():
    grid = Grid.around(LAT, LON, 3, 0.04)
    frames, times = moving_frames(grid, LON, 0.0)
    nc, _ = run_nowcast(frames, times, grid, LAT, LON, Config(rain_top_c=-60.0, storm_top_c=-70.0))
    assert nc.status == "no_rain" and not nc.storms_nearby


def test_mercator_rows_put_bermuda_where_leaflet_expects_it():
    from nowcast.run import to_mercator_rows
    grid = Grid.around(LAT, LON, 5, 0.04)
    rows_lat = np.repeat(grid.lats[:, None], len(grid.lons), axis=1)
    img = to_mercator_rows(rows_lat, grid)
    # Leaflet places image row i at Mercator-even spacing between the bounds; check the middle.
    merc = lambda lat: np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))  # noqa: E731
    north, south = grid.bounds[1][0], grid.bounds[0][0]
    target_y = merc(north) + (merc(south) - merc(north)) * 0.5
    expected_lat = np.degrees(2 * np.arctan(np.exp(target_y)) - np.pi / 2)
    assert img[img.shape[0] // 2, 0] == pytest.approx(expected_lat, abs=0.04)


def test_thin_cirrus_is_not_rain():
    grid = Grid.around(LAT, LON, 3, 0.04)
    frames, times = moving_frames(grid, LON, 0.0)        # cold blob sitting on Bermuda
    thick = [f - 1.0 for f in frames][-1]                # band 15 one kelvin colder: thick cloud
    thin = [f - 5.0 for f in frames][-1]                 # five kelvin colder: thin cirrus
    assert run_nowcast(frames, times, grid, LAT, LON, bt15_latest=thick)[0].status == "rain_now"
    assert run_nowcast(frames, times, grid, LAT, LON, bt15_latest=thin)[0].status == "no_rain"
    assert run_nowcast(frames, times, grid, LAT, LON)[0].status == "rain_now"   # no band 15: screen off

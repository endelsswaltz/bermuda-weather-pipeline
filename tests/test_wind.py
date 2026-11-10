from datetime import datetime, timezone

import numpy as np
import pytest

from nowcast.wind import build, grid_points, to_uv


def test_uv_points_downwind():
    u, v = to_uv(np.array([10.0, 10.0]), np.array([0.0, 270.0]))  # from N, from W
    assert u[0] == pytest.approx(0, abs=1e-9) and v[0] == pytest.approx(-10)
    assert u[1] == pytest.approx(10) and v[1] == pytest.approx(0, abs=1e-9)


def test_grid_is_north_first_west_to_east():
    lats, lons = grid_points(32.3, -64.78, 5)
    assert len(lats) == len(lons) == 11
    assert lats[0] == pytest.approx(37.3) and lats[-1] == pytest.approx(27.3)
    assert lons[0] == pytest.approx(-69.78) and lons[-1] == pytest.approx(-59.78)


def test_build_places_each_point_in_its_cell():
    lats, lons = [33.0, 32.0], [-65.0, -64.0, -63.0]
    # Point k has speed k, from the south (so v = +k, u = 0), at two times.
    resp = [{"hourly": {"time": [0, 3600], "wind_speed_10m": [k, k], "wind_direction_10m": [180, 180]}} for k in range(6)]
    out = build(resp, lats, lons, datetime(2026, 11, 10, tzinfo=timezone.utc))
    assert (out["ny"], out["nx"], out["north"], out["west"]) == (2, 3, 33.0, -65.0)
    assert out["v"][1] == [[0, 1, 2], [3, 4, 5]]
    assert all(x == 0 for row in out["u"][0] for x in row)

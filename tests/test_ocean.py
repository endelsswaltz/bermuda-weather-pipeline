import numpy as np

from nowcast.ocean import breaks, gradient_per_10km, parse_grid


def test_a_sharp_front_is_found_and_smooth_water_is_not():
    lats = np.arange(30.0, 34.0, 0.04)
    lons = np.arange(-67.0, -63.0, 0.04)
    sst = np.full((len(lats), len(lons)), 24.0)
    sst[:, 50:] = 26.0                       # 2 C step between two columns (about 3.8 km apart)
    cells = breaks(sst, lats, lons, 32.3)
    assert cells and all(abs(c[1] - lons[50]) < 0.09 for c in cells)   # only along the step
    assert cells[0][2] > 2                                              # strong: > 2 C per 10 km
    gentle = 24.0 + np.tile(np.linspace(0, 1, len(lons)), (len(lats), 1))  # 1 C over ~375 km
    assert breaks(gentle, lats, lons, 32.3) == []


def test_gradient_units():
    lats = np.arange(0, 1, 0.1)
    sst = np.tile(np.arange(10) * 1.0, (10, 1)).T   # 1 C per 0.1 deg of latitude = 11.12 km
    g = gradient_per_10km(sst, 0.0, 0.1)
    assert abs(g[5, 5] - 10 / 11.12) < 0.01


def test_parse_erddap_csv():
    text = "time,latitude,longitude,analysed_sst\nUTC,degrees_north,degrees_east,degree_C\n" + "".join(
        f"2026-11-09T09:00:00Z,{la},{lo},{v}\n" for la, lo, v in [(32.0, -65.0, 24.5), (32.0, -64.96, "NaN"), (32.04, -65.0, 24.7), (32.04, -64.96, 24.9)])
    when, lats, lons, grid = parse_grid(text)
    assert when == "2026-11-09T09:00:00Z" and list(lats) == [32.0, 32.04] and grid.shape == (2, 2)
    assert np.isnan(grid[0, 1]) and grid[1, 1] == 24.9

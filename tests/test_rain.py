import numpy as np

from nowcast.rain import LEGEND, colorize, estimate

K = 273.15


def test_colder_tops_rain_harder_and_warm_cloud_is_dry():
    bt = np.array([[K - 20, K - 35, K - 50, K - 65, K - 80]])
    r = estimate(bt)[0]
    assert r[0] == 0                       # warmer than -30 C: never rain
    assert 0 < r[1] < r[2] < r[3] <= r[4]  # monotonic with cooling
    assert r[4] == 50.0                    # capped
    assert 3 < r[2] < 6                    # about 4.5 mm/h at -50 C


def test_thin_cirrus_is_screened_out_and_gaps_stay_gaps():
    bt13 = np.array([[K - 55, K - 55, np.nan]])
    bt15 = np.array([[K - 56, K - 60, K - 60]])  # splits of 1 K (thick) and 5 K (thin cirrus)
    r = estimate(bt13, bt15)[0]
    assert r[0] > 0 and r[1] == 0 and np.isnan(r[2])


def test_colorize_is_transparent_when_dry_and_uses_the_legend():
    img = np.array(colorize(np.array([[0.0, 0.1, 0.5, 30.0, np.nan]])))
    assert list(img[0, :, 3]) == [0, 0, 205, 205, 0]
    assert tuple(img[0, 2, :3]) == LEGEND[0][1] and tuple(img[0, 3, :3]) == LEGEND[5][1]

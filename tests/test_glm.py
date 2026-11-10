from datetime import datetime, timezone

import h5py
import numpy as np

from nowcast.glm import Flashes, in_box, read_flashes, summary, to_geojson

LAT, LON = 32.3, -64.78


def write_glm(path, lats, lons, offsets_s, quality, base="2026-11-10 02:12:40.000"):
    """A minimal GLM LCFA file with the same packing as the real thing (unsigned int16 + scale)."""
    n = len(lats)
    with h5py.File(path, "w") as f:
        f["flash_lat"] = np.asarray(lats, dtype=np.float32)
        f["flash_lon"] = np.asarray(lons, dtype=np.float32)
        scale, offset = np.float32(0.00038148), np.float32(-5.0)
        packed = np.round((np.asarray(offsets_s) - offset) / scale).astype(np.uint16).view(np.int16)
        t = f.create_dataset("flash_time_offset_of_first_event", data=packed)
        t.attrs.update(scale_factor=np.array([scale]), add_offset=np.array([offset]), _Unsigned=np.bytes_(b"true"),
                       units=np.bytes_(f"seconds since {base}".encode()))
        for name, value, sc in [("flash_energy", 40000, 1e-15), ("flash_area", 60000, 152601.86)]:
            d = f.create_dataset(name, data=np.full(n, value, dtype=np.uint16).view(np.int16))
            d.attrs.update(scale_factor=np.array([sc], dtype=np.float32), add_offset=np.array([0.0], dtype=np.float32),
                           _Unsigned=np.bytes_(b"true"), _FillValue=np.array([-1], dtype=np.int16))
        q = f.create_dataset("flash_quality_flag", data=np.asarray(quality, dtype=np.int16))
        q.attrs.update(_Unsigned=np.bytes_(b"true"), _FillValue=np.array([-1], dtype=np.int16))


def test_read_flashes_unpacks_unsigned_fields_and_drops_bad_quality(tmp_path):
    p = tmp_path / "glm.nc"
    write_glm(p, [32.4, 30.0], [-64.7, -60.0], [10.0, 15.0], [0, 1])
    fl = read_flashes(str(p))
    assert len(fl) == 1
    base = datetime(2026, 11, 10, 2, 12, 40, tzinfo=timezone.utc).timestamp()
    assert abs(fl.t[0] - (base + 10.0)) < 0.01
    # 40000 counts (> int16 max, so only right if read as unsigned) × 1e-15 J = 40000 fJ.
    assert abs(fl.energy_fj[0] - 40000) < 0.1
    assert abs(fl.area_km2[0] - 60000 * 152601.86 / 1e6) < 1


def test_empty_file(tmp_path):
    p = tmp_path / "glm.nc"
    write_glm(p, [], [], [], [])
    assert len(read_flashes(str(p))) == 0


def flashes(lats, lons, ages_min, now):
    n = len(lats)
    return Flashes(np.array([now.timestamp() - a * 60 for a in ages_min], dtype=float), np.array(lats, float),
                   np.array(lons, float), np.ones(n), np.ones(n))


def test_summary_counts_and_nearest():
    now = datetime(2026, 11, 10, 3, tzinfo=timezone.utc)
    # 10 nm north (recent), 40 nm east (old), 300 nm away (recent).
    fl = flashes([LAT + 10 / 60, LAT, LAT + 5], [LON, LON + 40 / 60 / np.cos(np.radians(LAT)), LON], [5, 50, 2], now)
    s = summary(fl, LAT, LON, now)
    assert (s["count_25nm"], s["count_50nm"], s["recent_25nm"], s["recent_50nm"]) == (1, 2, 1, 1)
    assert s["nearest_nm"] == 10 and s["nearest_dir"] == "N"


def test_box_and_geojson_cap():
    now = datetime(2026, 11, 10, 3, tzinfo=timezone.utc)
    fl = flashes([LAT, LAT + 6, LAT], [LON, LON, LON - 1], [1, 1, 2], now)
    boxed = in_box(fl, LAT, LON, 5)
    assert len(boxed) == 2
    gj = to_geojson(boxed, max_features=1)
    assert len(gj["features"]) == 1 and gj["features"][0]["properties"]["t"] == int(now.timestamp() - 60)

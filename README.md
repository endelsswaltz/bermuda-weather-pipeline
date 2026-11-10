# Bermuda rain nowcast pipeline

Every 15 minutes a GitHub Actions job reads the latest GOES-19 infrared scans, estimates whether
shower clouds will reach Bermuda in the next 3 hours, and publishes a small JSON file (plus an
infrared image and motion vectors) to GitHub Pages. The Bermuda weather app reads that JSON.

Nothing here needs an account or API key. Raw satellite files are never stored in the repo.

## Output (GitHub Pages)

| File | Size | What |
| --- | --- | --- |
| `nowcast.json` | ~1 KB | Nowcast text, status, ETA, direction, speed (kt), cloud cover within 25/50 nm, coldest cloud top within 50 nm, storm flag, confidence, scan times |
| `ir.png` | ~60 KB | Band 13 infrared, ±5° around Bermuda, NASA GIBS colour table, rows spaced for Web Mercator; `nowcast.json.image.bounds` gives the corners |
| `motion.json` | ~12 KB | Cloud motion every 0.5° as `[lat, lon, east kt, north kt]` |
| `lightning.geojson` | ~0–500 KB | GOES-19 GLM flashes in the ±5° box from the last 60 min, newest first, capped at 5,000; properties `t` (epoch s), `e` (energy, fJ), `a` (area, km²) |
| `wind.json` | ~12 KB | Open-Meteo 10 m wind on a 1° grid (11×11), 3 h back to 6 h ahead, as `u`/`v` in kt indexed `[time][row][col]`, rows north to south |

`nowcast.json` also carries a `lightning` summary: flashes within 25/50 nm over the last 60 min and
the last 30 min, and the nearest flash. Any flash within 50 nm in the last 30 min sets `storms_nearby`.

`status` is one of `rain_now`, `rain_expected`, `no_rain`, `unknown`. Times are UTC ISO 8601.
The app treats the nowcast as stale when the newest scan is more than 45 minutes old.

## How it works

1. **Data**: GOES-19 ABI L2 Cloud and Moisture Imagery, band 13 (10.3 µm clean longwave IR),
   from NOAA's open-data bucket on AWS (`noaa-goes19`). The CONUS sector (5-minute scans) covers
   about 98% of the ±5° box; the full-disk product (10-minute scans) is the fallback. Files are read
   with HTTP range requests, so each scan costs about 1.6 MB of the 3.8 MB file.
2. **Grid**: brightness temperature resampled (nearest neighbour) to a 0.04° lat/lon grid,
   using the ABI fixed-grid projection from the GOES-R Product User Guide, vol. 5, §4.2.8.
3. **Motion**: OpenCV Farneback optical flow between three scans 10 minutes apart. Optical flow
   only sees motion where there is cloud, so the field is spread over clear gaps by normalised
   convolution, falling back to the box-wide mean where cloud is sparse.
4. **Rain proxy**: cloud tops colder than −40 °C (configurable) count as rain-bearing; specks
   under 4 pixels are dropped.
5. **Extrapolation**: the rain mask is advected forward in 10-minute steps for 3 hours. The first
   step where it comes within 8 nm of Bermuda is the ETA. Distance and direction are traced back
   along the motion.
6. **Cloud cover and storms**: cloud = colder than the scene's warm (95th percentile) reference
   minus 6 K. The storm flag is set when the coldest top within 50 nm is below −52 °C.
7. **Confidence**: high/medium/low from the lead time and how much the two motion estimates
   (scan 1→2 and 2→3) disagree at Bermuda.

## Lightning and wind

- **Lightning**: GOES-19 GLM L2 LCFA files (20 s each, ~320 KB) from the same NOAA bucket. Each run
  downloads the last hour (~180 files, in parallel), keeps good-quality flashes (`flash_quality_flag`
  0) in the box, and unpacks GLM's unsigned 16-bit fields. GLM sees in-cloud and cloud-to-ground
  lightning; detection efficiency over the western Atlantic is roughly 70–90% and positions are good
  to about 5–10 km.
- **Wind**: Open-Meteo has no gridded download, so the job asks for 121 points in one request. It
  refreshes about hourly (Open-Meteo's free tier allows 10,000 location-calls a day) and otherwise
  re-publishes the previous `wind.json`, because each Pages deploy replaces the whole site.
- If either source fails, the run still publishes the rain nowcast without it.

## Licences

- GOES-19 ABI and GLM: NOAA open data (NODD), no restrictions; credit NOAA.
- Open-Meteo: CC BY 4.0, free for non-commercial use; credit Open-Meteo.
- NASA GIBS colour table: NASA open data.

## Limits

- Cold cloud tops are a proxy. Thick cirrus can be cold without rain; shallow warm showers
  (common in Bermuda's trade-wind regime) are often warmer than −40 °C and are missed.
- Steady-motion extrapolation: cells that grow, decay or form on the spot aren't forecast.
- ETA resolution is the 10-minute step, and the synthetic test arrives a step late on average.
- GitHub can start scheduled runs 5–30 minutes late, or skip them when busy. Latency from scan
  to published JSON is otherwise about 15–20 minutes (NOAA upload ~10 min, then the run).
- See `nowcast/backtest.py` and the backtest section below for how it has scored.

## Run locally

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m nowcast.run --out site            # --rain-top-c -40 --storm-top-c -52
```

## Backtest

Truth is the hourly METARs from L.F. Wade International Airport (TXKF) via Iowa State's IEM
archive, saved as `data/txkf.csv`
(`https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=TXKF&data=wxcodes&data=metar&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4&year1=…`).
RA, DZ, SH or TS at the station counts as rain; VCSH (showers in the vicinity) doesn't.

```bash
.venv/bin/python -m nowcast.backtest run --days 2026-10-21 2026-10-22   # hourly, resumable
.venv/bin/python -m nowcast.backtest score
```

## Scheduling

GitHub's own cron (`*/15` in `.github/workflows/nowcast.yml`) turned out to be unreliable for this
repo: one scheduled run in about nine hours. The real trigger is an external scheduler
(cron-job.org) that calls GitHub's `workflow_dispatch` API every 15 minutes with a fine-grained
token limited to this repository's Actions. The GitHub cron stays as a backup. Each run re-enables
the workflow through the API, so it isn't switched off after 60 days without repository activity,
which would also block the external trigger.

## Attribution

GOES-19 imagery: NOAA, via the NOAA Open Data Dissemination (NODD) program on AWS. IR colour
table: NASA GIBS. Backtest observations: Iowa Environmental Mesonet, Iowa State University.

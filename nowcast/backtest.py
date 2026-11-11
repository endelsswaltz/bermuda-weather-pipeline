"""Backtest the nowcast against what the airport (TXKF) actually reported.

    python -m nowcast.backtest run --days 2026-10-21 2026-10-22 ...   # hourly nowcasts (slow, resumable)
    python -m nowcast.backtest score                                   # compare with METARs

Truth: L.F. Wade International Airport METARs from Iowa State's IEM archive (data/txkf.csv,
columns station,valid,wxcodes,metar). A report counts as rain when its present-weather group
has RA, DZ, SH or TS at the station; VCSH ("showers in the vicinity") does not count.
METARs are hourly, so timing can only be checked to about ±30 minutes.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .goes import Grid
from .model import Config, run_nowcast
from .run import HALF_WIDTH, LAT, LON, STEP, load_band, load_frames

DATA = Path("data")
RESULTS = DATA / "backtest.jsonl"
RAIN = re.compile(r"(?<![A-Z])[-+]?(?!VC)[A-Z]*(RA|DZ|SH|TS)")


def is_rain(wx: str) -> bool:
    return any(RAIN.fullmatch(tok) for tok in wx.split()) if wx and wx != "M" else False


def read_metars(path=DATA / "txkf.csv") -> list[tuple[datetime, bool, str]]:
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            t = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            out.append((t, is_rain(row["wxcodes"]), row["wxcodes"]))
    return sorted(out)


CACHE = DATA / "cache"
SPLIT_BAND = 15  # 12.3 um "dirty" longwave window; band 13 minus band 15 separates thin cirrus


def load(at: datetime):
    """Scans for one issue time, from the local cache when present: (times, bt13 x3, bt15 latest)."""
    grid = Grid.around(LAT, LON, HALF_WIDTH, STEP)
    times, bts, product = load_frames(grid, at, CACHE)
    return grid, times, bts, load_band(grid, times[-1], SPLIT_BAND, product, CACHE)


def one(at: datetime, cfg: Config | None = None) -> dict:
    try:
        grid, times, bts, bt15 = load(at)
        nc, _ = run_nowcast(bts, times, grid, LAT, LON, cfg or Config(), bt15_latest=bt15)
        return {"at": at.isoformat(), **{k: v for k, v in nc.__dict__.items() if k != "config"}}
    except Exception as e:  # missing scans etc.: record and move on
        return {"at": at.isoformat(), "error": str(e)}


def issue_times(days: list[str]) -> list[datetime]:
    out = []
    for d in days:
        start = datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
        # Issue at :50 so the 3-hour window lines up with the :55 METARs.
        out += [start + timedelta(hours=h, minutes=50) for h in range(24)]
    return out


def run(days: list[str], workers: int, cfg: Config | None = None, results: Path = RESULTS):
    """Run (or re-run, from the scan cache) every hourly nowcast for `days`; overwrites `results`."""
    todo = issue_times(days)
    print(f"{len(todo)} nowcasts to run")
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(one, t, cfg) for t in todo]), 1):
            rows.append(fut.result())
            if i % 20 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}]", flush=True)
    results.write_text("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["at"])))


def metrics(rows: list[dict], metars, horizon_h: float = 3.0) -> dict:
    """Contingency counts and skill for rain at TXKF within `horizon_h` of each nowcast."""
    hits = misses = false_alarms = quiet = 0
    eta_errors, by_conf, by_day = [], {}, {}
    for r in rows:
        if "error" in r:
            continue
        issued, at = datetime.fromisoformat(r["issued"]), datetime.fromisoformat(r["at"])
        window = [(t, rain) for t, rain, _ in metars if at < t <= issued + timedelta(hours=horizon_h)]
        observed = any(rain for _, rain in window)
        forecast = r["status"] in ("rain_now", "rain_expected")
        day = by_day.setdefault(r["at"][:10], {"hit": 0, "miss": 0, "false": 0, "quiet": 0})
        if forecast and observed:
            hits += 1; day["hit"] += 1
            first = next(t for t, rain in window if rain)
            if r["eta"]:
                eta_errors.append((first - datetime.fromisoformat(r["eta"])).total_seconds() / 60)
        elif observed:
            misses += 1; day["miss"] += 1
        elif forecast:
            false_alarms += 1; day["false"] += 1
        else:
            quiet += 1; day["quiet"] += 1
        c = by_conf.setdefault(r["confidence"], [0, 0])
        c[0] += 1
        c[1] += forecast == observed
    n = hits + misses + false_alarms + quiet
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    eta_errors.sort()
    return {
        "n": n, "hits": hits, "misses": misses, "false_alarms": false_alarms, "quiet": quiet,
        "pod": div(hits, hits + misses), "far": div(false_alarms, hits + false_alarms),
        "csi": div(hits, hits + misses + false_alarms), "accuracy": div(hits + quiet, n),
        "eta_median": eta_errors[len(eta_errors) // 2] if eta_errors else None,
        "eta_within_60": div(sum(abs(e) <= 60 for e in eta_errors), len(eta_errors)),
        "by_conf": by_conf, "by_day": by_day,
    }


def score(horizon_h: float = 3.0, results: Path = RESULTS):
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    m = metrics(rows, read_metars(), horizon_h)
    print(f"{m['n']} nowcasts ({len(rows) - m['n']} skipped for missing data), rain within {horizon_h:g} h at TXKF")
    print(f"  hits {m['hits']}  misses {m['misses']}  false alarms {m['false_alarms']}  correct no-rain {m['quiet']}")
    print(f"  POD {m['pod']:.2f}  FAR {m['far']:.2f}  CSI {m['csi']:.2f}  accuracy {m['accuracy']:.2f}")
    if m["eta_median"] is not None:
        print(f"  ETA (observed minus forecast) median {m['eta_median']:+.0f} min, within ±60 min {m['eta_within_60']:.0%}")
    for conf, (total, right) in sorted(m["by_conf"].items()):
        print(f"  confidence {conf}: {right}/{total} right")
    print("per day (hit / miss / false alarm / correct no-rain):")
    for d, c in sorted(m["by_day"].items()):
        print(f"  {d}: {c['hit']} / {c['miss']} / {c['false']} / {c['quiet']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--days", nargs="+", required=True)
    r.add_argument("--workers", type=int, default=6)
    sub.add_parser("score")
    args = ap.parse_args(argv)
    run(args.days, args.workers) if args.cmd == "run" else score()


if __name__ == "__main__":
    main()

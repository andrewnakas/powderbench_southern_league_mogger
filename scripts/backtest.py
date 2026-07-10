"""Dev-only: emit a hindcast-format submission CSV (extra round_date column)
for `powderbench hindcast`, using archived model runs — like-for-like with the
benchmark's own hindcast baselines.

    python scripts/backtest.py --league era5 --begin 2025-06-15 --end 2025-09-15 --out /tmp/bt.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from mogger.calibration import load_calibration  # noqa: E402
from mogger.config import load_config  # noqa: E402
from mogger.forecast import build_submission  # noqa: E402
from mogger.rounds import Round  # noqa: E402
from mogger.stations import load_stations  # noqa: E402
from mogger import openmeteo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=["era5", "resorts", "stations"])
    ap.add_argument("--begin", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    calib = load_calibration(args.league)
    stations = load_stations(args.league)
    sids = tuple(s.station_id for s in stations)
    begin, end = date.fromisoformat(args.begin), date.fromisoformat(args.end)

    # one bulk fetch per model over the whole window (cached), then slice per round
    members_all = {}
    for model in cfg.models:
        try:
            members_all[model] = openmeteo.hindcast_daily(stations, begin, end + timedelta(days=2), model=model)
        except Exception as exc:
            print(f"{model}: unavailable ({exc})")

    frames = []
    day = begin
    while day <= end:
        rnd = Round(
            round_id=day.isoformat(),
            league=args.league,
            cutoff_utc=datetime.now(timezone.utc),
            target_days=(day, day + timedelta(days=1), day + timedelta(days=2)),
            stations=sids,
        )
        members = {
            m: df[(df["date"] >= day) & (df["date"] <= day + timedelta(days=2))]
            for m, df in members_all.items()
        }
        sub = build_submission(rnd, cfg, calib, members=members)
        sub.insert(0, "round_date", day.isoformat())
        frames.append(sub)
        day += timedelta(days=1)

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows, {(end - begin).days + 1} rounds)")


if __name__ == "__main__":
    main()

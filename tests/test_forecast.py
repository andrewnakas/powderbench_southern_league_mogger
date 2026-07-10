import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mogger import HORIZONS
from mogger.config import load_config
from mogger.forecast import build_submission, climatology_submission
from mogger.rounds import Round
from mogger.validate import check
from test_calibration import make_calib

CFG = load_config()


def make_round(league="era5", stations=("portillo:CL:ERA5", "valle-nevado:CL:ERA5")) -> Round:
    return Round(
        round_id="2026-07-12",
        league=league,
        cutoff_utc=datetime(2026, 7, 11, 11, tzinfo=timezone.utc),
        target_days=(date(2026, 7, 12), date(2026, 7, 13), date(2026, 7, 14)),
        stations=tuple(stations),
    )


def member_frame(stations, days, value=2.0) -> pd.DataFrame:
    rows = [
        {"station_id": sid, "date": d, "snowfall_in": value}
        for sid in stations
        for d in days
    ]
    return pd.DataFrame(rows)


def test_build_submission_cumulative_and_valid():
    rnd = make_round()
    calib = make_calib(quantile_map={str(h): make_calib().quantile_map["24"] for h in HORIZONS})
    members = {"best_match": member_frame(rnd.stations, rnd.target_days, 2.0)}
    df = build_submission(rnd, CFG, calib, members=members)
    assert len(df) == len(rnd.stations) * 3
    for _, grp in df.groupby("station_id"):
        vals = grp.sort_values("horizon_h")["snowfall_in"].tolist()
        assert vals == sorted(vals)  # cumulative
        assert all(v >= 0 for v in vals)
    assert check(df, rnd) == []


def test_missing_station_falls_back_to_climatology():
    rnd = make_round()
    calib = make_calib(quantile_map={str(h): make_calib().quantile_map["24"] for h in HORIZONS})
    # only one of the two stations has model data
    members = {"best_match": member_frame(rnd.stations[:1], rnd.target_days, 1.0)}
    df = build_submission(rnd, CFG, calib, members=members)
    assert set(df["station_id"]) == set(rnd.stations)  # coverage kept via climo
    assert check(df, rnd) == []


def test_climatology_submission_full_coverage():
    rnd = make_round()
    df = climatology_submission(rnd)
    assert len(df) == len(rnd.stations) * 3
    assert check(df, rnd) == []


def test_validate_rejects_bad_frames():
    rnd = make_round()
    bad = pd.DataFrame(
        [
            {"station_id": "portillo:CL:ERA5", "horizon_h": 24, "snowfall_in": 5.0},
            {"station_id": "portillo:CL:ERA5", "horizon_h": 48, "snowfall_in": 3.0},  # not cumulative
        ]
    )
    problems = check(bad, rnd)
    assert any("cumulative" in p for p in problems)
    assert any("coverage" in p for p in problems)

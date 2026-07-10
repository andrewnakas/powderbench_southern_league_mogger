"""Build a full submission DataFrame for one round: calibrated ensemble point
forecast + empirical quantiles + prob_6in, with climatology fallback so
coverage never drops."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from . import HORIZONS, MAX_SNOWFALL_IN
from .calibration import Calibration
from .climatology import climo_rows
from .config import Config
from .rounds import Round
from .stations import load_stations
from . import openmeteo

log = logging.getLogger(__name__)


def fetch_members(rnd: Round, models: tuple[str, ...], mode: str = "live") -> dict[str, pd.DataFrame]:
    """Raw daily snowfall per ensemble member over the round's 3 target days.
    A member that errors out is dropped — the ensemble never requires any
    specific model."""
    stations = load_stations(rnd.league, list(rnd.stations))
    begin, end = rnd.target_days[0], rnd.target_days[-1]
    fetch = openmeteo.forecast_daily if mode == "live" else openmeteo.hindcast_daily
    members = {}
    for model in models:
        try:
            members[model] = fetch(stations, begin, end, model=model)
        except Exception as exc:
            log.warning("[%s] member %s failed, dropping: %s", rnd.league, model, exc)
    return members


def build_submission(
    rnd: Round,
    cfg: Config,
    calib: Calibration,
    members: dict[str, pd.DataFrame] | None = None,
    mode: str = "live",
) -> pd.DataFrame:
    if members is None:
        members = fetch_members(rnd, cfg.models, mode=mode)

    # member values indexed by (station_id, date)
    lookup: dict[str, dict] = {}
    for model, df in members.items():
        for r in df.itertuples():
            if r.snowfall_in is not None and not pd.isna(r.snowfall_in):
                lookup.setdefault(r.station_id, {}).setdefault(r.date, {})[model] = float(r.snowfall_in)

    rows = []
    for sid in rnd.stations:
        daily = []
        for j, day in enumerate(rnd.target_days):
            vals = lookup.get(sid, {}).get(day)
            if not vals:
                daily = None  # a hole in any target day -> whole station falls back
                break
            daily.append(calib.daily(sid, j, vals))
        if daily is None:
            fb = climo_rows(rnd.league, sid, rnd.target_days[0].timetuple().tm_yday)
            if fb is None:
                log.warning("[%s] %s: no model data and no climatology — skipping", rnd.league, sid)
                continue
            log.info("[%s] %s: climatology fallback", rnd.league, sid)
            rows.extend(fb)
            continue

        cum = 0.0
        for h, v in zip(HORIZONS, daily):
            cum = min(cum + v, MAX_SNOWFALL_IN)
            row = {"station_id": sid, "horizon_h": h, "snowfall_in": round(cum, 3)}
            row.update({k: round(v, 3) for k, v in calib.quantiles(h, cum).items()})
            if h == 24:
                row["prob_6in"] = round(calib.prob6(cum), 4)
            rows.append(row)

    cols = ["station_id", "horizon_h", "snowfall_in", "p10", "p25", "p50", "p75", "p90", "prob_6in"]
    return pd.DataFrame(rows, columns=cols)


def climatology_submission(rnd: Round) -> pd.DataFrame:
    """Total-outage fallback: the full climatology forecast (scores 0, never a miss)."""
    rows = []
    for sid in rnd.stations:
        fb = climo_rows(rnd.league, sid, rnd.target_days[0].timetuple().tm_yday)
        if fb:
            rows.extend(fb)
    cols = ["station_id", "horizon_h", "snowfall_in", "p10", "p25", "p50", "p75", "p90", "prob_6in"]
    return pd.DataFrame(rows, columns=cols)

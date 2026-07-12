"""Climatology: the benchmark's own per-station day-of-year tables, fetched
live from its main branch (so new stations are covered the day they appear),
with the committed snapshot as offline fallback.

Used as the fallback forecast when model data is missing — climatology scores
exactly 0 by construction, never a hole in coverage.
Regenerate the snapshot with scripts/refresh_static.py.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from io import StringIO

import pandas as pd
import requests

from . import HORIZONS, QUANTILE_COLS
from .config import CONFIG_PATH, REPO_ROOT, load_config

log = logging.getLogger(__name__)

TIMEOUT = 30


@lru_cache
def load_climatology(league: str) -> pd.DataFrame:
    try:
        raw_base = load_config(CONFIG_PATH).raw_base
        resp = requests.get(f"{raw_base}/data/climatology/{league}.csv", timeout=TIMEOUT)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text))
    except Exception as exc:
        log.warning("[%s] live climatology unavailable (%s), using snapshot", league, exc)
        return pd.read_csv(REPO_ROOT / "data" / "climatology" / f"{league}.csv")


def climo_rows(league: str, station_id: str, doy: int) -> list[dict] | None:
    """Full climatology submission rows (all horizons) for one station-day,
    or None if the table has no sample there."""
    climo = load_climatology(league)
    day = climo[(climo["station_id"] == station_id) & (climo["doy"] == min(doy, 365))]
    if not len(day):
        return None
    r = day.iloc[0]
    rows = []
    for h in HORIZONS:
        if pd.isna(r.get(f"h{h}_p50")):
            return None
        row = {
            "station_id": station_id,
            "horizon_h": h,
            "snowfall_in": float(r[f"h{h}_p50"]),
        }
        for col in QUANTILE_COLS.values():
            row[col] = float(r[f"h{h}_{col}"])
        if h == 24:
            row["prob_6in"] = float(r["h24_p6freq"])
        rows.append(row)
    return rows

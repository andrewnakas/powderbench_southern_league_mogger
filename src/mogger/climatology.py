"""Climatology snapshot (vendored copies of powderbench data/climatology/*.csv).

Used two ways: as the fallback forecast when model data is missing (climatology
scores exactly 0 by construction, never a hole in coverage), and nowhere else —
the calibrated ensemble does not blend toward it unless fitting said so.
Regenerate with scripts/refresh_static.py.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import HORIZONS, QUANTILE_COLS
from .config import REPO_ROOT


@lru_cache
def load_climatology(league: str) -> pd.DataFrame:
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

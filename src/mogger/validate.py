"""Pre-push validation mirroring powderbench's submission rules, plus a few
stricter bot-side checks (cumulative monotonicity, full coverage). A frame
that fails here is never pushed — the benchmark's CI must never see it."""

from __future__ import annotations

import pandas as pd

from . import HORIZONS, MAX_SNOWFALL_IN, QUANTILE_COLS
from .rounds import Round


def check(df: pd.DataFrame, rnd: Round) -> list[str]:
    """Return a list of problems; empty means good to push."""
    errors = []
    required = {"station_id", "horizon_h", "snowfall_in"}
    if missing := required - set(df.columns):
        return [f"missing required columns: {sorted(missing)}"]

    known = set(rnd.stations)
    if bad := sorted(set(df["station_id"]) - known):
        errors.append(f"unknown station_id(s): {bad[:5]}")
    if bad_h := sorted(set(df["horizon_h"]) - set(HORIZONS)):
        errors.append(f"bad horizon_h: {bad_h}")
    if df.duplicated(["station_id", "horizon_h"]).any():
        errors.append("duplicate (station_id, horizon_h) rows")

    vals = pd.to_numeric(df["snowfall_in"], errors="coerce")
    if vals.isna().any():
        errors.append("snowfall_in contains non-numeric values")
    elif ((vals < 0) | (vals > MAX_SNOWFALL_IN)).any():
        errors.append(f"snowfall_in outside [0, {MAX_SNOWFALL_IN}]")

    # cumulative horizons must be non-decreasing per station
    wide = df.pivot_table(index="station_id", columns="horizon_h", values="snowfall_in")
    for h0, h1 in zip(HORIZONS, HORIZONS[1:]):
        if h0 in wide.columns and h1 in wide.columns:
            if (wide[h1] - wide[h0] < -1e-9).any():
                errors.append(f"snowfall_in not cumulative between h{h0} and h{h1}")

    qcols = list(QUANTILE_COLS.values())
    present = [c for c in qcols if c in df.columns]
    if present and len(present) != len(qcols):
        errors.append(f"quantile columns must be all-or-none, got {present}")
    elif present:
        q = df[qcols].apply(pd.to_numeric, errors="coerce")
        filled = q.notna().all(axis=1)
        if q.notna().any(axis=1).sum() != filled.sum():
            errors.append("rows with partial quantiles")
        qq = q[filled]
        if len(qq) and not (qq.diff(axis=1).iloc[:, 1:] >= -1e-9).all().all():
            errors.append("quantiles must be non-decreasing")
        if len(qq) and ((qq < 0) | (qq > MAX_SNOWFALL_IN)).any().any():
            errors.append(f"quantiles outside [0, {MAX_SNOWFALL_IN}]")

    if "prob_6in" in df.columns:
        p = pd.to_numeric(df["prob_6in"], errors="coerce").dropna()
        if ((p < 0) | (p > 1)).any():
            errors.append("prob_6in outside [0, 1]")

    covered = df.drop_duplicates(["station_id", "horizon_h"])
    coverage = len(covered[covered["station_id"].isin(known)]) / (len(known) * len(HORIZONS))
    if coverage < 0.999:
        errors.append(f"coverage {coverage:.0%} < 100% — fallback should have filled the gaps")
    return errors

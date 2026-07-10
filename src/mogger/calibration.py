"""Load and apply fitted calibration artifacts (data/calibration/<league>.json).

The model, per station i, target day D+j, ensemble members m:

    c_j = lead_factor[j] * station_mult[i] * median_m( a_m * max(f_mij - t_m, 0) )

- (a_m, t_m) is a pooled per-model MAE-optimal transform (weighted-median fit).
- station_mult is a per-station multiplier shrunk hard toward 1.
- lead_factor discounts the (unfittable-offline) skill decay at longer leads.

Quantiles and prob_6in come from binned empirical conditional distributions of
truth given the calibrated cumulative forecast, fitted on the same data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from statistics import median

from . import MAX_SNOWFALL_IN, QUANTILE_COLS
from .config import REPO_ROOT

CALIB_DIR = REPO_ROOT / "data" / "calibration"


@dataclass(frozen=True)
class Calibration:
    league: str
    models: dict          # model -> {"a": float, "t": float}
    station_mult: dict    # station_id -> float
    lead_factors: tuple   # per target-day index 0..2
    quantile_map: dict    # str(horizon) -> [ {lo, hi, m, q: {p10..p90}, prob6} ]

    def transform(self, model: str, value: float) -> float:
        p = self.models.get(model) or self.models.get("best_match") or {"a": 1.0, "t": 0.0}
        return p["a"] * max(value - p["t"], 0.0)

    def daily(self, station_id: str, lead_j: int, member_values: dict[str, float]) -> float:
        """Calibrated point value for one station-day from raw member forecasts."""
        cal = [self.transform(m, v) for m, v in member_values.items() if v is not None and not math.isnan(v)]
        if not cal:
            raise ValueError("no ensemble members")
        lead = self.lead_factors[min(lead_j, len(self.lead_factors) - 1)]
        mult = self.station_mult.get(station_id, 1.0)
        return max(lead * mult * median(cal), 0.0)

    def _bins(self, horizon: int) -> list[dict]:
        return self.quantile_map[str(horizon)]

    def quantiles(self, horizon: int, yhat: float) -> dict[str, float]:
        """Empirical conditional quantiles around the point forecast: absolute
        in the dry bin, ratio-interpolated across wet-bin centers above it."""
        bins = self._bins(horizon)
        cols = list(QUANTILE_COLS.values())
        if yhat <= bins[0]["hi"]:
            q = {c: float(bins[0]["q"][c]) for c in cols}
        else:
            centers = [b["m"] for b in bins[1:]]
            q = {}
            for c in cols:
                ratios = [b["q"][c] / max(b["m"], 0.05) for b in bins[1:]]
                q[c] = yhat * _interp(yhat, centers, ratios)
        # anchor the median to the point forecast, keep order, stay in bounds
        q["p50"] = yhat
        prev = 0.0
        for c in cols:
            q[c] = min(max(q[c], prev, 0.0), MAX_SNOWFALL_IN)
            prev = q[c]
        return q

    def prob6(self, yhat24: float) -> float:
        bins = self._bins(24)
        centers = [b["m"] for b in bins]
        probs = [b["prob6"] for b in bins]
        return min(max(_interp(yhat24, centers, probs), 0.0), 1.0)


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear interpolation, clamped at the ends."""
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:]):
        if x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


@lru_cache
def load_calibration(league: str) -> Calibration:
    raw = json.loads((CALIB_DIR / f"{league}.json").read_text())
    return Calibration(
        league=raw["league"],
        models=raw["models"],
        station_mult=raw["station_mult"],
        lead_factors=tuple(raw["lead_factors"]),
        quantile_map=raw["quantile_map"],
    )

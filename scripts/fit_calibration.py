"""Dev-only: fit a league's calibration artifact from archived model runs vs
that league's truth, and write data/calibration/<league>.json.

    python scripts/fit_calibration.py --league era5
    python scripts/fit_calibration.py --league stations   # needs the powderbench checkout importable (SNOTEL truth)
    python scripts/fit_calibration.py --league resorts    # prior: softened era5-style fit at the resort coords

Training windows are hard-coded per league below; holdout seasons are left
untouched for the hindcast verification. The fit is deterministic: pooled
per-model (a, t) by exact weighted-median MAE minimization, per-station
multipliers shrunk toward 1, binned empirical quantile/prob6 maps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from mogger import HORIZONS, POWDER_ALERT_INCHES, QUANTILE_COLS  # noqa: E402
from mogger.config import load_config  # noqa: E402
from mogger.stations import load_stations  # noqa: E402
from mogger import openmeteo  # noqa: E402

# train on these; hold out 2025 austral winter / 2025-26 northern winter for verification
TRAIN_WINDOWS = {
    "era5": [
        ("2021-05-01", "2021-09-30"),
        ("2022-05-01", "2022-09-30"),
        ("2023-05-01", "2023-09-30"),
        ("2024-05-01", "2024-09-30"),
        ("2026-04-01", "2026-05-31"),
    ],
    "stations": [
        ("2021-11-01", "2022-04-30"),
        ("2022-11-01", "2023-04-30"),
        ("2023-11-01", "2024-04-30"),
        ("2024-11-01", "2025-04-30"),
    ],
}
LEAD_FACTORS = {"era5": [1.0, 0.9, 0.8], "resorts": [1.0, 0.9, 0.8], "stations": [1.0, 0.95, 0.9]}
BIN_EDGES = [0.0, 0.05, 0.5, 1.5, 3.0, 6.0, float("inf")]
THRESH_GRID = [round(0.02 * i, 2) for i in range(0, 21)]  # 0.00 .. 0.40 in
STATION_PRIOR_WET_DAYS = 60
MIN_WINTERS = 0.9  # a model with less usable history (in winters) inherits best_match's fit


def weighted_median(values: list[float], weights: list[float]) -> float:
    order = sorted(range(len(values)), key=lambda i: values[i])
    total = sum(weights)
    acc = 0.0
    for i in order:
        acc += weights[i]
        if acc >= total / 2:
            return values[i]
    return values[order[-1]] if order else 1.0


def fit_model_transform(f: pd.Series, y: pd.Series) -> dict:
    """Exact MAE-optimal (a, t) for g(f) = a * max(f - t, 0)."""
    best = None
    for t in THRESH_GRID:
        x = (f - t).clip(lower=0.0)
        wet = x > 0
        if not wet.any():
            continue
        a = weighted_median((y[wet] / x[wet]).tolist(), x[wet].tolist())
        a = min(max(a, 0.1), 2.0)
        mae = float((y - a * x).abs().mean())
        if best is None or mae < best["mae"]:
            best = {"a": round(a, 4), "t": t, "mae": mae}
    return best or {"a": 1.0, "t": 0.0, "mae": float((y - f).abs().mean())}


def truth_frame(league: str, stations, begin: date, end: date) -> pd.DataFrame:
    """columns: station_id, date, truth (NaN dropped)."""
    if league in ("era5", "resorts"):
        df = openmeteo.era5_daily(stations, begin, end)
        return df.rename(columns={"snowfall_in": "truth"}).dropna(subset=["truth"])
    # stations league: QC'd SNOTEL via the powderbench checkout (dev-time only)
    from powderbench.leagues import get_league
    from powderbench.truth_sources import daily_truth

    daily = daily_truth(get_league("stations"), [s.station_id for s in stations], begin, end)
    daily = daily[daily["valid"]]
    return daily.rename(columns={"snow24": "truth"})[["station_id", "date", "truth"]]


def collect_training(league: str, models: tuple[str, ...]) -> pd.DataFrame:
    """Long frame: station_id, date, truth, plus one column per model forecast."""
    stations = load_stations("resorts" if league == "resorts" else league)
    frames = []
    for begin_s, end_s in TRAIN_WINDOWS["era5" if league == "resorts" else league]:
        begin, end = date.fromisoformat(begin_s), date.fromisoformat(end_s)
        base = truth_frame(league, stations, begin, end)
        for model in models:
            try:
                fc = openmeteo.hindcast_daily(stations, begin, end, model=model)
            except Exception as exc:
                print(f"  {model} {begin_s}: unavailable ({exc})")
                continue
            fc = fc.rename(columns={"snowfall_in": model})
            base = base.merge(fc[["station_id", "date", model]], on=["station_id", "date"], how="left")
        frames.append(base)
        print(f"  window {begin_s}..{end_s}: {len(base)} station-days")
    return pd.concat(frames, ignore_index=True)


def fit(league: str, models: tuple[str, ...]) -> dict:
    data = collect_training(league, models)
    n_stations = data["station_id"].nunique()
    min_rows = int(MIN_WINTERS * 150 * n_stations)

    model_params = {}
    for model in models:
        if model not in data.columns:
            continue
        sub = data.dropna(subset=[model, "truth"])
        if len(sub) < min_rows:
            print(f"  {model}: only {len(sub)} rows — will inherit best_match")
            continue
        p = fit_model_transform(sub[model], sub["truth"])
        raw_mae = float((sub["truth"] - sub[model]).abs().mean())
        print(f"  {model}: a={p['a']} t={p['t']} mae={p['mae']:.4f} (raw {raw_mae:.4f}) n={len(sub)}")
        model_params[model] = {"a": p["a"], "t": p["t"]}
    if not model_params:
        raise SystemExit("no model has enough training data")
    if "best_match" not in model_params:
        model_params["best_match"] = next(iter(model_params.values()))

    def calibrated(row) -> float | None:
        vals = []
        for m, p in model_params.items():
            v = row.get(m)
            if v is not None and not pd.isna(v):
                vals.append(p["a"] * max(v - p["t"], 0.0))
        return median(vals) if vals else None

    data["cal"] = data.apply(calibrated, axis=1)
    pooled = data.dropna(subset=["cal", "truth"])
    print(f"  pooled calibrated MAE {float((pooled['truth'] - pooled['cal']).abs().mean()):.4f} n={len(pooled)}")

    station_mult = {}
    for sid, grp in pooled.groupby("station_id"):
        wet = grp[grp["cal"] > 0]
        n = len(wet)
        s = weighted_median((wet["truth"] / wet["cal"]).tolist(), wet["cal"].tolist()) if n else 1.0
        shrunk = (n * s + STATION_PRIOR_WET_DAYS) / (n + STATION_PRIOR_WET_DAYS)
        station_mult[sid] = round(min(max(shrunk, 0.5), 1.5), 4)

    # apply station multipliers, then build cumulative pairs for quantiles/prob6
    pooled = pooled.assign(cal_s=pooled.apply(lambda r: r["cal"] * station_mult[r["station_id"]], axis=1))
    qmap = {str(h): [] for h in HORIZONS}
    cum_frames = []
    for sid, grp in pooled.sort_values("date").groupby("station_id"):
        grp = grp.set_index("date")
        for col, out in (("cal_s", "yhat"), ("truth", "y")):
            s = grp[col]
            for h in HORIZONS:
                n = h // 24
                grp[f"{out}{h}"] = s[::-1].rolling(n, min_periods=n).sum()[::-1]
        cum_frames.append(grp.reset_index())
    cum = pd.concat(cum_frames, ignore_index=True)

    for h in HORIZONS:
        sub = cum.dropna(subset=[f"yhat{h}", f"y{h}"])
        bins = []
        for lo, hi in zip(BIN_EDGES, BIN_EDGES[1:]):
            sel = sub[(sub[f"yhat{h}"] > lo) & (sub[f"yhat{h}"] <= hi)] if lo > 0 else sub[sub[f"yhat{h}"] <= hi]
            if len(sel) < 20:
                continue
            entry = {
                "lo": lo,
                "hi": hi if hi != float("inf") else 999.0,
                "m": round(float(sel[f"yhat{h}"].median()), 3),
                "q": {c: round(float(sel[f"y{h}"].quantile(q)), 3) for q, c in QUANTILE_COLS.items()},
            }
            if h == 24:
                entry["prob6"] = round(float((sel[f"y{h}"] >= POWDER_ALERT_INCHES).mean()), 4)
            bins.append(entry)
        # monotonize prob6 across bins (higher forecast can't lower the powder odds)
        if h == 24:
            best_p = 0.0
            for b in bins:
                best_p = max(best_p, b["prob6"])
                b["prob6"] = best_p
        qmap[str(h)] = bins

    return {
        "league": league,
        "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_windows": TRAIN_WINDOWS["era5" if league == "resorts" else league],
        "models": model_params,
        "station_mult": station_mult,
        "lead_factors": LEAD_FACTORS[league],
        "quantile_map": qmap,
    }


def soften_for_resorts(artifact: dict) -> dict:
    """Resorts prior: halve the era5-style shrinkage (stake reports run closer
    to raw high-res NWP than to smoothed ERA5) and widen the outer quantiles."""
    artifact["league"] = "resorts"
    artifact["provenance"] = "prior"
    for p in artifact["models"].values():
        p["a"] = round((1.0 + p["a"]) / 2.0, 4)
        p["t"] = round(p["t"] / 2.0, 2)
    artifact["station_mult"] = {sid: 1.0 for sid in artifact["station_mult"]}
    for bins in artifact["quantile_map"].values():
        for b in bins:
            b["q"]["p10"] = round(b["q"]["p10"] / 1.5, 3)
            b["q"]["p90"] = round(b["q"]["p90"] * 1.5, 3)
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=["era5", "resorts", "stations"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg = load_config()
    print(f"fitting {args.league} …")
    artifact = fit(args.league, cfg.models)
    if args.league == "resorts":
        artifact = soften_for_resorts(artifact)

    out = args.out or HERE / "data" / "calibration" / f"{args.league}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

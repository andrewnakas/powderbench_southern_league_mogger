# southern-league-mogger

A forecasting bot for [PowderBench](https://github.com/andrewnakas/powderbench).
It competes as team **`southern-league-mogger`** in all three leagues — `era5`
and `resorts` (southern hemisphere, Jun–Oct) and `stations` (US SNOTEL,
Oct–May) — by generating a calibrated multi-model snowfall forecast every day
and pushing the submission CSV straight to the benchmark repo before each
league's cutoff.

## Method (honest description, per the benchmark's fair-play rules)

1. **Ensemble**: for each open round, fetch daily snowfall for the 3 target
   days from four free Open-Meteo models: `ecmwf_ifs025`, `gfs_seamless`,
   `icon_seamless`, `best_match` — the exact same request shape the
   benchmark's own NWP baselines use (station-local days, elevation-corrected).
2. **Calibration**: each member value passes through a pooled per-model
   transform `a·max(f − t, 0)` fitted offline to minimize MAE against the
   league's own truth history (ERA5 reanalysis for the southern leagues,
   QC'd SNOTEL for stations); the fit is an exact weighted-median solution,
   no optimizer. The point forecast is the **median** of calibrated members
   (MAE-optimal, matching the Powder Score), times a per-station multiplier
   shrunk hard toward 1, times a lead-day discount. Why this beats raw NWP in
   the era5 league: the truth there is a smoothed ~31 km reanalysis that
   high-res forecast models systematically over-shoot at mountain points, so
   the fitted transform learns exactly how much to shrink.
3. **Quantiles + powder alert**: `p10..p90` and `prob_6in` come from binned
   empirical conditional distributions of truth given the calibrated forecast,
   fitted on the same training data.
4. **Never miss a round**: any station/model gap falls back to the benchmark's
   own climatology (which scores 0 by definition); a full Open-Meteo outage
   submits pure climatology rather than nothing.
5. **PR-first submission**: each submission is pushed to a
   `submit/<team>-<league>-<round>` branch of the benchmark and opened as a PR
   for the benchmark's auto-merge gate, which validates it and squash-merges —
   the same audited path every competitor uses, and the path that keeps the
   team name bound to its owner in `data/teams.yaml`. If the gate hasn't
   merged and the cutoff is under 2 hours away, the bot falls back to a direct
   push to `main` so a gate outage can't cost a round.

Fitted coefficients are committed under `data/calibration/*.json` — the
`resorts` file is tagged `"provenance": "prior"` (softened era5 fit at the
resort coordinates) until enough real resort-report rounds resolve to refit.

## Daily automation

`.github/workflows/submit.yml` runs three crons (UTC):

| cron | purpose |
|---|---|
| `0 12 * * *` | first southern submit, right after the round opens at 11:05 |
| `30 9 * * *` | southern refresh with fresh 00Z runs, 90 min before the 11:00 cutoff |
| `30 21 * * *` | stations-league submit, 2.5 h before its 00:00 cutoff |

Every firing discovers *all* open rounds across the three leagues and
creates/updates whatever is submittable, so each cron also acts as a retry for
the others. Off-season leagues simply have no open rounds and no-op.

### One-time setup

Create a **fine-grained GitHub PAT** scoped to `andrewnakas/powderbench` with
*Contents: Read & Write* **and** *Pull requests: Read & Write*, and add it to
this repo as the Actions secret **`POWDERBENCH_TOKEN`**. That's the whole
setup — the bot PRs each submission to the benchmark's auto-merge gate (with a
direct-push-to-main fallback near the cutoff, which is what makes a submission
count).

## Running by hand

```bash
pip install -e .

# see what would be submitted right now, without pushing
mogger run --dry-run --out-dir /tmp/mogger

# submit one specific round
POWDERBENCH_TOKEN=... mogger run --league era5 --round 2026-07-12
```

## Refitting / development

```bash
pip install pyyaml   # dev-only extras
python scripts/refresh_static.py --powderbench ../powderbench   # stations + climatology snapshots
python scripts/fit_calibration.py --league era5                 # rewrites data/calibration/era5.json
python scripts/backtest.py --league era5 --begin 2025-06-15 --end 2025-09-15 --out /tmp/bt.csv
# then score it with the benchmark's own harness:
(cd ../powderbench && powderbench hindcast 2025-06-15 2025-09-15 --league era5 \
    --submission /tmp/bt.csv --team southern-league-mogger)
```

The `stations` fit needs a local powderbench checkout on `PYTHONPATH` (SNOTEL
truth comes through its QC pipeline); the southern fits are fully standalone.

Tests: `python -m pytest`.

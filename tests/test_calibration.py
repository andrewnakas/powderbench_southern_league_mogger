import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mogger.calibration import Calibration, _interp


def make_calib(**over) -> Calibration:
    base = dict(
        league="test",
        models={"best_match": {"a": 0.5, "t": 0.1}, "gfs_seamless": {"a": 0.8, "t": 0.0}},
        station_mult={"a:XX:ERA5": 1.2},
        lead_factors=(1.0, 0.9, 0.8),
        quantile_map={
            "24": [
                {"lo": 0.0, "hi": 0.05, "m": 0.0, "q": {"p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0.2}, "prob6": 0.001},
                {"lo": 0.05, "hi": 1.5, "m": 0.5, "q": {"p10": 0.1, "p25": 0.25, "p50": 0.5, "p75": 0.9, "p90": 1.5}, "prob6": 0.02},
                {"lo": 1.5, "hi": 999.0, "m": 4.0, "q": {"p10": 1.0, "p25": 2.0, "p50": 4.0, "p75": 6.0, "p90": 9.0}, "prob6": 0.3},
            ],
        },
    )
    base.update(over)
    return Calibration(**base)


def test_transform_threshold_and_slope():
    c = make_calib()
    assert c.transform("best_match", 1.1) == 0.5  # 0.5 * (1.1 - 0.1)
    assert c.transform("best_match", 0.05) == 0.0  # below threshold
    assert c.transform("unknown_model", 1.1) == 0.5  # falls back to best_match


def test_daily_median_lead_and_station_mult():
    c = make_calib()
    # members: best_match 1.1 -> 0.5 ; gfs 1.0 -> 0.8 ; median = 0.65
    v = c.daily("a:XX:ERA5", 1, {"best_match": 1.1, "gfs_seamless": 1.0})
    assert abs(v - 0.65 * 0.9 * 1.2) < 1e-9
    # unknown station -> multiplier 1
    v = c.daily("b:XX:ERA5", 0, {"best_match": 1.1, "gfs_seamless": 1.0})
    assert abs(v - 0.65) < 1e-9


def test_quantiles_anchor_median_and_order():
    c = make_calib()
    q = c.quantiles(24, 2.0)
    assert q["p50"] == 2.0
    vals = [q["p10"], q["p25"], q["p50"], q["p75"], q["p90"]]
    assert vals == sorted(vals)
    assert all(v >= 0 for v in vals)
    # dry forecast uses the dry bin's absolute quantiles
    q0 = c.quantiles(24, 0.0)
    assert q0["p50"] == 0.0 and q0["p90"] >= 0


def test_prob6_monotone_interp_and_bounds():
    c = make_calib()
    assert c.prob6(0.0) == 0.001
    assert c.prob6(10.0) == 0.3
    assert c.prob6(0.5) <= c.prob6(4.0)
    assert 0.0 <= c.prob6(2.0) <= 1.0


def test_interp_clamps():
    assert _interp(-1, [0, 1], [5, 10]) == 5
    assert _interp(2, [0, 1], [5, 10]) == 10
    assert _interp(0.5, [0, 1], [5, 10]) == 7.5


def test_committed_artifacts_load():
    calib_dir = Path(__file__).resolve().parents[1] / "data" / "calibration"
    for f in calib_dir.glob("*.json"):
        raw = json.loads(f.read_text())
        c = Calibration(
            league=raw["league"],
            models=raw["models"],
            station_mult=raw["station_mult"],
            lead_factors=tuple(raw["lead_factors"]),
            quantile_map=raw["quantile_map"],
        )
        assert c.daily(next(iter(c.station_mult)), 0, {"best_match": 1.0}) >= 0.0
        q = c.quantiles(24, 1.0)
        assert q["p50"] == 1.0

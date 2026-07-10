import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mogger.config import load_config
from mogger.rounds import discover_open_rounds

CFG = load_config()


def manifest(round_id, league, cutoff, status="open"):
    return {
        "round_id": round_id,
        "league": league,
        "cutoff_utc": cutoff,
        "target_days": [round_id, round_id, round_id],
        "horizons_h": [24, 48, 72],
        "stations": ["portillo:CL:ERA5"],
        "status": status,
    }


def test_southern_geometry_round_before_cutoff():
    # era5 round D=2026-07-12 locks 11:00 UTC on D-1; probing on D-2 at 12:00 finds it open
    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)

    def fake(cfg, league, day):
        if day.isoformat() == "2026-07-12":
            return manifest("2026-07-12", "era5", "2026-07-11T11:00:00+00:00")
        return None

    with patch("mogger.rounds._fetch_manifest", side_effect=fake):
        rounds = discover_open_rounds(CFG, "era5", now=now)
    assert [r.round_id for r in rounds] == ["2026-07-12"]


def test_stations_geometry_and_cutoff_margin():
    # stations round D locks 00:00 UTC on D; at 21:30 UTC on D-1 it is open,
    # at 23:55 (inside the 10-min margin) it is not
    def fake(cfg, league, day):
        if day.isoformat() == "2026-01-15":
            return manifest("2026-01-15", "stations", "2026-01-15T00:00:00+00:00")
        return None

    early = datetime(2026, 1, 14, 21, 30, tzinfo=timezone.utc)
    late = datetime(2026, 1, 14, 23, 55, tzinfo=timezone.utc)
    with patch("mogger.rounds._fetch_manifest", side_effect=fake):
        assert [r.round_id for r in discover_open_rounds(CFG, "stations", now=early)] == ["2026-01-15"]
        assert discover_open_rounds(CFG, "stations", now=late) == []


def test_offseason_league_yields_nothing():
    with patch("mogger.rounds._fetch_manifest", return_value=None):
        assert discover_open_rounds(CFG, "stations", now=datetime(2026, 7, 10, tzinfo=timezone.utc)) == []


def test_closed_round_skipped():
    def fake(cfg, league, day):
        return manifest(day.isoformat(), "era5", "2099-01-01T00:00:00+00:00", status="resolved")

    with patch("mogger.rounds._fetch_manifest", side_effect=fake):
        assert discover_open_rounds(CFG, "era5", now=datetime(2026, 7, 10, tzinfo=timezone.utc)) == []

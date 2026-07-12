import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mogger.config import load_config
from mogger import submit

CFG = load_config()
DF = pd.DataFrame([{"station_id": "portillo:CL:ERA5", "horizon_h": 24, "snowfall_in": 1.0}])


@pytest.fixture(autouse=True)
def _token_and_speed(monkeypatch):
    monkeypatch.setenv(submit.TOKEN_ENV, "test-token")
    monkeypatch.setattr(submit, "MERGE_WAIT_S", 0)
    monkeypatch.setattr(submit, "MERGE_POLL_S", 0)


def _common(monkeypatch, merged: bool):
    calls = {"main_put": 0}
    monkeypatch.setattr(submit, "_put_file_would_change", lambda *a: True)
    monkeypatch.setattr(submit, "_ensure_branch", lambda *a: None)

    def put_file(cfg, token, path, content, branch, message):
        if branch == "main":
            calls["main_put"] += 1
        return "created"

    monkeypatch.setattr(submit, "_put_file", put_file)
    monkeypatch.setattr(submit, "_open_or_find_pr", lambda *a: 7)
    monkeypatch.setattr(submit, "_wait_for_merge", lambda *a, **k: merged)
    return calls


def test_gate_merge_is_the_happy_path(monkeypatch):
    calls = _common(monkeypatch, merged=True)
    out = submit.push_csv(DF, "era5", "2026-07-14", CFG, cutoff_utc=datetime.now(timezone.utc) + timedelta(hours=20))
    assert out == "merged"
    assert calls["main_put"] == 0  # never touched main directly


def test_fallback_when_cutoff_near(monkeypatch):
    calls = _common(monkeypatch, merged=False)
    out = submit.push_csv(DF, "era5", "2026-07-14", CFG, cutoff_utc=datetime.now(timezone.utc) + timedelta(minutes=30))
    assert out == "fallback-created"
    assert calls["main_put"] == 1


def test_no_fallback_when_time_to_spare(monkeypatch):
    calls = _common(monkeypatch, merged=False)
    out = submit.push_csv(DF, "era5", "2026-07-14", CFG, cutoff_utc=datetime.now(timezone.utc) + timedelta(hours=20))
    assert out == "pr-pending"
    assert calls["main_put"] == 0


def test_unchanged_on_main_short_circuits(monkeypatch):
    monkeypatch.setattr(submit, "_put_file_would_change", lambda *a: False)
    out = submit.push_csv(DF, "era5", "2026-07-14", CFG, cutoff_utc=datetime.now(timezone.utc))
    assert out == "unchanged"


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv(submit.TOKEN_ENV)
    with pytest.raises(RuntimeError, match="POWDERBENCH_TOKEN"):
        submit.push_csv(DF, "era5", "2026-07-14", CFG)


def test_dry_run_never_calls_github(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network touched in dry-run")

    monkeypatch.setattr(submit.requests, "get", boom)
    monkeypatch.setattr(submit.requests, "post", boom)
    monkeypatch.setattr(submit.requests, "put", boom)
    assert submit.push_csv(DF, "era5", "2026-07-14", CFG, dry_run=True) == "dry-run"


def test_open_or_find_pr_reuses_existing(monkeypatch):
    class Resp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        submit.requests, "post", lambda *a, **k: Resp(422, {"message": "already exists"})
    )
    monkeypatch.setattr(
        submit.requests, "get", lambda *a, **k: Resp(200, [{"number": 42}])
    )
    assert submit._open_or_find_pr(CFG, "t", "submit/x", "title", "body") == 42

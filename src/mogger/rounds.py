"""Discover open PowderBench rounds over raw GitHub HTTP — no API tokens needed."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests

from .config import Config

log = logging.getLogger(__name__)

TIMEOUT = 30


@dataclass(frozen=True)
class Round:
    round_id: str
    league: str
    cutoff_utc: datetime
    target_days: tuple[date, ...]
    stations: tuple[str, ...]


def _fetch_manifest(cfg: Config, league: str, day: date) -> dict | None:
    url = f"{cfg.raw_base}/data/rounds/{league}/{day.isoformat()}/round.json"
    resp = requests.get(url, timeout=TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def discover_open_rounds(cfg: Config, league: str, now: datetime | None = None) -> list[Round]:
    """Rounds still open for submission: probe D in [today, today+2] — that
    covers both cutoff geometries (era5/resorts lock 11:00 UTC on D-1 with the
    round opened on D-2; stations lock 00:00 UTC on D with the round opened
    the day before). Off-season leagues 404 on every probe and yield []."""
    now = now or datetime.now(timezone.utc)
    margin = timedelta(minutes=cfg.cutoff_margin_min)
    out = []
    for offset in range(0, 3):
        day = now.date() + timedelta(days=offset)
        try:
            manifest = _fetch_manifest(cfg, league, day)
        except requests.RequestException as exc:
            log.warning("[%s] manifest fetch failed for %s: %s", league, day, exc)
            continue
        if not manifest or manifest.get("status") != "open":
            continue
        cutoff = datetime.fromisoformat(manifest["cutoff_utc"])
        if cutoff <= now + margin:
            log.info("[%s] round %s cutoff %s too close/past, skipping", league, day, cutoff)
            continue
        out.append(
            Round(
                round_id=manifest["round_id"],
                league=manifest["league"],
                cutoff_utc=cutoff,
                target_days=tuple(date.fromisoformat(d) for d in manifest["target_days"]),
                stations=tuple(manifest["stations"]),
            )
        )
    return out

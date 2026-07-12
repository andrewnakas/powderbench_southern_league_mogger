"""Station registry: fetched live from the benchmark's main branch so the bot
tracks registry growth automatically (leagues add stations mid-season), with
the committed snapshot (data/stations.json) as offline fallback.

Refresh the snapshot with scripts/refresh_static.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

import requests
import yaml

from .config import CONFIG_PATH, REPO_ROOT, load_config

log = logging.getLogger(__name__)

STATIONS_PATH = REPO_ROOT / "data" / "stations.json"
TIMEOUT = 30


@dataclass(frozen=True)
class Station:
    station_id: str
    league: str
    latitude: float
    longitude: float
    elevation_ft: float


@lru_cache
def _all_stations() -> tuple[Station, ...]:
    try:
        raw_base = load_config(CONFIG_PATH).raw_base
        resp = requests.get(f"{raw_base}/data/stations.yaml", timeout=TIMEOUT)
        resp.raise_for_status()
        registry = yaml.safe_load(resp.text)["stations"]
        return tuple(
            Station(
                station_id=s["station_id"],
                league=s.get("league", "stations"),
                latitude=s["latitude"],
                longitude=s["longitude"],
                elevation_ft=s["elevation_ft"],
            )
            for s in registry
        )
    except Exception as exc:
        log.warning("live station registry unavailable (%s), using snapshot", exc)
        raw = json.loads(STATIONS_PATH.read_text())
        return tuple(Station(**s) for s in raw)


def load_stations(league: str, station_ids: list[str] | None = None) -> list[Station]:
    """League registry, optionally narrowed to a round's station list."""
    out = [s for s in _all_stations() if s.league == league]
    if station_ids is not None:
        by_id = {s.station_id: s for s in out}
        out = [by_id[sid] for sid in station_ids if sid in by_id]
    return out

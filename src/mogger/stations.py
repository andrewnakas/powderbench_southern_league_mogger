"""Station registry snapshot (vendored from powderbench data/stations.yaml).

Regenerate with scripts/refresh_static.py when the benchmark registry changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .config import REPO_ROOT

STATIONS_PATH = REPO_ROOT / "data" / "stations.json"


@dataclass(frozen=True)
class Station:
    station_id: str
    league: str
    latitude: float
    longitude: float
    elevation_ft: float


@lru_cache
def _all_stations() -> tuple[Station, ...]:
    raw = json.loads(STATIONS_PATH.read_text())
    return tuple(Station(**s) for s in raw)


def load_stations(league: str, station_ids: list[str] | None = None) -> list[Station]:
    """League registry, optionally narrowed to a round's station list."""
    out = [s for s in _all_stations() if s.league == league]
    if station_ids is not None:
        by_id = {s.station_id: s for s in out}
        out = [by_id[sid] for sid in station_ids if sid in by_id]
    return out

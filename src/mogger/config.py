"""Bot configuration: benchmark repo location, leagues, ensemble members."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.json"


@dataclass(frozen=True)
class Config:
    team: str
    target_repo: str
    raw_base: str
    leagues: tuple[str, ...]
    models: tuple[str, ...]
    cutoff_margin_min: int = 10

    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / "data"


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    raw = json.loads(Path(path).read_text())
    return Config(
        team=raw["team"],
        target_repo=raw["target_repo"],
        raw_base=raw["raw_base"],
        leagues=tuple(raw["leagues"]),
        models=tuple(raw["models"]),
        cutoff_margin_min=int(raw.get("cutoff_margin_min", 10)),
    )

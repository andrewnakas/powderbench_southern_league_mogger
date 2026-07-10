"""Dev-only: regenerate the vendored station registry + climatology snapshots
from a local powderbench checkout.

    python scripts/refresh_static.py --powderbench ../powderbench
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
LEAGUES = ("era5", "resorts", "stations")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--powderbench", type=Path, default=HERE.parent / "powderbench")
    args = ap.parse_args()

    registry = yaml.safe_load((args.powderbench / "data" / "stations.yaml").read_text())["stations"]
    slim = [
        {
            "station_id": s["station_id"],
            "league": s.get("league", "stations"),
            "latitude": s["latitude"],
            "longitude": s["longitude"],
            "elevation_ft": s["elevation_ft"],
        }
        for s in registry
    ]
    out = HERE / "data" / "stations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slim, indent=1))
    print(f"wrote {out} ({len(slim)} stations)")

    for league in LEAGUES:
        src = args.powderbench / "data" / "climatology" / f"{league}.csv"
        dst = HERE / "data" / "climatology" / f"{league}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"copied {dst}")


if __name__ == "__main__":
    main()

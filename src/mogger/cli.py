"""CLI: `mogger run` — discover open rounds, forecast, validate, submit."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .calibration import load_calibration
from .config import load_config
from .forecast import build_submission, climatology_submission
from .rounds import discover_open_rounds
from .submit import push_csv
from .validate import check

log = logging.getLogger("mogger")


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mogger run")
    ap.add_argument("--league", action="append", help="restrict to league(s); default: all configured")
    ap.add_argument("--round", dest="round_id", help="restrict to one round date YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true", help="build + validate, print, do not push")
    ap.add_argument("--out-dir", type=Path, help="also write the CSVs here")
    args = ap.parse_args(argv)

    cfg = load_config()
    leagues = args.league or list(cfg.leagues)
    failed = False
    submitted = 0

    for league in leagues:
        try:
            rounds = discover_open_rounds(cfg, league)
        except Exception as exc:
            log.error("[%s] round discovery failed: %s", league, exc)
            failed = True
            continue
        if args.round_id:
            rounds = [r for r in rounds if r.round_id == args.round_id]
        if not rounds:
            log.info("[%s] no open rounds (off-season or past cutoff)", league)
            continue

        calib = load_calibration(league)
        for rnd in rounds:
            try:
                df = build_submission(rnd, cfg, calib)
                if not len(df):
                    log.warning("[%s %s] empty forecast, using climatology submission", league, rnd.round_id)
                    df = climatology_submission(rnd)
            except Exception as exc:
                log.error("[%s %s] forecast failed (%s), using climatology submission", league, rnd.round_id, exc)
                df = climatology_submission(rnd)

            problems = check(df, rnd)
            if problems:
                log.error("[%s %s] validation failed, NOT pushing: %s", league, rnd.round_id, problems)
                failed = True
                continue

            if args.out_dir:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                out = args.out_dir / f"{league}-{rnd.round_id}-{cfg.team}.csv"
                out.write_bytes(df.to_csv(index=False).encode())
                log.info("wrote %s", out)

            try:
                result = push_csv(df, league, rnd.round_id, cfg, cutoff_utc=rnd.cutoff_utc, dry_run=args.dry_run)
                log.info("[%s %s] %d rows -> %s", league, rnd.round_id, len(df), result)
                submitted += 1
            except Exception as exc:
                log.error("[%s %s] push failed: %s", league, rnd.round_id, exc)
                failed = True

    log.info("done: %d submission(s) handled%s", submitted, " (dry-run)" if args.dry_run else "")
    return 1 if failed else 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argv = sys.argv[1:]
    if not argv or argv[0] != "run":
        print("usage: mogger run [--league L] [--round YYYY-MM-DD] [--dry-run] [--out-dir DIR]", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run(argv[1:]))


if __name__ == "__main__":
    main()

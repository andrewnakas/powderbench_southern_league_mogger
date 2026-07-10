"""Push a submission CSV to the benchmark repo's main branch via the GitHub
contents API. sha-aware PUT makes re-runs idempotent updates (allowed until
the cutoff); identical content is skipped entirely."""

from __future__ import annotations

import base64
import logging
import os
import time

import pandas as pd
import requests

from .config import Config

log = logging.getLogger(__name__)

API = "https://api.github.com"
TIMEOUT = 30
TOKEN_ENV = "POWDERBENCH_TOKEN"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def submission_path(cfg: Config, league: str, round_id: str) -> str:
    return f"data/submissions/{league}/{round_id}/{cfg.team}.csv"


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def push_csv(df: pd.DataFrame, league: str, round_id: str, cfg: Config, dry_run: bool = False) -> str:
    path = submission_path(cfg, league, round_id)
    content = to_csv_bytes(df)
    if dry_run:
        log.info("[dry-run] would push %d rows to %s:%s", len(df), cfg.target_repo, path)
        return "dry-run"

    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        raise RuntimeError(f"{TOKEN_ENV} is not set — cannot push to {cfg.target_repo}")

    url = f"{API}/repos/{cfg.target_repo}/contents/{path}"
    for attempt in range(3):
        sha = None
        resp = requests.get(url, headers=_headers(token), params={"ref": "main"}, timeout=TIMEOUT)
        if resp.status_code == 200:
            existing = resp.json()
            sha = existing["sha"]
            if base64.b64decode(existing.get("content", "") or "") == content:
                log.info("%s already up to date on main", path)
                return "unchanged"
        elif resp.status_code != 404:
            resp.raise_for_status()

        payload = {
            "message": f"{cfg.team}: {league} round {round_id}",
            "content": base64.b64encode(content).decode(),
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha
        put = requests.put(url, headers=_headers(token), json=payload, timeout=TIMEOUT)
        if put.status_code in (200, 201):
            log.info("pushed %s (%s)", path, "updated" if sha else "created")
            return "updated" if sha else "created"
        if put.status_code == 409 and attempt < 2:  # sha race with another workflow
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"push failed ({put.status_code}): {put.text[:300]}")
    raise RuntimeError("push failed after retries")

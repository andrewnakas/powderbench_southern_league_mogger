"""Submit a CSV to the benchmark: PR-first, direct-push fallback.

The benchmark auto-merges valid submission-only PRs (and binds the team name
to the submitting login in data/teams.yaml), so the normal path is: push the
CSV to a bot branch, open a PR, and let the gate validate + squash-merge it.
Only if the gate hasn't merged and the cutoff is close does the bot fall back
to pushing straight to main (the owner PAT retains that right; a gate outage
must never cost a round — RULES only require the file on main before cutoff).
"""

from __future__ import annotations

import base64
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import Config

log = logging.getLogger(__name__)

API = "https://api.github.com"
TIMEOUT = 30
TOKEN_ENV = "POWDERBENCH_TOKEN"
MERGE_WAIT_S = 300          # how long to wait for the auto-merge gate
MERGE_POLL_S = 15
FALLBACK_WINDOW = timedelta(hours=2)  # direct-push only when cutoff is this close


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


def _put_file(cfg: Config, token: str, path: str, content: bytes, branch: str, message: str) -> str:
    """sha-aware create/update of one file on a branch. Returns
    'unchanged' | 'created' | 'updated'."""
    url = f"{API}/repos/{cfg.target_repo}/contents/{path}"
    for attempt in range(3):
        sha = None
        resp = requests.get(url, headers=_headers(token), params={"ref": branch}, timeout=TIMEOUT)
        if resp.status_code == 200:
            existing = resp.json()
            sha = existing["sha"]
            if base64.b64decode(existing.get("content", "") or "") == content:
                return "unchanged"
        elif resp.status_code != 404:
            resp.raise_for_status()

        payload = {"message": message, "content": base64.b64encode(content).decode(), "branch": branch}
        if sha:
            payload["sha"] = sha
        put = requests.put(url, headers=_headers(token), json=payload, timeout=TIMEOUT)
        if put.status_code in (200, 201):
            return "updated" if sha else "created"
        if put.status_code in (409, 422) and attempt < 2:  # sha race
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"put {path}@{branch} failed ({put.status_code}): {put.text[:300]}")
    raise RuntimeError(f"put {path}@{branch} failed after retries")


def _ensure_branch(cfg: Config, token: str, branch: str) -> None:
    """Create the branch from main's head if it doesn't exist."""
    resp = requests.get(
        f"{API}/repos/{cfg.target_repo}/git/ref/heads/{branch}", headers=_headers(token), timeout=TIMEOUT
    )
    if resp.status_code == 200:
        return
    main = requests.get(
        f"{API}/repos/{cfg.target_repo}/git/ref/heads/main", headers=_headers(token), timeout=TIMEOUT
    )
    main.raise_for_status()
    create = requests.post(
        f"{API}/repos/{cfg.target_repo}/git/refs",
        headers=_headers(token),
        json={"ref": f"refs/heads/{branch}", "sha": main.json()["object"]["sha"]},
        timeout=TIMEOUT,
    )
    if create.status_code not in (200, 201) and create.status_code != 422:  # 422: created concurrently
        raise RuntimeError(f"branch create failed ({create.status_code}): {create.text[:200]}")


def _open_or_find_pr(cfg: Config, token: str, branch: str, title: str, body: str) -> int:
    owner = cfg.target_repo.split("/")[0]
    resp = requests.post(
        f"{API}/repos/{cfg.target_repo}/pulls",
        headers=_headers(token),
        json={"title": title, "head": f"{owner}:{branch}", "base": "main", "body": body},
        timeout=TIMEOUT,
    )
    if resp.status_code in (200, 201):
        return resp.json()["number"]
    if resp.status_code == 422:  # already exists — find it
        listing = requests.get(
            f"{API}/repos/{cfg.target_repo}/pulls",
            headers=_headers(token),
            params={"head": f"{owner}:{branch}", "state": "open"},
            timeout=TIMEOUT,
        )
        listing.raise_for_status()
        prs = listing.json()
        if prs:
            return prs[0]["number"]
    raise RuntimeError(f"PR open failed ({resp.status_code}): {resp.text[:300]}")


def _wait_for_merge(cfg: Config, token: str, pr: int, wait_s: int = MERGE_WAIT_S) -> bool:
    deadline = time.monotonic() + wait_s
    url = f"{API}/repos/{cfg.target_repo}/pulls/{pr}"
    while time.monotonic() < deadline:
        resp = requests.get(url, headers=_headers(token), timeout=TIMEOUT)
        if resp.status_code == 200 and resp.json().get("merged"):
            return True
        time.sleep(MERGE_POLL_S)
    return False


def push_csv(
    df: pd.DataFrame,
    league: str,
    round_id: str,
    cfg: Config,
    cutoff_utc: datetime | None = None,
    dry_run: bool = False,
) -> str:
    path = submission_path(cfg, league, round_id)
    content = to_csv_bytes(df)
    if dry_run:
        log.info("[dry-run] would PR %d rows to %s:%s", len(df), cfg.target_repo, path)
        return "dry-run"

    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        raise RuntimeError(f"{TOKEN_ENV} is not set — cannot submit to {cfg.target_repo}")

    message = f"{cfg.team}: {league} round {round_id}"

    # already on main and identical? nothing to do (e.g. gate merged an earlier run)
    if _put_file_would_change(cfg, token, path, content) is False:
        log.info("%s already up to date on main", path)
        return "unchanged"

    branch = f"submit/{cfg.team}-{league}-{round_id}"
    _ensure_branch(cfg, token, branch)
    state = _put_file(cfg, token, path, content, branch, message)
    log.info("%s on %s: %s", path, branch, state)

    pr = _open_or_find_pr(
        cfg, token, branch, f"Submission: {cfg.team} ({league}/{round_id})",
        "Automated submission — see the team's method description in its repo. "
        "Expected to be handled by the auto-merge gate.",
    )
    log.info("PR #%d open, waiting for the auto-merge gate", pr)
    if _wait_for_merge(cfg, token, pr):
        log.info("PR #%d merged by the gate", pr)
        return "merged"

    now = datetime.now(timezone.utc)
    if cutoff_utc is not None and cutoff_utc - now <= FALLBACK_WINDOW:
        log.warning("gate did not merge PR #%d and cutoff is %s — direct-pushing to main", pr, cutoff_utc)
        state = _put_file(cfg, token, path, content, "main", message + " (gate fallback)")
        return f"fallback-{state}"

    log.warning("gate did not merge PR #%d yet; leaving it (cutoff %s is not close)", pr, cutoff_utc)
    return "pr-pending"


def _put_file_would_change(cfg: Config, token: str, path: str, content: bytes) -> bool | None:
    """False iff the file already has this exact content on main."""
    resp = requests.get(
        f"{API}/repos/{cfg.target_repo}/contents/{path}",
        headers=_headers(token),
        params={"ref": "main"},
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        return None
    return base64.b64decode(resp.json().get("content", "") or "") != content

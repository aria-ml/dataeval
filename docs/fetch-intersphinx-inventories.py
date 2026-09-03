#!/usr/bin/env python3
"""Pre-fetch intersphinx inventories with retries, before the docs build.

Why this exists
---------------
``sphinx-build --fail-on-warning`` fetches every ``intersphinx_mapping``
inventory live from the network (once per build, no retries). An intermittent
DNS/connect blip on docs.scipy.org & co. therefore fails the entire CI docs
job and forces a full rerun.

How it works
------------
1. This script downloads each inventory (retries + exponential backoff) into
   ``docs/source/_intersphinx/``.
2. ``docs/source/conf.py`` points each mapping's *location* at the local file,
   so the build reads the inventories from disk and makes **no** network
   calls at all. The target URI in the mapping is still remote; it only
   controls the external links Sphinx generates.
3. CI caches ``docs/source/_intersphinx/`` (keyed on ``conf.py``), so most
   builds never fetch anything.
4. If a fetch fails completely but a previous copy exists on disk, the stale
   copy is reused (with a warning) so a total outage degrades gracefully
   instead of blocking the build. The build only fails when there is no
   usable copy at all.

Usage
-----
    python docs/fetch-intersphinx-inventories.py
        [--conf docs/source/conf.py]
        [--max-age-days 14]
        [--attempts 5]
        [--force]

The nox ``docs`` session runs this automatically before ``sphinx-build``.

Stdlib only — no project dependencies required.
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

USER_AGENT = "dataeval-docs-intersphinx-fetch/1.0"
TTL_ENV_VAR = "DATAEVAL_DOCS_INV_TTL_DAYS"
DEFAULT_MAX_AGE_DAYS = 14
DEFAULT_ATTEMPTS = 5
DEFAULT_TIMEOUT_S = 30.0
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 30.0

# Sphinx inventory v2 files start with this comment line.
INV_HEADER = b"# Sphinx inventory version"


def _log(msg: str) -> None:
    print(f"[intersphinx-fetch] {msg}", flush=True)


def extract_intersphinx_mapping(conf_path: Path) -> dict[str, tuple[str, str | list[str]]]:
    """Parse ``intersphinx_mapping`` out of conf.py without importing it.

    conf.py imports project code (``import dataeval``), so parsing the AST and
    literal-evaluating the assignment is the safe route.
    """
    tree = ast.parse(conf_path.read_text(encoding="utf-8"), filename=str(conf_path))
    mapping = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "intersphinx_mapping":
                mapping = ast.literal_eval(node.value)
    if mapping is None:
        raise ValueError(f"intersphinx_mapping not found in {conf_path}")
    if not isinstance(mapping, dict):
        raise ValueError("intersphinx_mapping is not a dict")
    return mapping


def inv_url_for(uri: str) -> str:
    """Inventory URL for a project target URI (Sphinx's default: uri + objects.inv)."""
    return uri.rstrip("/") + "/objects.inv"


def fetch_with_retries(url: str, attempts: int, timeout: float) -> bytes:
    """GET *url* with retries and exponential backoff. Raises on final failure."""
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            if raw.startswith(INV_HEADER):
                return raw
            # Downloaded something that is not an inventory (e.g. an HTML error
            # page with a 200 status). Treat as a failure so we don't cache it.
            raise ValueError(f"unexpected content (first {min(80, len(raw))} bytes: {raw[:80]!r})")
        except Exception as err:
            last_err = err
            if attempt < attempts:
                delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_MAX_S)
                delay += random.uniform(0, 0.5)
                _log(f"  attempt {attempt}/{attempts} failed: {err}; retrying in {delay:.1f}s")
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def ensure_inventory(
    name: str,
    uri: str,
    location: str,
    src_dir: Path,
    *,
    max_age_days: float,
    attempts: int,
    force: bool,
) -> str:
    """Ensure the local inventory file for one mapping entry exists.

    Returns 'fresh', 'fetched', 'skipped-fresh', or 'stale-fallback'.
    Raises if the fetch fails and no previous copy exists.
    """
    dest = src_dir / location
    now = time.time()
    max_age_s = max_age_days * 86400

    if not force and dest.is_file() and (now - dest.stat().st_mtime) < max_age_s:
        age_h = (now - dest.stat().st_mtime) / 3600
        _log(f"{name}: fresh on disk ({age_h:.1f}h old), skipping fetch of {inv_url_for(uri)}")
        return "skipped-fresh"

    url = inv_url_for(uri)
    try:
        raw = fetch_with_retries(url, attempts=attempts, timeout=DEFAULT_TIMEOUT_S)
    except Exception as err:
        if dest.is_file():
            age_d = (now - dest.stat().st_mtime) / 86400
            _log(
                f"WARNING {name}: all {attempts} fetch attempts failed ({err}); "
                f"reusing stale copy from {age_d:.1f} days ago. "
                f"Links to {name} may be out of date."
            )
            return "stale-fallback"
        raise RuntimeError(
            f"failed to fetch {url} after {attempts} attempts and no cached copy "
            f"exists at {dest}. Fix the network issue or delete the docs CI cache."
        ) from err

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, dest)
    _log(f"{name}: fetched {len(raw) / 1024:.1f} KiB from {url} -> {dest}")
    return "fetched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conf", default="docs/source/conf.py", help="path to Sphinx conf.py")
    parser.add_argument(
        "--src-dir",
        default=None,
        help="Sphinx source dir (default: directory containing --conf)",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=float(os.environ.get(TTL_ENV_VAR, DEFAULT_MAX_AGE_DAYS)),
        help=f"reuse a file younger than this (default: {DEFAULT_MAX_AGE_DAYS}, env: {TTL_ENV_VAR})",
    )
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--force", action="store_true", help="refetch even if fresh on disk")
    args = parser.parse_args()

    conf_path = Path(args.conf)
    src_dir = Path(args.src_dir) if args.src_dir else conf_path.parent

    try:
        mapping = extract_intersphinx_mapping(conf_path)
    except (ValueError, SyntaxError) as err:
        _log(f"ERROR: could not parse intersphinx_mapping from {conf_path}: {err}")
        return 2

    failures: list[str] = []
    for name, value in mapping.items():
        try:
            uri, location = value
        except (TypeError, ValueError):
            _log(f"ERROR: malformed intersphinx_mapping[{name!r}] = {value!r}")
            failures.append(name)
            continue
        if not isinstance(uri, str) or not uri.startswith(("http://", "https://")):
            _log(f"ERROR: intersphinx_mapping[{name!r}] target URI must be a remote URL")
            failures.append(name)
            continue
        locations = location if isinstance(location, list) else [location]
        for loc in locations:
            if not isinstance(loc, str) or not loc:
                _log(f"ERROR: intersphinx_mapping[{name!r}] location must be a non-empty string")
                failures.append(name)
                continue
            if loc.startswith(("http://", "https://")):
                _log(
                    f"ERROR: intersphinx_mapping[{name!r}] location must be a local path "
                    f"(relative to the Sphinx source dir), not a URL. Pre-fetching is "
                    f"required for the build to stay network-free."
                )
                failures.append(name)
                continue
            try:
                ensure_inventory(
                    name,
                    uri,
                    loc,
                    src_dir,
                    max_age_days=args.max_age_days,
                    attempts=args.attempts,
                    force=args.force,
                )
            except RuntimeError as err:
                _log(f"ERROR {err}")
                failures.append(name)

    if failures:
        _log(f"FAILED for: {', '.join(failures)}")
        return 1
    _log("all inventories available")
    return 0


if __name__ == "__main__":
    sys.exit(main())

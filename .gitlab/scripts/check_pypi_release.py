#!/usr/bin/env python3
"""Wait until the release named by a version tag is on PyPI, or fail after the timeout.

The GitHub workflow that publishes to PyPI starts when the tag reaches GitHub, so the package appears some minutes
after the tag pipeline starts.
"""

import json
import sys
import time
import urllib.error
import urllib.request

PACKAGE = "dataeval"
TIMEOUT_SECONDS = 30 * 60
INTERVAL_SECONDS = 30


def published_files(version: str) -> list[dict] | None:
    """Release files PyPI lists for the version, or None while it is not there yet."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{PACKAGE}/{version}/json", timeout=30) as response:
            return json.load(response)["urls"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def main() -> int:
    """Poll PyPI for the tag's version."""
    tag = sys.argv[1]
    version = tag.removeprefix("v")
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while True:
        files = published_files(version)
        if files:
            kinds = sorted({f["packagetype"] for f in files})
            print(f"{PACKAGE} {version} is on PyPI ({', '.join(kinds)})")
            return 0 if "bdist_wheel" in kinds and "sdist" in kinds else 1
        if time.monotonic() > deadline:
            print(f"{PACKAGE} {version} did not appear on PyPI within {TIMEOUT_SECONDS // 60} minutes")
            return 1
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())

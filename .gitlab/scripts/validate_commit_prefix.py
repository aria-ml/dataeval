#!/usr/bin/env python3
"""
Require a known `[tag]` prefix on the merge request title.

The title becomes the commit subject on main, and `scripts/release.py` reads those
subjects to build the changelog. A missing or misspelled tag silently files the change
under Miscellaneous - which is how `[imrp] Split internal utilities` shipped - so it is
rejected here, where the author can still fix it.
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from release import TAG_CATEGORIES

if __name__ == "__main__":
    # A draft is not going to be merged as-is, so do not block it on its title yet
    title = re.sub(r"^\s*(Draft|WIP):\s*", "", os.getenv("CI_MERGE_REQUEST_TITLE", ""))
    match = re.match(r"\[([a-zA-Z]+)\]\s*\S", title)

    if match is None:
        sys.exit(
            f"ERROR: merge request title must start with a [tag] prefix, got: {title!r}\n"
            f"       Valid tags: {', '.join(sorted(TAG_CATEGORIES))}"
        )

    tag = match[1].lower()
    if tag not in TAG_CATEGORIES:
        sys.exit(
            f"ERROR: unknown tag [{match[1]}] in merge request title.\n"
            f"       Valid tags: {', '.join(sorted(TAG_CATEGORIES))}"
        )

    print(f"Title tag [{tag}] -> {TAG_CATEGORIES[tag]}")

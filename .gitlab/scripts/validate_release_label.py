#!/usr/bin/env python3

import os
import sys

if __name__ == "__main__":
    labels = os.getenv("CI_MERGE_REQUEST_LABELS")
    exitcode = 0

    if labels is None or "release::" not in labels:
        print("ERROR: Release label not present.  Please ensure the Merge Request has an appropriate label.")
        exitcode = 1

    sys.exit(exitcode)

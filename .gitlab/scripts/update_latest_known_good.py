#!/usr/bin/env python3

if __name__ == "__main__":
    from os import environ

    from gitlab import LATEST_KNOWN_GOOD, Gitlab

    ref = environ.get("CI_COMMIT_SHA") or "main"

    gl = Gitlab(verbose=True)
    gl.delete_tag(LATEST_KNOWN_GOOD)
    gl.add_tag(LATEST_KNOWN_GOOD, ref)

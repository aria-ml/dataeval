#!/usr/bin/env python3

if __name__ == "__main__":
    from gitlab import LATEST_KNOWN_GOOD, Gitlab

    gl = Gitlab(verbose=True)
    gl.delete_tag(LATEST_KNOWN_GOOD)
    gl.add_tag(LATEST_KNOWN_GOOD)

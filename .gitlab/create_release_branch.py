#!/usr/bin/env python3

if __name__ == "__main__":
    from gitlab import LATEST_KNOWN_GOOD, Gitlab
    from versiontag import VersionTag

    gl = Gitlab(verbose=True)
    vt = VersionTag(gl)

    gl.create_repository_branch(f"releases/{vt.pending}", LATEST_KNOWN_GOOD)

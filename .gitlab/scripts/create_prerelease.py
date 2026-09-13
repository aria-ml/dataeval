#!/usr/bin/env python3
"""
Creates a prerelease version tag (e.g., v1.0.0-a0, v1.0.0-rc0, v1.0.0-rc1).

Trigger by setting CREATE_PRERELEASE=true in a scheduled pipeline. PRERELEASE_TYPE
selects the kind: 'rc' (default) for a release candidate, 'a' for an early alpha
snapshot of main that downstream projects can build against.

Behavior:
- If current version is a prerelease of the same kind (v1.0.0-a0), increments to v1.0.0-a1
- If current version is an earlier kind (v1.0.0-a3 with PRERELEASE_TYPE=rc), promotes
  to v1.0.0-rc0 on the same base version
- If current version is a standard release (v0.99.0), creates v1.0.0-<kind>0
  based on MR labels (MAJOR/MINOR/PATCH)
- Updates CHANGELOG.md with the prerelease version
"""

if __name__ == "__main__":
    import os

    from gitlab import Gitlab
    from releasegen import ReleaseGen
    from versiontag import VersionTag

    gl = Gitlab(verbose=True)
    rg = ReleaseGen(gl)
    vt = VersionTag(gl)

    # Get the version type from MR labels
    version_type = rg.get_version_type()

    # Calculate next prerelease version of the requested kind
    prerelease_type = os.environ.get("PRERELEASE_TYPE", "rc")
    version_tag = vt.next_prerelease(version_type, "a" if prerelease_type == "a" else "rc")

    # Bail out before committing if the tag already exists (e.g. a concurrent
    # pipeline won the race) - otherwise the commit lands on main untagged
    if version_tag and gl.tag_exists(version_tag):
        raise SystemExit(f"Tag {version_tag} already exists - another pipeline created it. Nothing to do.")

    # Generate changelog with prerelease version
    _, payload = rg.generate_prerelease(version_tag)

    if version_tag and payload:
        print(f"Creating prerelease {version_tag}:")
        commit_id = gl.commit("main", f"Prerelease {version_tag}", payload)["id"]
        # Tag before triggering pipeline so push-docs-cache.sh can detect the version tag
        gl.add_tag(version_tag, commit_id, message=f"DataEval {version_tag} (prerelease)")
        print(f"Created prerelease tag: {version_tag}")
        # Trigger API pipeline on main for docs build and artifact publishing
        gl.create_pipeline("main")
        print("Triggered pipeline on main")
    else:
        print("No changes to commit and tag.")

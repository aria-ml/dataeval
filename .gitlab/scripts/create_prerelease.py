#!/usr/bin/env python3
"""
Creates a pre-release version tag (e.g., v1.0.0-rc0, v1.0.0-rc1).

Trigger by setting CREATE_PRE_RELEASE=true in a scheduled pipeline.

Behavior:
- If current version is a pre-release (v1.0.0-rc0), increments to v1.0.0-rc1
- If current version is a standard release (v0.99.0), creates v1.0.0-rc0
  based on MR labels (MAJOR/MINOR/PATCH)
- Creates a release branch (release/vX.Y) if one does not already exist
- Updates CHANGELOG.md with the pre-release version
"""

if __name__ == "__main__":
    from gitlab import Gitlab
    from releasegen import ReleaseGen
    from versiontag import VersionTag

    gl = Gitlab(verbose=True)
    rg = ReleaseGen(gl)
    vt = VersionTag(gl)

    # Get the version type from MR labels
    version_type = rg.get_version_type()

    # Calculate next pre-release version
    version_tag = vt.next_prerelease(version_type)

    # Generate changelog with pre-release version
    _, payload = rg.generate_prerelease(version_tag)

    if version_tag and payload:
        print(f"Creating pre-release {version_tag}:")
        commit_id = gl.commit("main", f"Pre-release {version_tag}", payload)["id"]
        # Create release branch with vX.Y format (strip patch version and rc suffix)
        major_minor = ".".join(version_tag.split("-")[0].split(".")[:2])
        release_branch = f"release/{major_minor}"
        try:
            gl.create_repository_branch(release_branch, commit_id)
            print(f"Created release branch: {release_branch}")
        except ConnectionError:
            print(f"Release branch '{release_branch}' already exists, updating to new commit")
            gl.delete_branch(release_branch)
            gl.create_repository_branch(release_branch, commit_id)
            print(f"Updated release branch: {release_branch}")
        # Tag before triggering pipeline so push-docs-cache.sh can detect the version tag
        gl.add_tag(version_tag, commit_id, message=f"DataEval {version_tag} (pre-release)")
        print(f"Created pre-release tag: {version_tag}")
        # Trigger pipeline on release branch for docs build and artifact publishing
        gl.create_pipeline(release_branch)
        print(f"Triggered pipeline on {release_branch}")
    else:
        print("No changes to commit and tag.")

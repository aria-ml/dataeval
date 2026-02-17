#!/usr/bin/env python3
"""
Creates a release branch on-demand from the latest tag matching a given version prefix.

Triggered by setting CREATE_RELEASE_BRANCH=vX.Y in a web pipeline.

Example:
    CREATE_RELEASE_BRANCH=v1.2  ->  finds latest v1.2.* tag  ->  creates release/v1.2 from it

Behavior:
- Validates the CREATE_RELEASE_BRANCH variable format (vX.Y)
- Finds the latest tag matching vX.Y.* (e.g., v1.2.0, v1.2.1)
- Creates a release/vX.Y branch pointing at that tag's commit
- If the branch already exists, exits with an informational message
"""

if __name__ == "__main__":
    import os
    import re
    import sys

    from gitlab import Gitlab

    # Get the version prefix from environment variable
    version_prefix = os.getenv("CREATE_RELEASE_BRANCH", "").strip()

    if not version_prefix:
        print("ERROR: CREATE_RELEASE_BRANCH environment variable is not set")
        sys.exit(1)

    # Validate format: must be vX.Y (e.g., v1.2, v0.99)
    prefix_pattern = re.compile(r"^v\d+\.\d+$")
    if not prefix_pattern.match(version_prefix):
        print(f"ERROR: Invalid version prefix format: '{version_prefix}'")
        print("Expected format: vX.Y (e.g., v1.2, v0.99)")
        sys.exit(1)

    print(f"Creating release branch for version prefix: {version_prefix}")

    gl = Gitlab(verbose=True)

    # Check if branch already exists
    release_branch = f"release/{version_prefix}"
    try:
        gl.get_single_repository_branch(release_branch)
        print(f"INFO: Release branch '{release_branch}' already exists")
        print("No action needed. Use the existing branch for patch releases.")
        sys.exit(0)
    except ConnectionError as e:
        status_code = int(str(e))
        if status_code != 404:
            raise

    # Find the latest tag matching vX.Y.*
    tags = gl.list_tags()
    tag_pattern = re.compile(rf"^{re.escape(version_prefix)}\.(\d+)$")

    matching_tags = []
    for tag in tags:
        match = tag_pattern.match(tag["name"])
        if match:
            patch_num = int(match.group(1))
            matching_tags.append((patch_num, tag))

    if not matching_tags:
        print(f"ERROR: No tags found matching {version_prefix}.* pattern")
        print("Available version tags:")
        version_tag_pattern = re.compile(r"^v\d+\.\d+\.\d+")
        for tag in tags[:20]:
            if version_tag_pattern.match(tag["name"]):
                print(f"  - {tag['name']}")
        sys.exit(1)

    # Sort by patch number descending and take the latest
    matching_tags.sort(key=lambda x: x[0], reverse=True)
    latest_patch, latest_tag = matching_tags[0]
    tag_name = latest_tag["name"]
    commit_id = latest_tag["commit"]["id"]

    print(f"Latest tag for {version_prefix}: {tag_name} (commit: {commit_id[:8]})")

    # Create the release branch from the tag's commit
    gl.create_repository_branch(release_branch, commit_id)
    print(f"Successfully created release branch: {release_branch}")
    print(f"  From tag: {tag_name}")
    print(f"  At commit: {commit_id[:8]}")
    print()
    print("Next steps:")
    print(f"  1. Cherry-pick or merge fixes to {release_branch}")
    print(f"  2. Patch releases will be auto-created when fixes merge to {release_branch}")
    print(f"  3. Future 'release::fix' merges to main will auto-cherry-pick to {release_branch}")

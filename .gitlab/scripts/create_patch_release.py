#!/usr/bin/env python3

if __name__ == "__main__":
    import os
    import re
    import sys

    from gitlab import Gitlab
    from releasegen import CHANGELOG_FILE, HOWTO_INDEX_FILE, TUTORIAL_INDEX_FILE, ReleaseGen, _Category, _Merge

    # Get current branch name from CI environment
    branch_name = os.getenv("CI_COMMIT_BRANCH", "")

    # Validate we're on a release branch
    if not branch_name.startswith("release/v"):
        print(f"ERROR: This script should only run on release/v* branches, got: {branch_name}")
        sys.exit(1)

    # Extract the base version from branch name (e.g., "release/v1.2" -> "v1.2")
    base_version_match = re.match(r"release/(v\d+\.\d+)", branch_name)
    if not base_version_match:
        print(f"ERROR: Could not extract version from branch name: {branch_name}")
        sys.exit(1)

    base_version = base_version_match.group(1)
    print(f"Base version from branch: {base_version}")

    gl = Gitlab(verbose=True)
    rg = ReleaseGen(gl)

    # Get the commit hash of the last changelog update
    current = rg._read_changelog()
    last_hash = rg._get_last_hash(current[0]) if current else ""

    # Get all merges to this release branch since last release
    merges = gl.list_merge_requests(state="merged", target_branch=branch_name)

    if not merges:
        print("No merge requests found for this release branch.")
        sys.exit(0)

    # Check that all merges are fixes only and collect them
    categorized_merges = []
    merges_sorted = [_Merge(m) for m in merges]
    merges_sorted.sort(reverse=True)

    for merge in merges_sorted:
        # Stop if we've reached the last hash
        if merge.hash == last_hash:
            break

        categorized_merges.append(merge)

        # Validate that only fix labels are present
        if merge.category != _Category.FIX:
            print(f"ERROR: Non-fix merge request found: {merge.description}")
            print(f"       Category: {_Category(merge.category).name}")
            print("       Only 'release::fix' labeled MRs are allowed on release branches.")
            sys.exit(1)

    if not categorized_merges:
        print("No new changes to release since last tag.")
        sys.exit(0)

    # Find the latest patch version for this release branch
    tags = gl.list_tags()
    patch_version = 0
    version_pattern = re.compile(rf"{re.escape(base_version)}\.(\d+)")

    for tag in tags:
        match = version_pattern.match(tag["name"])
        if match:
            patch_num = int(match.group(1))
            patch_version = max(patch_version, patch_num)

    # Calculate next patch version
    next_patch = patch_version + 1
    next_version = f"{base_version}.{next_patch}"
    print(f"Next patch version: {next_version}")

    # Build changelog content
    lines = ["", _Category.to_markdown(_Category.FIX)]
    for merge in categorized_merges:
        lines.append(merge.to_markdown())

    latest_merge = merges_sorted[0]
    header = [f"[//]: # ({latest_merge.hash})", "", "# DataEval Change Log", "", f"## {next_version}"]
    changelog_content = "\n".join(header + lines) + "\n"

    for oldline in current[3:]:
        changelog_content += oldline

    # Create actions list
    actions = []

    # Update jupyter cache
    jupyter_cache_actions = rg._generate_jupyter_cache_actions()
    actions.extend(jupyter_cache_actions)

    # Update documentation index files
    actions.extend(
        [
            rg._generate_index_markdown_update_action(HOWTO_INDEX_FILE, next_version),
            rg._generate_index_markdown_update_action(TUTORIAL_INDEX_FILE, next_version),
        ],
    )

    # Add changelog update
    actions.append(
        {
            "action": "update",
            "file_path": CHANGELOG_FILE,
            "encoding": "text",
            "content": changelog_content,
        },
    )

    # Filter out empty actions
    payload = [action for action in actions if action]

    if not payload:
        print("No changes to commit and tag.")
        sys.exit(0)

    print(f"Updating changelog and documentation for patch release {next_version}:")
    commit_id = gl.commit(branch_name, f"Release {next_version}", payload)["id"]
    gl.add_tag(next_version, commit_id, message=f"DataEval {next_version}")
    print(f"Successfully created patch release {next_version}")

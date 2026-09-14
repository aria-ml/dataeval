#!/usr/bin/env python3

if __name__ == "__main__":
    import os
    import re
    import sys

    from gitlab import Gitlab
    from releasegen import CHANGELOG_FILE, HOWTO_INDEX_FILE, TUTORIAL_INDEX_FILE, ReleaseGen, _Category

    # Categories that change the public API and therefore belong on a minor or major
    # release cut from main, never on a patch release off a release branch.
    DISALLOWED = (_Category.MAJOR, _Category.FEATURE, _Category.DEPRECATION)

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

    # Collect everything that landed on the release branch since that hash. Hotfixes
    # are routinely pushed straight to a release branch, so merge requests alone miss
    # most of them; _get_entries pairs them with the branch's direct commits.
    latest, entries = rg._get_entries(last_hash, current, branch=branch_name)

    # UNKNOWN covers merge requests with no release::* label; they are dropped from the
    # changelog on main too, so they must not gate or shape the patch release either.
    entries.pop(_Category.UNKNOWN, None)

    for category in DISALLOWED:
        for entry in entries.get(category, []):
            print(f"ERROR: {_Category(category).name} change found on release branch: {entry.description}")
            print("       Release branches only carry fixes, improvements and miscellaneous changes.")
            sys.exit(1)

    if latest is None or not any(entries.values()):
        print("No new changes to release since last tag.")
        sys.exit(0)

    # Find the latest patch version for this release branch
    tags = gl.list_tags()
    # Anchored so a pre-release tag (v1.1.2-rc0) cannot be read as a released patch
    version_pattern = re.compile(rf"{re.escape(base_version)}\.(\d+)$")
    patch_version = max(
        (int(match.group(1)) for tag in tags if (match := version_pattern.match(tag["name"]))),
        default=0,
    )

    # Calculate next patch version
    next_version = f"{base_version}.{patch_version + 1}"
    print(f"Next patch version: {next_version}")

    # Build changelog content
    lines: list[str] = []
    for category in sorted(entries):
        if not entries[category]:
            continue
        lines.append("")
        lines.append(_Category.to_markdown(category))
        lines.append("")
        for entry in entries[category]:
            lines.append(entry.to_markdown())
            print(f"Adding - {entry.to_markdown()}")

    header = [f"[//]: # ({latest.hash})", "", "# DataEval Change Log", "", f"## {next_version}"]
    changelog_content = "\n".join(header + lines) + "\n"

    for oldline in current[3:]:
        changelog_content += oldline

    # Create actions list
    actions = []

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

    # Trigger API pipeline on the release branch for docs build and artifact publishing
    # (push-triggered pipelines are skipped due to CHANGELOG.md change filter)
    gl.create_pipeline(branch_name)
    print(f"Triggered pipeline on {branch_name}")

#!/usr/bin/env python3
"""
Automatically cherry-pick release::fix commits from main to all active release branches.

This script identifies the most recent commit on main that has the 'release::fix' label
(via its associated merge request) and cherry-picks it to all active release/* branches,
creating a merge request for each release branch.
"""

if __name__ == "__main__":
    import contextlib
    import os
    import re
    import subprocess
    import sys
    from typing import Any

    from gitlab import Gitlab

    # Ensure we're running on main branch
    branch_name = os.getenv("CI_COMMIT_BRANCH", "")
    if branch_name != "main":
        print(f"INFO: This script only runs on main branch, got: {branch_name}")
        sys.exit(0)

    # Get the commit SHA that triggered this pipeline
    commit_sha = os.getenv("CI_COMMIT_SHA", "")
    commit_before_sha = os.getenv("CI_COMMIT_BEFORE_SHA", "")

    if not commit_sha:
        print("ERROR: CI_COMMIT_SHA environment variable not set")
        sys.exit(1)

    print(f"Processing commit {commit_sha[:8]} on main branch")

    gl = Gitlab(verbose=True)

    # Check if this is a merge commit by seeing if it has 2 parents
    try:
        parents_output = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", commit_sha],
            text=True,
        ).strip()
        parent_count = len(parents_output.split()) - 1  # First element is the commit itself

        if parent_count < 2:
            print(f"INFO: Commit {commit_sha[:8]} is not a merge commit (has {parent_count} parent(s))")
            print("Skipping cherry-pick automation (only runs on merge commits)")
            sys.exit(0)

        print(f"✓ Commit {commit_sha[:8]} is a merge commit with {parent_count} parents")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Could not check commit parents: {e}")

    # Find the merge request associated with this commit
    # We need to check if this commit was part of an MR with release::fix label
    print(f"\nSearching for merge request that created merge commit {commit_sha[:8]}...")

    # Get all recently merged MRs to main
    recent_mrs = gl.list_merge_requests(state="merged", target_branch="main", order_by="merged_at")

    # Find the MR that contains this commit
    target_mr: dict[str, Any] | None = None

    for mr in recent_mrs[:20]:  # Check the last 20 merged MRs
        # Only check merge_commit_sha (not regular sha) since we verified this is a merge commit
        if mr.get("merge_commit_sha") == commit_sha:
            target_mr = mr
            break

    if not target_mr:
        print(f"INFO: No merge request found for merge commit {commit_sha[:8]}")
        print("This might be a direct push to main or a fast-forward merge.")
        sys.exit(0)

    print(f"Found MR !{target_mr['iid']}: {target_mr['title']}")

    # Check if the MR has the release::fix label
    labels = target_mr.get("labels", [])
    print(f"MR labels: {labels}")

    has_fix_label = "release::fix" in labels

    if not has_fix_label:
        print(f"INFO: MR !{target_mr['iid']} does not have 'release::fix' label")
        print("Skipping cherry-pick to release branches")
        sys.exit(0)

    print(f"✓ MR !{target_mr['iid']} has 'release::fix' label")

    try:
        branches_output = subprocess.check_output(["git", "branch", "-r"], text=True).strip()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Could not list branches: {e}")
        sys.exit(1)

    # Filter for release branches
    release_branches = []
    release_pattern = re.compile(r"origin/(release/v\d+\.\d+)$")

    for line in branches_output.split("\n"):
        line = line.strip()
        match = release_pattern.match(line)
        if match:
            release_branches.append(match.group(1))

    if not release_branches:
        print("INFO: No active release branches found")
        sys.exit(0)

    print(f"\nFound {len(release_branches)} active release branch(es):")
    for branch in release_branches:
        print(f"  - {branch}")

    # For each release branch, cherry-pick the commit
    results = []

    for release_branch in release_branches:
        print(f"\n{'=' * 60}")
        print(f"Processing release branch: {release_branch}")
        print(f"{'=' * 60}")

        # Extract version from branch name
        version_match = re.match(r"release/(v\d+\.\d+)", release_branch)
        if not version_match:
            print(f"WARNING: Could not extract version from {release_branch}")
            continue

        base_version = version_match.group(1)

        # Create a cherry-pick branch for this release
        cherry_pick_branch = f"cherry-pick/fix-to-{base_version.replace('.', '-')}"

        # Check if the commit already exists in the release branch
        try:
            check_result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit_sha, release_branch],
                capture_output=True,
            )

            if check_result.returncode == 0:
                print(f"INFO: Commit {commit_sha[:8]} already exists in {release_branch}")
                results.append(
                    {
                        "release_branch": release_branch,
                        "status": "skipped",
                        "reason": "Commit already exists in release branch",
                    },
                )
                continue
        except Exception as e:
            print(f"WARNING: Could not check if commit exists: {e}")
        # Create or recreate the cherry-pick branch from the release branch
        try:
            # First, try to delete the branch if it exists
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["git", "push", "origin", "--delete", cherry_pick_branch],
                    capture_output=True,
                    check=False,  # Don't fail if branch doesn't exist
                )

            # Create the branch from the release branch
            result = gl.create_repository_branch(cherry_pick_branch, release_branch)
            print(f"Created branch {cherry_pick_branch} from {release_branch}")
        except Exception as e:
            print(f"ERROR: Could not create branch {cherry_pick_branch}: {e}")
            results.append(
                {"release_branch": release_branch, "status": "failed", "reason": f"Failed to create branch: {e}"},
            )
            continue

        # Cherry-pick the commit
        try:
            print(f"Cherry-picking {commit_sha[:8]} to {cherry_pick_branch}")
            cherry_result = gl.cherry_pick(commit_sha, cherry_pick_branch)
            print("✓ Successfully cherry-picked commit")
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Failed to cherry-pick: {error_msg}")
            results.append(
                {"release_branch": release_branch, "status": "failed", "reason": f"Cherry-pick failed: {error_msg}"},
            )
            continue

        # Create merge request
        mr_title = f"[Hotfix] Cherry-pick fix to {base_version}: {target_mr['title']}"

        mr_description = f"""## Hotfix Cherry-pick to {release_branch}

This MR cherry-picks a fix from main to the {release_branch} release branch.

**Original MR:** !{target_mr["iid"]} - {target_mr["title"]}
**Commit:** `{commit_sha[:8]}`
**Labels:** {", ".join(labels)}

### Changes
{target_mr.get("description", "No description provided")}

---
🤖 Auto-generated by release fix automation
"""

        # Check if MR already exists
        existing_mrs = gl.list_merge_requests(
            state="opened",
            source_branch=cherry_pick_branch,
            target_branch=release_branch,
        )

        try:
            if existing_mrs:
                # Update existing MR
                mr_iid = existing_mrs[0]["iid"]
                print(f"Updating existing MR !{mr_iid}")
                gl.update_merge_request(mr_iid, mr_title, mr_description)
                mr_url = existing_mrs[0]["web_url"]
                print(f"✓ Updated MR: {mr_url}")
                results.append(
                    {"release_branch": release_branch, "status": "updated", "mr_url": mr_url, "mr_iid": mr_iid},
                )
            else:
                # Create new MR
                print(f"Creating MR from {cherry_pick_branch} to {release_branch}")
                mr = gl.create_merge_request(
                    title=mr_title,
                    description=mr_description,
                    source_branch=cherry_pick_branch,
                    target_branch=release_branch,
                )
                mr_url = mr["web_url"]
                print(f"✓ Created MR: {mr_url}")
                results.append(
                    {"release_branch": release_branch, "status": "created", "mr_url": mr_url, "mr_iid": mr["iid"]},
                )
        except Exception as e:
            print(f"✗ Failed to create/update MR: {e}")
            results.append(
                {"release_branch": release_branch, "status": "failed", "reason": f"Failed to create MR: {e}"},
            )

    # Print summary
    print(f"\n\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    created = [r for r in results if r["status"] == "created"]
    updated = [r for r in results if r["status"] == "updated"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    if created:
        print(f"\n✓ Created {len(created)} new MR(s):")
        for r in created:
            print(f"  - {r['release_branch']}: !{r['mr_iid']} - {r['mr_url']}")

    if updated:
        print(f"\n↻ Updated {len(updated)} existing MR(s):")
        for r in updated:
            print(f"  - {r['release_branch']}: !{r['mr_iid']} - {r['mr_url']}")

    if skipped:
        print(f"\n⊘ Skipped {len(skipped)} release branch(es):")
        for r in skipped:
            print(f"  - {r['release_branch']}: {r['reason']}")

    if failed:
        print(f"\n✗ Failed {len(failed)} release branch(es):")
        for r in failed:
            print(f"  - {r['release_branch']}: {r['reason']}")

    # Exit with error if all failed
    if failed and not created and not updated:
        print("\n✗ All cherry-picks failed")
        sys.exit(1)

    print("\n✓ Release fix automation completed")

#!/usr/bin/env python3

import re

if __name__ == "__main__":
    from gitlab import Gitlab

    gl = Gitlab(verbose=True)

    # Get source branches from merged and opened MRs targeting main
    merged_set = {mr["source_branch"] for mr in gl.list_merge_requests("merged", "main")}
    opened_set = {mr["source_branch"] for mr in gl.list_merge_requests("opened", "main")}
    delete_set = merged_set - opened_set

    # List all docs-artifacts/* branches
    artifact_branches = gl.list_branches(search="docs-artifacts/")

    # Delete artifact branches whose source branch has been merged
    for branch_info in artifact_branches:
        branch_name = branch_info["name"]
        if not branch_name.startswith("docs-artifacts/"):
            continue
        source_branch = branch_name[len("docs-artifacts/") :]
        # Preserve versioned artifact branches (used for Colab links)
        if re.match(r"^v\d+\.\d+\.\d+", source_branch):
            continue
        # Preserve main and release artifact branches
        if source_branch == "main" or source_branch.startswith("release/"):
            continue
        if source_branch in delete_set:
            print(f"Removing artifact branch: {branch_name}")
            gl.delete_branch(branch_name)

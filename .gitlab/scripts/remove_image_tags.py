#!/usr/bin/env python3

if __name__ == "__main__":
    import re

    from gitlab import Gitlab
    from harbor import Harbor

    gl = Gitlab(verbose=True)
    hb = Harbor(verbose=True)

    merged_set = {mr["source_branch"] for mr in gl.list_merge_requests("merged", "main")}
    opened_set = {mr["source_branch"] for mr in gl.list_merge_requests("opened", "main")}
    delete_set = {re.sub(r"[^a-zA-Z0-9]+", "-", branch_name) for branch_name in (merged_set - opened_set)}

    # Build list of registry image tags that should be deleted
    repositories = []
    for branch_name in delete_set:
        for repository in hb.list_repositories(branch_name):
            repositories.append(repository["name"])

    # Delete the collected registry image tags
    for repository in repositories:
        print(f"Removing {repository}...")
        hb.delete_repository(repository)

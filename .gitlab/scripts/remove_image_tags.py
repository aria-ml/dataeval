#!/usr/bin/env python3

if __name__ == "__main__":
    import re

    from gitlab import Gitlab
    from harbor import Harbor

    gl = Gitlab(verbose=True)
    hb = Harbor(verbose=True)

    merged_set = {mr["source_branch"] for mr in gl.list_merge_requests("merged", "main")}
    opened_set = {mr["source_branch"] for mr in gl.list_merge_requests("opened", "main")}
    delete_set = {re.sub(r"[^a-zA-Z0-9]+", "-", branch) for branch in (merged_set - opened_set)}

    # Build list of registry image tags that should be deleted
    registry_tags = {"cache": set(), "dev": set()}
    for repository in registry_tags:
        for artifact in hb.list_artifacts(repository):
            for tag in artifact["tags"] or []:
                if any(branch in tag["name"] for branch in delete_set):
                    registry_tags[repository].add(tag["name"])

    # Delete the collected registry image tags
    for repository, tags in registry_tags.items():
        for tag in tags:
            print(f"Removing {tag} from {repository}...")
            hb.delete_tag(repository, tag)

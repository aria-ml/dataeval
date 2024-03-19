#!/usr/bin/env python3

if __name__ == "__main__":
    from changegen import CHANGELOG_FILE, ChangeGen
    from gitlab import Gitlab
    from versiontag import VersionTag

    gl = Gitlab(verbose=True)
    changelog = ChangeGen(gl).generate()
    if changelog:
        pending_version = VersionTag(gl).pending
        print(f"Updating changelog and tagging to {pending_version}:")
        print(changelog["content"])
        branch = "main"
        gl.push_file(CHANGELOG_FILE, branch, **changelog)
        response = gl.get_single_repository_branch(branch)
        commit_id = response["commit"]["id"]
        gl.add_tag(pending_version, commit_id, message=f"DAML {pending_version}")
    else:
        print("No changes to commit and tag.")

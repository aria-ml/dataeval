#!/usr/bin/env python3

if __name__ == "__main__":
    from commitgen import CommitGen
    from gitlab import Gitlab
    from versiontag import VersionTag

    gl = Gitlab(verbose=True)
    pending_version = VersionTag(gl).pending
    payload = CommitGen(pending_version, gl).generate()
    if payload:
        print(f"Updating jupyter cache and changelog and tagging to {pending_version}:")
        branch = "main"
        response = gl.commit(
            branch,
            f"Release {pending_version}",
            payload,
        )
        commit_id = response["id"]
        gl.add_tag(pending_version, commit_id, message=f"DAML {pending_version}")
    else:
        print("No changes to commit and tag.")

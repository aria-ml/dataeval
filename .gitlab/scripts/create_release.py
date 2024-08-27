#!/usr/bin/env python3

if __name__ == "__main__":
    from commitgen import CommitGen
    from gitlab import Gitlab

    gl = Gitlab(verbose=True)
    commit_gen = CommitGen(gl)
    payload = commit_gen.generate()  # This computes and sets the pending version
    pending_version = commit_gen._pending
    if payload:
        print(f"Updating jupyter cache and changelog and tagging to {pending_version}:")
        branch = "main"
        response = gl.commit(
            branch,
            f"Release {pending_version}",
            payload,
        )
        commit_id = response["id"]
        gl.add_tag(pending_version, commit_id, message=f"DataEval {pending_version}")
    else:
        print("No changes to commit and tag.")

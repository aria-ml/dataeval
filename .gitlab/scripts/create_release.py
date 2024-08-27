#!/usr/bin/env python3

if __name__ == "__main__":
    from gitlab import Gitlab
    from releasegen import ReleaseGen

    gl = Gitlab(verbose=True)
    rg = ReleaseGen(gl)
    tag, payload = rg.generate()  # This computes and sets the pending version
    if payload:
        print(f"Updating jupyter cache and changelog and tagging to {tag}:")
        branch = "main"
        response = gl.commit(
            branch,
            f"Release {tag}",
            payload,
        )
        commit_id = response["id"]
        gl.add_tag(tag, commit_id, message=f"DataEval {tag}")
    else:
        print("No changes to commit and tag.")

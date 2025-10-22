#!/usr/bin/env python3

if __name__ == "__main__":
    from gitlab import Gitlab
    from releasegen import ReleaseGen

    gl = Gitlab(verbose=True)
    rg = ReleaseGen(gl)
    version_tag, payload = rg.generate()  # This computes and sets the pending version
    if version_tag and payload:
        print(f"Updating jupyter cache and changelog and tagging to {version_tag}:")
        commit_id = gl.commit("main", f"Release {version_tag}", payload)["id"]
        # Create release branch with vX.X format (strip patch version)
        major_minor = ".".join(version_tag.split(".")[:2])
        gl.create_repository_branch(f"release/{major_minor}", commit_id)
        gl.add_tag(version_tag, commit_id, message=f"DataEval {version_tag}")
    else:
        print("No changes to commit and tag.")

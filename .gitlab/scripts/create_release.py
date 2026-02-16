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
        # Branch may already exist from a pre-release
        major_minor = ".".join(version_tag.split(".")[:2])
        release_branch = f"release/{major_minor}"
        try:
            gl.create_repository_branch(release_branch, commit_id)
            print(f"Created release branch: {release_branch}")
        except ConnectionError:
            print(f"Release branch '{release_branch}' already exists, skipping creation")
        # Tag before triggering pipeline so push-docs-cache.sh can detect the version tag
        gl.add_tag(version_tag, commit_id, message=f"DataEval {version_tag}")
        print(f"Created tag: {version_tag}")
        # Trigger pipeline on release branch for docs build and artifact publishing
        gl.create_pipeline(release_branch)
        print(f"Triggered pipeline on {release_branch}")
    else:
        print("No changes to commit and tag.")

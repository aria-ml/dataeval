#!/usr/bin/env python3

if __name__ == "__main__":
    from gitlab import Gitlab
    from releasegen import ReleaseGen

    gl = Gitlab(verbose=True)
    rg = ReleaseGen(gl)
    version_tag, payload = rg.generate()  # This computes and sets the pending version

    # Bail out before committing if the tag already exists (e.g. a concurrent
    # pipeline won the race) - otherwise the commit lands on main untagged
    if version_tag and gl.tag_exists(version_tag):
        raise SystemExit(f"Tag {version_tag} already exists - another pipeline created it. Nothing to do.")

    if version_tag and payload:
        print(f"Updating jupyter cache and changelog and tagging to {version_tag}:")
        commit_id = gl.commit("main", f"Release {version_tag}", payload)["id"]
        # Tag before triggering pipeline so push-docs-cache.sh can detect the version tag
        gl.add_tag(version_tag, commit_id, message=f"DataEval {version_tag}")
        print(f"Created tag: {version_tag}")

        # Automatically create the release branch for the new major/minor version
        import re

        match = re.match(r"^(v\d+\.\d+)", version_tag)
        if match:
            release_branch = f"release/{match.group(1)}"
            try:
                gl.create_repository_branch(release_branch, commit_id)
                print(f"Created release branch: {release_branch}")
            except Exception as e:
                print(f"Warning: Could not create release branch {release_branch}: {e}")

        # Trigger API pipeline on main for docs build and artifact publishing
        gl.create_pipeline("main")
        print("Triggered pipeline on main")
    else:
        print("No changes to commit and tag.")

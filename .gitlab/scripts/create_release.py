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
        # Tag before triggering pipeline so push-docs-cache.sh can detect the version tag
        gl.add_tag(version_tag, commit_id, message=f"DataEval {version_tag}")
        print(f"Created tag: {version_tag}")
        # Trigger API pipeline on main for docs build and artifact publishing
        gl.create_pipeline("main")
        print("Triggered pipeline on main")
    else:
        print("No changes to commit and tag.")

#!/usr/bin/env python3

if __name__ == "__main__":
    from gitlab import Gitlab
    from harbor import Harbor

    gl = Gitlab(verbose=True)
    hb = Harbor(verbose=True)

    merged_branch = gl.list_merge_requests("merged", "main")[0]["source_branch"]
    repository_tags = {"cache": [], "dev": []}
    for repository in repository_tags:
        for artifacts in hb.list_artifacts(repository, merged_branch):
            for tag in artifacts["tags"]:
                if merged_branch in tag["name"]:
                    repository_tags[repository].append(tag["name"])

    for repository, tags in repository_tags.items():
        for tag in tags:
            print(f"Remove {tag} from {repository}")
            # Harbor 2.10 does not provide full permissions to OIDC robot accounts
            # https://github.com/goharbor/harbor/issues/8723
            # hb.delete_tag(repository, tag)

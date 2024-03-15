#!/usr/bin/env python3

import argparse

from changegen import CHANGELOG_FILE, ChangeGen
from gitlab import Gitlab
from verboselog import set_verbose
from versiontag import VersionTag

BUMP_VERSION = "bump_version"
UPDATE_CHANGELOG = "update_changelog"
CREATE_MR = "create_mr"
ACTIONS = [BUMP_VERSION, UPDATE_CHANGELOG, CREATE_MR]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAML Release Utilities")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose")
    parser.add_argument("--commit", action="store_true", dest="commit")
    parser.add_argument("action", choices=ACTIONS)
    args = parser.parse_args()

    set_verbose(args.verbose)
    gl = Gitlab(verbose=args.verbose)
    vt = VersionTag(gl)
    action = args.action
    response = dict()
    if action == BUMP_VERSION:
        if args.commit:
            print(f"Bumping version from {vt.current} to {vt.pending}...")
            response = gl.add_tag(vt.pending, message=f"DAML {vt.pending}")
        else:
            print(f"Current version: {vt.current} Pending version: {vt.pending}")
    elif action == UPDATE_CHANGELOG:
        cg = ChangeGen(gl)
        change = cg.generate("changelog")
        if change:
            print("Updating changelog file with following content:")
            print(change["content"])
            if args.commit:
                branch = "main"
                gl.push_file(CHANGELOG_FILE, branch, **change)
                response = gl.get_single_repository_branch(branch)
                commit_id = response["commit"]["id"]
                gl.create_repository_branch(f"releases/{vt.pending}", commit_id)
        else:
            print("Current changelog is up to date.")
    elif action == CREATE_MR:
        cg = ChangeGen(gl)
        title = f"Release {vt.pending}"
        merge = cg.generate("merge")
        existing = gl.list_merge_requests(
            "opened",
            target_branch="main",
            source_branch="develop",
            search_title="Release",
        )
        if merge:
            if existing:
                print("Updating existing merge request with following content:")
                print(merge["description"])
                if args.commit:
                    response = gl.update_mr(
                        existing[0]["iid"], title, merge["description"]
                    )
            else:
                print("Creating merge request with following content:")
                print(merge["description"])
                if args.commit:
                    response = gl.create_mr(title, merge["description"])
        else:
            print("No changes to merge.")

    if args.verbose:
        print(response)

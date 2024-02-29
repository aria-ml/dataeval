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
    action = args.action
    response = dict()
    if action == BUMP_VERSION:
        vt = VersionTag(gl)
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
                response_push = gl.push_file(CHANGELOG_FILE, "develop", **change)
                file_info = gl.get_file_info(CHANGELOG_FILE, "develop")
                response_cherry_pick = gl.cherry_pick(file_info["commit_id"])
                response = {"push": response_push, "cherry-pick": response_cherry_pick}
        else:
            print("Current changelog is up to date.")
    elif action == CREATE_MR:
        cg = ChangeGen(gl)
        vt = VersionTag(gl)
        title = f"Release {vt.pending}"
        merge = cg.generate("merge")
        if merge:
            print("Creating merge request with following content:")
            print(merge["description"])
            if args.commit:
                response = gl.create_mr(title, merge["description"])
        else:
            print("No changes to merge.")
    if args.verbose:
        print(response)

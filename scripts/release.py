#!/usr/bin/env python3
"""
Cut a release from the branch you are standing on.

    python scripts/release.py                 # main -> minor, release/vN.N -> patch
    python scripts/release.py major           # main only
    python scripts/release.py prerelease      # -rc0, or bump the current prerelease
    python scripts/release.py prerelease a    # -a0 alpha snapshot
    python scripts/release.py --dry-run       # print what would happen, touch nothing

Updates CHANGELOG.md and the notebook links in the docs indexes, commits, and tags.
It never pushes: review the commit, then `git push --follow-tags` to publish. Pushing
the tag is what triggers PyPI publication and the docs build.

Needs nothing but git and a Python interpreter - no credentials, no GitLab API.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHANGELOG_FILE = REPO / "CHANGELOG.md"
DOC_INDEX_FILES = (REPO / "docs/source/how-to/index.md", REPO / "docs/source/tutorials/index.md")

# Prerelease kinds in PEP 440 precedence order: v1.0.0-a0 < v1.0.0-rc0 < v1.0.0
PRERELEASE_KINDS = ("a", "rc")
VERSION_RE = re.compile(rf"^v(\d+)\.(\d+)\.(\d+)(?:-({'|'.join(PRERELEASE_KINDS)})(\d+))?$")

# Commit titles this script writes for its own bookkeeping - never changelog entries.
RELEASE_COMMIT_RE = re.compile(r"^(Release|Prerelease) v\d+\.\d+\.\d+")

# Changelog section headings, ordered by precedence. The first category present decides
# whether an unqualified release off main is a major or a minor bump.
CATEGORIES = (
    ("major", "🚀 **Major Release**"),
    ("feature", "🌟 **Feature Release**"),
    ("deprecation", "🚧 **Deprecations and Removals**"),
    ("improvement", "🛠️ **Improvements and Enhancements**"),
    ("fix", "👾 **Fixes**"),
    ("misc", "📝 **Miscellaneous**"),
)
CATEGORY_ORDER = [name for name, _ in CATEGORIES]

# Commit message `[tag]` prefix -> changelog category. This is the whole accepted
# vocabulary: `.gitlab/scripts/validate_commit_prefix.py` rejects a merge request whose
# title uses anything else, which is what stops a typo like `[imrp]` reaching main.
# Reading history is more forgiving - an unrecognized prefix on an old commit falls
# through to misc rather than being dropped.
TAG_CATEGORIES = {
    "major": "major",
    "feat": "feature",
    "feature": "feature",
    "depr": "deprecation",
    "deprecate": "deprecation",
    "deprecation": "deprecation",
    "impr": "improvement",
    "improvement": "improvement",
    "enh": "improvement",
    "perf": "improvement",
    "fix": "fix",
    "bugfix": "fix",
    # Housekeeping - real changes, but not ones that move the public API
    "deps": "misc",
    "devops": "misc",
    "devsecops": "misc",
    "docs": "misc",
    "lint": "misc",
    "misc": "misc",
    "test": "misc",
    "type": "misc",
}

# Categories that change the public API, so they may not ship in a patch off a release branch.
API_CHANGING = ("major", "feature", "deprecation")


def git(*args: str) -> str:
    """Run a git command, or abort with git's own message - a bare traceback hides it."""
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"git {' '.join(args)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def git_ok(*args: str) -> str | None:
    """Run a git command that is allowed to fail; None when it does."""
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def fail(message: str) -> None:
    sys.exit(f"ERROR: {message}")


def parse_version(tag: str) -> tuple[int, int, int, int, int] | None:
    """Sort key for a version tag, or None if it is not one.

    A prerelease sorts before its own release, so the kind index defaults to len(kinds)
    for a full release and the sequence number is only a tiebreaker within a kind.
    """
    match = VERSION_RE.match(tag)
    if match is None:
        return None
    major, minor, patch, kind, number = match.groups()
    kind_rank = PRERELEASE_KINDS.index(kind) if kind else len(PRERELEASE_KINDS)
    return (int(major), int(minor), int(patch), kind_rank, int(number or 0))


def latest_tag(series: str | None = None) -> str | None:
    """Highest version tag reachable from HEAD, optionally limited to a `vN.N` series."""
    tags = [t for t in git("tag", "--merged", "HEAD").splitlines() if parse_version(t)]
    if series is not None:
        tags = [t for t in tags if t.startswith(f"{series}.")]
    return max(tags, key=lambda t: parse_version(t), default=None) if tags else None


def collect_entries(since: str | None) -> dict[str, list[tuple[str, str]]]:
    """Group `(shorthash, description)` per category for first-parent commits since `since`."""
    span = f"{since}..HEAD" if since else "HEAD"
    # \x1f separates fields, \x1e separates records, so subjects containing either are safe
    raw = git("log", "--first-parent", span, "--format=%h%x1f%p%x1f%B%x1e")

    entries: dict[str, list[tuple[str, str]]] = {name: [] for name in CATEGORY_ORDER}
    for record in filter(None, (r.strip("\n") for r in raw.split("\x1e"))):
        shorthash, parents, body = record.strip().split("\x1f", 2)
        lines = body.strip().splitlines()

        # A GitLab merge commit reads "Merge branch 'x' into 'main'" with the merge request
        # title on the third line; that title is the change, the merge subject is noise.
        description = lines[2].strip() if len(parents.split()) > 1 and len(lines) > 2 else lines[0].strip()

        if RELEASE_COMMIT_RE.match(description):
            continue

        prefix = re.match(r"\s*\[([a-zA-Z]+)\]", description)
        entries[TAG_CATEGORIES.get(prefix[1].lower() if prefix else "", "misc")].append((shorthash, description))

    return {name: found for name, found in entries.items() if found}


def next_version(current: str | None, bump: str, kind: str | None) -> str:
    """Apply `bump` to `current`, as a full release or as a prerelease of `kind`."""
    major, minor, patch, kind_rank, number = parse_version(current) if current else (0, 0, 0, len(PRERELEASE_KINDS), 0)
    current_kind = PRERELEASE_KINDS[kind_rank] if kind_rank < len(PRERELEASE_KINDS) else None

    if current_kind is not None:
        base = f"v{major}.{minor}.{patch}"
        if kind is None:
            return base  # finalizing: v1.2.0-rc3 -> v1.2.0
        if kind == current_kind:
            return f"{base}-{kind}{number + 1}"
        if PRERELEASE_KINDS.index(kind) > kind_rank:
            return f"{base}-{kind}0"
        # Going back down the ladder publishes a version that sorts before one already out
        fail(f"cannot cut a '{kind}' prerelease after {current}: {base}-{kind}0 sorts before it")

    bumped = {
        "major": f"v{major + 1}.0.0",
        "minor": f"v{major}.{minor + 1}.0",
        "patch": f"v{major}.{minor}.{patch + 1}",
    }[bump]
    return f"{bumped}-{kind}0" if kind else bumped


def render_section(version: str, entries: dict[str, list[tuple[str, str]]]) -> str:
    lines = [f"## {version}"]
    for name, heading in CATEGORIES:
        if name not in entries:
            continue
        lines += ["", heading, ""]
        lines += [f"- `{shorthash}` - {description}" for shorthash, description in entries[name]]
    return "\n".join(lines)


def update_changelog(version: str, section: str, head: str) -> None:
    lines = CHANGELOG_FILE.read_text().splitlines(keepends=True)
    # ponytail: the `[//]: # (sha)` marker is only read by the GitLab CI release scripts.
    # Drop the marker, and this line, once those are deleted.
    header = f"[//]: # ({head})\n\n# DataEval Change Log\n"
    CHANGELOG_FILE.write_text(header + f"\n{section}\n" + "".join(lines[3:]))


def update_doc_indexes(version: str) -> list[Path]:
    pattern = re.compile(
        rf"aria-ml/dataeval/blob/docs-artifacts/(?:main|v\d+\.\d+\.\d+(?:-(?:{'|'.join(PRERELEASE_KINDS)})\d+)?)/notebooks"
    )
    replacement = f"aria-ml/dataeval/blob/docs-artifacts/{version}/notebooks"

    changed = []
    for index_file in DOC_INDEX_FILES:
        original = index_file.read_text()
        updated = pattern.sub(replacement, original)
        if updated != original:
            index_file.write_text(updated)
            changed.append(index_file)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "bump", nargs="?", choices=["major", "prerelease"], help="default: minor on main, patch on a release branch"
    )
    parser.add_argument(
        "kind", nargs="?", choices=list(PRERELEASE_KINDS), default="rc", help="prerelease kind (default: rc)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the release and change nothing")
    args = parser.parse_args()

    if git("status", "--porcelain"):
        fail("working tree is not clean - commit or stash first")

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    release_series = re.match(r"^release/(v\d+\.\d+)$", branch)
    if branch != "main" and not release_series:
        fail(f"release from 'main' or a 'release/vN.N' branch, not {branch!r}")

    if args.bump == "major" and release_series:
        fail("a release branch only carries patches - cut a major release from main")

    # A stale branch recomputes a version that is already published, because the tags that
    # decide the next version sit on commits it has not fetched yet.
    upstream = git_ok("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if upstream and git("rev-list", "--count", f"HEAD..{upstream}") != "0":
        fail(f"{branch} is behind {upstream} - `git pull` before cutting a release")

    series = release_series[1] if release_series else None
    current = latest_tag(series)
    if current and git("rev-list", "-n", "1", current) == git("rev-parse", "HEAD"):
        fail(f"HEAD is already tagged {current}")

    entries = collect_entries(current)
    if not entries:
        print(f"Nothing new since {current or 'the start of history'} - no release to cut.")
        return

    if release_series:
        blocked = [d for name in API_CHANGING for _, d in entries.get(name, [])]
        if blocked:
            fail("release branches carry fixes only; these belong on main:\n  " + "\n  ".join(blocked))

    # An unqualified release off main follows its entries: a [major] commit forces the
    # caller to say so out loud rather than shipping a breaking change as a minor.
    if args.bump != "prerelease" and "major" in entries and args.bump != "major":
        fail("a [major] change is pending - run `release.py major` to confirm the bump")

    bump = "patch" if release_series else ("major" if args.bump == "major" else "minor")
    version = next_version(current, bump, args.kind if args.bump == "prerelease" else None)
    # Checked before anything is written: tagging is the last step, and discovering the
    # collision there would leave a release commit behind with no tag on it.
    if git_ok("rev-parse", "-q", "--verify", f"refs/tags/{version}") is not None:
        fail(f"tag {version} already exists - fetch tags and check what has been released")
    section = render_section(version, entries)

    print(f"{branch}: {current or 'no tag'} -> {version}\n\n{section}\n")

    if args.dry_run:
        print("--dry-run: nothing written.")
        return

    update_changelog(version, section, git("rev-parse", "HEAD"))
    changed = update_doc_indexes(version)

    label = "Prerelease" if args.bump == "prerelease" else "Release"
    git("add", str(CHANGELOG_FILE), *(str(p) for p in changed))
    git("commit", "-m", f"{label} {version}")
    git("tag", "-a", version, "-m", f"DataEval {version}")

    print(f"Committed and tagged {version}. Review it:\n")
    print(f"    git show {version}\n")
    print("Then publish - pushing the tag is what triggers PyPI and the docs build:\n")
    print(f"    git push --follow-tags origin {branch}\n")
    print(f"To back out:\n\n    git tag -d {version} && git reset --hard HEAD~1")


if __name__ == "__main__":
    main()

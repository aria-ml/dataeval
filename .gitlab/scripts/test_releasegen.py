#!/usr/bin/env python3
"""
Self-check for changelog entry collection. Run directly: `python .gitlab/scripts/test_releasegen.py`

Not picked up by the project test suite (pyproject restricts testpaths to `tests`), so it stays
runnable without pulling the release scripts into the library's dependency set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from releasegen import ReleaseGen, _Category

BRANCH = "release/v1.1"
BOUNDARY = "a" * 40
MERGE_SHA = "beeeeeef".ljust(40, "0")

CHANGELOG = [
    f"[//]: # ({BOUNDARY})\n",
    "\n",
    "# DataEval Change Log\n",
    "\n",
    "## v1.1.0\n",
    "\n",
    "👾 **Fixes**\n",
    "\n",
    "- `0943a51b` - [fix] Pass dtype to ensure behavior stable on all versions of NumPy\n",
]


def _commit(sha: str, title: str, day: int, parents: int = 1) -> dict:
    return {
        "id": sha.ljust(40, "0"),
        "title": title,
        "committed_date": f"2026-09-{day:02d}T12:00:00.000+00:00",
        "parent_ids": ["p"] * parents,
    }


BRANCH_COMMITS = [
    _commit(BOUNDARY, "Merge branch 'doc-updates' into 'main'", 1, parents=2),
    _commit("0943a51b", "[fix] Pass dtype to ensure behavior stable on all versions of NumPy", 2),
    _commit("41bf3a72", "Release v1.1.0", 3),
    _commit("02bacb78", "[fix] Allow merge_datasets to accept packed collections of datasets", 4),
    _commit(MERGE_SHA, "Merge branch 'hotfix' into 'release/v1.1'", 5, parents=2),
    _commit("0ff0fbf3", "[misc] Mirror MAITE protocols to enable stable isinstance checks", 6),
    _commit("a36422a9", "[fix] MaskedTarget is not properly wrapping MAITE targets", 7),
]

MERGE_REQUEST = {
    "merged_at": "2026-09-05T12:00:00.000Z",
    "merge_commit_sha": MERGE_SHA,
    "title": 'Resolve "A merged hotfix"',
    "labels": ["release::fix"],
    "description": "",
    "target_branch": BRANCH,
}


class FakeGitlab:
    """Minimal stand-in for the endpoints _get_entries touches."""

    def __init__(self, merge_requests: list[dict], commits: list[dict]) -> None:
        self.merge_requests = merge_requests
        self.commits = commits
        self.listed_refs: list[str] = []

    def list_merge_requests(self, state=None, target_branch=None, **_):
        return [mr for mr in self.merge_requests if mr["target_branch"] == target_branch]

    def get_commit(self, sha):
        return next(c for c in self.commits if c["id"] == sha)

    def list_commits(self, ref_name="main", since=None, first_parent=False):
        self.listed_refs.append(ref_name)
        return [c for c in self.commits if since is None or c["committed_date"] >= since]


def _descriptions(entries):
    return [entry.description for entry in entries]


def test_release_branch_collects_pushes_and_merges():
    gl = FakeGitlab([MERGE_REQUEST], BRANCH_COMMITS)
    latest, entries = ReleaseGen(gl)._get_entries(BOUNDARY, CHANGELOG, branch=BRANCH)

    assert gl.listed_refs == [BRANCH], f"walked {gl.listed_refs}, expected the release branch"

    # newest first, direct pushes and the merge request interleaved by commit time
    assert _descriptions(entries[_Category.FIX]) == [
        "[fix] MaskedTarget is not properly wrapping MAITE targets",
        "A merged hotfix",
        "[fix] Allow merge_datasets to accept packed collections of datasets",
    ], _descriptions(entries[_Category.FIX])
    assert _descriptions(entries[_Category.MISCELLANEOUS]) == [
        "[misc] Mirror MAITE protocols to enable stable isinstance checks",
    ]

    # the boundary commit, the release commit and the already-recorded fix stay out,
    # and the merge commit is counted once (as the merge request, not as a commit)
    recorded = [d for cat in entries.values() for d in _descriptions(cat)]
    assert len(recorded) == 4, recorded
    assert not any("Release v1.1.0" in d or "Pass dtype" in d or "doc-updates" in d for d in recorded), recorded

    # the boundary for the next release is the newest entry, not the newest merge request
    assert latest is not None, "expected a boundary entry"
    assert latest.shorthash == "a36422a9", latest


def test_no_new_changes_yields_no_release():
    released = [c for c in BRANCH_COMMITS if c["committed_date"] <= "2026-09-03"]
    gl = FakeGitlab([], released)
    latest, entries = ReleaseGen(gl)._get_entries(BOUNDARY, CHANGELOG, branch=BRANCH)

    assert latest is None, latest
    assert not any(entries.values()), dict(entries)


def test_feature_pushes_are_categorized_for_rejection():
    # create_patch_release.py refuses to cut a patch when one of these shows up
    assert _Category.from_commit_message("[feat] Add a thing") == _Category.FEATURE
    assert _Category.from_commit_message("[fix] Fix a thing") == _Category.FIX
    assert _Category.from_commit_message("[type] Adjust a protocol") == _Category.MISCELLANEOUS


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok - {name}")

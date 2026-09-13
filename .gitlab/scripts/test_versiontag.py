#!/usr/bin/env python3
"""
Self-check for prerelease version arithmetic. Not collected by pytest (testpaths
is "tests"); run it directly:

    python .gitlab/scripts/test_versiontag.py
"""

import sys
from os import path

sys.path.insert(0, path.dirname(path.abspath(__file__)))

from releasegen import RELEASE_COMMIT_PREFIXES, _get_version_tuple, prerelease_entry_pattern
from versiontag import VersionTag


class FakeGitlab:
    """Stands in for Gitlab - VersionTag only ever calls list_tags()."""

    def __init__(self, *tags: str) -> None:
        self.tags = [{"name": t} for t in tags]

    def list_tags(self) -> list[dict[str, str]]:
        return self.tags


def vt(*tags: str) -> VersionTag:
    return VersionTag(FakeGitlab(*tags))  # type: ignore[arg-type]


def test_parsing() -> None:
    assert vt("v1.2.0").current_base == "v1.2.0"
    assert vt("v1.2.0-a3").current_base == "v1.2.0"
    assert vt("v1.2.0-rc3").current_base == "v1.2.0"

    assert vt("v1.2.0").prerelease_kind is None
    assert vt("v1.2.0-a3").prerelease_kind == "a"
    assert vt("v1.2.0-rc3").prerelease_kind == "rc"

    assert not vt("v1.2.0").is_prerelease
    assert vt("v1.2.0-a3").is_prerelease

    # non-version tags are skipped in favour of the first real version
    assert vt("latest-known-good", "v1.2.0-a0").current == "v1.2.0-a0"


def test_new_prerelease_from_release() -> None:
    assert vt("v1.1.0").next_prerelease("MINOR", "a") == "v1.2.0-a0"
    assert vt("v1.1.0").next_prerelease("MINOR", "rc") == "v1.2.0-rc0"
    assert vt("v1.1.0").next_prerelease("PATCH", "a") == "v1.1.1-a0"
    assert vt("v1.1.0").next_prerelease("MAJOR", "a") == "v2.0.0-a0"
    # rc is the default kind, preserving the pre-alpha behaviour
    assert vt("v1.1.0").next_prerelease("MINOR") == "v1.2.0-rc0"
    # multi-digit components survive the round trip
    assert vt("v9.10.11").next_prerelease("MAJOR", "a") == "v10.0.0-a0"


def test_increment_same_kind() -> None:
    assert vt("v1.2.0-a0").next_prerelease("MINOR", "a") == "v1.2.0-a1"
    assert vt("v1.2.0-a9").next_prerelease("MINOR", "a") == "v1.2.0-a10"
    assert vt("v1.2.0-rc1").next_prerelease("MINOR", "rc") == "v1.2.0-rc2"
    # the version type is ignored once a prerelease line is open
    assert vt("v1.2.0-a0").next_prerelease("MAJOR", "a") == "v1.2.0-a1"


def test_promote_alpha_to_rc() -> None:
    # promoting restarts the sequence on the SAME base version, not the next one
    assert vt("v1.2.0-a3").next_prerelease("MINOR", "rc") == "v1.2.0-rc0"


def test_demotion_is_refused() -> None:
    for kind, tag in (("a", "v1.2.0-rc1"), ("b", "v1.1.0")):
        try:
            vt(tag).next_prerelease("MINOR", kind)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for kind={kind!r} after {tag}")


def test_finalize() -> None:
    assert vt("v1.2.0-a2").next("MINOR") == "v1.2.0"
    assert vt("v1.2.0-rc1").next("MINOR") == "v1.2.0"
    # a full release still bumps normally
    assert vt("v1.1.0").next("MINOR") == "v1.2.0"
    assert vt("v1.1.0").next("PATCH") == "v1.1.1"
    assert vt("v1.1.0").next("MAJOR") == "v2.0.0"


def test_changelog_version_ordering() -> None:
    versions = ["v1.2.0", "v1.2.0-rc0", "v1.2.0-a1", "v1.2.0-a0", "v1.1.9"]
    ordered = sorted(versions, key=lambda v: _get_version_tuple(v) or ())
    assert ordered == ["v1.1.9", "v1.2.0-a0", "v1.2.0-a1", "v1.2.0-rc0", "v1.2.0"]
    assert _get_version_tuple("not-a-version") is None


def test_bookkeeping_entries_are_recognized() -> None:
    # both the current and the pre-rename spelling, for both kinds
    for line in (
        "- `000b2d21` - Prerelease v1.1.0-a5",
        "- `000b2d21` - Prerelease v1.1.0-rc5",
        "- `000b2d21` - Pre-release v1.1.0-rc5",
    ):
        assert prerelease_entry_pattern.match(line), line
    assert not prerelease_entry_pattern.match("- `000b2d21` - Fix the prerelease v1.1.0-rc5 handling")

    for title in ("Release v1.1.0", "Prerelease v1.1.0-a0", "Pre-release v1.1.0-rc5"):
        assert title.startswith(RELEASE_COMMIT_PREFIXES), title
    assert not "Releasing the hounds".startswith(RELEASE_COMMIT_PREFIXES)


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok - {name}")
    print("all passed")

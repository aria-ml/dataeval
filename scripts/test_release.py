#!/usr/bin/env python3
"""
Self-check for release's version arithmetic and categorization.

Run directly: `python scripts/test_release.py`. Deliberately not a pytest suite -
pyproject restricts testpaths to `tests`, and this needs to run anywhere git does.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import release
from release import TAG_CATEGORIES, next_version, parse_version, render_section


def test_version_ladder():
    cases = {
        # (current, bump, prerelease kind): expected
        ("v1.1.1", "patch", None): "v1.1.2",
        ("v1.1.0", "minor", None): "v1.2.0",
        ("v1.9.3", "major", None): "v2.0.0",
        ("v9.0.0", "major", None): "v10.0.0",  # multi-digit majors keep counting
        ("v1.1.0", "minor", "rc"): "v1.2.0-rc0",
        ("v1.2.0-rc0", "minor", "rc"): "v1.2.0-rc1",
        ("v1.2.0-a3", "minor", "a"): "v1.2.0-a4",
        ("v1.2.0-a3", "minor", "rc"): "v1.2.0-rc0",  # promotion restarts the sequence
        ("v1.2.0-rc2", "minor", None): "v1.2.0",  # finalizing drops the suffix
        ("v1.2.0-rc2", "patch", None): "v1.2.0",  # ...and does not also bump
        ("v1.1.0", "major", "rc"): "v2.0.0-rc0",  # a prerelease carries the major bump with it
    }
    for (current, bump, kind), expected in cases.items():
        actual = next_version(current, bump, kind)
        assert actual == expected, f"{current} +{bump} kind={kind}: expected {expected}, got {actual}"


def test_prerelease_sorts_before_its_release():
    order = ["v1.1.0", "v1.2.0-a0", "v1.2.0-a1", "v1.2.0-rc0", "v1.2.0", "v1.2.1"]
    keys = [parse_version(tag) for tag in order]
    assert keys == sorted(keys), f"tags sort out of order: {order}"
    assert parse_version("not-a-tag") is None
    assert parse_version("v1.2.0-beta0") is None  # unknown kinds are not version tags


def test_tag_vocabulary():
    assert TAG_CATEGORIES["fix"] == "fix"
    assert TAG_CATEGORIES["feat"] == "feature"
    assert TAG_CATEGORIES["depr"] == "deprecation"
    assert TAG_CATEGORIES["perf"] == "improvement"
    # housekeeping tags are valid but never move the public API
    for tag in ("docs", "test", "deps", "type", "devops", "devsecops", "lint", "misc"):
        assert TAG_CATEGORIES[tag] == "misc", tag
    assert "imrp" not in TAG_CATEGORIES  # the typo stays invalid on purpose


def test_section_renders_in_precedence_order():
    section = render_section(
        "v1.2.0",
        {"misc": [("cccccccc", "[misc] Tidy")], "fix": [("bbbbbbbb", "[fix] Repair")]},
    )
    assert section.splitlines()[0] == "## v1.2.0"
    assert section.index("👾 **Fixes**") < section.index("📝 **Miscellaneous**"), section
    assert "- `bbbbbbbb` - [fix] Repair" in section
    # a category with no entries contributes no heading
    assert "🌟" not in section


def test_changelog_splice_keeps_history():
    original = "# DataEval Change Log\n\n## v1.1.0\n\n\U0001f47e **Fixes**\n\n- `aaaaaaaa` - [fix] Old\n"
    with tempfile.TemporaryDirectory() as tmp:
        changelog = Path(tmp) / "CHANGELOG.md"
        changelog.write_text(original)
        release.CHANGELOG_FILE = changelog
        release.update_changelog(render_section("v1.2.0", {"fix": [("bbbbbbbb", "[fix] New")]}))
        result = changelog.read_text()

    assert result.startswith("# DataEval Change Log\n\n## v1.2.0\n"), result
    # the previous release survives intact, one blank line below the new section
    assert "- `bbbbbbbb` - [fix] New\n\n## v1.1.0\n" in result, result
    assert "- `aaaaaaaa` - [fix] Old" in result


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok - {name}")

#!/usr/bin/env python3
"""
Test script to validate pre-release logic without triggering actual releases.
Run from the .gitlab/scripts directory.
"""

import sys
from unittest.mock import MagicMock, patch


def test_version_patterns():
    """Test that version patterns match correctly."""
    print("\n=== Testing Version Patterns ===")

    from versiontag import VERSION_PATTERN, PRERELEASE_PATTERN

    # Standard versions should match VERSION_PATTERN
    standard_versions = ["v1.0.0", "v0.99.0", "v12.34.56"]
    for v in standard_versions:
        match = VERSION_PATTERN.match(v)
        assert match is not None, f"VERSION_PATTERN should match {v}"
        assert PRERELEASE_PATTERN.match(v) is None, f"PRERELEASE_PATTERN should NOT match {v}"
        print(f"  ✓ {v} - correctly identified as standard version")

    # Pre-release versions should match PRERELEASE_PATTERN
    prerelease_versions = ["v1.0.0-rc0", "v1.0.0-rc1", "v2.5.0-rc99"]
    for v in prerelease_versions:
        match = PRERELEASE_PATTERN.match(v)
        assert match is not None, f"PRERELEASE_PATTERN should match {v}"
        print(f"  ✓ {v} - correctly identified as pre-release version")

    print("  All pattern tests passed!")


def test_releasegen_version_pattern():
    """Test the releasegen version pattern handles both formats."""
    print("\n=== Testing ReleaseGen Version Pattern ===")

    from releasegen import version_pattern, _get_version_tuple

    test_cases = [
        ("v1.0.0", (1, 0, 0, None)),
        ("v0.99.0", (0, 99, 0, None)),
        ("v1.0.0-rc0", (1, 0, 0, 0)),
        ("v1.0.0-rc1", (1, 0, 0, 1)),
        ("v2.5.3-rc99", (2, 5, 3, 99)),
    ]

    for version_str, expected in test_cases:
        result = _get_version_tuple(version_str)
        assert result == expected, f"Expected {expected} for {version_str}, got {result}"
        print(f"  ✓ {version_str} -> {result}")

    print("  All version tuple tests passed!")


def create_mock_gitlab(current_version: str):
    """Create a mock GitLab instance that returns the specified current version."""
    mock_gl = MagicMock()
    mock_gl.list_tags.return_value = [
        {"name": current_version, "commit": {"id": "abc123", "short_id": "abc123", "committed_date": "2024-01-01T00:00:00.000+00:00"}}
    ]
    return mock_gl


def test_versiontag_standard_release():
    """Test VersionTag behavior with standard releases."""
    print("\n=== Testing VersionTag with Standard Releases ===")

    from versiontag import VersionTag

    # Test MINOR bump from standard version
    mock_gl = create_mock_gitlab("v0.99.0")
    vt = VersionTag(mock_gl)

    assert vt.current == "v0.99.0", f"Expected v0.99.0, got {vt.current}"
    assert vt.is_prerelease == False, "v0.99.0 should not be a pre-release"
    assert vt.current_base == "v0.99.0", f"Expected v0.99.0, got {vt.current_base}"
    print(f"  ✓ current={vt.current}, is_prerelease={vt.is_prerelease}")

    # Test next version calculations
    next_minor = vt.next("MINOR")
    assert next_minor == "v0.100.0", f"Expected v0.100.0, got {next_minor}"
    print(f"  ✓ next(MINOR) = {next_minor}")

    # Reset for MAJOR test
    vt._current = None
    mock_gl.list_tags.return_value = [{"name": "v0.99.0"}]
    next_major = vt.next("MAJOR")
    assert next_major == "v1.0.0", f"Expected v1.0.0, got {next_major}"
    print(f"  ✓ next(MAJOR) = {next_major}")

    # Test next_prerelease from standard version
    vt._current = None
    next_pre = vt.next_prerelease("MINOR")
    assert next_pre == "v0.100.0-rc0", f"Expected v0.100.0-rc0, got {next_pre}"
    print(f"  ✓ next_prerelease(MINOR) = {next_pre}")

    vt._current = None
    next_pre_major = vt.next_prerelease("MAJOR")
    assert next_pre_major == "v1.0.0-rc0", f"Expected v1.0.0-rc0, got {next_pre_major}"
    print(f"  ✓ next_prerelease(MAJOR) = {next_pre_major}")

    print("  All standard release tests passed!")


def test_versiontag_prerelease():
    """Test VersionTag behavior with pre-releases."""
    print("\n=== Testing VersionTag with Pre-releases ===")

    from versiontag import VersionTag

    # Test with existing pre-release
    mock_gl = create_mock_gitlab("v1.0.0-rc0")
    vt = VersionTag(mock_gl)

    assert vt.current == "v1.0.0-rc0", f"Expected v1.0.0-rc0, got {vt.current}"
    assert vt.is_prerelease == True, "v1.0.0-rc0 should be a pre-release"
    assert vt.current_base == "v1.0.0", f"Expected v1.0.0, got {vt.current_base}"
    print(f"  ✓ current={vt.current}, is_prerelease={vt.is_prerelease}, base={vt.current_base}")

    # Test incrementing pre-release (rc0 -> rc1)
    next_pre = vt.next_prerelease("MINOR")  # version_type is ignored for existing pre-releases
    assert next_pre == "v1.0.0-rc1", f"Expected v1.0.0-rc1, got {next_pre}"
    print(f"  ✓ next_prerelease() from rc0 = {next_pre}")

    # Test with rc5
    vt._current = None
    mock_gl.list_tags.return_value = [{"name": "v1.0.0-rc5"}]
    next_pre = vt.next_prerelease("MINOR")
    assert next_pre == "v1.0.0-rc6", f"Expected v1.0.0-rc6, got {next_pre}"
    print(f"  ✓ next_prerelease() from rc5 = {next_pre}")

    # Test finalizing pre-release (rc -> final)
    vt._current = None
    mock_gl.list_tags.return_value = [{"name": "v1.0.0-rc2"}]
    final = vt.next("MINOR")  # version_type is ignored when finalizing
    assert final == "v1.0.0", f"Expected v1.0.0, got {final}"
    print(f"  ✓ next() from rc2 (finalize) = {final}")

    print("  All pre-release tests passed!")


def test_version_flow():
    """Test a complete version flow scenario."""
    print("\n=== Testing Complete Version Flow ===")
    print("  Scenario: v0.99.0 -> rc0 -> rc1 -> rc2 -> v1.0.0")

    from versiontag import VersionTag

    versions = []

    # Start at v0.99.0
    mock_gl = create_mock_gitlab("v0.99.0")
    vt = VersionTag(mock_gl)
    versions.append(vt.current)
    print(f"  1. Starting version: {vt.current}")

    # Create first pre-release (MAJOR bump)
    rc0 = vt.next_prerelease("MAJOR")
    versions.append(rc0)
    print(f"  2. First pre-release: {rc0}")

    # Increment to rc1
    vt._current = None
    mock_gl.list_tags.return_value = [{"name": rc0}]
    rc1 = vt.next_prerelease("MAJOR")
    versions.append(rc1)
    print(f"  3. Second pre-release: {rc1}")

    # Increment to rc2
    vt._current = None
    mock_gl.list_tags.return_value = [{"name": rc1}]
    rc2 = vt.next_prerelease("MAJOR")
    versions.append(rc2)
    print(f"  4. Third pre-release: {rc2}")

    # Finalize to v1.0.0
    vt._current = None
    mock_gl.list_tags.return_value = [{"name": rc2}]
    final = vt.next("MAJOR")
    versions.append(final)
    print(f"  5. Final release: {final}")

    expected = ["v0.99.0", "v1.0.0-rc0", "v1.0.0-rc1", "v1.0.0-rc2", "v1.0.0"]
    assert versions == expected, f"Expected {expected}, got {versions}"

    print(f"  ✓ Complete flow: {' -> '.join(versions)}")
    print("  Version flow test passed!")


def test_releasegen_get_version_type():
    """Test ReleaseGen.get_version_type() method."""
    print("\n=== Testing ReleaseGen.get_version_type() ===")

    from releasegen import ReleaseGen, _Category

    mock_gl = MagicMock()
    rg = ReleaseGen(mock_gl)

    # Mock _read_changelog to return minimal changelog
    rg._read_changelog = MagicMock(return_value=["[//]: # (abc123)", "", "# DataEval Change Log"])

    # Test with MAJOR category
    rg._get_entries = MagicMock(return_value=(None, {_Category.MAJOR: []}))
    vt = rg.get_version_type()
    assert vt == "MAJOR", f"Expected MAJOR, got {vt}"
    print(f"  ✓ MAJOR label -> {vt}")

    # Test with FEATURE category
    rg._get_entries = MagicMock(return_value=(None, {_Category.FEATURE: []}))
    vt = rg.get_version_type()
    assert vt == "MINOR", f"Expected MINOR, got {vt}"
    print(f"  ✓ FEATURE label -> {vt}")

    # Test with FIX category
    rg._get_entries = MagicMock(return_value=(None, {_Category.FIX: []}))
    vt = rg.get_version_type()
    assert vt == "MINOR", f"Expected MINOR, got {vt}"
    print(f"  ✓ FIX label -> {vt}")

    # Test with no entries
    rg._get_entries = MagicMock(return_value=(None, {}))
    vt = rg.get_version_type()
    assert vt == "MINOR", f"Expected MINOR (default), got {vt}"
    print(f"  ✓ No entries -> {vt} (default)")

    print("  All get_version_type tests passed!")


def test_changelog_consolidation():
    """Test that pre-release sections are consolidated when finalizing."""
    print("\n=== Testing Changelog Consolidation ===")

    from releasegen import ReleaseGen

    mock_gl = MagicMock()
    rg = ReleaseGen(mock_gl)

    # Simulate a changelog with multiple rc sections
    changelog_lines = [
        "## v1.0.0-rc1\n",
        "\n",
        "🌟 **Feature Release**\n",
        "- `abc12345` - Feature from rc1\n",
        "\n",
        "## v1.0.0-rc0\n",
        "\n",
        "🌟 **Feature Release**\n",
        "- `def67890` - Feature from rc0\n",
        "\n",
        "👾 **Fixes**\n",
        "- `fix12345` - Fix from rc0\n",
        "\n",
        "## v0.99.0\n",
        "\n",
        "🌟 **Feature Release**\n",
        "- `old12345` - Old feature\n",
    ]

    # Consolidate rc sections for v1.0.0
    result = rg._consolidate_prerelease_sections(changelog_lines, "v1.0.0")

    # The rc headers should be removed
    result_str = "".join(result)
    assert "## v1.0.0-rc0" not in result_str, "rc0 header should be removed"
    assert "## v1.0.0-rc1" not in result_str, "rc1 header should be removed"
    print("  ✓ Pre-release headers removed")

    # The content from rc sections should remain
    assert "Feature from rc0" in result_str, "rc0 content should remain"
    assert "Feature from rc1" in result_str, "rc1 content should remain"
    assert "Fix from rc0" in result_str, "rc0 fix should remain"
    print("  ✓ Pre-release content preserved")

    # The v0.99.0 section should be untouched
    assert "## v0.99.0" in result_str, "v0.99.0 header should remain"
    assert "Old feature" in result_str, "v0.99.0 content should remain"
    print("  ✓ Previous release section preserved")

    # Test with no pre-release sections (should return unchanged)
    no_rc_lines = [
        "## v0.99.0\n",
        "\n",
        "🌟 **Feature Release**\n",
        "- `old12345` - Old feature\n",
    ]
    result2 = rg._consolidate_prerelease_sections(no_rc_lines, "v1.0.0")
    assert "".join(result2) == "".join(no_rc_lines), "Should not modify changelog without rc sections"
    print("  ✓ No changes when no pre-release sections exist")

    print("  All consolidation tests passed!")


def test_finalize_prerelease_changelog():
    """Test the full finalization flow with changelog consolidation."""
    print("\n=== Testing Finalize Pre-release Changelog ===")

    from releasegen import ReleaseGen, _Category
    from versiontag import VersionTag

    # Mock GitLab with a pre-release version
    mock_gl = MagicMock()
    mock_gl.list_tags.return_value = [
        {"name": "v1.0.0-rc1", "commit": {"id": "latest123", "short_id": "lat", "committed_date": "2024-01-17T00:00:00.000+00:00"}}
    ]

    rg = ReleaseGen(mock_gl)
    vt = VersionTag(mock_gl)

    # Verify we're in pre-release state
    assert vt.is_prerelease == True, "Should detect pre-release"
    assert vt.current_base == "v1.0.0", f"Base should be v1.0.0, got {vt.current_base}"
    print(f"  ✓ Detected pre-release: {vt.current} (base: {vt.current_base})")

    # Mock changelog with rc sections
    mock_changelog = [
        "[//]: # (rc1hash123)\n",
        "\n",
        "# DataEval Change Log\n",
        "\n",
        "## v1.0.0-rc1\n",
        "\n",
        "🌟 **Feature Release**\n",
        "- `rc1feat` - RC1 feature\n",
        "\n",
        "## v1.0.0-rc0\n",
        "\n",
        "🌟 **Feature Release**\n",
        "- `rc0feat` - RC0 feature\n",
        "\n",
        "## v0.99.0\n",
        "\n",
        "📝 **Miscellaneous**\n",
        "- `misc123` - Old misc\n",
    ]

    rg._read_changelog = MagicMock(return_value=mock_changelog)

    # Mock _get_entries to return no new entries (common case for finalization)
    rg._get_entries = MagicMock(return_value=(MagicMock(hash="rc1hash123"), {}))

    # Call the method
    version, action = rg._generate_version_and_changelog_action()

    assert version == "v1.0.0", f"Expected v1.0.0, got {version}"
    print(f"  ✓ Version finalized to: {version}")

    assert action, "Should generate changelog action even with no new entries"
    content = action["content"]

    # Check that the final version header is present
    assert "## v1.0.0\n" in content or "## v1.0.0" in content.split("\n"), "Should have v1.0.0 header"
    print("  ✓ Final version header present")

    # Check that rc headers are removed
    assert "## v1.0.0-rc0" not in content, "rc0 header should be consolidated"
    assert "## v1.0.0-rc1" not in content, "rc1 header should be consolidated"
    print("  ✓ Pre-release headers consolidated")

    # Check that rc content is preserved
    assert "RC0 feature" in content, "rc0 content should be preserved"
    assert "RC1 feature" in content, "rc1 content should be preserved"
    print("  ✓ Pre-release content preserved")

    # Check that old version is untouched
    assert "## v0.99.0" in content, "v0.99.0 should remain"
    print("  ✓ Previous release preserved")

    print("  All finalize tests passed!")


def main():
    print("=" * 60)
    print("Pre-release Logic Test Suite")
    print("=" * 60)

    try:
        test_version_patterns()
        test_releasegen_version_pattern()
        test_versiontag_standard_release()
        test_versiontag_prerelease()
        test_version_flow()
        test_releasegen_get_version_type()
        test_changelog_consolidation()
        test_finalize_prerelease_changelog()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

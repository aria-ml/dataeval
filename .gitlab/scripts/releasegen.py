import re
from collections import defaultdict
from datetime import UTC, datetime
from enum import IntEnum
from os import path, remove
from typing import Any, Literal

from gitlab import Gitlab
from rest import verbose
from versiontag import VersionTag

CHANGELOG_FILE = "CHANGELOG.md"
# need to read this to update the doc links
HOWTO_INDEX_FILE = "docs/source/how-to/index.md"
TUTORIAL_INDEX_FILE = "docs/source/tutorials/index.md"
TAB = "    "

# Pattern to match versions with optional pre-release suffix (e.g., v1.0.0 or v1.0.0-rc0)
version_pattern = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)(?:-rc([0-9]+))?")

"""
Multiline pattern that matches for the following content in MR description:

    ## Release Notes

    This is the release notes intended for capture.
    Includes multiline content.
    ```python
    test = 123  # And potential code comments
    ```

Capture:

    (?P<content>[\\s\\S]+?) - Lazy named capture group <content> for any whitespace
    or non-whitespace characters one or more times until hitting one of the post
    capture end conditions below.

Post capture end conditions:

    ^#+.+$ - Markdown heading indicators (#) at start of line
    ^Closes #\\d+.*$ - Merge request shortcut to close related issue
    \\Z - End of string input
"""
release_notes_pattern = re.compile(
    r"[\s\S]*^#+ Release Notes$(?P<content>[\s\S]+?)(?:^#+.+$|^Closes #\d+.*$|\Z)",
    re.MULTILINE,
)


def _get_version_tuple(version: str) -> tuple[int, int, int, int | None] | None:
    """Parse version string into tuple. Returns (major, minor, patch, rc) where rc is None for non-prerelease."""
    result = version_pattern.match(version)
    groups = None if result is None else result.groups()
    if groups is None or len(groups) < 3:
        return None
    rc = int(groups[3]) if len(groups) > 3 and groups[3] is not None else None
    return (int(groups[0]), int(groups[1]), int(groups[2]), rc)


class _Category(IntEnum):
    MAJOR = 0
    FEATURE = 1
    DEPRECATION = 2
    IMPROVEMENT = 3
    FIX = 4
    MISCELLANEOUS = 5
    UNKNOWN = 6
    TAG = 7

    @classmethod
    def from_label(cls, value: str) -> "_Category":
        if "release::feature" in value:
            return _Category.FEATURE
        if "release::improvement" in value:
            return _Category.IMPROVEMENT
        if "release::fix" in value:
            return _Category.FIX
        if "release::deprecation" in value:
            return _Category.DEPRECATION
        if "release::major" in value:
            return _Category.MAJOR
        if "release::misc" in value:
            return _Category.MISCELLANEOUS
        return _Category.UNKNOWN

    @classmethod
    def to_markdown(cls, value: "_Category") -> str:
        if value == _Category.FEATURE:
            return "🌟 **Feature Release**"
        if value == _Category.IMPROVEMENT:
            return "🛠️ **Improvements and Enhancements**"
        if value == _Category.FIX:
            return "👾 **Fixes**"
        if value == _Category.DEPRECATION:
            return "🚧 **Deprecations and Removals**"
        if value == _Category.MAJOR:
            return "🚀 **Major Release**"
        if value == _Category.MISCELLANEOUS:
            return "📝 **Miscellaneous**"
        raise ValueError("Do not generate release markdown for UNKNOWN entries")

    @classmethod
    def version_type(cls, value: "_Category") -> Literal["MAJOR", "MINOR", "PATCH"]:
        match value:
            case _Category.MAJOR:
                return "MAJOR"
            case _:
                return "MINOR"


class _Tag:
    def __init__(self, response: dict[str, Any] | None = None, pending: bool | None = None) -> None:
        if response is not None:
            time = response["commit"]["committed_date"]
            self.time: datetime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f%z")
            self.hash: str = response["commit"]["id"]
            self.shorthash: str = response["commit"]["short_id"]
            self.description: str = response["name"]
        elif pending:
            self.time: datetime = datetime.now(UTC)
            self.hash: str = "pending"
            self.shorthash: str = "pending"
            self.description: str = "pending"
        else:
            raise ValueError("Must provide a response or specify pending")

    def to_markdown(self) -> str:
        return f"## {self.description}"

    def __lt__(self, other: "_Merge") -> bool:
        self_version = _get_version_tuple(self.description)
        other_version = _get_version_tuple(other.description)

        if self_version is not None and other_version is not None:
            return self_version < other_version

        return self.time < other.time

    def __hash__(self):
        return self.hash.__hash__()

    def __repr__(self) -> str:
        return f"Entry(TAG, {self.time}, {self.hash}, {self.description})"


class _Merge:
    """Extracts elements of merge commits to create the change history contents."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.time: datetime = datetime.strptime(response["merged_at"][0:19], "%Y-%m-%dT%H:%M:%S").astimezone(UTC)
        self.hash: str = response["merge_commit_sha"]
        self.shorthash: str = response["merge_commit_sha"][0:8]
        self.description: str = response["title"].replace('Resolve "', "").replace('"', "")
        self.category: _Category = _Category.from_label(response["labels"])
        match = release_notes_pattern.match(response["description"])
        groupdict = {} if match is None else match.groupdict()
        self.details: str = groupdict.get("content", "").strip(" \t\n")

    def to_markdown(self) -> str:
        md = f"- `{self.shorthash}` - {self.description}"
        if len(self.details) > 0 and not self.details.startswith("Placeholder"):
            md += f"\n\n{TAB}" + f"\n{TAB}".join(self.details.splitlines())
        return md

    def __lt__(self, other: "_Merge") -> bool:
        return self.time < other.time

    def __repr__(self) -> str:
        entry_type = _Category(self.category).name
        return f"Entry({entry_type}, {self.time}, {self.hash}, {self.description})"


class ReleaseGen:
    """Generates commit payload for changelog and documentation link updates."""

    def __init__(self, gitlab: Gitlab) -> None:
        self.gl = gitlab

    def _read_changelog(self) -> list[str]:
        temp = False
        if not path.exists(CHANGELOG_FILE):
            verbose(f"{CHANGELOG_FILE} not found, pulling from Gitlab")
            temp = True
            self.gl.get_file(CHANGELOG_FILE, CHANGELOG_FILE)
        with open(CHANGELOG_FILE) as file:
            lines = file.readlines()
        if temp:
            verbose(f"Removing temp {CHANGELOG_FILE}")
            remove(CHANGELOG_FILE)
        return lines

    def _read_doc_file(self, file_name: str) -> None | list[str]:
        temp = False
        if not path.exists(file_name):
            verbose(f"{file_name} not found, pulling from Gitlab")
            temp = True
            self.gl.get_file(file_name, file_name)
        try:
            with open(file_name) as file:
                lines = file.readlines()
            if temp:
                verbose(f"Removing temp {file_name}")
                remove(file_name)
            return lines
        except Exception:
            return None

    def _get_last_hash(self, line: str) -> str:
        start = line.find("(") + 1
        end = line.find(")")
        return line[start:end]

    def _get_entries(self, last_hash: str) -> tuple[_Merge, dict[_Category, list[_Merge]]]:
        # get merges in to develop and main and sort
        merges: list[_Merge] = [_Merge(m) for m in self.gl.list_merge_requests(state="merged", target_branch="main")]
        merges.sort(reverse=True)
        latest = merges[0]

        # populate the categorized merge issues
        categorized: dict[_Category, list[_Merge]] = defaultdict(lambda: [])

        for merge in merges:
            # return early if hash is already present
            if merge.hash == last_hash:
                return latest, categorized

            merge_log = f"COMMITGEN: {merge.description} @ {merge.hash}"
            verbose(merge_log + f" - ADDED as {_Category(merge.category).name}")
            categorized[merge.category].append(merge)

        return latest, categorized

    def _extract_prerelease_entries(
        self, lines: list[str], base_version: str
    ) -> tuple[dict[_Category, list[str]], list[str]]:
        """
        Extract categorized entries from pre-release sections and return remaining non-RC lines.

        Parses all rc sections for the base version, groups their entries by category,
        and returns the grouped entries along with the non-RC changelog lines.
        """
        prerelease_pattern = re.compile(rf"^## {re.escape(base_version)}-rc\d+$")

        # Build a lookup from markdown header to _Category
        header_to_category: dict[str, _Category] = {}
        for cat in _Category:
            if cat in (_Category.UNKNOWN, _Category.TAG):
                continue
            header_to_category[_Category.to_markdown(cat)] = cat

        rc_entries: dict[_Category, list[str]] = defaultdict(list)
        non_rc_lines: list[str] = []

        in_rc_section = False
        current_category: _Category | None = None

        for line in lines:
            stripped = line.strip()

            # RC header - enter RC section
            if prerelease_pattern.match(stripped):
                verbose(f"Extracting pre-release section: {stripped}")
                in_rc_section = True
                current_category = None
                continue

            # Non-RC version header - exit RC section
            if stripped.startswith("## v"):
                in_rc_section = False
                current_category = None
                non_rc_lines.append(line)
                continue

            if not in_rc_section:
                non_rc_lines.append(line)
                continue

            # In RC section - check for category header
            matched = header_to_category.get(stripped)
            if matched is not None:
                current_category = matched
                continue

            # Skip blank lines before first category
            if current_category is None:
                continue

            # Collect entry lines (items, continuation lines, and blank lines between entries)
            rc_entries[current_category].append(line)

        # Clean up entries per category: strip trailing blank lines and remove
        # stray blank lines between entry groups (from different RCs) while
        # preserving blank lines that are part of multi-line entry details
        for cat in rc_entries:
            raw = rc_entries[cat]
            cleaned: list[str] = []
            for i, line in enumerate(raw):
                if line.strip() == "":
                    # Keep blank line only if it's part of a multi-line entry detail
                    # (i.e., followed by an indented continuation line)
                    if i + 1 < len(raw) and raw[i + 1].startswith((" ", "\t")):
                        cleaned.append(line)
                else:
                    cleaned.append(line)
            rc_entries[cat] = cleaned

        return rc_entries, non_rc_lines

    def _generate_version_and_changelog_action(self) -> tuple[str, dict[str, str]]:
        current = self._read_changelog()
        last_hash = self._get_last_hash(current[0]) if current else ""

        vt = VersionTag(self.gl)

        # Check if we're finalizing a pre-release
        is_finalizing_prerelease = vt.is_prerelease
        base_version = vt.current_base if is_finalizing_prerelease else None

        # Return empty dict if nothing to update
        latest, entries = self._get_entries(last_hash)

        # When finalizing a pre-release, we may not have new entries but still need to
        # consolidate the changelog sections
        if not entries and not is_finalizing_prerelease:
            return "", {}

        # Get the latest hash - use current if no new entries (finalizing without new MRs)
        latest_hash = latest.hash if entries else last_hash

        # Get remaining changelog content (after the header lines)
        remaining_lines = current[3:]

        if is_finalizing_prerelease and base_version:
            verbose(f"Finalizing pre-release: consolidating {base_version}-rcX sections")

            # Extract categorized entries from all RC sections
            rc_entries, remaining_lines = self._extract_prerelease_entries(remaining_lines, base_version)

            # Merge new entries (since last RC) into the RC entries
            next_category = _Category.UNKNOWN
            for category in sorted(entries):
                if category == _Category.UNKNOWN:
                    continue
                for merge in entries[category]:
                    if merge.hash == last_hash:
                        break
                    rc_entries[category].append(merge.to_markdown() + "\n")
                    verbose(f"Adding - {merge.to_markdown()}")
                    next_category = min(next_category, category)

            # Include RC categories when determining version type
            for category in rc_entries:
                if category != _Category.UNKNOWN:
                    next_category = min(next_category, category)

            next_version = vt.next(_Category.version_type(next_category))

            # Build merged categorized content in correct category order
            lines: list[str] = []
            for category in sorted(rc_entries):
                cat_entries = rc_entries[category]
                # Strip trailing blank lines
                while cat_entries and cat_entries[-1].strip() == "":
                    cat_entries.pop()
                if cat_entries:
                    lines.append("")
                    lines.append(_Category.to_markdown(category))
                    lines.append("")
                    for entry_line in cat_entries:
                        lines.append(entry_line.rstrip("\n"))
        else:
            lines = []
            next_category = _Category.UNKNOWN

            categories = sorted(entries)
            for category in categories:
                # skip unknown categories
                if category == _Category.UNKNOWN:
                    continue
                merges = entries[category]
                lines.append("")
                lines.append(_Category.to_markdown(category))
                lines.append("")
                for merge in merges:
                    if merge.hash == last_hash:
                        break

                    lines.append(merge.to_markdown())
                    verbose(f"Adding - {merge.to_markdown()}")
                    next_category = min(next_category, category)

            next_version = vt.next(_Category.version_type(next_category))

        header = [f"[//]: # ({latest_hash})", "", "# DataEval Change Log", "", f"## {next_version}"]
        content = "\n".join(header + lines) + "\n"

        for oldline in remaining_lines:
            content += oldline

        return next_version, {
            "action": "update",
            "file_path": CHANGELOG_FILE,
            "encoding": "text",
            "content": content,
        }

    def _generate_index_markdown_update_action(self, file_name: str, pending_version: str) -> dict[str, str]:
        howto_index_file = self._read_doc_file(file_name)
        if howto_index_file:
            pattern = re.compile(
                r"aria-ml/dataeval/blob/docs-artifacts/(?:main|v[0-9]+\.[0-9]+\.[0-9]+(?:-rc[0-9]+)?)/notebooks",
            )
            new_path = f"aria-ml/dataeval/blob/docs-artifacts/{pending_version}/notebooks"

            verbose(f"Substituting markdown links for new version {pending_version}")
            content = "".join([re.sub(pattern, new_path, line) for line in howto_index_file])
            return {
                "action": "update",
                "file_path": file_name,
                "encoding": "text",
                "content": content,
            }
        return {}

    def generate(self) -> tuple[str, list[dict[str, str]]]:
        version, changelog_action = self._generate_version_and_changelog_action()
        if not changelog_action:
            return "", []

        actions = [
            self._generate_index_markdown_update_action(f, version) for f in [HOWTO_INDEX_FILE, TUTORIAL_INDEX_FILE]
        ]
        actions.append(changelog_action)

        return version, [action for action in actions if action]

    def get_version_type(self) -> Literal["MAJOR", "MINOR", "PATCH"]:
        """
        Determine the version bump type based on merged MR labels.
        Returns the highest priority version type found.
        """
        current = self._read_changelog()
        last_hash = self._get_last_hash(current[0]) if current else ""
        _, entries = self._get_entries(last_hash)

        if not entries:
            return "MINOR"  # Default to MINOR if no entries

        # Find the highest priority (lowest enum value) category
        next_category = _Category.UNKNOWN
        for category in entries:
            if category != _Category.UNKNOWN:
                next_category = min(next_category, category)

        return _Category.version_type(next_category)

    def generate_prerelease(self, version: str) -> tuple[str, list[dict[str, str]]]:
        """
        Generate changelog and doc link updates for a pre-release version.
        Similar to generate() but uses provided version instead of calculating it.
        Does not update jupyter cache for pre-releases.
        """
        current = self._read_changelog()
        last_hash = self._get_last_hash(current[0]) if current else ""

        latest, entries = self._get_entries(last_hash)
        if not entries:
            return "", []

        lines: list[str] = []

        categories = sorted(entries)
        for category in categories:
            if category == _Category.UNKNOWN:
                continue
            merges = entries[category]
            lines.append("")
            lines.append(_Category.to_markdown(category))
            lines.append("")
            for merge in merges:
                if merge.hash == last_hash:
                    break
                lines.append(merge.to_markdown())
                verbose(f"Adding - {merge.to_markdown()}")

        header = [f"[//]: # ({latest.hash})", "", "# DataEval Change Log", "", f"## {version}"]
        content = "\n".join(header + lines) + "\n"

        for oldline in current[3:]:
            content += oldline

        changelog_action = {
            "action": "update",
            "file_path": CHANGELOG_FILE,
            "encoding": "text",
            "content": content,
        }

        # Update documentation links (colab links) to point to the pre-release version
        actions = [
            self._generate_index_markdown_update_action(f, version) for f in [HOWTO_INDEX_FILE, TUTORIAL_INDEX_FILE]
        ]
        actions.append(changelog_action)

        return version, [action for action in actions if action]

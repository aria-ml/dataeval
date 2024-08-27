from base64 import b64encode
from collections import defaultdict
from datetime import UTC, datetime
from enum import IntEnum
from os import path, remove, walk
from re import MULTILINE, compile
from shutil import move, rmtree
from typing import Any, Dict, List, Optional, Tuple

from gitlab import Gitlab
from rest import verbose
from versiontag import VersionTag

CHANGELOG_FILE = "CHANGELOG.md"
TAB = "    "

version_pattern = compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)")

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
release_notes_pattern = compile(
    r"[\s\S]*^#+ Release Notes$(?P<content>[\s\S]+?)(?:^#+.+$|^Closes #\d+.*$|\Z)",
    MULTILINE,
)


def _get_version_tuple(version: str) -> Optional[Tuple[int, int, int]]:
    result = version_pattern.match(version)
    groups = None if result is None else result.groups()
    if groups is None or len(groups) != 3:
        return None
    return (int(groups[0]), int(groups[1]), int(groups[2]))


class _Category(IntEnum):
    TAG = 0
    UNKNOWN = 1
    IMPROVEMENT = 2
    FIX = 3
    FEATURE = 4
    DEPRECATION = 5
    MAJOR = 6

    @classmethod
    def from_label(cls, value: str) -> "_Category":
        if "release::feature" in value:
            return _Category.FEATURE
        elif "release::improvement" in value:
            return _Category.IMPROVEMENT
        elif "release::fix" in value:
            return _Category.FIX
        elif "release::deprecation" in value:
            return _Category.DEPRECATION
        elif "release::major" in value:
            return _Category.MAJOR
        return _Category.UNKNOWN

    @classmethod
    def to_markdown(cls, value: "_Category"):
        if value == _Category.FEATURE:
            return "🌟 **Feature Release**"
        elif value == _Category.IMPROVEMENT:
            return "🛠️ **Improvements and Enhancements**"
        elif value == _Category.FIX:
            return "👾 **Fixes**"
        elif value == _Category.DEPRECATION:
            return "🚧 **Deprecations and Removals**"
        elif value == _Category.MAJOR:
            return "🚀 **Major Release**"
        else:
            return "📝 **Miscellaneous**"


class _Tag:
    def __init__(
        self,
        response: Optional[Dict[str, Any]] = None,
        pending_version: Optional[Tuple[str, str]] = None,
    ):
        if response is not None:
            time = response["commit"]["committed_date"]
            self.time: datetime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f%z")
            self.hash: str = response["commit"]["id"]
            self.shorthash: str = response["commit"]["short_id"]
            self.description: str = response["name"]
        elif pending_version is not None:
            self.time: datetime = datetime.now(UTC)
            self.hash: str = pending_version[1]
            self.shorthash: str = pending_version[1][0:8]
            self.description: str = pending_version[0]
        else:
            raise ValueError("Must provide a response or pending version string")

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
    """
    Extracts elements of merge commits to create the change history contents.
    """

    def __init__(self, response: Dict[str, Any]):
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


class CommitGen:
    """
    Generates commit payload for changes for used in the documentation cache
    and changelog updates
    """

    def __init__(self, gitlab: Gitlab):
        self.gl = gitlab
        self._pending = "0.0.0"

    def _read_changelog(self) -> List[str]:
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

    def _get_last_hash(self, line: str) -> str:
        start = line.find("(") + 1
        end = line.find(")")
        return line[start:end]

    def _get_entries(self, last_hash: str) -> Dict[_Tag, Dict[_Category, List[_Merge]]]:
        # get merges in to develop and main and sort
        merges: List[_Merge] = []
        for merge in self.gl.list_merge_requests(state="merged", target_branch="main", order_by="merged_at"):
            merges.append(_Merge(merge))
        merges.sort(reverse=True)

        # get version buckets and sort
        tags: List[_Tag] = []
        for tag in self.gl.list_tags():
            if _get_version_tuple(tag["name"]) is not None:
                tags.append(_Tag(tag))
        tags.sort(reverse=True)

        # populate the categorized merge issues
        categorized: Dict[_Tag, Dict[_Category, List[_Merge]]] = defaultdict(lambda: defaultdict(lambda: []))

        tag_it = iter(tags)
        tag = next(tag_it)
        next_tag = next(tag_it, None)
        for merge in merges:
            # check if we need to move to next tag
            while next_tag is not None and (merge.hash == next_tag.hash or merge.time <= next_tag.time):
                # return early if hash is already present
                if next_tag.hash == last_hash:
                    return categorized

                # move to next tag
                tag = next_tag
                next_tag = next(tag_it, None)

            # return early if hash is already present
            if merge.hash == last_hash:
                return categorized

            # drop merges that are not categorized
            merge_log = f"COMMITGEN: {merge.description} @ {merge.hash}"
            if merge.category != _Category.UNKNOWN:
                verbose(merge_log + f" - ADDED as {_Category(merge.category).name}")
                categorized[tag][merge.category].append(merge)
            else:
                verbose(merge_log + " - SKIPPED")

        return categorized

    def _generate_changelog_action(self) -> Dict[str, str]:
        current = self._read_changelog()
        last_hash = self._get_last_hash(current[0]) if current else ""

        # Return empty dict if nothing to update
        entries = self._get_entries(last_hash)
        tags = list(entries)
        vTag = VersionTag(self.gl)
        # generate the new pending version and set it on VersionTag and locally for use.
        pending_version = self.get_next_version_type(entries, last_hash, vTag)
        self._pending = pending_version  # get this value for the new release tag
        vTag.pending = pending_version
        lines: List[str] = []

        for tag in tags:
            if tag.hash == last_hash:
                break

            if not entries[tag]:
                continue

            lines.append("")
            lines.append(tag.to_markdown())

            categories = sorted(entries[tag])
            for category in categories:
                merges = entries[tag][category]
                lines.append("")
                lines.append(_Category.to_markdown(category))
                for merge in merges:
                    if merge.hash == last_hash:
                        break

                    lines.append(merge.to_markdown())

        if not lines:
            return {}

        header = [f"[//]: # ({tags[0].hash})", "", "# DataEval Change Log"]
        content = "\n".join(header + lines) + "\n"

        for oldline in current[3:]:
            content += oldline

        return {"action": "update", "file_path": CHANGELOG_FILE, "encoding": "text", "content": content}

    def _is_binary_file(self, file: str) -> bool:
        try:
            with open(file) as f:
                f.read()
                return False
        except Exception:
            return True

    def _generate_actions(self, old_files: List[str], new_files: List[str]) -> List[Dict[str, str]]:
        actions: List[Dict[str, str]] = []

        for old_file in old_files:
            if old_file not in new_files:
                actions.append(
                    {
                        "action": "delete",
                        "file_path": old_file,
                    }
                )

        for new_file in new_files:
            if self._is_binary_file(new_file):
                encoding = "base64"
                with open(new_file, "rb") as f:
                    content = b64encode(f.read()).decode()
            else:
                encoding = "text"
                with open(new_file) as f:
                    content = f.read()

            if new_file in old_files:
                actions.append(
                    {
                        "action": "update",
                        "file_path": new_file,
                        "encoding": encoding,
                        "content": content,
                    }
                )
            else:
                actions.append(
                    {
                        "action": "create",
                        "file_path": new_file,
                        "encoding": encoding,
                        "content": content,
                    }
                )

        return actions

    def _get_files(self, file_path: str) -> List[str]:
        file_paths: List[str] = []
        for root, _, files in walk(file_path):
            for filename in files:
                file_paths.append(path.join(root, filename))
        return file_paths

    def _generate_jupyter_cache_actions(self) -> List[Dict[str, str]]:
        ref = "main"
        cache_path = "docs/.jupyter_cache"
        output_path = path.join("output", cache_path)
        self.gl.get_artifacts(job="docs", dest="./", ref=ref)
        if not path.exists(output_path):
            raise FileNotFoundError(f"Artifacts downloaded from {ref} does not contain {output_path}")
        if not path.exists(cache_path):
            raise FileNotFoundError(f"Current path does not contain {cache_path}")

        old_files = self._get_files(cache_path)
        rmtree(cache_path)
        move(output_path, cache_path)
        actions = self._generate_actions(old_files, self._get_files(cache_path))
        return actions

    def generate(self) -> List[Dict[str, str]]:
        changelog_action = self._generate_changelog_action()
        if not changelog_action:
            return []

        actions = self._generate_jupyter_cache_actions()
        actions.append(changelog_action)
        return actions

    def get_next_version_type(self, entries, last_hash, vTag: VersionTag) -> str:
        # Get the last hash from the repository. Used for getting all the changes since the last release.
        # create a list of Categories from the entries. Use the largest one for the overall label.
        tags = list(entries)  # convert to a list
        # loop through the list
        change = _Category.TAG  # 0
        for tag in tags:
            if tag.hash == last_hash:  # stop if no changes
                break
            if not entries[tag]:  # if no tag skip logic
                continue
            categories = sorted(entries[tag])  # sorted by tag
            for category in categories:
                if (
                    category > change
                ):  # use strictly greater than to limit changes on duplicates. Change types are ordered
                    change = category
        match change:
            case _Category.TAG | _Category.UNKNOWN:  # 0,1
                version = vTag.current
            case _Category.FIX | _Category.IMPROVEMENT:  # 2,3
                version = vTag.next("PATCH")
            case _Category.FEATURE | _Category.DEPRECATION:  # 4,5
                version = vTag.next("MINOR")
            case _Category.MAJOR:  # 6
                version = vTag.next("MAJOR")
            case _:
                verbose(f"Default Case: No Version Change. change is {change}")
                version = vTag.current  # default is no change

        self._pending_version = version
        return version

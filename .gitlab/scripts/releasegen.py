import re
from base64 import b64encode
from collections import defaultdict
from datetime import UTC, datetime
from enum import IntEnum
from os import path, remove, walk
from shutil import move, rmtree
from typing import Any, Literal

from gitlab import Gitlab
from rest import verbose
from versiontag import VersionTag

CHANGELOG_FILE = "CHANGELOG.md"
# need to read this to update the doc links
HOWTO_INDEX_FILE = "docs/source/how_to/index.md"
TUTORIAL_INDEX_FILE = "docs/source/tutorials/index.md"
NOTEBOOK_DIRECTORY = "docs/source/how_to"
TUTORIAL_DIRECTORY = "docs/source/tutorials"
TAB = "    "

version_pattern = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)")

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


def _get_version_tuple(version: str) -> tuple[int, int, int] | None:
    result = version_pattern.match(version)
    groups = None if result is None else result.groups()
    if groups is None or len(groups) != 3:
        return None
    return (int(groups[0]), int(groups[1]), int(groups[2]))


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
            case _Category.FEATURE | _Category.DEPRECATION:
                return "MINOR"
            case _:
                return "PATCH"


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
    """
    Extracts elements of merge commits to create the change history contents.
    """

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
    """
    Generates commit payload for changes for used in the documentation cache
    and changelog updates
    """

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

    def _generate_version_and_changelog_action(self) -> tuple[str, dict[str, str]]:
        current = self._read_changelog()
        last_hash = self._get_last_hash(current[0]) if current else ""

        # Return empty dict if nothing to update
        latest, entries = self._get_entries(last_hash)
        if not entries:
            return "", {}

        lines: list[str] = []
        next_category = _Category.UNKNOWN

        categories = sorted(entries)
        for category in categories:
            # skip unknown categories
            if category == _Category.UNKNOWN:
                continue
            merges = entries[category]
            lines.append("")
            lines.append(_Category.to_markdown(category))
            for merge in merges:
                if merge.hash == last_hash:
                    break

                lines.append(merge.to_markdown())
                verbose(f"Adding - {merge.to_markdown()}")
                next_category = min(next_category, category)

        vt = VersionTag(self.gl)
        next_version = vt.next(_Category.version_type(next_category))

        header = [f"[//]: # ({latest.hash})", "", "# DataEval Change Log", "", f"## {next_version}"]
        content = "\n".join(header + lines) + "\n"

        for oldline in current[3:]:
            content += oldline

        return next_version, {
            "action": "update",
            "file_path": CHANGELOG_FILE,
            "encoding": "text",
            "content": content,
        }

    def _is_binary_file(self, file: str) -> bool:
        try:
            with open(file) as f:
                f.read()
                return False
        except Exception:
            return True

    def _generate_actions(self, old_files: list[str], new_files: list[str]) -> list[dict[str, str]]:
        actions: list[dict[str, str]] = []
        # will need this as an input for permanent solution
        # current_tag = pending_version

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
                # comment out this code for now. Will need for permanent solution.
                # if new_file.endswith("ipynb"):
                #    content = self._update_cache_file_path(new_file, current_tag=current_tag)
            if new_file in old_files:
                if content:
                    actions.append(
                        {
                            "action": "update",
                            "file_path": new_file,
                            "encoding": encoding,
                            "content": content,
                        }
                    )
            else:
                if content:
                    actions.append(
                        {
                            "action": "create",
                            "file_path": new_file,
                            "encoding": encoding,
                            "content": content,
                        }
                    )

        return actions

    def _get_files(self, file_path: str) -> list[str]:
        file_paths: list[str] = []
        for root, _, files in walk(file_path):
            for filename in files:
                file_paths.append(path.join(root, filename))
        return file_paths

    def _generate_index_markdown_update_action(self, file_name: str, pending_version: str) -> dict[str, str]:
        howto_index_file = self._read_doc_file(file_name)
        if howto_index_file:
            pattern = re.compile(r"aria-ml/dataeval/blob/v([0-9]+)\.([0-9]+)\.([0-9]+)/docs")
            new_path = f"aria-ml/dataeval/blob/{pending_version}/docs"

            verbose(f"Substituting markdown links for new version {pending_version}")
            content = "".join([re.sub(pattern, new_path, line) for line in howto_index_file])
            return {
                "action": "update",
                "file_path": file_name,
                "encoding": "text",
                "content": content,
            }
        return {}

    def _update_cache_file_path(self, file_name: str, current_tag: str) -> None | str:
        search_pattern = r"(%|\!)+(pip install -q dataeval){1}(\[\w+\])*"
        if path.isfile(file_name):
            new_list: list[str] = []
            lines = self._read_doc_file(file_name)  # return none if file is unreadable.
            if lines:
                for line in lines:
                    result = re.search(search_pattern, line)
                    if result:
                        pos = result.string.find("==v")
                        if pos == -1:
                            new_string = (
                                result.string[0 : result.regs[0][1]]
                                + "=="
                                + current_tag
                                + result.string[result.regs[0][1] :]
                            )
                        else:
                            new_string = result.string[0:pos] + "==" + current_tag + '\\n",\n'
                        new_list.append(new_string)
                    else:
                        new_list.append(line)
                content = "".join(new_list)
            else:
                content = None
        else:
            content = None
        return content

    def _generate_notebook_update_actions(self, pending_version: str) -> list[dict[str, str]]:
        # get all the ipynb file from the notebook directory
        # may need to read .jupytercache files and update them instead of the original files.
        file_path = NOTEBOOK_DIRECTORY
        files: list[str] = []
        notebook_files = self._get_files(file_path)
        for f in notebook_files:
            if f.lower().endswith("ipynb"):
                files.append(f)
        # get the notebooks in the docs/tutorials directory
        more_file_path = TUTORIAL_DIRECTORY
        more_files = self._get_files(more_file_path)
        # add them to the list if they are notebook files.
        for f in more_files:
            if f.lower().endswith("ipynb"):
                files.append(f)
        current_tag = pending_version
        action_list: list[dict[str, str]] = []
        for file_name in files:
            content = self._update_cache_file_path(file_name, current_tag=current_tag)
            new_action = {
                "action": "update",
                "file_path": file_name,
                "encoding": "text",
                "content": content,
            }
            action_list.append(new_action)

        return action_list

    # removed pending version. Will need to add back for permanant solution.
    def _generate_jupyter_cache_actions(self) -> list[dict[str, str]]:
        # cannot use 'latest-known-good' because 404 on download - investigate
        ref = "main"
        cache_path = "docs/source/.jupyter_cache"
        output_path = "output/docs/.jupyter_cache"
        self.gl.get_artifacts(job="docs", dest="./", ref=ref)
        if not path.exists(output_path):
            raise FileNotFoundError(f"Artifacts downloaded from {ref} does not contain {output_path}")
        if not path.exists(cache_path):
            raise FileNotFoundError(f"Current path does not contain {cache_path}")

        old_files = self._get_files(cache_path)
        rmtree(cache_path)
        move(output_path, cache_path)
        # removed pending version for now. will need for permanent solution.
        return self._generate_actions(old_files, self._get_files(cache_path))

    def generate(self) -> tuple[str, list[dict[str, str]]]:
        version, changelog_action = self._generate_version_and_changelog_action()
        if not changelog_action:
            return "", []

        # creating actions for updating notebook cache, documentation links, and changelog content
        actions = self._generate_jupyter_cache_actions()
        actions.extend(
            [self._generate_index_markdown_update_action(f, version) for f in [HOWTO_INDEX_FILE, TUTORIAL_INDEX_FILE]]
        )

        # comment out for now - will need for permanent solution
        # creating actions for updating python notebook pip install statements
        # actions.extend(self._generate_notebook_update_actions(pending_version=version))

        actions.append(changelog_action)

        return version, [action for action in actions if action]

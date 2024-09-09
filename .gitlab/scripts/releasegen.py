from base64 import b64encode
from collections import defaultdict
from datetime import UTC, datetime
from enum import IntEnum
from os import path, remove, walk
from re import MULTILINE, compile, search, sub
from shutil import move, rmtree
from typing import Any, Dict, List, Literal, Tuple

from gitlab import Gitlab
from rest import verbose
from versiontag import VersionTag

CHANGELOG_FILE = "CHANGELOG.md"
# need to read this to update the doc links
HOWTO_INDEX_FILE = "docs/how_to/index.md"
TUTORIAL_INDEX_FILE = "docs/tutorials/index.md"
NOTEBOOK_DIRECTORY = "docs/how_to/notebooks"
TUTORIAL_DIRECTORY = "docs/tutorials"
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


def _get_version_tuple(version: str) -> Tuple[int, int, int] | None:
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
    UNKNOWN = 5
    TAG = 6

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
    def __init__(self, response: Dict[str, Any] | None = None, pending: bool | None = None):
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


class ReleaseGen:
    """
    Generates commit payload for changes for used in the documentation cache
    and changelog updates
    """

    def __init__(self, gitlab: Gitlab):
        self.gl = gitlab

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

    def _read_doc_file(self, file_name: str) -> None | List[str]:
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
        except:
            return None

    def _get_last_hash(self, line: str) -> str:
        start = line.find("(") + 1
        end = line.find(")")
        return line[start:end]

    def _get_entries(self, last_hash: str) -> Dict[_Tag, Dict[_Category, List[_Merge]]]:
        # get merges in to develop and main and sort
        merges: List[_Merge] = [_Merge(m) for m in self.gl.list_merge_requests(state="merged", target_branch="main")]
        merges.sort(reverse=True)

        # get version buckets and sort
        tags: List[_Tag] = [_Tag(t) for t in self.gl.list_tags() if _get_version_tuple(t["name"]) is not None]
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

            merge_log = f"COMMITGEN: {merge.description} @ {merge.hash}"
            verbose(merge_log + f" - ADDED as {_Category(merge.category).name}")
            categorized[tag][merge.category].append(merge)

        return categorized

    def _generate_version_and_changelog_action(self) -> Tuple[str, Dict[str, str]]:
        current = self._read_changelog()
        last_hash = self._get_last_hash(current[0]) if current else ""

        # Return empty dict if nothing to update
        entries = self._get_entries(last_hash)
        tags = list(entries)
        lines: List[str] = []
        next_category = _Category.UNKNOWN

        for tag in tags:
            if tag.hash == last_hash:
                break

            if not entries[tag]:
                continue

            categories = sorted(entries[tag])
            for category in categories:
                merges = entries[tag][category]
                lines.append("")
                lines.append(_Category.to_markdown(category))
                for merge in merges:
                    if merge.hash == last_hash:
                        break

                    lines.append(merge.to_markdown())
                    verbose(f"Adding - {merge.to_markdown()}")
                    next_category = min(next_category, category)

        if not lines:
            return "", {}

        vt = VersionTag(self.gl)
        next_version = vt.next(_Category.version_type(next_category))

        header = [f"[//]: # ({tags[0].hash})", "", "# DataEval Change Log", "", f"## {next_version}"]
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

    def _generate_index_markdown_update_action(self, file_name: str) -> None | Dict[str, str]:
        howto_index_file = self._read_doc_file(file_name)
        if howto_index_file:
            vt = VersionTag(self.gl)
            new_ref = vt.current
            new_sub_1 = "aria-ml/dataeval/blob/" + new_ref + "/docs"

            verbose(f"substitution for new version  {new_sub_1}")
            pattern_1 = "aria-ml/dataeval/blob/main/docs"
            lines: List[str] = []
            for line in howto_index_file:
                new_line_1 = sub(pattern_1, new_sub_1, line)
                if new_line_1 != line:
                    lines.append(new_line_1)  # replace the old with the new line.
                else:
                    lines.append(line)
            content = "".join(lines)
            return {
                "action": "update",
                "file_path": file_name,
                "encoding": "text",
                "content": content,
            }
        else:
            return None

    def _generate_notebook_update_actions(self) -> List[Dict[str, str]]:
        # get all the ipynb file from the notebook directory
        # may need to read .jupytercache files and update them instead of the original files. TEST THIS
        file_path = NOTEBOOK_DIRECTORY
        files = self._get_files(file_path)
        # get the notebooks in the docs/tutorials directory
        more_file_path = TUTORIAL_DIRECTORY
        more_files = self._get_files(more_file_path)
        # add them to the list if they are notebook files.
        for f in more_files:
            if f.lower().endswith("ipynb"):
                files.append(f)

        vt = VersionTag(self.gl)
        # file_lines: List[List[str]] = [[] for i in range(len(files))]
        current_tag = vt.current
        search_pattern = r"(%|\!)+(pip install -q dataeval){1}(\[\w+\])*"
        # Second example search pattern !pip install -q dataeval[torch] torchmetrics torchvision
        action_list: List[Dict[str, str]] = []
        # index = 0
        for file_name in files:
            if path.isfile(file_name):
                new_list: List[str] = []
                lines = self._read_doc_file(file_name)  # return none if file is unreadable.
                if lines:
                    for line in lines:
                        result = search(search_pattern, line)
                        if result:
                            new_string = (
                                result.string[0 : result.regs[0][1]]
                                + "=="
                                + current_tag
                                + result.string[result.regs[0][1] :]
                            )
                            new_list.append(new_string)
                        else:
                            new_list.append(line)
                    content = "".join(new_list)
                    new_action = {
                        "action": "update",
                        "file_path": file_name,
                        "encoding": "text",
                        "content": content,
                    }
                    action_list.append(new_action)

        return action_list

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

    def generate(self) -> Tuple[str, List[Dict[str, str]]]:
        version, changelog_action = self._generate_version_and_changelog_action()
        if not changelog_action:
            return "", []
        # creating actions for updating python notebook pip install statements
        file_update_actions = self._generate_notebook_update_actions()
        actions = self._generate_jupyter_cache_actions()
        # creating actions for updating colab references in markdown locations.
        howto_index_action = self._generate_index_markdown_update_action(HOWTO_INDEX_FILE)
        tutorital_index_action = self._generate_index_markdown_update_action(TUTORIAL_INDEX_FILE)
        if howto_index_action:  # can return None
            actions.append(howto_index_action)
        if tutorital_index_action:  # can return None
            actions.append(tutorital_index_action)
        if file_update_actions:  # can return None
            actions.extend(file_update_actions)

        actions.append(changelog_action)
        return version, actions

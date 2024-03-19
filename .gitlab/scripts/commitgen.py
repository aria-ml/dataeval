from base64 import b64encode
from datetime import datetime, timedelta
from os import path, remove, walk
from re import match
from shutil import move, rmtree
from typing import Any, Dict, List

from gitlab import Gitlab
from verboselog import verbose

CHANGELOG_FILE = "CHANGELOG.md"


class _Entry:
    """
    Extracts the common elements of merge commits and git tags used to
    create the change history contents.  Elements extracted from the
    response are:

    hash - commit hash of commit or tag
    shorthash - first 8 chars of the hash
    description - description in the commit or the name of the tag
    is_tag - whether the hash is a tag or not
    """

    def __init__(self, response: Dict[str, Any]):
        if "merge_commit_sha" in response.keys():
            self.time: datetime = datetime.strptime(
                response["merged_at"][0:19], "%Y-%m-%dT%H:%M:%S"
            ).astimezone()
            self.hash: str = response["merge_commit_sha"]
            self.shorthash: str = response["merge_commit_sha"][0:8]
            self.description: str = (
                response["title"].replace('Resolve "', "").replace('"', "")
            )
            self.is_tag: bool = False
        elif "target" in response.keys():
            self.time: datetime = datetime.strptime(
                response["commit"]["committed_date"], "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            self.hash: str = response["commit"]["id"]
            self.shorthash: str = response["commit"]["short_id"]
            self.description: str = response["name"]
            self.is_tag: bool = True

    def to_markdown(self) -> str:
        if self.is_tag:
            return f"## {self.description}\n"
        else:
            return f"- ```{self.shorthash} - {self.description}```\n"

    def __lt__(self, other: "_Entry") -> bool:
        # handle erroneous case with identical tags"
        if self.is_tag and other.is_tag and self.time == other.time:
            return self.description < other.description
        else:
            return self.time < other.time

    def __repr__(self) -> str:
        entry_type = "TAG" if self.is_tag else "COM"
        return f"Entry({entry_type}, {self.time}, {self.hash}, {self.description})"


class CommitGen:
    """
    Generates commit payload for changes for used in the documentation cache
    and changelog updates
    """

    def __init__(self, gitlab: Gitlab, verbose: bool = False):
        self.gl = gitlab
        self.verbose = verbose

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

    def _get_entries(self) -> List[_Entry]:
        entries: List[_Entry] = list()

        for response in self.gl.list_merge_requests(
            state="merged", target_branch="main"
        ):
            entries.append(_Entry(response))

        for response in self.gl.list_tags():
            if not match(r"v[0-9]+.[0-9]+.?[0-9]*", response["name"]):
                continue
            entry = _Entry(response)
            hashmatch = list(filter(lambda x: x.hash == entry.hash, entries))
            if hashmatch:
                entry.time = hashmatch[0].time + timedelta(seconds=1)
            entries.append(entry)

        # Entries retrieved are not in chronological order, sort before check
        entries.sort(reverse=True)
        return entries

    def _generate_changelog_action(self) -> Dict[str, str]:
        current = self._read_changelog()
        last_hash = self._get_last_hash(current[0]) if current else ""

        # Return empty dict if nothing to update
        entries = self._get_entries()
        if last_hash == entries[0].hash:
            return dict()

        lines: List[str] = list()

        lines.append(f"[//]: # ({entries[0].hash})\n")
        lines.append("\n")
        lines.append("# DAML Change Log\n")

        # If we have a change and no release yet
        if not entries[0].is_tag:
            lines.append("## Pending Release\n")

        for entry in entries:
            if entry.hash == last_hash:
                break
            lines.append((entry.to_markdown()))

        # If we had a pending release we can drop this as there are new changes
        for oldline in current[3:]:
            if oldline == "## Pending Release\n":
                continue
            lines.append(oldline)

        content = "".join(lines)

        return {
            "action": "update",
            "file_path": CHANGELOG_FILE,
            "encoding": "text",
            "content": content,
        }

    def _is_binary_file(self, file: str) -> bool:
        try:
            with open(file, "rt") as f:
                f.read()
                return False
        except Exception:
            return True

    def _generate_actions(
        self, old_files: List[str], new_files: List[str]
    ) -> List[Dict[str, str]]:
        actions: List[Dict[str, str]] = list()

        for old_file in old_files:
            if old_file not in new_files:
                actions.append(
                    {
                        "actions": "delete",
                        "file_path": old_file,
                    }
                )

        for new_file in new_files:
            if self._is_binary_file(new_file):
                encoding = "base64"
                content = b64encode(open(new_file, "rb").read()).decode()
            else:
                encoding = "text"
                content = open(new_file, "rt").read()

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
        file_paths: List[str] = list()
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
            raise FileNotFoundError(
                f"Artifacts downloaded from {ref} does not contain {output_path}"
            )
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
            return list()

        actions = self._generate_jupyter_cache_actions()
        actions.append(changelog_action)
        return actions

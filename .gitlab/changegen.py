from datetime import datetime, timedelta
from os import path, remove
from typing import Any, Dict, List, Literal

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
    skip - whether to skip this entry or not (skips merges from develop into main)
    """

    def __init__(self, response: Dict[str, Any]):
        if "merge_commit_sha" in response.keys():
            self.time = datetime.strptime(
                response["merged_at"][0:19], "%Y-%m-%dT%H:%M:%S"
            ).astimezone()
            self.hash = response["merge_commit_sha"]
            self.shorthash = response["merge_commit_sha"][0:8]
            self.description = (
                response["title"].replace('Resolve "', "").replace('"', "")
            )
            self.is_tag = False
            self.skip = (
                response["source_branch"] == "develop"
                and response["target_branch"] == "main"
            )
        elif "target" in response.keys():
            self.time = datetime.strptime(
                response["commit"]["committed_date"], "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            self.hash = response["commit"]["id"]
            self.shorthash = response["commit"]["short_id"]
            self.description = response["name"]
            self.is_tag = True
            self.skip = False

    def to_markdown(self) -> str:
        if self.is_tag:
            return f"## {self.description}\n"
        else:
            return f"- ```{self.shorthash} - {self.description}```\n"

    def __lt__(self, other) -> bool:
        return self.time < other.time

    def __repr__(self) -> str:
        entry_type = "TAG" if self.is_tag else "COM"
        return f"Entry({entry_type}, {self.time}, {self.hash}, {self.description})"


class ChangeGen:
    """
    Generates lists of changes for use in the changelog updates or for
    merge request descriptions
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
            state="merged", target_branch="develop"
        ):
            entries.append(_Entry(response))

        for response in self.gl.list_tags():
            entry = _Entry(response)
            match = list(filter(lambda x: x.hash == entry.hash, entries))
            if match:
                entry.time = match[0].time + timedelta(seconds=1)
            entries.append(entry)

        # Entries retrieved are not in chronological order, sort before check
        entries.sort(reverse=True)
        return entries

    def generate(self, target: Literal["merge", "changelog"]) -> Dict[str, str]:
        """
        Generates content to use for merges or changelogs.

        Content are merge commit titles in chronological order. Changelog
        generation will recreate the full changelog file content. Merge
        generation will create a snippet of new merge commits since the
        last release.

        Parameters
        ---------
        target : Literal["merge", "changelog"]
            The type of content to generate

        Returns
        -------
        Dict[str, str]
            The content required to update the changelog file or to post an MR
        """
        current = self._read_changelog()
        last_hash = self._get_last_hash(current[0]) if current else ""

        # Return empty dict if nothing to update
        entries = self._get_entries()
        if last_hash == entries[0].hash:
            return dict()

        lines: List[str] = list()

        if target == "changelog":
            lines.append(f"[//]: # ({entries[0].hash})\n")
            lines.append("\n")
            lines.append("# DAML Change Log\n")

        # If we have a change and no release yet
        if not entries[0].is_tag:
            lines.append("## Pending Release\n")

        for entry in entries:
            if entry.hash == last_hash:
                break
            if entry.skip:
                continue
            lines.append((entry.to_markdown()))

        if target == "changelog":
            # If we had a pending release we can drop this as there are new changes
            for oldline in current[3:]:
                if oldline == "## Pending Release\n":
                    continue
                lines.append(oldline)

        # fmt: off
        if target == "merge":
            lines.append("\n")
            lines.append("## Criteria For Approval\n")
            lines.append("- [ ] Ensure all features are complete and ready to ship\n")
            lines.append("- [ ] Review features released for test coverage and documentation\n")  # noqa: E501
            lines.append("- [ ] Review documentation generated in pipeline\n")
            lines.append("- [ ] Ensure all security scans have no critical issues and review any pre-existing issues\n")  # noqa: E501
        # fmt: on

        content = "".join(lines)

        new_shorthash = entries[0].shorthash
        change: Dict[str, str] = dict()
        if target == "changelog":
            change["commit_message"] = f"Update CHANGELOG.md with {new_shorthash}"
            change["content"] = content
        elif target == "merge":
            change["description"] = content

        return change

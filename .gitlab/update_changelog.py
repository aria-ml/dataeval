from datetime import datetime, timedelta
from os import environ
from typing import Any, Dict, List, Optional

from requests import Response, get, put


class ChangelogEntry:
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
            self.isTag = False
        elif "target" in response.keys():
            self.time = datetime.strptime(
                response["commit"]["committed_date"], "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            self.hash = response["target"]
            self.shorthash = response["commit"]["short_id"]
            self.description = response["name"]
            self.isTag = True

    def to_markdown(self) -> str:
        if self.isTag:
            return f"## {self.description}\n"
        else:
            return f"- ```{self.shorthash} - {self.description}```\n"

    def __lt__(self, other) -> bool:
        return self.time < other.time

    def __repr__(self) -> str:
        return f"ChangelogEntry({self.time}, {self.hash}, {self.description})"


def request(cmd: str, urlPath: str, data: Optional[Dict[str, Any]] = None) -> Response:
    # $DAML_BUILD_PAT is only available in production environments
    headers = {"PRIVATE-TOKEN": environ["DAML_BUILD_PAT"]}
    url = "https://gitlab.jatic.net/api/v4/projects/151/" + urlPath

    if cmd == "put":
        return put(url, headers=headers, data=body, timeout=10)
    elif cmd == "get":
        return get(url, headers=headers, timeout=10)
    else:
        raise ValueError("Unsupported request type")


if __name__ == "__main__":
    entries: List[ChangelogEntry] = list()

    for response in request("get", "merge_requests?state=merged").json():
        entries.append(ChangelogEntry(response))

    for response in request("get", "repository/tags").json():
        entry = ChangelogEntry(response)
        match = list(filter(lambda x: x.hash == entry.hash, entries))
        if match:
            entry.time = match[0].time + timedelta(seconds=1)
        entries.append(entry)

    with open("CHANGELOG.md") as file:
        current = file.readlines()

    if current:
        start = current[0].find("(") + 1
        end = current[0].find(")")
        lastHash = current[0][start:end]
    else:
        lastHash = ""

    # Just bail early if everything is up to date
    if lastHash == entries[0].hash:
        exit()

    lines: List[str] = list()
    entries.sort(reverse=True)
    lines.append(f"[//]: # ({entries[0].hash})\n")
    lines.append("\n")
    lines.append("# DAML Change Log\n")

    # If we have a change and no release yet
    if not entries[0].isTag:
        lines.append("## Pending Release\n")

    for entry in entries:
        if entry.hash == lastHash:
            break
        lines.append((entry.to_markdown()))

    # If we had a pending release we can drop this as there are new changes
    for oldline in current[3:]:
        if oldline == "## Pending Release\n":
            continue
        lines.append(oldline)

    content = "".join(lines)

    body: Dict[str, str] = dict()
    body["branch"] = "main"
    body["commit_message"] = f"Update CHANGELOG.md with {entries[0].shorthash}"
    body["content"] = content

    print(body["commit_message"])

    # GPG signature requirements prevent REST API commits
    # response = request("put", "repository/files/CHANGELOG.md", body)
    # print(response.json())

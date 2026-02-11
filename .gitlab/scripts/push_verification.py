#!/usr/bin/env python3
"""Push generated verification artifacts to the DataEval meta repo.

Reads markdown files from verification/reports/metarepo/ and commits them
to the meta repo (project 409) via the GitLab API.

Requires:
  - DATAEVAL_BUILD_PAT environment variable (GitLab personal access token)
  - Generated artifacts from verification/generate_metarepo.py

Usage:
  python .gitlab/scripts/push_verification.py [--dry-run]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from requests import get, post
from rest import RestWrapper

METAREPO_PROJECT_URL = "https://gitlab.jatic.net/api/v4/projects/409/"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "verification" / "reports" / "metarepo"


class MetaRepo(RestWrapper):
    def __init__(self) -> None:
        super().__init__(METAREPO_PROJECT_URL, "DATAEVAL_BUILD_PAT", verbose=True)
        self.headers = {"PRIVATE-TOKEN": self.token}

    def list_tree(self, path: str = "", ref: str = "main") -> list[dict]:
        return self._request(get, "repository/tree", {"path": path, "ref": ref, "per_page": "100"})

    def commit(self, branch: str, message: str, actions: list[dict]) -> dict:
        return self._request(
            post,
            "repository/commits",
            None,
            {
                "branch": branch,
                "commit_message": message,
                "actions": actions,
            },
        )


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    repo = MetaRepo()

    # Discover existing files in the meta repo so we know create vs update
    existing_paths: set[str] = set()
    for tree_path in ("", "test-cases"):
        try:
            tree = repo.list_tree(tree_path)
            for item in tree:
                existing_paths.add(item["path"])
        except ConnectionError:
            pass  # directory may not exist yet

    # Build commit actions from generated files
    actions: list[dict] = []

    # Test case files
    tc_dir = OUTPUT_DIR / "test-cases"
    if tc_dir.exists():
        for f in sorted(tc_dir.glob("*.md")):
            file_path = f"test-cases/{f.name}"
            action = "update" if file_path in existing_paths else "create"
            actions.append({"action": action, "file_path": file_path, "content": f.read_text()})

    # VCRM
    vcrm_path = OUTPUT_DIR / "vcrm.md"
    if vcrm_path.exists():
        action = "update" if "vcrm.md" in existing_paths else "create"
        actions.append({"action": action, "file_path": "vcrm.md", "content": vcrm_path.read_text()})

    if not actions:
        print("No files to push.")
        return

    # Build commit message
    version = os.environ.get("CI_COMMIT_TAG", os.environ.get("DATAEVAL_VERSION", "dev"))
    message = f"Update verification artifacts for DataEval {version}"

    print(f"Commit message: {message}")
    print(f"Pushing {len(actions)} file(s) to meta repo:")
    for a in actions:
        print(f"  {a['action']}: {a['file_path']}")

    if dry_run:
        print("\n--dry-run: skipping commit")
        return

    result = repo.commit("main", message, actions)
    commit_id = result.get("id", "unknown")
    print(f"\nCommitted: {commit_id}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Push generated verification artifacts to the DataEval meta repo.

Commits the output of ``verification/generate_metarepo.py`` to the meta repo
via the GitLab API. Target project and directory come from
``verification/registry.yaml``:

    metarepo:
      project_id: 409
      path: DataEval

Only files this project generates are ever written:

    output/metarepo/vcrm.md              -> <path>/vcrm.md
    output/metarepo/requirements/*.md    -> <path>/requirements/*.md
    output/metarepo/test-cases/*.md      -> <path>/test-cases/*.md

Requirements, test cases, and the VCRM are all generated from the registry.
Nothing is ever deleted by default: files that exist remotely but are no longer
generated are reported as stale and left alone. ``--prune`` (or
``PRUNE_STALE=1``) opts into deleting stale ``requirements/FR-*.md``,
``requirements/NFR-*.md`` and ``test-cases/test-case-*.md`` files only, and
never touches anything else, such as ``archive/`` or ``proposed/``.

Requires:
  - DATAEVAL_BUILD_PAT environment variable (GitLab personal access token)
  - Generated artifacts from verification/generate_metarepo.py

Usage:
  python3 .gitlab/scripts/push_verification.py [--dry-run] [--prune]
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml
from requests import get, post
from rest import RestError, RestWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "verification" / "registry.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output" / "metarepo"

# Generated files are requirements/FR-*.md, requirements/NFR-*.md, and
# test-cases/test-case-<n>-<k>.md. Only files matching these patterns, directly
# inside those two directories, are eligible for --prune.
MANAGED_RE = {
    "requirements": re.compile(r"^N?FR-[\w.-]+\.md$"),
    "test-cases": re.compile(r"^test-case-[\w.-]+\.md$"),
}


def load_metarepo_config() -> tuple[int, str]:
    """Return the meta repo project id and the directory this project owns."""
    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)
    metarepo = registry["metarepo"]
    path = str(metarepo.get("path", "")).strip("/")
    if not path:
        raise SystemExit(f"error: metarepo.path is not set in {REGISTRY_PATH}")
    return int(metarepo["project_id"]), path


class MetaRepo(RestWrapper):
    """GitLab REST client scoped to the meta repo project."""

    def __init__(self, project_id: int) -> None:
        """Authenticate against the meta repo project using DATAEVAL_BUILD_PAT."""
        project_url = f"https://gitlab.jatic.net/api/v4/projects/{project_id}/"
        super().__init__(project_url, "DATAEVAL_BUILD_PAT", verbose=True)
        self.headers = {"PRIVATE-TOKEN": self.token}

    def list_tree(self, path: str = "", ref: str = "main") -> list[dict]:
        """List blobs under ``path``, recursively. Returns [] if it doesn't exist yet."""
        try:
            return self._request(
                get,
                "repository/tree",
                {"path": path, "ref": ref, "recursive": "true", "per_page": "100"},
            )
        except RestError as e:
            if e.status_code == 404:
                return []  # directory not created yet — every file is a create
            raise

    def commit(self, branch: str, message: str, actions: list[dict]) -> dict:
        """Apply ``actions`` to ``branch`` as a single commit."""
        return self._request(
            post,
            "repository/commits",
            None,
            {"branch": branch, "commit_message": message, "actions": actions},
        )


def collect_generated(base: str) -> dict[str, Path]:
    """Map meta repo file path -> local source file for everything CI generates."""
    generated: dict[str, Path] = {}

    for sub in MANAGED_RE:
        src = OUTPUT_DIR / sub
        if src.exists():
            for f in sorted(src.glob("*.md")):
                generated[f"{base}/{sub}/{f.name}"] = f

    vcrm = OUTPUT_DIR / "vcrm.md"
    if vcrm.exists():
        generated[f"{base}/vcrm.md"] = vcrm

    return generated


def find_stale(existing: set[str], generated: set[str], base: str) -> list[str]:
    """Remote files of the managed kinds that the registry no longer generates."""
    stale = []
    for path in sorted(existing - generated):
        parts = Path(path).relative_to(base).parts
        if len(parts) == 2 and (rx := MANAGED_RE.get(parts[0])) is not None and rx.match(parts[1]):
            stale.append(path)
    return stale


def refuse_default_branch() -> None:
    """Exit unless this run is allowed to write to the meta repo.

    The meta repo keeps one copy of the requirements, test cases, and VCRM, so whichever
    pipeline publishes last wins. The default branch runs ahead of the release, and its
    results would overwrite the evidence recorded for the released version. Publish from a
    release tag, or from a release branch.
    """
    branch = os.environ.get("CI_COMMIT_BRANCH", "")
    default = os.environ.get("CI_DEFAULT_BRANCH", "main")
    if branch and branch in {default, "main"} and not os.environ.get("CI_COMMIT_TAG"):
        raise SystemExit(
            f"error: refusing to publish verification artifacts from the default branch ({branch}). "
            "Publish from a release tag or a release branch; --dry-run is allowed here."
        )


def main() -> None:
    """Generate the commit plan and, unless ``--dry-run``, push it to the meta repo."""
    dry_run = "--dry-run" in sys.argv
    prune = "--prune" in sys.argv or os.environ.get("PRUNE_STALE") == "1"
    if not dry_run:
        refuse_default_branch()

    project_id, base = load_metarepo_config()
    generated = collect_generated(base)
    if not generated:
        print(f"No generated artifacts under {OUTPUT_DIR} — run `nox -s verify` first.")
        return

    repo = MetaRepo(project_id)
    existing = {item["path"] for item in repo.list_tree(base) if item.get("type") == "blob"}
    print(f"Meta repo project {project_id}, managing {base}/ ({len(existing)} existing file(s))")

    actions = [
        {
            "action": "update" if remote in existing else "create",
            "file_path": remote,
            "content": local.read_text(),
        }
        for remote, local in generated.items()
    ]

    # Anything present remotely that CI does not generate. Files of the managed
    # kinds are stale; everything else is left alone.
    stale = find_stale(existing, set(generated), base)
    preserved = sorted(existing - set(generated) - set(stale))

    version = os.environ.get("CI_COMMIT_TAG") or os.environ.get("DATAEVAL_VERSION") or "unreleased"
    message = f"Update verification artifacts for DataEval {version}"

    print(f"Commit message: {message}")
    print(f"Pushing {len(actions)} file(s):")
    for a in sorted(actions, key=lambda a: a["file_path"]):
        print(f"  {a['action']}: {a['file_path']}")

    if preserved:
        print(f"\nLeaving {len(preserved)} other file(s) untouched:")
        for p in preserved:
            print(f"  keep: {p}")

    if stale:
        verb = "Deleting" if prune else "Stale (use --prune to delete)"
        print(f"\n{verb}: {len(stale)} requirement or test case file(s) no longer in registry.yaml:")
        for p in stale:
            print(f"  {'delete' if prune else 'stale'}: {p}")
        if prune:
            actions += [{"action": "delete", "file_path": p} for p in stale]

    if dry_run:
        print("\n--dry-run: skipping commit")
        return

    result = repo.commit("main", message, actions)
    print(f"\nCommitted: {result.get('id', 'unknown')}")


if __name__ == "__main__":
    main()

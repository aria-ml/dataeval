#!/usr/bin/env python3
"""Validate the inputs that Jupyter's per-notebook code-cell hashes cannot see.

GitLab records this manifest only after a successful docs build. Checking the
manifest from the restored cache (rather than the previous commit) also handles
missed builds, older branch caches, and fallbacks to docs-artifacts/main.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Notebook code cells have their own hashes. Keep the shared execution inputs
# here, so editing one notebook or a Markdown page doesn't invalidate every one.
INPUT_PATHS = (
    "src",
    "uv.lock",
    "pyproject.toml",
    "noxfile.py",
    ".cuda-version",
    ".gitlab-ci.yml",
    ".gitlab/ci/base.yml",
    ".gitlab/ci/docs.yml",
    "docs/notebook_cache_inputs.py",
    "docs/check_notebook_cache.py",
    "docs/source/conf.py",
    "docs/source/data.py",
    "docs/source/notebooks/jupytext.toml",
    "docs/source/notebooks/models",
)
INPUT_ENV = ("DOCS_PYTHON_VERSION", "DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "DOCS_RUNNER_TAG")
MANIFEST = Path("docs/source/.jupyter_cache/execution-inputs.json")


def execution_inputs(root: Path) -> dict:
    """Hash tracked and new, non-ignored inputs, including working-tree edits."""
    names = (
        subprocess
        .check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *INPUT_PATHS],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    return {
        "files": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(set(names) - {""})
            if (root / name).is_file()
        },
        "environment": {name: os.environ.get(name, "") for name in INPUT_ENV},
    }


def cache_miss_reasons(previous: object, current: dict) -> list[str]:
    """Describe differences without exposing long, unhelpful file hashes."""
    if (
        not isinstance(previous, dict)
        or previous.keys() != current.keys()
        or any(
            not isinstance(previous[group], dict)
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in previous[group].items())
            for group in current
        )
    ):
        return ["Invalid execution-input manifest: expected 'files' and 'environment' mappings of strings."]

    reasons = []
    old_files, new_files = previous["files"], current["files"]
    for name in sorted(old_files.keys() | new_files.keys()):
        if name not in old_files:
            reasons.append(f"File added: {name}")
        elif name not in new_files:
            reasons.append(f"File removed: {name}")
        elif old_files[name] != new_files[name]:
            reasons.append(f"File changed: {name}")

    old_env, new_env = previous["environment"], current["environment"]
    for name in sorted(old_env.keys() | new_env.keys()):
        if name not in old_env or name not in new_env or old_env[name] != new_env[name]:
            before = repr(old_env[name]) if name in old_env else "<missing>"
            after = repr(new_env[name]) if name in new_env else "<missing>"
            reasons.append(f"Environment changed: {name}: {before} -> {after}")
    return reasons


def main() -> int:
    """Check before building, or record the inputs after a successful build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "record"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    manifest = root / MANIFEST
    current = execution_inputs(root)

    if args.action == "record":
        # Do not create a cache for a skipped or unsuccessful docs build.
        manifest.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print("Recorded notebook execution inputs.")
        return 0

    try:
        previous = json.loads(manifest.read_text())
    except FileNotFoundError:
        reasons = [f"Missing execution-input manifest: {MANIFEST}"]
    except OSError as error:
        reasons = [f"Cannot read execution-input manifest: {MANIFEST} ({error})"]
    except ValueError as error:
        reasons = [f"Invalid execution-input manifest: {MANIFEST} ({error})"]
    else:
        if previous == current:
            print("Notebook execution inputs match; reusing cached outputs.")
            return 0
        reasons = cache_miss_reasons(previous, current)

    print("Notebook execution cache miss; a clean build is required.")
    try:
        source = manifest.with_name(".cache_source").read_text().strip()
    except OSError:
        source = ""
    if source:
        print(f"  Restored cache source: {source}")
    for reason in reasons:
        print(f"  - {reason}")
    print("All notebooks will be re-executed because the shared execution inputs could not be validated.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

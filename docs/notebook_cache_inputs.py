#!/usr/bin/env python3
"""Validate the inputs that Jupyter's per-notebook code-cell hashes cannot see.

GitLab records this manifest only after a successful docs build. Checking the
manifest from the restored cache (rather than the previous commit) also handles
missed builds, older branch caches, and fallbacks to docs-artifacts/main.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Notebook code cells have their own hashes. Keep the shared execution inputs
# here, so editing one notebook or a Markdown page doesn't invalidate every one.
INPUT_PATHS = ("src", "uv.lock", "noxfile.py", "docs/datasets.toml")
INPUT_ENV = ("DOCS_PYTHON_VERSION", "DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "DOCS_RUNNER_TAG", "CI_JOB_IMAGE")
# Sessions the docs job never runs. An edit to one of them must not re-execute the notebooks. A session
# that is not listed counts as relevant, so a new session costs one extra clean build until it is listed.
IGNORED_SESSIONS = ("dev", "test", "verify", "type", "deps", "lint", "doctest", "lock", "docsync", "check")
MANIFEST = Path("docs/source/.jupyter_cache/execution-inputs.json")


def module_name(path: str) -> str:
    """Map a repository path such as `src/dataeval/a/__init__.py` to its module name `dataeval.a`."""
    parts = Path(path).with_suffix("").parts[1:]
    return ".".join(parts[:-1] if parts and parts[-1] == "__init__" else parts)


def _with_parents(name: str, known: set[str]) -> set[str]:
    parts = name.split(".")
    return {".".join(parts[: i + 1]) for i in range(len(parts))} & known


def imports_of(text: str, module: str, is_package: bool, known: set[str]) -> set[str]:
    """Return the modules in `known` that `text` imports, each with its parent packages.

    Every import statement counts wherever it sits (module level, inside a function or under
    `TYPE_CHECKING`), so this over-approximates what runs. Raises `SyntaxError` if `text` does not parse.
    """
    # ponytail: file-level closure, so a shared file such as protocols.py re-executes every notebook.
    # Symbol level needs recorded execution or a call graph that Python's dynamic dispatch makes unsound.
    found: set[str] = set()
    package = module if is_package else module.rpartition(".")[0]
    package_parts = package.split(".") if package else []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found |= _with_parents(alias.name, known)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package_parts[: max(len(package_parts) - (node.level - 1), 0)]
                base = ".".join([*base_parts, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            found |= _with_parents(base, known)
            for alias in node.names:
                candidate = f"{base}.{alias.name}" if base else alias.name
                if alias.name != "*" and candidate in known:
                    found |= _with_parents(candidate, known)
    return found


def closure(seeds: set[str], graph: dict[str, set[str]]) -> set[str]:
    """Return everything reachable from `seeds` through `graph`, seeds included."""
    seen: set[str] = set()
    todo = list(seeds)
    while todo:
        node = todo.pop()
        if node not in seen:
            seen.add(node)
            todo.extend(graph.get(node, ()))
    return seen


def noxfile_digest(text: str) -> str:
    """Hash `noxfile.py` without the sessions in `IGNORED_SESSIONS`, which the docs job never runs.

    Module-level code, helpers, the `docs` session and any unlisted session stay in. The digest hashes
    raw source lines, so it does not depend on the Python version. Raises `SyntaxError` if `text` does not parse.
    """
    dropped: set[int] = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in IGNORED_SESSIONS and node.decorator_list:
            dropped.update(range(node.decorator_list[0].lineno, (node.end_lineno or node.lineno) + 1))
    kept = "".join(line for number, line in enumerate(text.splitlines(keepends=True), start=1) if number not in dropped)
    return hashlib.sha256(kept.encode()).hexdigest()


def execution_inputs(root: Path) -> dict:
    """Hash tracked and new, non-ignored inputs, including working-tree edits.

    Raises `ValueError` if `noxfile.py` cannot be parsed, because its digest cannot be trusted.
    """
    names = (
        subprocess
        .check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *INPUT_PATHS],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(set(names) - {""})
        if (root / name).is_file()
    }
    if "noxfile.py" in files:
        try:
            files["noxfile.py"] = noxfile_digest((root / "noxfile.py").read_text())
        except (SyntaxError, UnicodeDecodeError) as error:
            raise ValueError(f"Cannot parse noxfile.py: {error}") from error
    return {"files": files, "environment": {name: os.environ.get(name, "") for name in INPUT_ENV}}


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


def report_miss(manifest: Path, reasons: list[str]) -> int:
    """Print why the cache cannot be reused and return the exit code that asks for a clean build."""
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


def main() -> int:
    """Check before building, or record the inputs after a successful build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "record"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    manifest = root / MANIFEST
    try:
        current = execution_inputs(root)
    except ValueError as error:
        if args.action == "record":
            raise
        return report_miss(manifest, [str(error)])

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

    return report_miss(manifest, reasons)


if __name__ == "__main__":
    raise SystemExit(main())

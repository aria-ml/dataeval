#!/usr/bin/env python3
"""Decide which cached notebook outputs the docs build can reuse.

A notebook is re-executed when a shared input changed or when a source file it imports changed. GitLab records
the manifest only after a successful docs build. Checking the manifest from the restored cache (rather than the
previous commit) also handles missed builds, older branch caches, and fallbacks to docs-artifacts/main.

`check` exits 1 when everything must be re-executed. It exits 0 when the cache can be reused, after deleting the
cached outputs of the notebooks that import a changed source file.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

# Inputs that can change what a notebook prints. Each notebook's code cells have their own hashes and the source
# files it imports are tracked per notebook, so editing one notebook or a Markdown page does not invalidate every
# one. File hashes are raw bytes on purpose: cached warnings embed `src/...py:LINE`, so even a comment edit must
# re-execute the notebooks that import the file.
# Deliberately not tracked, because none of these changes a notebook's output:
#   pyproject.toml          uv.lock pins the dependencies, and `nox -s check` keeps the two in step
#   .cuda-version           the docs job sets DATAEVAL_NOX_UV_EXTRAS_OVERRIDE itself, which is tracked
#   the CI YAML files       the Python version, extras, runner tag and image name are tracked as environment values
#   docs/source/conf.py     sets no kernel environment, and a failing notebook still fails the build
#   jupytext.toml, the two docs cache scripts   file pairing and tooling
#   other nox sessions      see IGNORED_SESSIONS
# Not tracked and not detectable here: imports the static analysis cannot see, a mutable image tag, and packages
# fetched at run time. The nightly clean, version-tag builds and the `DOCS_CLEAN` variable are the backstops.
INPUT_PATHS = ("src", "uv.lock", "noxfile.py", "docs/datasets.toml")
INPUT_ENV = ("DOCS_PYTHON_VERSION", "DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "DOCS_RUNNER_TAG", "CI_JOB_IMAGE")
# Sessions the docs job never runs. An edit to one of them must not re-execute the notebooks. A session
# that is not listed counts as relevant, so a new session costs one extra clean build until it is listed.
IGNORED_SESSIONS = ("dev", "test", "verify", "type", "deps", "lint", "doctest", "lock", "docsync", "check")
MANIFEST = Path("docs/source/.jupyter_cache/execution-inputs.json")
CACHE = MANIFEST.parent
NOTEBOOKS = Path("docs/source/notebooks")


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
                if alias.name == "*":
                    # A star import brings in any submodule that `__all__` names, imported or not.
                    found |= {name for name in known if name.rpartition(".")[0] == base}
                    continue
                candidate = f"{base}.{alias.name}" if base else alias.name
                if candidate in known:
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
        # An added or changed source file is decided per notebook. A removed one is not: the current tree
        # cannot show who imported it (a guarded import changes behavior without changing its own text).
        per_notebook = name.startswith("src/") and name.endswith(".py")
        if name not in old_files:
            if not per_notebook:
                reasons.append(f"File added: {name}")
        elif name not in new_files:
            reasons.append(f"File removed: {name}")
        elif old_files[name] != new_files[name] and not per_notebook:
            reasons.append(f"File changed: {name}")

    old_env, new_env = previous["environment"], current["environment"]
    for name in sorted(old_env.keys() | new_env.keys()):
        if name not in old_env or name not in new_env or old_env[name] != new_env[name]:
            before = repr(old_env[name]) if name in old_env else "<missing>"
            after = repr(new_env[name]) if name in new_env else "<missing>"
            reasons.append(f"Environment changed: {name}: {before} -> {after}")
    return reasons


def changed_sources(previous: dict[str, str], current: dict[str, str]) -> set[str]:
    """Return the `src/**/*.py` files in `current` that are new or differ from `previous`."""
    return {
        name
        for name, digest in current.items()
        if name.startswith("src/") and name.endswith(".py") and previous.get(name) != digest
    }


def _imports_from_file(root: Path, path: str, module: str, is_package: bool, known: set[str]) -> set[str]:
    try:
        return imports_of((root / path).read_text(), module, is_package, known)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ValueError(f"Cannot parse {path}: {error}") from error


def affected_notebooks(root: Path, files: dict[str, str], changed: set[str]) -> dict[str, list[str]]:
    """Map each notebook whose import closure contains a changed file to those changed files.

    The closure is built from the current tree: `files` lists the `src/**/*.py` modules and `root` holds them.
    Raises `ValueError("Cannot parse <path>: ...")` if a source file or notebook cannot be parsed.
    """
    sources = {module_name(path): path for path in files if path.startswith("src/") and path.endswith(".py")}
    known = set(sources)
    graph = {
        module: _imports_from_file(root, path, module, path.endswith("__init__.py"), known)
        for module, path in sources.items()
    }
    affected: dict[str, list[str]] = {}
    for notebook in sorted((root / NOTEBOOKS).glob("*.py")):
        seeds = _imports_from_file(root, f"{NOTEBOOKS}/{notebook.name}", "__notebook__", False, known)
        hit = sorted(sources[module] for module in closure(seeds, graph) if sources[module] in changed)
        if hit:
            affected[notebook.name] = hit
    return affected


def prune_cache(root: Path, stale: Iterable[str], current: Iterable[str]) -> list[str]:
    """Delete the cached executions of the `stale` notebooks, found through jupyter-cache's own database.

    jupyter-cache keys an entry on the notebook's code cells and keeps the notebook name of the build that created
    it, so a notebook renamed without a code change still has its entry under the old name. An entry whose name is
    not in `current` cannot be attributed to a notebook, so it is deleted too, at the cost of one extra execution.
    Two notebooks with identical code cells share one entry and one name; that case is not covered.

    Returns the hash keys whose `executed/` folder was deleted. Nothing stale, no database, no folder or no matching
    row is a no-op; `--fix` later drops the orphaned rows. Raises `ValueError` if the database cannot be read.
    """
    names = set(stale)
    database = root / CACHE / "global.db"
    if not names or not database.is_file():
        return []
    known = set(current)
    try:
        with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
            rows = connection.execute("SELECT hashkey, uri FROM nbcache").fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"Cannot read cache database: {error}") from error
    deleted = []
    for hashkey, uri in rows:
        folder = root / CACHE / "executed" / hashkey
        name = os.path.basename(uri or "")
        if (name in names or name not in known) and folder.is_dir():
            shutil.rmtree(folder)
            deleted.append(hashkey)
    return sorted(deleted)


def reuse_lines(affected: dict[str, list[str]], total: int) -> list[str]:
    """Explain the re-executions: one sorted line per changed file, naming the notebooks unless it is all of them."""
    by_file: dict[str, list[str]] = {}
    for notebook, paths in affected.items():
        for path in paths:
            by_file.setdefault(path, []).append(notebook)
    lines = []
    for path in sorted(by_file):
        names = sorted(by_file[path])
        if len(names) == total:
            lines.append(f"  - {path} changed: re-executing all {total} notebooks")
        else:
            lines.append(f"  - {path} changed: re-executing {len(names)} of {total} notebooks: {', '.join(names)}")
    return lines


def check_notebooks(root: Path, manifest: Path, previous: dict[str, str], current: dict[str, str]) -> int:
    """Prune the notebooks that import a changed source file and report what is re-executed."""
    changed = changed_sources(previous, current)
    notebooks = sorted(path.name for path in (root / NOTEBOOKS).glob("*.py"))
    try:
        affected = affected_notebooks(root, current, changed) if changed else {}
        prune_cache(root, affected, notebooks)
    except ValueError as error:
        return report_miss(manifest, [str(error)])
    if not affected:
        print("Notebook execution inputs changed, but no notebook imports a changed file; reusing cached outputs.")
        return 0
    total = len(notebooks)
    print(f"Notebook execution inputs changed; re-executing {len(affected)} of {total} notebooks and reusing the rest.")
    for line in reuse_lines(affected, total):
        print(line)
    return 0


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
        if not reasons:
            return check_notebooks(root, manifest, previous["files"], current["files"])

    return report_miss(manifest, reasons)


if __name__ == "__main__":
    raise SystemExit(main())

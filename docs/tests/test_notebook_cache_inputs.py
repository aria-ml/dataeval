"""Exercise docs cache invalidation against real Git checkouts and saved manifests."""

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "docs/notebook_cache_inputs.py"
MANIFEST = "docs/source/.jupyter_cache/execution-inputs.json"


def load_script():
    spec = importlib.util.spec_from_file_location("notebook_cache_inputs", ROOT / SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache_inputs = load_script()


def write(root: Path, name: str, content: str = "original\n") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def run(root: Path, action: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, SCRIPT, action],
        cwd=root,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


NOX = """import nox

DOCS_ENVS = {"LANG": "C"}


@nox.session
def test(session):
    session.run("pytest")


@nox.session
def docs(session):
    session.run("sphinx-build")
"""


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for name in (
        "src/dataeval/example.py",
        "uv.lock",
        "pyproject.toml",
        "noxfile.py",
        ".cuda-version",
        ".gitlab-ci.yml",
        ".gitlab/ci/base.yml",
        ".gitlab/ci/docs.yml",
        "docs/check_notebook_cache.py",
        "docs/source/conf.py",
        "docs/datasets.toml",
        "docs/source/notebooks/jupytext.toml",
        "docs/source/notebooks/example.py",
        "docs/source/guide.md",
        "tests/example.py",
    ):
        write(tmp_path, name)
    write(tmp_path, "noxfile.py", NOX)
    shutil.copyfile(ROOT / SCRIPT, tmp_path / SCRIPT)
    write(tmp_path, ".gitignore", "__pycache__/\n_version.py\n.jupyter_cache/\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    (tmp_path / MANIFEST).parent.mkdir(parents=True)
    assert run(tmp_path, "record").returncode == 0
    assert run(tmp_path, "check").returncode == 0
    return tmp_path


@pytest.mark.parametrize(
    "name",
    ["uv.lock", "docs/datasets.toml"],
)
def test_shared_execution_input_changes_require_clean_build(checkout: Path, name: str):
    with (checkout / name).open("a") as stream:
        stream.write("\n# changed\n")
    result = run(checkout, "check")
    assert result.returncode == 1
    assert f"File changed: {name}" in result.stdout
    assert "All notebooks will be re-executed" in result.stdout


@pytest.mark.parametrize(
    "name",
    [
        "pyproject.toml",
        ".cuda-version",
        ".gitlab-ci.yml",
        ".gitlab/ci/base.yml",
        ".gitlab/ci/docs.yml",
        SCRIPT,
        "docs/check_notebook_cache.py",
        "docs/source/conf.py",
        "docs/source/notebooks/jupytext.toml",
    ],
)
def test_dropped_inputs_do_not_force_clean_build(checkout: Path, name: str):
    with (checkout / name).open("a") as stream:
        stream.write("\n# changed\n")
    result = run(checkout, "check")
    assert result.returncode == 0
    assert "cache miss" not in result.stdout


@pytest.mark.parametrize("name", ["src/dataeval/data.json"])
def test_new_execution_input_requires_clean_build(checkout: Path, name: str):
    write(checkout, name)
    result = run(checkout, "check")
    assert result.returncode == 1
    assert f"File added: {name}" in result.stdout


def test_deleted_execution_input_requires_clean_build(checkout: Path):
    (checkout / "src/dataeval/example.py").unlink()
    result = run(checkout, "check")
    assert result.returncode == 1
    assert "File removed: src/dataeval/example.py" in result.stdout


@pytest.mark.parametrize(
    "name",
    [
        "docs/source/notebooks/example.py",
        "docs/source/guide.md",
        "tests/example.py",
        "src/dataeval/_version.py",
        "src/dataeval/__pycache__/example.pyc",
    ],
)
def test_notebook_prose_tests_and_generated_files_do_not_force_clean_build(checkout: Path, name: str):
    # Jupyter handles a changed notebook's own code cells separately.
    write(checkout, name, "changed\n")
    assert run(checkout, "check").returncode == 0


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "Missing execution-input manifest"),
        ("invalid json", "Invalid execution-input manifest"),
        ("{}", "Invalid execution-input manifest"),
        ("[]", "Invalid execution-input manifest"),
        ('{"files": [], "environment": {}}', "Invalid execution-input manifest"),
        ('{"files": {"src/example.py": null}, "environment": {}}', "Invalid execution-input manifest"),
    ],
)
def test_legacy_or_invalid_manifest_requires_clean_build(checkout: Path, value: str | None, reason: str):
    manifest = checkout / MANIFEST
    if value is None:
        manifest.unlink()
    else:
        manifest.write_text(value)
    result = run(checkout, "check")
    assert result.returncode == 1
    assert reason in result.stdout
    assert "All notebooks will be re-executed" in result.stdout


def test_unreadable_manifest_explains_why_cache_cannot_be_validated(checkout: Path):
    manifest = checkout / MANIFEST
    manifest.unlink()
    manifest.mkdir()
    result = run(checkout, "check")
    assert result.returncode == 1
    assert f"Cannot read execution-input manifest: {MANIFEST}" in result.stdout


@pytest.mark.parametrize(
    "name", ["DOCS_PYTHON_VERSION", "DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "DOCS_RUNNER_TAG", "CI_JOB_IMAGE"]
)
def test_execution_environment_changes_require_clean_build(checkout: Path, name: str):
    previous = json.loads((checkout / MANIFEST).read_text())["environment"][name]
    result = run(checkout, "check", **{name: "different"})
    assert result.returncode == 1
    assert f"Environment changed: {name}: {previous!r} -> 'different'" in result.stdout


def test_cache_miss_reports_all_changes_and_restored_source(checkout: Path):
    write(checkout, "docs/datasets.toml", "changed\n")
    write(checkout, "src/dataeval/data.json")
    (checkout / "uv.lock").unlink()
    write(checkout, "docs/source/.jupyter_cache/.cache_source", "docs-artifacts/main\n")
    result = run(checkout, "check", DOCS_PYTHON_VERSION="different")
    assert result.returncode == 1
    assert "Restored cache source: docs-artifacts/main" in result.stdout
    assert "File changed: docs/datasets.toml" in result.stdout
    assert "File added: src/dataeval/data.json" in result.stdout
    assert "File removed: uv.lock" in result.stdout
    assert "Environment changed: DOCS_PYTHON_VERSION:" in result.stdout


def test_cache_hit_reports_reuse(checkout: Path):
    result = run(checkout, "check")
    assert result.returncode == 0
    assert "reusing cached outputs" in result.stdout
    assert "cache miss" not in result.stdout


def test_missed_builds_remain_invalid_until_a_successful_build_records_inputs(checkout: Path):
    write(checkout, "uv.lock", "changed lock\n")
    assert run(checkout, "check").returncode == 1
    # A later, unrelated change must not make the old execution cache valid.
    write(checkout, "docs/source/guide.md", "changed guide\n")
    assert run(checkout, "check").returncode == 1
    assert run(checkout, "record").returncode == 0
    assert run(checkout, "check").returncode == 0


def test_manifest_survives_artifact_publish_and_main_fallback(checkout: Path, tmp_path: Path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=checkout, check=True)
    env = {
        **os.environ,
        "CI": "true",
        "DATAEVAL_BUILD_PAT": "",  # Always publish to the temporary local remote.
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GITLAB_USER_NAME": "Cache test",
        "GITLAB_USER_EMAIL": "cache-test@example.invalid",
    }
    manifest = (checkout / MANIFEST).read_bytes()
    subprocess.run(["bash", str(ROOT / "docs/push-docs-cache.sh"), "main"], cwd=checkout, env=env, check=True)
    shutil.rmtree((checkout / MANIFEST).parent)
    # A new feature branch has no artifacts and falls back to main's cache.
    subprocess.run(["bash", str(ROOT / "docs/fetch-docs-cache.sh"), "feature"], cwd=checkout, env=env, check=True)
    assert (checkout / MANIFEST).read_bytes() == manifest
    assert run(checkout, "check").returncode == 0


KNOWN = {"dataeval", "dataeval.a", "dataeval.a.b", "dataeval.a.c", "dataeval.d"}


def imports(text: str, module: str = "x", is_package: bool = False) -> set[str]:
    return cache_inputs.imports_of(text, module, is_package, KNOWN)


def test_module_name_maps_paths_to_modules():
    assert cache_inputs.module_name("src/dataeval/a/__init__.py") == "dataeval.a"
    assert cache_inputs.module_name("src/dataeval/_x.py") == "dataeval._x"
    assert cache_inputs.module_name("src/conftest.py") == "conftest"


def test_imports_resolve_absolute_forms_with_their_parent_packages():
    assert imports("import dataeval.a.b") == {"dataeval", "dataeval.a", "dataeval.a.b"}
    assert imports("from dataeval.a import b, nothing") == {"dataeval", "dataeval.a", "dataeval.a.b"}
    assert imports("from dataeval.a import *") == {"dataeval", "dataeval.a", "dataeval.a.b", "dataeval.a.c"}
    assert imports("from dataeval import *") == {"dataeval", "dataeval.a", "dataeval.d"}
    assert imports("import numpy\nfrom os import path") == set()


def test_imports_resolve_relative_forms_from_a_package_and_from_a_module():
    assert imports("from . import c", "dataeval.a", is_package=True) == {"dataeval", "dataeval.a", "dataeval.a.c"}
    assert imports("from . import c", "dataeval.a.b") == {"dataeval", "dataeval.a", "dataeval.a.c"}
    assert imports("from .. import d", "dataeval.a.b") == {"dataeval", "dataeval.d"}


def test_imports_include_function_level_and_type_checking_imports():
    assert imports("def f():\n    import dataeval.d\n") == {"dataeval", "dataeval.d"}
    assert imports("if TYPE_CHECKING:\n    from dataeval.a import b\n") == {"dataeval", "dataeval.a", "dataeval.a.b"}


def test_imports_raise_syntax_error_for_unparsable_source():
    with pytest.raises(SyntaxError):
        imports("def (")


def test_closure_follows_edges_and_tolerates_cycles():
    graph = {"a": {"b"}, "b": {"a", "c"}, "c": set()}
    assert cache_inputs.closure({"a"}, graph) == {"a", "b", "c"}
    assert cache_inputs.closure({"c"}, graph) == {"c"}


def test_noxfile_digest_ignores_listed_sessions_only():
    digest = cache_inputs.noxfile_digest
    base = digest(NOX)
    assert digest(NOX.replace('"pytest"', '"pytest", "-x"')) == base
    assert digest(NOX.replace('"sphinx-build"', '"sphinx-build", "-W"')) != base
    assert digest(NOX.replace('"C"', '"en"')) != base
    assert digest(NOX + "\n\n@nox.session\ndef brand_new(session):\n    pass\n") != base
    with pytest.raises(SyntaxError):
        digest("def (")


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [('"pytest"', '"pytest", "-x"', 0), ('"sphinx-build"', '"sphinx-build", "-W"', 1), ('"C"', '"en"', 1)],
)
def test_noxfile_edits_force_a_clean_build_only_outside_ignored_sessions(checkout: Path, old: str, new: str, code: int):
    path = checkout / "noxfile.py"
    path.write_text(path.read_text().replace(old, new))
    assert run(checkout, "check").returncode == code


def test_unparsable_noxfile_requires_clean_build(checkout: Path):
    write(checkout, "noxfile.py", "def (\n")
    result = run(checkout, "check")
    assert result.returncode == 1
    assert "Cannot parse noxfile.py" in result.stdout


def test_manifest_from_before_this_change_requires_one_clean_build(checkout: Path):
    manifest = json.loads((checkout / MANIFEST).read_text())
    del manifest["environment"]["CI_JOB_IMAGE"]
    (checkout / MANIFEST).write_text(json.dumps(manifest))
    result = run(checkout, "check")
    assert result.returncode == 1
    assert "Environment changed: CI_JOB_IMAGE: <missing> -> ''" in result.stdout


ENTRIES = {"nb_a.py": "hash_a", "nb_b.py": "hash_b"}


@pytest.fixture
def package(checkout: Path) -> Path:
    files = {
        "src/dataeval/__init__.py": "",
        "src/dataeval/shared.py": "X = 1\n",
        "src/dataeval/alpha.py": "from dataeval import shared\n",
        "src/dataeval/beta.py": "VALUE = 2\n",
        "src/conftest.py": "# fixtures\n",
        "docs/source/notebooks/nb_a.py": "from dataeval import alpha\n",
        "docs/source/notebooks/nb_b.py": "from dataeval import beta\n",
    }
    for name, text in files.items():
        write(checkout, name, text)
    (checkout / "docs/source/notebooks/example.py").unlink()  # the two notebooks above are the only ones
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    with sqlite3.connect(checkout / "docs/source/.jupyter_cache/global.db") as db:
        db.execute("CREATE TABLE nbcache (pk INTEGER PRIMARY KEY, hashkey TEXT NOT NULL UNIQUE, uri TEXT NOT NULL)")
    for name, key in ENTRIES.items():
        add_entry(checkout, name, key)
    assert run(checkout, "record").returncode == 0
    return checkout


def add_entry(root: Path, notebook: str, key: str) -> None:
    cache = root / "docs/source/.jupyter_cache"
    with closing(sqlite3.connect(cache / "global.db")) as db, db:
        db.execute(
            "INSERT INTO nbcache (hashkey, uri) VALUES (?, ?)", (key, f"/builds/x/docs/source/notebooks/{notebook}")
        )
    write(cache / "executed" / key, "base.ipynb", "{}")


def entries(root: Path) -> list[str]:
    return sorted(p.name for p in (root / "docs/source/.jupyter_cache/executed").iterdir())


def test_a_changed_file_re_executes_only_the_notebooks_that_import_it(package: Path):
    write(package, "src/dataeval/shared.py", "X = 2\n")
    result = run(package, "check")
    assert result.returncode == 0
    assert "src/dataeval/shared.py changed: re-executing 1 of 2 notebooks: nb_a.py" in result.stdout
    assert "re-executing 1 of 2 notebooks and reusing the rest" in result.stdout
    assert entries(package) == ["hash_b"]


def test_a_file_every_notebook_imports_re_executes_all(package: Path):
    write(package, "src/dataeval/__init__.py", "# edited\n")
    result = run(package, "check")
    assert "src/dataeval/__init__.py changed: re-executing all 2 notebooks" in result.stdout
    assert entries(package) == []


def test_a_comment_only_edit_still_re_executes_dependents(package: Path):
    write(package, "src/dataeval/alpha.py", "from dataeval import shared  # note\n")
    assert run(package, "check").returncode == 0
    assert entries(package) == ["hash_b"]


@pytest.mark.parametrize("name", ["src/conftest.py", "src/dataeval/unused.py", "src/dataeval/_version.py"])
def test_files_no_notebook_imports_re_execute_nothing(package: Path, name: str):
    write(package, name, "changed\n")
    result = run(package, "check")
    assert result.returncode == 0
    assert "reusing cached outputs" in result.stdout
    assert entries(package) == ["hash_a", "hash_b"]


def test_a_new_file_imported_by_a_changed_file_re_executes_the_importer(package: Path):
    write(package, "src/dataeval/gamma.py", "G = 1\n")
    write(package, "src/dataeval/beta.py", "from dataeval import gamma\n")
    result = run(package, "check")
    assert "src/dataeval/gamma.py changed: re-executing 1 of 2 notebooks: nb_b.py" in result.stdout
    assert entries(package) == ["hash_a"]


def test_removing_a_source_file_requires_clean_build(package: Path):
    (package / "src/dataeval/beta.py").unlink()
    result = run(package, "check")
    assert result.returncode == 1
    assert "File removed: src/dataeval/beta.py" in result.stdout
    assert entries(package) == ["hash_a", "hash_b"]


def test_a_changed_non_python_file_under_src_requires_clean_build(package: Path):
    write(package, "src/dataeval/data.json", "{}\n")
    result = run(package, "check")
    assert result.returncode == 1
    assert "File added: src/dataeval/data.json" in result.stdout


@pytest.mark.parametrize("name", ["src/dataeval/alpha.py", "docs/source/notebooks/nb_a.py"])
def test_a_file_that_does_not_parse_requires_clean_build(package: Path, name: str):
    write(package, "src/dataeval/shared.py", "X = 2\n")
    write(package, name, "def (\n")
    result = run(package, "check")
    assert result.returncode == 1
    assert f"Cannot parse {name}" in result.stdout
    assert "Traceback" not in result.stderr


def test_pruning_without_a_database_or_entry_is_a_no_op(package: Path):
    (package / "docs/source/.jupyter_cache/global.db").unlink()
    write(package, "src/dataeval/shared.py", "X = 2\n")
    assert run(package, "check").returncode == 0


def test_a_corrupt_database_requires_clean_build(package: Path):
    (package / "docs/source/.jupyter_cache/global.db").write_bytes(b"not sqlite")
    write(package, "src/dataeval/shared.py", "X = 2\n")
    result = run(package, "check")
    assert result.returncode == 1
    assert "Cannot read cache database" in result.stdout


def test_record_is_deterministic_and_reasons_are_sorted(package: Path):
    first = (package / MANIFEST).read_bytes()
    assert run(package, "record").returncode == 0
    assert (package / MANIFEST).read_bytes() == first
    write(package, "src/dataeval/shared.py", "X = 2\n")
    write(package, "src/dataeval/alpha.py", "from dataeval import shared  # note\n")
    lines = [line for line in run(package, "check").stdout.splitlines() if " changed: " in line]
    assert len(lines) == 2
    assert lines == sorted(lines)


def test_a_partial_check_stays_stale_until_a_build_records_inputs(package: Path):
    write(package, "src/dataeval/shared.py", "X = 2\n")
    assert run(package, "check").returncode == 0
    write(package, "docs/source/guide.md", "unrelated\n")
    assert " changed: " in run(package, "check").stdout  # a missed build must not make the cache look valid
    assert run(package, "record").returncode == 0
    again = run(package, "check")
    assert "reusing cached outputs" in again.stdout
    assert " changed: " not in again.stdout


def test_an_entry_that_names_no_current_notebook_is_removed_with_the_stale_ones(package: Path):
    # jupyter-cache keys entries on code cells and keeps the name of the build that made them, so a renamed
    # notebook with unchanged code still has its entry under the old name. It cannot be attributed, so it goes.
    add_entry(package, "nb_old.py", "hash_old")
    write(package, "src/dataeval/shared.py", "X = 2\n")
    assert run(package, "check").returncode == 0
    assert entries(package) == ["hash_b"]


def test_an_entry_that_names_no_current_notebook_is_kept_when_nothing_is_stale(package: Path):
    add_entry(package, "nb_old.py", "hash_old")
    write(package, "src/conftest.py", "changed\n")
    assert run(package, "check").returncode == 0
    assert entries(package) == ["hash_a", "hash_b", "hash_old"]


@pytest.mark.parametrize("decorator", ["@nox.session", '@session(uv_groups=["x"], uv_extras=["cpu"])'])
@pytest.mark.parametrize("session", cache_inputs.IGNORED_SESSIONS)
def test_noxfile_digest_ignores_every_listed_session(session: str, decorator: str):
    text = (
        f'import nox\n\n\n{decorator}\ndef {session}(session):\n    session.run("a")\n\n\n'
        '@nox.session\ndef docs(session):\n    session.run("b")\n'
    )
    assert cache_inputs.noxfile_digest(text) == cache_inputs.noxfile_digest(text.replace('"a"', '"changed"'))

"""Exercise docs cache invalidation against real Git checkouts and saved manifests."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.infrastructure

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = "docs/notebook_cache_inputs.py"
MANIFEST = "docs/source/.jupyter_cache/execution-inputs.json"


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
        "docs/source/data.py",
        "docs/source/notebooks/jupytext.toml",
        "docs/source/notebooks/models/ae",
        "docs/source/notebooks/example.py",
        "docs/source/guide.md",
        "tests/example.py",
    ):
        write(tmp_path, name)
    shutil.copyfile(ROOT / SCRIPT, tmp_path / SCRIPT)
    write(tmp_path, ".gitignore", "__pycache__/\n_version.py\n.jupyter_cache/\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    (tmp_path / MANIFEST).parent.mkdir(parents=True)
    assert run(tmp_path, "record").returncode == 0
    assert run(tmp_path, "check").returncode == 0
    return tmp_path


@pytest.mark.parametrize(
    "name",
    [
        "src/dataeval/example.py",
        "uv.lock",
        "pyproject.toml",
        "noxfile.py",
        ".cuda-version",
        ".gitlab-ci.yml",
        ".gitlab/ci/base.yml",
        ".gitlab/ci/docs.yml",
        SCRIPT,
        "docs/check_notebook_cache.py",
        "docs/source/conf.py",
        "docs/source/data.py",
        "docs/source/notebooks/jupytext.toml",
        "docs/source/notebooks/models/ae",
    ],
)
def test_shared_execution_input_changes_require_clean_build(checkout: Path, name: str):
    with (checkout / name).open("a") as stream:
        stream.write("\n# changed\n")
    result = run(checkout, "check")
    assert result.returncode == 1
    assert f"File changed: {name}" in result.stdout
    assert "All notebooks will be re-executed" in result.stdout


@pytest.mark.parametrize("name", ["src/dataeval/new_module.py", "docs/source/notebooks/models/new_weights"])
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


@pytest.mark.parametrize("name", ["DOCS_PYTHON_VERSION", "DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "DOCS_RUNNER_TAG"])
def test_execution_environment_changes_require_clean_build(checkout: Path, name: str):
    previous = json.loads((checkout / MANIFEST).read_text())["environment"][name]
    result = run(checkout, "check", **{name: "different"})
    assert result.returncode == 1
    assert f"Environment changed: {name}: {previous!r} -> 'different'" in result.stdout


def test_cache_miss_reports_all_changes_and_restored_source(checkout: Path):
    write(checkout, "src/dataeval/example.py", "changed\n")
    write(checkout, "src/dataeval/new_module.py")
    (checkout / "uv.lock").unlink()
    write(checkout, "docs/source/.jupyter_cache/.cache_source", "docs-artifacts/main\n")
    result = run(checkout, "check", DOCS_PYTHON_VERSION="different")
    assert result.returncode == 1
    assert "Restored cache source: docs-artifacts/main" in result.stdout
    assert "File changed: src/dataeval/example.py" in result.stdout
    assert "File added: src/dataeval/new_module.py" in result.stdout
    assert "File removed: uv.lock" in result.stdout
    assert "Environment changed: DOCS_PYTHON_VERSION:" in result.stdout


def test_cache_hit_reports_reuse(checkout: Path):
    result = run(checkout, "check")
    assert result.returncode == 0
    assert "reusing cached outputs" in result.stdout
    assert "cache miss" not in result.stdout


def test_missed_builds_remain_invalid_until_a_successful_build_records_inputs(checkout: Path):
    write(checkout, "src/dataeval/example.py", "changed library\n")
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

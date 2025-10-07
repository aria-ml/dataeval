import os
import re
from sys import version_info

import nox
import nox_uv

PYTHON_VERSION = f"{version_info[0]}.{version_info[1]}"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
PYTHON_RE_PATTERN = re.compile(r"\d\.\d{1,2}")
IS_CI = bool(os.environ.get("CI"))

os.environ["TQDM_DISABLE"] = "1"
nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test", "type", "deps", "lint", "doctest", "check"]

DOCS_ENVS = {"LANG": "C", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
DOCTEST_ENVS = {"NB_EXECUTION_MODE_OVERRIDE": "off"}
RESTORE_CMD = """
if (which git) > /dev/null; then
    git restore ./reference/autoapi/dataeval/index.rst
    if [[ ! $(git status --porcelain | grep docs/source/.jupyter_cache | grep --invert-match global.db) ]]; then
        echo "No cache changes - reverting global.db";
        git restore ./.jupyter_cache/global.db;
    fi;
fi
"""


def get_python_version(session: nox.Session) -> str:
    matches = PYTHON_RE_PATTERN.search(session.name)
    return matches.group(0) if matches else PYTHON_VERSION


@nox_uv.session(uv_groups=["base"])
def dev(session: nox.Session) -> None:
    """Set up a python development environment at `.venv-{version}`. Specify version using `nox -P {version} -e dev`."""
    arch_extras = {"cpu", "cu118", "cu126"}
    arch_posargs = arch_extras & set(session.posargs)
    arch_args = [] if not arch_posargs else [f"--extra={list(arch_posargs)[0]}"]

    python_version = get_python_version(session)
    venv_path = f".venv-{python_version}"
    session.run("rm", "-rf", venv_path, external=True)
    session.run("uv", "venv", "-p", python_version, "--seed", venv_path, external=True)
    session.run("uv", "sync", "-p", venv_path, "--extra=all", *arch_args, external=True)


@nox_uv.session(uv_groups=["test"], uv_extras=["cpu", "all"])
def test(session: nox.Session) -> None:
    """Run unit tests with coverage reporting. Specify version using `nox -P {version} -e test`."""
    python_version = get_python_version(session)
    pytest_args = ["-m", "not cuda"]
    xdist_args = ["-n4", "--dist", "loadfile"]
    cov_args = ["--cov", f"--junitxml=output/junit.{python_version}.xml"]
    cov_term_args = ["--cov-report", "term"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.{python_version}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.{python_version}"]

    session.run(
        "pytest",
        *pytest_args,
        *xdist_args,
        *cov_args,
        *cov_term_args,
        *cov_xml_args,
        *cov_html_args,
        *session.posargs,
    )
    session.run("mv", ".coverage", f"output/.coverage.{python_version}", external=True)


@nox_uv.session
def unit(session: nox.Session) -> None:
    """Alias for `test` session."""
    test(session)


@nox_uv.session(uv_groups=["type"], uv_extras=["cpu", "all"])
def type(session: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types. Specify version using `nox -P {version} -e type`."""
    session.run("pyright", "--stats", "src/", "tests/")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "dataeval")


@nox_uv.session(uv_only_groups=["base"], reuse_venv=False)
def deps(session: nox.Session) -> None:
    """Run unit tests against standard installation."""
    session.run_install("uv", "pip", "install", ".[cpu]", "--resolution=lowest-direct")
    session.run_install("uv", "pip", "install", "pytest")
    session.run("pytest", "-m", "not (requires_all or optional)")


@nox_uv.session(uv_only_groups=["lint"], uv_no_install_project=True)
def lint(session: nox.Session) -> None:
    """Perform linting and spellcheck."""
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("ruff", "format", "--check" if IS_CI else ".")
    session.run("codespell")


@nox_uv.session(uv_groups=["test"], uv_extras=["cpu"])
def doctest(session: nox.Session) -> None:
    """Run docstring tests."""
    target = session.posargs if session.posargs else ["src/dataeval"]
    session.run(
        "pytest",
        "--doctest-modules",
        "--doctest-continue-on-failure",
        "--disable-warnings",
        "--ignore=src/dataeval/detectors/drift/_nml",
        *target,
    )


@nox_uv.session(uv_groups=["docs"], uv_extras=["matplotlib"])
def docs(session: nox.Session) -> None:
    """Generate documentation. Clear the jupyter cache by calling `nox -e docs -- clean`."""
    session.chdir("docs/source")
    session.run("rm", "-rf", "../../output/docs", external=True)
    if "clean" in session.posargs:
        session.run("rm", "-rf", ".jupyter_cache", external=True)
    session.run(
        "sphinx-build",
        "--fail-on-warning",
        "--keep-going",
        "--fresh-env",
        "--show-traceback",
        "--jobs",
        "4",
        "--builder",
        "html",
        "--doctree-dir",
        "../build/doctrees",
        "--define",
        "language=en",
        ".",
        "../../output/docs/html",
        env={**DOCS_ENVS},
    )
    session.run("cp", "-R", ".jupyter_cache", "../../output/docs", external=True)
    session.run_always("bash", "-c", RESTORE_CMD, external=True)


@nox_uv.session(uv_only_groups=["lock"], uv_sync_locked=False)
def lock(session: nox.Session) -> None:
    """Lock dependencies in "uv.lock". Update dependencies by calling `nox -e lock -- upgrade`."""
    upgrade_args = ["--upgrade"] if "upgrade" in session.posargs else []
    session.run("uv", "lock", *upgrade_args)
    session.run("uv", "export", "--extra=all", "--no-emit-project", "-o", "requirements.txt")
    session.run("poetry", "lock")
    session.run("p2c", "y", "-f", "pyproject.toml", "-e", "all", "--python-include", "infer", "-o", "environment.yaml")


@nox_uv.session(uv_only_groups=["base"])
def check(session: nox.Session) -> None:
    """Validate lock file and exported dependency files are up to date."""
    session.run("uv", "lock", "--check")

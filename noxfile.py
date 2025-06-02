import os
import re
from sys import version_info

import nox

PYTHON_VERSION = f"{version_info[0]}.{version_info[1]}"
IS_CI = bool(os.environ.get("CI"))

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test", "type", "deps", "lint", "doctest", "check"]

COMMON_ENVS = {"TQDM_DISABLE": "1"}
DOCS_ENVS = {"LANG": "C", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
DOCTEST_ENVS = {"NB_EXECUTION_MODE_OVERRIDE": "off"}

SUPPORTED_VERSIONS = ("3.9", "3.10", "3.11", "3.12")

RESTORE_CMD = """
if (which git) > /dev/null; then
    git restore ./reference/autoapi/dataeval/index.rst
    if [[ ! $(git status --porcelain | grep docs/source/.jupyter_cache | grep --invert-match global.db) ]]; then
        echo "No cache changes - reverting global.db";
        git restore ./.jupyter_cache/global.db;
    fi;
fi
"""


def prep(session: nox.Session) -> str:
    session.env["UV_PROJECT_ENVIRONMENT"] = session.env["VIRTUAL_ENV"]
    version = session.name
    pattern = re.compile(r".*(3.\d+)$")
    matches = pattern.match(version)
    version = matches.groups()[0] if matches is not None and len(matches.groups()) > 0 else PYTHON_VERSION
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Specified python version {version} is not supported.")
    return version


@nox.session
def dev(session: nox.Session) -> None:
    """Set up a python development environment at `.venv-{version}`. Specify version using `nox -P {version} -e dev`."""
    python_version = prep(session)
    session.env["UV_PROJECT_ENVIRONMENT"] = f".venv-{python_version}"
    session.run("rm", "-rf", f".venv-{python_version}", external=True)
    session.run("uv", "venv", "-p", python_version, "--seed", external=True)
    session.run("uv", "sync", "--extra", "all", external=True)


@nox.session
def test(session: nox.Session) -> None:
    """Run unit tests with coverage reporting. Specify version using `nox -P {version} -e test`."""
    python_version = prep(session)
    pytest_args = ["-m", "not cuda"]
    xdist_args = ["-n4", "--dist", "loadfile"]
    cov_args = ["--cov", f"--junitxml=output/junit.{python_version}.xml"]
    cov_term_args = ["--cov-report", "term"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.{python_version}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.{python_version}"]

    session.run_install("uv", "sync", "--no-dev", "--extra=all", "--group=test")
    session.run(
        "pytest",
        *pytest_args,
        *xdist_args,
        *cov_args,
        *cov_term_args,
        *cov_xml_args,
        *cov_html_args,
        *session.posargs,
        env={**COMMON_ENVS},
    )
    session.run("mv", ".coverage", f"output/.coverage.{python_version}", external=True)


@nox.session
def unit(session: nox.Session) -> None:
    """Alias for `test` session."""
    test(session)


@nox.session
def type(session: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types. Specify version using `nox -P {version} -e type`."""
    prep(session)
    session.run_install("uv", "sync", "--no-dev", "--extra=all", "--group=type")
    session.run("pyright", "--stats", "src/", "tests/")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "dataeval")


@nox.session(reuse_venv=False)
def deps(session: nox.Session) -> None:
    """Run unit tests against standard installation."""
    prep(session)
    session.run_install("uv", "pip", "install", "pytest")
    session.run_install("uv", "pip", "install", ".", "--resolution=lowest-direct")
    session.run("pytest", "-m", "not (requires_all or optional)")


@nox.session
def lint(session: nox.Session) -> None:
    """Perform linting and spellcheck."""
    prep(session)
    session.run_install("uv", "sync", "--only-group=lint")
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("ruff", "format", "--check" if IS_CI else ".")
    session.run("codespell")


@nox.session
def doctest(session: nox.Session) -> None:
    """Run docstring tests."""
    prep(session)
    target = session.posargs if session.posargs else ["src/dataeval"]
    session.run_install("uv", "sync", "--no-dev", "--group=test")
    session.run(
        "pytest",
        "--doctest-modules",
        "--doctest-continue-on-failure",
        "--disable-warnings",
        "--ignore=src/dataeval/detectors/drift/_nml",
        *target,
        env={**COMMON_ENVS},
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Generate documentation. Clear the jupyter cache by calling `nox -e docs -- clean`."""
    prep(session)
    session.run_install("uv", "sync", "--no-dev", "--extra=all", "--group=docs")
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
        env={**DOCS_ENVS, **COMMON_ENVS},
    )
    session.run("cp", "-R", ".jupyter_cache", "../../output/docs", external=True)
    session.run_always("bash", "-c", RESTORE_CMD, external=True)


@nox.session
def lock(session: nox.Session) -> None:
    """Lock dependencies in "uv.lock". Update dependencies by calling `nox -e lock -- upgrade`."""
    prep(session)
    upgrade_args = ["--upgrade"] if "upgrade" in session.posargs else []
    session.install("uv", "pyproject2conda", "poetry")
    session.run("uv", "lock", *upgrade_args)
    session.run("uv", "export", "--extra=all", "--no-emit-project", "-o", "requirements.txt")
    session.run("poetry", "lock")
    session.run("p2c", "y", "-f", "pyproject.toml", "-e", "all", "--python-include", "infer", "-o", "environment.yaml")


@nox.session
def check(session: nox.Session) -> None:
    """Validate lock file and exported dependency files are up to date."""
    prep(session)
    session.install("uv")
    session.run("uv", "lock", "--check")

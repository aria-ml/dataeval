import re
from sys import version_info

import nox

PYTHON_VERSION = f"{version_info[0]}.{version_info[1]}"

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test", "type", "deps", "lint", "doctest", "check"]

INSTALL_ARGS = ["--no-deps", "-e", ".", "-r", "environment/requirements.txt", "-r", "environment/requirements-dev.txt"]
INSTALL_ENVS = {"UV_INDEX_STRATEGY": "unsafe-best-match", "POETRY_DYNAMIC_VERSIONING_BYPASS": "0.0.0"}
COMMON_ENVS = {"TQDM_DISABLE": "1"}
DOCS_ENVS = {"LANG": "C", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
DOCTEST_ENVS = {"NB_EXECUTION_MODE_OVERRIDE": "off"}
TEST_ENVS = {"CUDA_VISIBLE_DEVICES": "-1"}

REQUIREMENTS_OPTION_MAP = {"requirements.txt": "--all-extras", "requirements-dev.txt": "--only=dev"}
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


def check_version(version: str) -> str:
    pattern = re.compile(r".*(3.\d+)$")
    matches = pattern.match(version)
    version = matches.groups()[0] if matches is not None and len(matches.groups()) > 0 else PYTHON_VERSION
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Specified python version {version} is not supported.")
    return version


@nox.session
def dev(session: nox.Session) -> None:
    """Set up a python development environment at `.venv-{version}`. Specify version using `nox -P {version} -e dev`."""
    python_version = check_version(session.name)
    session.run("uv", "venv", "--python", python_version, f".venv-{python_version}", "--seed", external=True)
    session.run("uv", "pip", "install", "--python", f".venv-{python_version}", *INSTALL_ARGS, env=INSTALL_ENVS)


@nox.session
def test(session: nox.Session) -> None:
    """Run unit tests with coverage reporting. Specify version using `nox -P {version} -e test`."""
    python_version = check_version(session.name)
    pytest_args = ["--cov", "-n8", "--dist", "loadgroup", f"--junitxml=output/junit.{python_version}.xml"]
    cov_term_args = ["--cov-report", "term"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.{python_version}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.{python_version}"]

    session.install(*INSTALL_ARGS, env=INSTALL_ENVS)
    session.run("pytest", *pytest_args, *cov_term_args, *cov_xml_args, *cov_html_args, env={**TEST_ENVS, **COMMON_ENVS})
    session.run("mv", ".coverage", f"output/.coverage.{python_version}", external=True)


@nox.session
def unit(session: nox.Session) -> None:
    """Alias for `test` session."""
    test(session)


@nox.session
def type(session: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types. Specify version using `nox -P {version} -e type`."""
    check_version(session.name)
    session.install(*INSTALL_ARGS, env=INSTALL_ENVS)
    session.run("pyright", "--stats", "src/", "tests/")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "dataeval")


@nox.session(reuse_venv=False)
def deps(session: nox.Session) -> None:
    """Run unit tests against standard installation."""
    check_version(session.name)
    session.install(".", "pytest", env=INSTALL_ENVS)
    session.run("pytest", "tests/test_mindeps.py")


@nox.session
def lint(session: nox.Session) -> None:
    """Perform linting and spellcheck."""
    check_version(session.name)
    session.install("ruff", "codespell[toml]")
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("codespell")


@nox.session
def doctest(session: nox.Session) -> None:
    """Run docstring tests."""
    check_version(session.name)
    session.install(*INSTALL_ARGS, env=INSTALL_ENVS)
    session.run(
        "pytest",
        "--doctest-modules",
        "--doctest-continue-on-failure",
        "--disable-warnings",
        "src/dataeval",
        env={**TEST_ENVS, **COMMON_ENVS},
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Generate documentation. Clear the jupyter cache by calling `nox -e docs -- clean`."""
    check_version(session.name)
    session.install(*INSTALL_ARGS, env=INSTALL_ENVS)
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
    """Lock dependencies in "poetry.lock" with --no-update. Update dependencies by calling `nox -e lock -- update`."""
    update_args = [] if "update" in session.posargs else ["--no-update"]
    session.install("poetry<2", "poetry-lock-groups-plugin", "poetry2conda")
    session.run("cp", "-f", "environment/poetry.lock", "poetry.lock", external=True)
    session.run("poetry", "lock", "--with=dev", *update_args)
    for file, option in REQUIREMENTS_OPTION_MAP.items():
        session.run("poetry", "export", option, "--without-hashes", "-o", f"environment/{file}")
    session.run("cp", "-f", "poetry.lock", "environment/poetry.lock", external=True)
    session.run("poetry", "lock", *update_args)
    session.run("poetry2conda", "pyproject.toml", "environment/environment.yaml", "-E", "all")


@nox.session
def check(session: nox.Session) -> None:
    """Validate lock file and exported dependency files are up to date."""
    session.install("poetry<2")
    session.run_always("cp", "-f", "poetry.lock", "poetry.tmp", external=True)
    session.run_always("mkdir", "-p", "output/tmp", external=True)

    session.run("poetry", "config", "warnings.export", "false")
    session.run("poetry", "check")
    session.run("cp", "-f", "environment/poetry.lock", "poetry.lock", external=True)
    session.run("poetry", "check")
    for file, option in REQUIREMENTS_OPTION_MAP.items():
        session.run("poetry", "export", option, "--without-hashes", "-o", f"output/tmp/{file}")
        session.run("diff", f"environment/{file}", f"output/tmp/{file}", external=True)
        session.run("cmp", "-s", f"environment/{file}", f"output/tmp/{file}", external=True)

    session.run_always("cp", "-f", "poetry.tmp", "poetry.lock", external=True)
    session.run_always("rm", "-f", "poetry.tmp", external=True)
    session.run_always("rm", "-rf", "output/tmp", external=True)

import re
from sys import version_info

import nox

python_version = f"{version_info[0]}.{version_info[1]}"

nox.options.default_venv_backend = "uv"
nox.options.sessions = [f"test-{python_version}", f"type-{python_version}", "deps", "lint", "doctest", "check"]

INSTALL_ARGS = ["-e", ".", "-r", "environment/requirements.txt", "-r", "environment/requirements-dev.txt"]
INSTALL_ENVS = {"UV_INDEX_STRATEGY": "unsafe-best-match", "POETRY_DYNAMIC_VERSIONING_BYPASS": "0.0.0"}
COMMON_ENVS = {"TQDM_DISABLE": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}
DOCS_ENVS = {"LANG": "C", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
DOCTEST_ENVS = {"NB_EXECUTION_MODE_OVERRIDE": "off"}
TEST_ENVS = {"CUDA_VISIBLE_DEVICES": "-1", **COMMON_ENVS}

REQUIREMENTS_OPTION_MAP = {"requirements.txt": "--all-extras", "requirements-dev.txt": "--only=dev"}
SUPPORTED_VERSIONS = ("3.9", "3.10", "3.11", "3.12")

RESTORE_CMD = """
if (which git) > /dev/null; then
    if [[ ! $(git status --porcelain | grep docs/.jupyter_cache | grep --invert-match global.db) ]]; then
        echo "No cache changes - reverting global.db";
        git restore .jupyter_cache/global.db;
    fi;
fi
"""


@nox.session
def dev(session: nox.Session) -> None:
    """Set up a python development environment at ".venv-[python_version]". Specify version using positional args."""
    pattern = re.compile(r"3.\d+$")
    versions = [python_version] if not session.posargs else [s for s in session.posargs if pattern.match(s)]
    for version in versions:
        if version not in SUPPORTED_VERSIONS:
            print(f"Only python {SUPPORTED_VERSIONS} is supported. {version} provided. Skipping venv creation.")
            continue
        session.run("uv", "venv", "-p", version, f".venv-{version}", "--seed", external=True)
        session.run("uv", "pip", "install", "-p", f".venv-{version}", *INSTALL_ARGS, env=INSTALL_ENVS, silent=False)


@nox.session(python=SUPPORTED_VERSIONS)
def test(session: nox.Session) -> None:
    """Run unit tests with coverage reporting."""
    pytest_args = ["--cov", "-n8", "--dist", "loadgroup", f"--junitxml=output/junit.{session.name}.xml"]
    cov_term_args = ["--cov-report", "term"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.{session.name}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.{session.name}"]

    session.install(*INSTALL_ARGS, env=INSTALL_ENVS, silent=False)
    session.run("pytest", *pytest_args, *cov_term_args, *cov_xml_args, *cov_html_args, env={**TEST_ENVS, **COMMON_ENVS})
    session.run("mv", ".coverage", f"output/.coverage.{session.name}", external=True)


@nox.session(python=SUPPORTED_VERSIONS)
def type(session: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types."""
    session.install(*INSTALL_ARGS, env=INSTALL_ENVS, silent=False)
    session.run("pyright", "--stats", "src/", "tests/")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "dataeval")


@nox.session(reuse_venv=False)
def deps(session: nox.Session) -> None:
    """Run minimal unit tests against baseline installation."""
    session.install(".", "pytest", env=INSTALL_ENVS, silent=False)
    session.run("pytest", "tests/test_mindeps.py")


@nox.session
def lint(session: nox.Session) -> None:
    """Perform linting and spellcheck."""
    session.install("ruff", "codespell[toml]", silent=False)
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("codespell")


@nox.session
def doctest(session: nox.Session) -> None:
    """Run docstring tests."""
    session.install(*INSTALL_ARGS, env=INSTALL_ENVS, silent=False)
    session.chdir("docs")
    session.run("rm", "-rf", "../output/docs", external=True)
    session.run("sphinx-build", "-M", "doctest", ".", "../output/docs", env={**DOCTEST_ENVS, **COMMON_ENVS})


@nox.session
def docs(session: nox.Session) -> None:
    """Generate documentation. Clear the cache using "clean" as a positional arg."""
    session.install(*INSTALL_ARGS, env=INSTALL_ENVS, silent=False)
    session.chdir("docs")
    session.run("rm", "-rf", "../output/docs", external=True)
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
        "_build/doctrees",
        "--define",
        "language=en",
        ".",
        "../output/docs/html",
        env={**DOCS_ENVS, **COMMON_ENVS},
    )
    session.run("cp", "-R", ".jupyter_cache", "../output/docs", external=True)
    session.run("bash", "-c", RESTORE_CMD, external=True)


@nox.session
def lock(session: nox.Session) -> None:
    """Lock dependencies in "poetry.lock" with --no-update. Can also use "update" as a positional arg."""
    update_args = "" if "update" in session.posargs else "--no-update"
    session.install("poetry", "poetry-lock-groups-plugin", "poetry2conda")
    session.run("cp", "-f", "environment/poetry.lock", "poetry.lock", external=True)
    session.run("poetry", "config", "warnings.export", "false")
    session.run("poetry", "lock", "--with=dev", update_args)
    for file, option in REQUIREMENTS_OPTION_MAP.items():
        session.run("poetry", "export", option, "--without-hashes", "-o", f"environment/{file}")
    session.run("cp", "-f", "poetry.lock", "environment/poetry.lock", external=True)
    session.run("poetry", "lock", update_args)
    session.run("poetry2conda", "pyproject.toml", "environment/environment.yaml", "-E", "all")


@nox.session
def check(session: nox.Session) -> None:
    """Validate lock file and exported dependency files are up to date."""
    session.install("poetry")
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

import functools
import glob
import os
import re
from pathlib import Path
from sys import version_info

import nox

# Try importing nox_uv. If it fails, define a fallback decorator.
try:
    import nox_uv
except ImportError:
    nox_uv = None


# Session compatibility shim for nox/nox-uv
def session(**kwargs):
    """
    Custom session decorator that works with or without nox-uv.
    If nox-uv is missing, it strips 'uv_*' arguments and falls back to standard nox.
    """

    def decorator(func):
        if nox_uv is not None:
            # If nox-uv is installed, pass everything through directly
            return nox_uv.session(**kwargs)(func)
        # Extract uv_* options (use .get() to avoid mutating kwargs)
        uv_groups = kwargs.get("uv_groups", [])
        uv_extras = kwargs.get("uv_extras", [])
        uv_only_groups = kwargs.get("uv_only_groups", [])
        uv_no_install_project = kwargs.get("uv_no_install_project", False)
        # Strip all uv_* args to avoid kwargs errors in standard nox
        clean_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("uv_")}

        # Define a wrapper that runs the install command before the actual session
        @functools.wraps(func)
        def wrapper(session: nox.Session):
            # Ensure pip >= 25.1 for --group support (PEP 735)
            session.install("pip>=25.1")

            # Build install command for the project with extras
            if not uv_no_install_project and not uv_only_groups:
                # Install the project itself, optionally with extras
                if uv_extras:
                    extras_str = ",".join(uv_extras)
                    session.install("-e", f".[{extras_str}]")
                else:
                    session.install("-e", ".")

            # Handle dependency groups (uv_groups installs project + groups,
            # uv_only_groups installs only the groups without the project)
            groups = uv_only_groups if uv_only_groups else uv_groups
            if groups:
                group_args = []
                for group in groups:
                    group_args.extend(["--group", group])
                session.install(*group_args)

            # Run the original function
            return func(session)

        # Register the wrapper with standard nox
        return nox.session(**clean_kwargs)(wrapper)

    return decorator


PYTHON_VERSION = f"{version_info[0]}.{version_info[1]}"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
PYTHON_RE_PATTERN = re.compile(r"\d\.\d{1,2}")
IS_CI = bool(os.environ.get("CI"))
DATAEVAL_NOX_UV_EXTRAS_OVERRIDE = os.environ.get("DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "")
if not DATAEVAL_NOX_UV_EXTRAS_OVERRIDE:
    if os.path.exists(".cuda-version"):
        with open(".cuda-version") as f:
            DATAEVAL_NOX_UV_EXTRAS_OVERRIDE = f.read().strip()
    if DATAEVAL_NOX_UV_EXTRAS_OVERRIDE not in ["cpu", "cu118", "cu124", "cu128"]:
        DATAEVAL_NOX_UV_EXTRAS_OVERRIDE = "cu118"

UV_EXTRAS = [DATAEVAL_NOX_UV_EXTRAS_OVERRIDE]

# Configure Numba disk caching
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba"))
os.environ.setdefault("NUMBA_ENABLE_CACHING", "1")

# Configure UV to always clear the venv
os.environ.setdefault("UV_VENV_CLEAR", "1")

# Standard nox options
nox.options.default_venv_backend = "uv" if nox_uv is not None else "virtualenv"
nox.options.sessions = ["test", "type", "deps", "lint", "doclint", "doctest", "check"]

DOCS_ENVS = {"LANG": "C", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
DOCTEST_ENVS = {"NB_EXECUTION_MODE_OVERRIDE": "off"}


def get_python_version(session: nox.Session) -> str:
    matches = PYTHON_RE_PATTERN.search(session.name)
    return matches.group(0) if matches else PYTHON_VERSION


def with_onnx(extras: list[str]) -> list[str]:
    if "cpu" in extras:
        return extras + ["onnx"]
    if any(extra.startswith("cu") for extra in extras):
        return extras + ["onnx-gpu"]
    return extras


@session(uv_groups=["test"], uv_extras=["cpu", "onnx", "opencv"])
def test(session: nox.Session) -> None:
    """Run unit tests with coverage reporting. Specify version using `nox -P {version} -e test`.

    Pass 'clear-cache' to clear the Numba cache before running tests: `nox -e test -- clear-cache`
    """
    python_version = get_python_version(session)

    # Handle clear-cache argument
    if "clear-cache" in session.posargs:
        numba_cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba")))
        if numba_cache_dir.exists():
            session.log(f"Clearing Numba cache at {numba_cache_dir}...")
            session.run("rm", "-rf", str(numba_cache_dir), external=True)
        # Remove 'clear-cache' from posargs so it doesn't get passed to pytest
        remaining_posargs = [arg for arg in session.posargs if arg != "clear-cache"]
    else:
        remaining_posargs = list(session.posargs)

    # Standard pytest configuration
    pytest_args = ["-m", "not cuda"]
    xdist_args = ["-n4", "--dist", "loadfile"]
    cov_args = ["--cov", f"--junitxml=output/junit.{python_version}.xml"]
    cov_term_args = ["--cov-report", "term-missing"]
    cov_xml_args = ["--cov-report", f"xml:output/coverage.{python_version}.xml"]
    cov_html_args = ["--cov-report", f"html:output/htmlcov.{python_version}"]

    # Pre-warm Numba JIT cache only if cache doesn't exist or is empty
    numba_cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba")))
    cache_exists = numba_cache_dir.exists() and any(numba_cache_dir.iterdir())

    if not cache_exists:
        session.log("Pre-warming Numba JIT compilation cache (no cache found)...")
        session.run("python", "-m", "dataeval._warm_cache")
    else:
        session.log("Skipping cache pre-warm (cache already exists)")

    session.run(
        "pytest",
        *pytest_args,
        *xdist_args,
        *cov_args,
        *cov_term_args,
        *cov_xml_args,
        *cov_html_args,
        *remaining_posargs,
    )
    session.run("mv", ".coverage", f"output/.coverage.{python_version}", external=True)


@session(uv_groups=["verify"], uv_extras=["cpu", "onnx", "opencv"])
def verify(session: nox.Session) -> None:
    """Run verification tests for FR/NFR compliance. Specify version using `nox -P {version} -e verify`."""
    # uv sync doesn't trigger hatch-vcs build hook, so _version.py may not exist.
    # Force an editable install to generate it via the build hook.
    session.run_install("uv", "pip", "install", "-e", ".", "--no-deps")
    python_version = get_python_version(session)
    session.run(
        "pytest",
        "verification/",
        "--tb=short",
        f"--junitxml=output/verify.{python_version}.xml",
        *session.posargs,
    )
    session.run("python", "verification/generate_metarepo.py")


@session(uv_groups=["type"], uv_extras=with_onnx(["cpu"]))
def type(session: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types. Specify version using `nox -P {version} -e type`."""
    session.run("pyright", "--stats")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "dataeval")


@session(python=PYTHON_VERSIONS[0], uv_only_groups=["base"], reuse_venv=False)
def deps(session: nox.Session) -> None:
    """Run unit tests against minimum supported Python with lowest declared dependencies."""
    session.run_install("uv", "pip", "install", ".[cpu]", "--resolution=lowest-direct")
    session.run_install("uv", "pip", "install", "pytest")
    session.run("pytest", "-m", "not (optional)")


@session(uv_only_groups=["lint"], uv_no_install_project=True)
def lint(session: nox.Session) -> None:
    """Perform linting and spellcheck."""
    session.run("ruff", "check", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("ruff", "format", "--check" if IS_CI else ".")
    session.run("codespell")


@session(uv_groups=["test"], uv_extras=with_onnx(["cpu", "opencv"]))
def doctest(session: nox.Session) -> None:
    """Run docstring tests."""
    target = session.posargs if session.posargs else ["src/dataeval"]
    session.run(
        "pytest",
        "--doctest-modules",
        "--doctest-continue-on-failure",
        "--disable-warnings",
        *target,
    )


@session(uv_groups=["docs"], uv_extras=with_onnx(UV_EXTRAS) + ["opencv"])
def docs(session: nox.Session) -> None:
    """Generate documentation.

    Pass 'clean' to clear the jupyter cache: `nox -e docs -- clean`
    Pass 'skip' to skip notebook execution: `nox -e docs -- skip`
    """
    skip_notebooks = "skip" in session.posargs
    clean_notebooks = "clean" in session.posargs

    if {"chart", "charts"} & set(session.posargs):
        try:
            session.run(
                "python",
                "docs/generate_charts.py",
                "--data-file",
                "docs/charts.json",
                "--output-dir",
                "docs/source/_static/charts",
                "--include-js",
                "cdn",
                "--quiet",  # Suppress verbose output in automated builds
                external=False,  # Run with session's Python environment
            )
            session.log("✅ Charts generated successfully")
        except Exception as e:
            session.warn(f"⚠️  Chart generation failed: {e}")
            session.log("Continuing with documentation build...")

    # Convert py:percent notebooks to ipynb (py is source of truth, ignores timestamps)
    notebook_dir = "docs/source/notebooks"
    session.run("jupytext", "--to", "notebook", "--update", notebook_dir + "/*.py")

    if clean_notebooks:
        # Clear local jupyter cache to force re-execution of all notebooks
        cache_dir = "docs/source/.jupyter_cache"
        session.log(f"Clearing jupyter cache at {cache_dir} to force re-execution...")
        session.run("rm", "-rf", cache_dir, external=True)
    elif not skip_notebooks:
        # Fetch cached notebook results from orphan artifact branch
        session.run("bash", "docs/fetch-docs-cache.sh", external=True)

    session.run("rm", "-rf", "output/docs", external=True)
    session.chdir("docs/source")

    if not skip_notebooks:
        # Fix any inconsistent cache state before building (e.g., db records without folders or vice versa)
        session.run("python", "../../docs/check_notebook_cache.py", "--fix")

    session.run(
        "sphinx-build",
        "--fail-on-warning",
        "--keep-going",
        "--fresh-env",
        "--show-traceback",
        "--builder",
        "html",
        "--doctree-dir",
        "../build/doctrees",
        "--define",
        "language=en",
        ".",
        "../../output/docs/html",
        env={**DOCS_ENVS, **({"NB_EXECUTION_MODE_OVERRIDE": "off"} if skip_notebooks else {})},
    )

    if not skip_notebooks:
        # Clean up stale cache entries after sphinx-build updates the cache
        session.run("python", "../../docs/check_notebook_cache.py", "--clean")


@session(python=PYTHON_VERSIONS[0], uv_only_groups=["lock"], uv_sync_locked=False)
def lock(session: nox.Session) -> None:
    """Lock dependencies in "uv.lock". Update dependencies by calling `nox -e lock -- upgrade`."""
    upgrade_args = ["--upgrade"] if "upgrade" in session.posargs else []
    session.run("uv", "lock", *upgrade_args)
    session.run("uv", "export", "--no-emit-project", "-o", "requirements.txt")
    session.run("poetry", "lock")
    session.run("p2c", "y", "-f", "pyproject.toml", "--python-include", "infer", "-o", "environment.yaml")


@session(uv_only_groups=["docsync"], uv_no_install_project=True)
def docsync(session: nox.Session) -> None:
    """Sync notebook .py/.ipynb pairs."""
    notebook_dir = "docs/source/notebooks"

    # Generate .py for any new .ipynb files without a script pair
    ipynb_stems = {Path(f).stem for f in glob.glob(f"{notebook_dir}/*.ipynb")}
    py_stems = {Path(f).stem for f in glob.glob(f"{notebook_dir}/*.py")}
    for stem in sorted(ipynb_stems - py_stems):
        session.log(f"Generating script for new notebook: {stem}.ipynb")
        session.run("jupytext", "--to", "py:percent", f"{notebook_dir}/{stem}.ipynb")

    # Bidirectional sync: updates whichever side is stale (uses jupytext.toml pairing)
    # If ipynb is newer -> updates py; if py is newer -> updates ipynb
    session.run("jupytext", "--sync", notebook_dir + "/*.py")


@session(python=PYTHON_VERSIONS[0], uv_only_groups=["lock"])
def check(session: nox.Session) -> None:
    """Validate lock file and exported dependency files are up to date."""
    session.run("uv", "lock", "--check")
    session.run("poetry", "check", "--lock")

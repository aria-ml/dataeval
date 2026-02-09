import functools
import glob
import os
import re
import shutil
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
        else:
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
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
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
nox.options.sessions = ["test", "type", "deps", "lint", "docsync", "doclint", "doctest", "check"]

DOCS_ENVS = {"LANG": "C", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
DOCTEST_ENVS = {"NB_EXECUTION_MODE_OVERRIDE": "off"}
RESTORE_CMD = """
if (which git) > /dev/null; then
    git restore ./reference/autoapi/dataeval/index.rst;
fi
"""


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


@session(uv_groups=["type"], uv_extras=with_onnx(["cpu"]))
def type(session: nox.Session) -> None:  # noqa: A001
    """Run type checks and verify external types. Specify version using `nox -P {version} -e type`."""
    session.run("pyright", "--stats", "src/", "tests/")
    session.run("pyright", "--ignoreexternal", "--verifytypes", "dataeval")


@session(uv_only_groups=["base"], reuse_venv=False)
def deps(session: nox.Session) -> None:
    """Run unit tests against standard installation."""
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
    """Generate documentation. Clear the jupyter cache by calling `nox -e docs -- clean`."""
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

    # Convert markdown notebooks to ipynb (md is source of truth, ignores timestamps)
    notebook_dir = "docs/source/notebooks"
    session.run("jupytext", "--to", "notebook", "--update", notebook_dir + "/*.md")

    if {"synconly", "sync-only"} & set(session.posargs):
        session.log("Sync-only mode: notebooks have been synced but docs build is skipped.")
        return

    # Fetch cached notebook results from orphan artifact branch
    session.run("bash", "docs/fetch-docs-cache.sh", external=True)

    session.run("rm", "-rf", "output/docs", external=True)
    session.chdir("docs/source")
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
        env={**DOCS_ENVS},
    )
    # Clean up stale cache entries after sphinx-build updates the cache
    session.run("python", "../../docs/check_notebook_cache.py", "--clean")
    session.run_always("bash", "-c", RESTORE_CMD, external=True)


@session(uv_only_groups=["lock"], uv_sync_locked=False)
def lock(session: nox.Session) -> None:
    """Lock dependencies in "uv.lock". Update dependencies by calling `nox -e lock -- upgrade`."""
    upgrade_args = ["--upgrade"] if "upgrade" in session.posargs else []
    session.run("uv", "lock", *upgrade_args)
    session.run("uv", "export", "--no-emit-project", "-o", "requirements.txt")
    session.run("poetry", "lock")
    session.run("p2c", "y", "-f", "pyproject.toml", "--python-include", "infer", "-o", "environment.yaml")


def _clean_notebook_script(script_path: str) -> None:
    """Remove IPython magics, cell markers, and markdown from converted notebook script."""
    script = Path(script_path)
    if not script.exists():
        return

    content = script.read_text()
    lines = content.split("\n")
    cleaned_lines = []
    in_markdown_cell = False

    for line in lines:
        stripped = line.strip()
        # Skip IPython magic commands
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        if stripped.startswith("get_ipython()"):
            continue
        # Track markdown cell blocks (jupytext percent format)
        if stripped.startswith("# %% [markdown]"):
            in_markdown_cell = True
            continue
        if stripped.startswith("# %%"):
            in_markdown_cell = False
            continue
        # Skip markdown content lines (commented text inside markdown cells)
        if in_markdown_cell:
            continue
        # Skip jupytext header block (lines starting with "# ---" or "# jupyter:")
        if stripped.startswith("# ---") or stripped.startswith("# jupyter:"):
            continue
        cleaned_lines.append(line)

    script.write_text("\n".join(cleaned_lines))


def _run_doclint_tests(session: nox.Session, output_dir: str, scripts: list) -> list:
    """Run all lint tests and return list of failures."""
    test_failures = []

    # Run ruff check (lint)
    session.log("Running ruff check on generated scripts...")
    try:
        session.run("ruff", "check", "--ignore=E402,E501,E703,I001,RUF100,SIM105,UP009", output_dir)
    except Exception as e:
        test_failures.append(("ruff", str(e)))

    # Run pyright (typecheck)
    session.log("Running pyright on generated scripts...")
    try:
        session.run("pyright", output_dir)
    except Exception as e:
        test_failures.append(("pyright", str(e)))

    return test_failures


@session(uv_only_groups=["docsync"], uv_no_install_project=True)
def docsync(session: nox.Session) -> None:
    """Sync notebook .md/.ipynb pairs and format markdown."""
    notebook_dir = "docs/source/notebooks"

    # Generate .md for any new .ipynb files without a markdown pair
    ipynb_stems = {Path(f).stem for f in glob.glob(f"{notebook_dir}/*.ipynb")}
    md_stems = {Path(f).stem for f in glob.glob(f"{notebook_dir}/*.md")}
    for stem in sorted(ipynb_stems - md_stems):
        session.log(f"Generating markdown for new notebook: {stem}.ipynb")
        session.run("jupytext", "--to", "myst", f"{notebook_dir}/{stem}.ipynb")

    # Bidirectional sync: updates whichever side is stale (uses jupytext.toml pairing)
    # If ipynb is newer -> updates md; if md is newer -> updates ipynb
    session.run("jupytext", "--sync", notebook_dir + "/*.md")

    # Format markdown notebooks (fix locally, check in CI)
    mdformat_args = ["mdformat", "--wrap", "120"]
    if IS_CI:
        mdformat_args.append("--check")
    session.run(*mdformat_args, notebook_dir)

    # Regenerate ipynb to match formatted md (mdformat may have modified md)
    session.run("jupytext", "--to", "notebook", "--update", notebook_dir + "/*.md")


@session(uv_only_groups=["doclint"])
def doclint(session: nox.Session) -> None:
    """Extract scripts from notebooks in docs and run lint, typecheck, and compile tests."""
    # Setup output directory - clear it first
    output_dir = Path("output/nb_scripts")
    if output_dir.exists():
        session.log(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all notebooks in docs/source
    notebooks = glob.glob("docs/source/notebooks/*.md")
    if not notebooks:
        session.warn("No notebooks found in docs/source/notebooks/")
        return

    session.log(f"Found {len(notebooks)} notebooks to process")

    # Convert notebooks to Python scripts using jupytext
    for notebook in notebooks:
        nb_path = Path(notebook)
        output_script = output_dir / f"{nb_path.stem}.py"
        session.log(f"Converting {nb_path.name} -> {output_script.name}")
        session.run("jupytext", "--to", "py:percent", "--output", str(output_script), str(nb_path))
        _clean_notebook_script(str(output_script))

    # Get all generated Python scripts
    scripts = list(output_dir.glob("*.py"))
    if not scripts:
        session.warn("No Python scripts were generated")
        return

    # Run all tests and collect failures
    test_failures = _run_doclint_tests(session, str(output_dir), scripts)

    # Report results
    if test_failures:
        session.error(
            f"Doclint failed with {len(test_failures)} test failure(s):\n"
            + "\n  - ".join(f"{test_name}: {error}" for test_name, error in test_failures)
        )


@session(uv_only_groups=["lock"])
def check(session: nox.Session) -> None:
    """Validate lock file and exported dependency files are up to date."""
    session.run("uv", "lock", "--check")
    session.run("poetry", "check", "--lock")

import argparse
import functools
import glob
import os
import re
import shutil
import sys
from pathlib import Path

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


PYTHON_VERSION = f"{sys.version_info[0]}.{sys.version_info[1]}"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
PYTHON_DEFAULT = "3.11"
PYTHON_RE_PATTERN = re.compile(r"\d\.\d{1,2}")
DEVICE_VARIANTS = ["cpu", "cu126", "cu130"]
DEVICE_DEFAULT = "cpu"
VENV_DEFAULT = ".venv"
CUDA_VERSION_FILE = ".cuda-version"
IS_CI = bool(os.environ.get("CI"))
DATAEVAL_NOX_UV_EXTRAS_OVERRIDE = os.environ.get("DATAEVAL_NOX_UV_EXTRAS_OVERRIDE", "")
if not DATAEVAL_NOX_UV_EXTRAS_OVERRIDE:
    if os.path.exists(CUDA_VERSION_FILE):
        with open(CUDA_VERSION_FILE) as f:
            DATAEVAL_NOX_UV_EXTRAS_OVERRIDE = f.read().strip()
    if DATAEVAL_NOX_UV_EXTRAS_OVERRIDE not in DEVICE_VARIANTS:
        DATAEVAL_NOX_UV_EXTRAS_OVERRIDE = DEVICE_DEFAULT

UV_EXTRAS = [DATAEVAL_NOX_UV_EXTRAS_OVERRIDE]

# Configure Numba disk caching. The cache is kept in the checkout rather than in
# a user-wide directory so it is scoped to this branch's sources and can be
# cached by CI. Warming is handled by the test suite (tests/fixtures/numba_cache.py).
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.abspath(".numba-cache"))
os.environ.setdefault("NUMBA_ENABLE_CACHING", "1")

# Configure UV to always clear the venv
os.environ.setdefault("UV_VENV_CLEAR", "1")

# Standard nox options
nox.options.default_venv_backend = "uv" if nox_uv is not None else "virtualenv"
nox.options.sessions = ["test", "type", "deps", "lint", "doctest", "check"]

DOCS_ENVS = {
    "LANG": "C",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYDEVD_DISABLE_FILE_VALIDATION": "1",
    "IPYTHONDIR": os.path.abspath("docs/source/.ipython"),
}
DOCTEST_ENVS = {"NB_EXECUTION_MODE_OVERRIDE": "off"}


def get_python_version(session: nox.Session) -> str:
    matches = PYTHON_RE_PATTERN.search(session.name)
    return matches.group(0) if matches else PYTHON_VERSION


def with_onnx(extras: list[str]) -> list[str]:
    if cuda_extra := next((extra for extra in extras if extra.startswith("cu")), None):
        return extras + [f"onnx-{cuda_extra}"]
    return extras + ["onnx"]


def resolve_option(session: nox.Session, label: str, provided: "str | None", allowed: list[str], default: str) -> str:
    """Validate an option value, prompting for it when it was not supplied on the command line.

    A value typed at the prompt falls back to `default` when unrecognized, but a value passed
    as a flag is an error, so that a typo such as `-d cu124` cannot silently install cu130.
    """
    value = provided
    interactive = value is None
    if interactive:
        prompt = f"Enter desired {label} [supported: {' '.join(allowed)}] [default: {default}]: "
        value = input(prompt).strip() if sys.stdin.isatty() else ""
    if not value:
        return default
    # Accept a full patch version such as "3.11.4" for a "3.11" option (a no-op for device names).
    value = ".".join(value.split(".")[:2])
    if value not in allowed:
        if not interactive:
            session.error(f"Unrecognized {label} '{value}' (supported: {', '.join(allowed)})")
        session.warn(f"Unrecognized {label} '{value}', defaulting to {default}")
        return default
    return value


# Declared with nox.session rather than the nox-uv shim: this session builds the project
# environment itself, so it must not ask nox to create one first.
@nox.session(venv_backend="none")
def dev(session: nox.Session) -> None:
    """Create a local development environment. Prompts for any option not passed on a terminal.

    Usage: `uvx --with nox-uv nox -s dev -- [-p VERSION] [-d DEVICE] [-n NAME]`

    Bootstrap this one with `uvx`, not `uv run nox -s dev` -- nox itself lives in the
    environment the session rebuilds. `--with nox-uv` is not needed here (this session
    creates no nox environment of its own) but is what every other session wants, so it
    is the invocation worth keeping in muscle memory.
    """
    parser = argparse.ArgumentParser(prog="nox -s dev --", add_help=False)
    parser.add_argument("-p", "--python", dest="python")
    parser.add_argument("-d", "--device", dest="device")
    parser.add_argument("-n", "--name", dest="name", default=VENV_DEFAULT)
    try:
        args = parser.parse_args(session.posargs)
    except SystemExit:
        session.error("Usage: nox -s dev -- [-p|--python VERSION] [-d|--device DEVICE] [-n|--name NAME]")
        raise  # unreachable; keeps `args` bound for type checkers that cannot see nox's NoReturn

    if shutil.which("uv") is None:
        session.error("Install uv to continue: https://docs.astral.sh/uv/")

    python_version = resolve_option(session, "version of python", args.python, PYTHON_VERSIONS, PYTHON_DEFAULT)
    device = resolve_option(session, "CUDA version", args.device, DEVICE_VARIANTS, DEVICE_DEFAULT)
    venv_path = Path(args.name).resolve()

    # `uv sync` recreates the environment, which would pull the interpreter out from under a
    # nox running inside it.
    if Path(sys.prefix).resolve() == venv_path:
        session.error(
            f"Refusing to rebuild '{args.name}' while running from it. "
            "Use `uvx --with nox-uv nox -s dev` instead of `uv run nox -s dev`."
        )
    if venv_path.exists():
        if not (venv_path / "pyvenv.cfg").exists():
            session.error(f"'{args.name}' exists but is not a virtual environment; refusing to remove it.")
        session.log(f"Removing existing virtual environment at {args.name}...")
        shutil.rmtree(venv_path)

    session.log(f"Installing Python {python_version}+{device} to {args.name}...")
    extras: list[str] = []
    for extra in with_onnx([device]):
        extras += ["--extra", extra]
    session.run(
        "uv",
        "sync",
        "-p",
        python_version,
        *extras,
        external=True,
        env={"UV_PROJECT_ENVIRONMENT": str(venv_path)},
    )
    # Recorded for DATAEVAL_NOX_UV_EXTRAS_OVERRIDE so later sessions pick up the same device.
    Path(CUDA_VERSION_FILE).write_text(f"{device}\n")

    session.log(f"Finished installing dataeval for python {python_version} to {args.name}")
    if os.environ.get("REMOTE_CONTAINERS"):
        session.log(f"Activate the environment (e.g. `source {args.name}/bin/activate`) before proceeding.")
    else:
        session.log(
            "It is now safe to reload the window ('Developer: Reload Window') and select a Python "
            "interpreter ('Python: Select Interpreter') from the Command Palette."
        )


@session(uv_groups=["test-onnx"], uv_extras=with_onnx(["cpu", "opencv", "ontology"]))
def test(session: nox.Session) -> None:
    """Run unit tests with coverage reporting. Specify version using `nox -P {version} -e test`.

    Pass 'clear-cache' to clear the Numba cache before running tests: `nox -e test -- clear-cache`
    """
    python_version = get_python_version(session)

    # Handle clear-cache argument
    if "clear-cache" in session.posargs:
        numba_cache_dir = Path(os.environ["NUMBA_CACHE_DIR"])
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


@session(uv_groups=["type"], uv_extras=with_onnx(["cpu", "opencv", "ontology"]))
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
    session.run("ruff", "check", "--preview", "--show-fixes", "--exit-non-zero-on-fix", "--fix")
    session.run("ruff", "format", "--preview", "--check" if IS_CI else ".")
    session.run("codespell")
    session.run("typos")


@session(uv_groups=["test-onnx"], uv_extras=with_onnx(["cpu", "opencv"]))
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

    # Pre-fetch intersphinx inventories (with retries + backoff) so the build
    # reads them from disk and can't fail on an intermittent network blip.
    session.run(
        "python",
        "docs/fetch-intersphinx-inventories.py",
        "--conf",
        "docs/source/conf.py",
        "--src-dir",
        "docs/source",
        external=False,
    )

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


# Files derived from pyproject.toml + uv.lock and committed alongside them. They
# are what the pip and conda install lanes validate against, so a stale copy means
# those lanes certify an environment nobody ships.
EXPORTED_DEPENDENCY_FILES = [f"requirements.{variant}.txt" for variant in DEVICE_VARIANTS] + [
    "requirements.dev.txt",
    "environment.yml",
]


def _export_dependency_files(session: nox.Session) -> None:
    """Regenerate `EXPORTED_DEPENDENCY_FILES` from pyproject.toml and uv.lock.

    Shared by ``lock`` and ``check`` so the two cannot disagree about what
    "up to date" means. The commands must match byte for byte, arguments
    included: both `uv export` and `p2c` record the invoking command line in the
    file header, so regenerating with different arguments would differ on the
    header alone.
    """
    for variant in DEVICE_VARIANTS:
        out = Path(f"requirements.{variant}.txt")
        session.run(
            "uv",
            "export",
            "--no-emit-project",
            "--no-dev",
            "--extra",
            variant,
            "--extra",
            "onnx" if variant == "cpu" else f"onnx-{variant}",
            "--extra",
            "opencv",
            "-o",
            str(out),
        )
        # uv export does not emit index directives; prepend the matching PyTorch index.
        out.write_text(f"--extra-index-url https://download.pytorch.org/whl/{variant}\n{out.read_text()}")

    session.run(
        "uv",
        "export",
        "--no-emit-project",
        "--extra",
        "onnx",
        "--extra",
        "opencv",
        "-o",
        "requirements.dev.txt",
    )
    session.run(
        "p2c",
        "yaml",
        "--pyproject",
        "pyproject.toml",
        "--python-include",
        "infer",
        "--name",
        "dataeval",
        "--output",
        "environment.yml",
    )


@session(python=PYTHON_VERSIONS[0], uv_only_groups=["lock"], uv_sync_locked=False)
def lock(session: nox.Session) -> None:
    """Lock dependencies in "uv.lock". Update dependencies by calling `nox -e lock -- upgrade`."""
    upgrade_args = ["--upgrade"] if "upgrade" in session.posargs else []
    session.run("uv", "lock", *upgrade_args)
    _export_dependency_files(session)


@session(uv_only_groups=["docsync"], uv_no_install_project=True)
def docsync(session: nox.Session) -> None:
    """
    Sync notebook .py/.ipynb pairs.

    Posargs:
        adopt [stem ...] -- generate .py scripts for orphan .ipynb files
        prune [stem ...] -- delete orphan .ipynb files
    """
    notebook_dir = "docs/source/notebooks"

    adopt = "adopt" in session.posargs
    prune = "prune" in session.posargs
    if adopt and prune:
        session.error("Pass either 'adopt' or 'prune', not both")
    selected = {arg for arg in session.posargs if arg not in {"adopt", "prune"}}
    if selected and not (adopt or prune):
        session.error(f"Notebook names require 'adopt' or 'prune': {', '.join(sorted(selected))}")

    # The .py scripts are the committed source of truth and the .ipynb files are gitignored
    # build artifacts, so an .ipynb without a script pair is usually left over from another
    # branch rather than a new notebook. Never touch those implicitly -- adopting one into a
    # script (or deleting it) is opt-in.
    ipynb_stems = {Path(f).stem for f in glob.glob(f"{notebook_dir}/*.ipynb")}
    py_stems = {Path(f).stem for f in glob.glob(f"{notebook_dir}/*.py")}
    orphans = sorted(ipynb_stems - py_stems)
    if selected:
        unknown = selected - set(orphans)
        if unknown:
            session.error(f"Not an orphan notebook: {', '.join(sorted(unknown))}")
        orphans = sorted(selected)

    for stem in orphans:
        if adopt:
            session.log(f"Generating script for notebook: {stem}.ipynb")
            session.run("jupytext", "--to", "py:percent", f"{notebook_dir}/{stem}.ipynb")
        elif prune:
            session.log(f"Removing orphan notebook: {stem}.ipynb")
            Path(f"{notebook_dir}/{stem}.ipynb").unlink()
        else:
            session.warn(f"Skipping orphan notebook (no {stem}.py pair): {stem}.ipynb")
    if orphans and not (adopt or prune):
        session.warn(
            "Orphan notebooks are usually left over from another branch. Delete them with "
            "'nox -s docsync -- prune [name ...]', or keep a new one with "
            "'nox -s docsync -- adopt [name ...]'."
        )

    # Bidirectional sync: updates whichever side is stale (uses jupytext.toml pairing)
    # If ipynb is newer -> updates py; if py is newer -> updates ipynb
    session.run("jupytext", "--sync", notebook_dir + "/*.py")


@session(python=PYTHON_VERSIONS[0], uv_only_groups=["lock"])
def check(session: nox.Session) -> None:
    """Validate lock file and exported dependency files are up to date.

    The exported half of that docstring was previously unenforced: only uv.lock
    was checked, so the committed requirements.*.txt and environment.yml could
    drift from pyproject.toml unnoticed -- and the conda lane then validated an
    environment.yml that no longer matched.

    Regenerating in place rather than into a scratch directory mirrors the `lint`
    session: a local run repairs the tree, and CI still fails because the diff is
    non-empty and the container is discarded either way.
    """
    session.run("uv", "lock", "--check")
    _export_dependency_files(session)
    session.run("git", "diff", "--exit-code", "--", *EXPORTED_DEPENDENCY_FILES, external=True)

"""Container smoke test for the DataEval base image.

Run against the interpreter that the image actually ships:

    python docker/smoke.py

The repo-level suites (``nox -s test``, ``lint``, ``type``) already cover the
source. What they cannot cover is the artifact: a frozen, variant-specific venv
built from wheels for one platform and one interpreter. The failures this
catches are container-shaped -- a missing extra, a CPU wheel in a CUDA image, a
native ABI mismatch between numpy and a compiled dependency, an entry point
that no longer resolves once the project is installed non-editable.

Every check prints a line and raises on failure, so the Docker build stops
before anything is pushed.
"""

from __future__ import annotations

import importlib
import io
import logging
import os
import sys
from importlib.metadata import entry_points

FAILURES: list[str] = []


def check(label: str):
    """Run the decorated function as a named check, recording rather than raising on failure."""

    def decorator(fn):
        try:
            detail = fn()
        except Exception as exc:
            FAILURES.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}", flush=True)
        else:
            print(f"  ok    {label}{f' ({detail})' if detail else ''}", flush=True)
        return fn

    return decorator


# flush=True throughout: these run interleaved with dataeval's own logging on
# stderr, and block-buffered stdout would report the checks out of order in a
# CI log, which is exactly when the ordering matters.
print("DataEval container smoke test", flush=True)
print(f"  python {sys.version.split()[0]} at {sys.executable}", flush=True)
print(f"  variant {os.environ.get('DATAEVAL_SMOKE_VARIANT', '<unset>')}", flush=True)


@check("dataeval imports and reports a real version")
def _version() -> str:
    import dataeval

    version = dataeval.__version__
    if not version or version == "unknown":
        raise AssertionError(f"__version__ is {version!r}; hatch-vcs did not resolve a version at build time")
    return version


@check("public submodules import")
def _submodules() -> str:
    names = [
        "dataeval.bias",
        "dataeval.config",
        "dataeval.core",
        "dataeval.data",
        "dataeval.exceptions",
        "dataeval.extractors",
        "dataeval.flags",
        "dataeval.models",
        "dataeval.performance",
        "dataeval.protocols",
        "dataeval.quality",
        "dataeval.scope",
        "dataeval.shift",
        "dataeval.types",
        "dataeval.utils",
    ]
    for name in names:
        module = importlib.import_module(name)
        # A wheel that shipped an empty directory, or a stale one left in a
        # working tree, still "imports" as an implicit namespace package with
        # __file__ set to None. That is never a real DataEval module, and it is
        # the one import failure a source checkout cannot reproduce.
        if getattr(module, "__file__", None) is None:
            raise AssertionError(f"{name} resolved to a namespace package, not a real module")
    return f"{len(names)} modules"


@check("MAITE entry points resolve [IR-1-H-3]")
def _entry_points() -> str:
    groups = [
        "maite.tasks",
        "maite.protocols.image_classification.Model",
        "maite.protocols.object_detection.Model",
    ]
    loaded = 0
    for group in groups:
        found = [ep for ep in entry_points(group=group) if ep.name.startswith("dataeval")]
        if not found:
            raise AssertionError(f"no dataeval entry points advertised in group {group!r}")
        for ep in found:
            # Loading is the real check: a non-editable install that lost its
            # metadata, or a moved symbol, fails here rather than at user runtime.
            ep.load()
            loaded += 1
    return f"{loaded} entry points"


@check("numeric stack executes")
def _numeric_stack() -> str:
    import numpy as np
    import scipy.stats
    import sklearn.decomposition

    rng = np.random.default_rng(0)
    data = rng.normal(size=(64, 8))
    sklearn.decomposition.PCA(n_components=2).fit_transform(data)
    scipy.stats.entropy(np.abs(data[0]) + 1e-9)
    return f"numpy {np.__version__}"


@check("torch executes and matches the image variant")
def _torch() -> str:
    import torch

    torch.matmul(torch.ones(4, 4), torch.ones(4, 4))
    build = torch.__version__

    # The wheel's local version segment records which PyTorch index it was
    # resolved from: `+cpu`, `+cu126`, `+cu130`. A CPU wheel in a CUDA image
    # (or the reverse) is the most likely way these builds go wrong, and it is
    # invisible until someone tries to use a GPU, so assert it rather than
    # just reporting it.
    #
    # VARIANT is set by the Dockerfile's test stage and by `nox -s docker_smoke`.
    # It is deliberately absent from the prod image, which sets no environment
    # variables at all. When unset the check degrades to reporting.
    expected = os.environ.get("DATAEVAL_SMOKE_VARIANT")
    if expected:
        _, _, local = build.partition("+")
        if not local:
            raise AssertionError(
                f"torch {build} carries no local version segment; expected '+{expected}'. "
                "The per-extra PyTorch index in pyproject.toml did not apply."
            )
        if local != expected:
            raise AssertionError(f"torch {build} is a '+{local}' build, but this is the '{expected}' image")

    return f"{build}, cuda_available={torch.cuda.is_available()}"


@check("onnxruntime loads with providers")
def _onnxruntime() -> str:
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    if not providers:
        raise AssertionError("onnxruntime reports no execution providers")
    return f"{onnxruntime.__version__}: {', '.join(providers)}"


@check("compiled dependencies import")
def _compiled() -> str:
    import cv2
    import lightgbm
    import numba
    import polars  # noqa: ICN001
    import xxhash

    numba.njit(lambda x: x + 1)(1)
    versions = {}
    for module in (numba, lightgbm, cv2, polars, xxhash):
        # xxhash spells it `VERSION`; it reserves `XXHASH_VERSION` for the
        # bundled C library. Everything else here uses `__version__`.
        version = next(
            (getattr(module, attr) for attr in ("__version__", "VERSION") if hasattr(module, attr)),
            None,
        )
        if version is None:
            raise AssertionError(f"{module.__name__} exposes neither __version__ nor VERSION")
        versions[module.__name__] = version

    return str.join(", ", (f"{name} {ver}" for name, ver in versions.items()))


@check("logs go to stderr, never stdout [IR-2.2-H-1]")
def _logging() -> str:
    import dataeval
    from dataeval._log import _ROOT

    logger = logging.getLogger(_ROOT)

    # A library must not emit anything until the consumer opts in.
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        raise AssertionError(f"logger {_ROOT!r} has no NullHandler; import-time output is possible")

    # dataeval.log() attaches a StreamHandler. Confirm it writes to stderr:
    # IR-2.2-H-1 permits stdout or stderr, but a library that writes to stdout
    # corrupts any pipeline whose real output is on stdout.
    dataeval.log(level=logging.DEBUG)
    streams = [h.stream for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    if not streams:
        raise AssertionError("dataeval.log() attached no StreamHandler")
    if any(s is sys.stdout for s in streams):
        raise AssertionError("dataeval.log() writes to stdout; expected stderr")

    # Prove a record actually reaches the stream.
    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    logger.addHandler(handler)
    logger.debug("smoke test log record")
    logger.removeHandler(handler)
    if "smoke test log record" not in captured.getvalue():
        raise AssertionError("log record did not reach the attached handler")

    return f"{len(streams)} stream handler(s), all stderr"


if FAILURES:
    print(f"\n{len(FAILURES)} check(s) failed:", flush=True)
    for failure in FAILURES:
        print(f"  - {failure}", flush=True)
    sys.exit(1)

print("\nAll checks passed.", flush=True)

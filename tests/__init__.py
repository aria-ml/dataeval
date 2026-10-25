"""Test package.

Keep native thread pools small before conftest imports numerical libraries. Each
xdist worker runs small workloads; letting every worker start a pool for every
CPU spends more time coordinating threads than computing. Environment settings
also reach subprocesses and can be overridden for local experiments.

Pins the Numba disk cache to a single location for the whole suite. Numba reads
``NUMBA_CACHE_DIR`` when it is first imported and otherwise writes cache entries
into the ``__pycache__`` beside each source file, which splits the cache between
``nox`` runs and bare ``pytest`` runs and puts build artifacts under ``src/``.
This module is imported before ``tests.conftest``, and dataeval delays every
Numba import to first use, so the setting always lands in time.

The cache itself is warmed by ``tests.fixtures.numba_cache``.
"""

import os
from pathlib import Path

for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "NUMBA_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
):
    os.environ.setdefault(_variable, "1")

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".numba-cache"))

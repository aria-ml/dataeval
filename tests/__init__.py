"""Test package.

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

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".numba-cache"))

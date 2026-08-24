"""Session fixture that warms the Numba on-disk JIT cache before tests run.

The clustering kernels ported from ``fast_hdbscan`` are decorated with
``@numba.njit(cache=True)``, so their machine code is compiled once and reused
from disk (``$NUMBA_CACHE_DIR``) on later runs. Compiling every kernel costs
~19s; loading them from a populated cache costs ~0.5s.

Without this each xdist worker pays the compile independently the first time a
test touches clustering, which both multiplies the cost and skews test timings
depending on which worker happened to get the first clustering test. Compiling
in ``pytest_configure`` runs it once in the controller before any worker exists,
so the workers only ever load the result.

Warming goes through the public call paths rather than invoking the njit
functions directly. Numba specializes on argument types, so a cache entry only
helps if it was compiled from the same types the real call site passes -- see
``_warm`` for the specializations this covers.
"""

import numpy as np
import pytest

# Every njit kernel dataeval calls from Python -- i.e. the full set imported by
# the `from dataeval.core._fast_hdbscan... import` sites outside the package.
# Only these need warming: kernels reached from inside another kernel are
# compiled into their caller and restored with it.
#
# Numba silently recompiles anything the warm misses, so a gap shows up only as
# a slow test run. Asserting the list keeps drift in the call paths loud.
EXPECTED = (
    ("dataeval.core._fast_hdbscan._disjoint_set", ("ds_find", "ds_rank_create", "ds_union_by_rank")),
    ("dataeval.core._fast_hdbscan._mst", ("_flatten_and_sort", "_update_tree", "compare_links_to_cluster_std")),
    (
        "dataeval.core._fast_hdbscan._cluster_trees",
        (
            "cluster_tree_from_condensed_tree",
            "condense_tree",
            "extract_eom_clusters",
            "get_cluster_label_vector",
            "get_point_membership_strength_vector",
            "mst_to_linkage_tree",
        ),
    ),
)


_warmed = False


def _warm() -> None:
    """Run every clustering call path that reaches a cached njit kernel."""
    global _warmed

    from dataeval.core._clusterer import cluster
    from dataeval.core._mst import minimum_spanning_tree
    from dataeval.quality._duplicates import _find_cluster_duplicates

    rng = np.random.default_rng(0)

    # Full HDBSCAN pipeline: the MST kernels plus the _cluster_trees kernels.
    # Rows are repeated so the tree carries zero-length edges and duplicate
    # detection below finds pairs -- without them it returns early and never
    # reaches the disjoint-set kernels.
    result = cluster(np.repeat(rng.random((10, 3)), 2, axis=0))

    # Duplicate detection. cluster_sensitivity is passed positionally to match
    # _find_cluster_duplicates -- omitting it compiles an unrelated
    # `omitted(default=1.0)` specialization that the real call site never hits.
    _find_cluster_duplicates(result["mst"], result["clusters"], 1.0)

    # Two well-separated blobs leave the k-NN graph disconnected, so the MST
    # build falls back to inter-cluster neighbors. Those arrays are transposed,
    # giving _flatten_and_sort and _update_tree their F-contiguous
    # specializations in addition to the C-contiguous ones above.
    minimum_spanning_tree(np.vstack([rng.random((20, 3)), rng.random((20, 3)) + 1000.0]), k=15)

    _warmed = True


def _uncompiled() -> list[str]:
    """Return the qualified names of expected dispatchers with no compiled signature."""
    from importlib import import_module

    missing = []
    for module_name, function_names in EXPECTED:
        module = import_module(module_name)
        missing.extend(f"{module_name}.{name}" for name in function_names if not getattr(module, name).signatures)
    return missing


def pytest_configure(config: pytest.Config) -> None:
    """Compile the cache before xdist spawns its workers."""
    # `workerinput` is only set on xdist workers, and xdist spawns them from
    # `pytest_sessionstart`, which runs after this hook -- so this is the
    # controller and nothing else is running yet. The controller never collects
    # (`DSession.pytest_collection` short-circuits it), so this is the only
    # point in a parallel run where the compile is guaranteed to be sequential.
    # Workers reach the fixture below with `$NUMBA_CACHE_DIR` already populated
    # and only have to load it.
    if hasattr(config, "workerinput") or config.option.collectonly:
        return
    _warm()


@pytest.fixture(scope="session", autouse=True)
def _numba_cache() -> None:
    """Load the cache into this process, then check the warm covered every kernel."""
    # Already warmed in-process by `pytest_configure` unless this is an xdist
    # worker, where this loads the controller's compiled kernels from disk.
    if not _warmed:
        _warm()

    missing = _uncompiled()
    if missing:
        raise RuntimeError(
            "Numba cache warming missed the following kernels, so tests will recompile them:\n  "
            + "\n  ".join(missing)
            + "\nUpdate tests/fixtures/numba_cache.py to match the current call paths."
        )

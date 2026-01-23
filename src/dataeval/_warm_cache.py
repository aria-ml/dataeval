"""Private module for warming Numba JIT compilation cache.

This module provides functionality to pre-compile all Numba JIT-decorated functions
by calling them with minimal test data. This ensures compiled functions are cached
to disk before running tests in parallel, avoiding redundant compilation overhead.

This is an internal implementation detail and should not be imported by users.

Notes
-----
- Failures are caught and logged as warnings, allowing tests to continue
- The first call (cold cache) takes ~15-20 seconds
- Subsequent calls (warm cache) takes ~3-5 seconds
- Saves 10-15 seconds per test worker process

To execute from command line:
```
$ python -m dataeval._warm_cache
```
"""

__all__ = []

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np

from dataeval.core._clusterer import cluster
from dataeval.core._fast_hdbscan._disjoint_set import ds_rank_create, ds_union_by_rank
from dataeval.core._fast_hdbscan._mst import (
    _cluster_edges,
    _flatten_and_sort,
    _update_tree,
    compare_links_to_cluster_std,
)

F = TypeVar("F", bound=Callable[..., None])


def print_elapsed(description: str) -> Callable[[F], F]:
    """Decorator that prints elapsed time for a function and handles errors."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            print(f"  - {description}...", end="", flush=True)
            start = time.perf_counter()
            try:
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                print(f" ({elapsed:.2f}s)")
            except Exception:
                print(" (warning: skipped)")

        return wrapper  # type: ignore[return-value]

    return decorator


@print_elapsed("Compiling disjoint set functions")
def warm_disjoint_set() -> None:
    """Warm disjoint set functions."""
    disjoint_set = ds_rank_create(np.int64(5))
    ds_union_by_rank(disjoint_set, np.intp(0), np.intp(1))


@print_elapsed("Compiling MST construction functions")
def warm_mst_construction() -> None:
    """Warm MST construction functions."""
    n_neighbors = np.array([[1], [0], [3], [2], [4]], dtype=np.intp)
    n_distance = np.array([[0.1], [0.1], [0.2], [0.2], [0.3]], dtype=np.float32)
    nbrs_sorted, dist_sorted, point_sorted = _flatten_and_sort(n_neighbors, n_distance)

    size = np.int64(n_neighbors.shape[0])
    tree = np.zeros((size - 1, 3), dtype=np.float32)
    total_edge = 0
    disjoint_set = (np.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    merge_tracker = np.full((n_neighbors.shape[1] + 1, n_neighbors.shape[0]), -1, dtype=np.int64)
    tree, total_edge, _, merge_tracker[0] = _update_tree(
        tree, total_edge, disjoint_set, merge_tracker[0], nbrs_sorted, dist_sorted, point_sorted
    )


@print_elapsed("Compiling cluster edge detection")
def warm_cluster_edge_detection() -> None:
    """Warm cluster edge detection."""
    tracker = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    cluster_distances = np.random.rand(5, 10).astype(np.float32)
    _cluster_edges(tracker, 2, cluster_distances)


@print_elapsed("Compiling duplicate detection")
def warm_duplicate_detection() -> None:
    """Warm duplicate detection functions."""
    mst = np.array([[0, 1, 0.1], [1, 2, 0.2], [2, 3, 0.15], [3, 4, 0.25]], dtype=np.float32)
    clusters = np.array([0, 0, 1, 1, 1], dtype=np.intp)
    compare_links_to_cluster_std(mst, clusters)


@print_elapsed("Compiling main clustering pipeline")
def warm_cluster_pipeline() -> None:
    """Warm main clustering pipeline (includes cluster_trees functions)."""
    cluster(np.random.rand(20, 3))


def main() -> None:
    """Command-line entry point for warming the cache."""
    print("Warming Numba cache...")

    warm_disjoint_set()
    warm_mst_construction()
    warm_cluster_edge_detection()
    warm_duplicate_detection()
    warm_cluster_pipeline()

    print("Numba cache warmed successfully!")


if __name__ == "__main__":
    main()

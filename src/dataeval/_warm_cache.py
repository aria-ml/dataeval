"""Private module for warming Numba JIT compilation cache.

This module provides functionality to pre-compile all Numba JIT-decorated functions
by calling them with minimal test data. This ensures compiled functions are cached
to disk before running tests in parallel, avoiding redundant compilation overhead.

This is an internal implementation detail and should not be imported by users.
"""

from __future__ import annotations

__all__ = []


def warm_cache() -> None:
    """
    Pre-warm Numba JIT compilation cache.

    Triggers compilation of all Numba-decorated functions by calling them with
    minimal test data. This populates the disk cache (~/.cache/numba) so that
    subsequent processes can load pre-compiled versions instead of recompiling.

    This function is designed to be called before running parallel tests to avoid
    redundant compilation across multiple test workers.

    Notes
    -----
    - Failures are caught and logged as warnings, allowing tests to continue
    - The first call (cold cache) takes ~15-20 seconds
    - Subsequent calls (warm cache) take ~3-5 seconds
    - Saves 10-15 seconds per test worker process

    To execute from command line:
    ```
    $ python -m dataeval._warm_cache
    ```
    """
    import numpy as np

    try:
        # Import Numba modules to trigger compilation
        from dataeval.core._fast_hdbscan._disjoint_set import ds_rank_create, ds_union_by_rank
        from dataeval.core._fast_hdbscan._mst import (
            _cluster_edges,
            _init_tree,
            _update_tree_by_distance,
            compare_links_to_cluster_std,
        )

        # Warm disjoint set functions
        disjoint_set = ds_rank_create(np.int32(10))
        ds_union_by_rank(disjoint_set, np.intp(0), np.intp(1))

        # Warm MST construction functions
        n_neighbors = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8], dtype=np.intp)
        n_distance = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5], dtype=np.float32)
        tree, int_tree, disjoint_set, _ = _init_tree(n_neighbors, n_distance)

        n_neighbors_uint = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1], dtype=np.uint32)
        n_distance_new = np.array([0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0], dtype=np.float32)
        tree, int_tree, disjoint_set, _ = _update_tree_by_distance(
            tree, int_tree, disjoint_set, n_neighbors_uint, n_distance_new
        )

        # Warm cluster edge detection
        tracker = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        cluster_distances = np.random.rand(5, 10).astype(np.float32)
        _cluster_edges(tracker, 2, cluster_distances)

        # Warm duplicate detection
        mst = np.array([[0, 1, 0.1], [1, 2, 0.2], [2, 3, 0.15], [3, 4, 0.25]], dtype=np.float32)
        clusters = np.array([0, 0, 1, 1, 1], dtype=np.intp)
        compare_links_to_cluster_std(mst, clusters)

        # Warm main clustering pipeline (includes cluster_trees functions)
        from dataeval.core._clusterer import cluster

        cluster(np.random.rand(20, 3))

    except Exception as e:
        import warnings

        warnings.warn(
            f"Failed to warm Numba cache: {e}\nTests will still run but may take longer on first execution.",
            RuntimeWarning,
            stacklevel=2,
        )


def main() -> None:
    """Command-line entry point for warming the cache."""
    print("Warming Numba cache...")
    print("  - Compiling disjoint set functions...")
    print("  - Compiling MST construction functions...")
    print("  - Compiling cluster edge detection...")
    print("  - Compiling duplicate detection...")
    print("  - Compiling main clustering pipeline...")

    warm_cache()

    print("✓ Numba cache warmed successfully!")


if __name__ == "__main__":
    main()

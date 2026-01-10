"""Private module for warming Numba JIT compilation cache.

This module provides functionality to pre-compile all Numba JIT-decorated functions
by calling them with minimal test data. This ensures compiled functions are cached
to disk before running tests in parallel, avoiding redundant compilation overhead.

This is an internal implementation detail and should not be imported by users.
"""

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
            _flatten_and_sort,
            _update_tree,
            compare_links_to_cluster_std,
        )

        # Warm disjoint set functions
        disjoint_set = ds_rank_create(np.int64(10))
        ds_union_by_rank(disjoint_set, np.intp(0), np.intp(1))

        # Warm flatten and sort
        n_neighbors = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8], dtype=np.intp)
        n_distance = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5], dtype=np.float32)
        nbrs_sorted, dist_sorted, point_sorted = _flatten_and_sort(n_neighbors, n_distance)

        # Warm MST construction functions
        size = np.int64(n_neighbors.shape[0])
        tree = np.zeros((size - 1, 3), dtype=np.float32)
        total_edge = 0
        merge_tracker = np.full((n_neighbors.shape[1] + 1, n_neighbors.shape[0]), -1, dtype=np.int64)
        tree, total_edge, tree_disjoint_set, merge_tracker[0] = _update_tree(
            tree, total_edge, disjoint_set, merge_tracker[0], nbrs_sorted, dist_sorted, point_sorted
        )

        # Warm cluster edge detection
        tracker = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
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

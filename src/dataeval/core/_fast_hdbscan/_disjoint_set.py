"""Disjoint set (union-find) data structure with Numba JIT compilation.

This module is adapted from fast_hdbscan.disjoint_set v0.2.0:
    https://github.com/TutteInstitute/fast_hdbscan
    Copyright (c) 2020, Leland McInnes
    License: BSD 2-Clause

Modifications:
    - Added cache=True to all @numba.njit() decorators for disk caching
    - Enhanced documentation and updated typing
"""

__all__ = []

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True)
def ds_find(disjoint_set: tuple[NDArray[np.int64], NDArray[np.int64]], x: np.int64) -> np.int64:
    """
    Find the root of the set containing element x with path compression.

    This implements the union-find "find" operation with path compression optimization.
    As it traverses up the tree to find the root, it compresses the path by making
    each node point directly to its grandparent, thereby flattening the tree structure.

    Parameters
    ----------
    disjoint_set : tuple[NDArray[np.int64], NDArray[np.int64]]
        Tuple of (parent, rank) arrays representing the disjoint set forest.
        parent[i] gives the parent of node i.
    x : np.int64
        The element whose set root we want to find

    Returns
    -------
    np.int64
        The root element of the set containing x
    """
    parent = disjoint_set[0]
    while parent[x] != x:
        x, parent[x] = parent[x], parent[parent[x]]
    return x


@numba.njit(cache=True)
def ds_rank_create(n_elements: np.int64) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Create a new disjoint set data structure for n_elements.

    Initializes a disjoint set where each element is in its own set.
    Uses union-by-rank strategy for efficient merging.

    Parameters
    ----------
    n_elements : np.int64
        The number of elements in the disjoint set

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.int64]]
        Tuple of (parent, rank) arrays where:
        - parent[i] = i initially (each element is its own parent/root)
        - rank[i] = 0 initially (all trees have height 0)
    """
    parent = np.arange(n_elements, dtype=np.int64)
    rank = np.zeros(n_elements, dtype=np.int64)
    return (parent, rank)


@numba.njit(cache=True)
def ds_union_by_rank(disjoint_set: tuple[NDArray[np.int64], NDArray[np.int64]], point: np.int64, nbr: np.int64) -> bool:
    """
    Perform union-by-rank on two points in a disjoint set data structure.

    This operation merges the sets containing 'point' and 'nbr' using the union-by-rank
    heuristic for efficiency. The smaller rank tree is attached under the root of the
    larger rank tree.

    This function modifies disjoint_set in-place. The union-by-rank optimization
    ensures the tree depth remains logarithmic, giving nearly constant-time operations.

    Parameters
    ----------
    disjoint_set : tuple[NDArray[np.int64], NDArray[np.int64]]
        Tuple of (parent, rank) arrays representing the disjoint set forest.
        parent[i] gives the parent of node i, rank[i] gives the rank (depth bound) of node i.
    point : int
        Index of the first point to union
    nbr : int
        Index of the second point (neighbor) to union

    Returns
    -------
    bool
        True if the union was successful (sets were different), False if points were already
        in the same set (no union performed)
    """
    x = ds_find(disjoint_set, point)
    y = ds_find(disjoint_set, nbr)

    # Already in same set
    if x == y:
        return False

    # Union by rank: attach smaller rank tree under root of larger rank tree
    if disjoint_set[1][x] < disjoint_set[1][y]:
        x, y = y, x

    disjoint_set[0][y] = x
    if disjoint_set[1][x] == disjoint_set[1][y]:
        disjoint_set[1][x] += 1
    return True

from __future__ import annotations

__all__ = []

from typing import TYPE_CHECKING

from dataeval.typing import ArrayLike

if TYPE_CHECKING:
    from dataeval.utils._clusterer import ClusterData


def clusterer(data: ArrayLike) -> ClusterData:
    # Delay load numba compiled functions
    from dataeval.utils._clusterer import cluster

    return cluster(data)

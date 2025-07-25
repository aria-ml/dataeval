from __future__ import annotations

__all__ = []

import itertools
from collections.abc import Sequence

import numpy as np

from dataeval.config import EPSILON
from dataeval.typing import Array
from dataeval.utils._array import ensure_embeddings


def completeness(embeddings: Array, quantiles: int) -> tuple[float, Sequence[tuple[float, float]]]:
    # Ensure proper data format
    embeddings = ensure_embeddings(embeddings, dtype=np.float64, unit_interval=False)

    # Get data dimensions
    n, p = embeddings.shape
    if quantiles > 2 or quantiles <= 0:
        raise ValueError(
            f"Number of quantiles ({quantiles}) is greater than 2 or is nonpositive. \
            The metric scales exponentially in this value. Please 1 or 2 quantiles."
        )
    if p > 10:
        raise ValueError(
            f"Dimension of embeddings ({p}) is greater than 10. \
            The metric scales exponentially in this value. Please reduce the embedding dimension."
        )
    if n == 0 or p == 0:
        raise ValueError("Your provided embeddings do not contain any data!")
    # n+2 edges partition the embedding dimension (e.g. [0,0.5,1] for quantiles = 1)
    quantile_vec = np.linspace(0, 1, quantiles + 2)

    # Calculate the bin edges for each dimension based on quantiles
    bin_edges = []
    for dim in range(p):
        # Calculate the quantile values for this feature
        edges = np.array(np.quantile(embeddings[:, dim], quantile_vec))
        # Make sure the last bin contains all the remaining points
        edges[-1] += EPSILON
        bin_edges.append(edges)
    # Convert each data point into its corresponding grid cell indices
    grid_indices = []
    for dim in range(p):
        # For each dimension, find which bin each data point belongs to
        # Digitize is 1 indexed so we subtract 1
        indices = np.digitize(embeddings[:, dim], bin_edges[dim]) - 1
        grid_indices.append(indices)

    # Make the rows the data point and the column the grid index
    grid_coords = np.array(grid_indices).T

    # Use set to find unique tuple of grid coordinates
    occupied_cells = set(map(tuple, grid_coords))

    # For the fraction
    num_occupied_cells = len(occupied_cells)

    # Calculate total possible cells in the grid
    num_bins_per_dim = [len(edges) - 1 for edges in bin_edges]
    total_possible_cells = np.prod(num_bins_per_dim)

    # Generate all possible grid cells
    all_cells = set(itertools.product(*[range(bins) for bins in num_bins_per_dim]))

    # Find the empty cells (cells with no data points)
    empty_cells = all_cells - occupied_cells

    # Calculate center points of empty boxes
    empty_box_centers = []
    for cell in empty_cells:
        center_coords = []
        for dim, idx in enumerate(cell):
            # Calculate center of the bin as midpoint between edges
            center = (bin_edges[dim][idx] + bin_edges[dim][idx + 1]) / 2
            center_coords.append(center)
        empty_box_centers.append(center_coords)

    # Calculate the fraction
    fraction = float(num_occupied_cells / total_possible_cells)
    return fraction, empty_box_centers

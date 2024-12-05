import numpy as np
from numpy.typing import NDArray

from fast_cluster import Clusterer

# Function to add to metadata_preprocessing once the clustering in fast_cluster is moved into production

def _binning_by_clusters(data: NDArray[np.number]):
    """
    Bins continuous data by using the Clusterer to identify clusters
    and incorporates outliers by adding them to the nearest bin.
    """
    # Create initial clusters
    groupings = Clusterer(data)
    clusters = groupings.create_clusters()

    # Create bins from clusters
    bin_edges = np.zeros(clusters.max() + 2)
    for group in range(clusters.max() + 1):
        points = np.nonzero(clusters == group)[0]
        bin_edges[group] = data[points].min()

    # Get the outliers
    outliers = np.nonzero(clusters == -1)[0]

    # Identify non-outlier neighbors
    nbrs = groupings._kneighbors[outliers]
    nbrs = np.where(np.isin(nbrs, outliers), -1, nbrs)

    # Find the nearest non-outlier neighbor for each outlier
    nn = np.full(outliers.size, -1, dtype=np.int32)
    for row in range(outliers.size):
        non_outliers = nbrs[row, nbrs[row] != -1]
        if non_outliers.size > 0:
            nn[row] = non_outliers[0]

    # Group outliers by their neighbors
    unique_nnbrs, same_nbr, counts = np.unique(nn, return_inverse=True, return_counts=True)

    # Adjust bin_edges based on each unique neighbor group
    extend_bins = []
    for i, nnbr in enumerate(unique_nnbrs):
        outlier_indices = np.nonzero(same_nbr == i)[0]
        min2add = data[outliers[outlier_indices]].min()
        if counts[i] >= 4:
            extend_bins.append(min2add)
        else:
            if min2add < data[nnbr]:
                cluster = clusters[nnbr]
                bin_edges[cluster] = min2add
    if extend_bins:
        bin_edges = np.concatenate([bin_edges, extend_bins])

    bin_edges = np.sort(bin_edges)
    return bin_edges

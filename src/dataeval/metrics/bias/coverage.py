from __future__ import annotations

__all__ = ["CoverageOutput", "coverage"]

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform

from dataeval.interop import to_numpy
from dataeval.metrics.bias.metadata import coverage_plot
from dataeval.output import OutputMetadata, set_metadata
from dataeval.utils.shared import flatten

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class CoverageOutput(OutputMetadata):
    """
    Output class for :func:`coverage` :term:`bias<Bias>` metric

    Attributes
    ----------
    indices : NDArray
        Array of uncovered indices
    radii : NDArray
        Array of critical value radii
    critical_value : float
        Radius for :term:`coverage<Coverage>`
    """

    indices: NDArray[np.intp]
    radii: NDArray[np.float64]
    critical_value: float

    def plot(
        self,
        images: NDArray[Any],
        top_k: int = 6,
    ) -> Figure:
        """
        Plot the top k images together for visualization

        Parameters
        ----------
        images : ArrayLike
            Original images (not embeddings) in (N, C, H, W) or (N, H, W) format
        top_k : int, default 6
            Number of images to plot (plotting assumes groups of 3)
        """
        # Determine which images to plot
        highest_uncovered_indices = self.indices[:top_k]

        # Grab the images
        images = to_numpy(images)
        selected_images = images[highest_uncovered_indices]

        # Plot the images
        fig = coverage_plot(selected_images, top_k)

        return fig


@set_metadata()
def coverage(
    embeddings: ArrayLike,
    radius_type: Literal["adaptive", "naive"] = "adaptive",
    k: int = 20,
    percent: float = 0.01,
) -> CoverageOutput:
    """
    Class for evaluating :term:`coverage<Coverage>` and identifying images/samples that are in undercovered regions.

    Parameters
    ----------
    embeddings : ArrayLike, shape - (N, P)
        A dataset in an ArrayLike format.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimesionial space.
    radius_type : Literal["adaptive", "naive"], default "adaptive"
        The function used to determine radius.
    k: int, default 20
        Number of observations required in order to be covered.
        [1] suggests that a minimum of 20-50 samples is necessary.
    percent: float, default 0.01
        Percent of observations to be considered uncovered. Only applies to adaptive radius.

    Returns
    -------
    CoverageOutput
        Array of uncovered indices, critical value radii, and the radius for coverage

    Raises
    ------
    ValueError
        If length of :term:`embeddings<Embeddings>` is less than or equal to k
    ValueError
        If radius_type is unknown

    Note
    ----
    Embeddings should be on the unit interval [0-1].

    Example
    -------
    >>> results = coverage(embeddings)
    >>> results.indices
    array([447, 412,   8,  32,  63])
    >>> results.critical_value
    0.8459038956941765

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """

    # Calculate distance matrix, look at the (k+1)th farthest neighbor for each image.
    embeddings = to_numpy(embeddings)
    n = len(embeddings)
    if n <= k:
        raise ValueError(
            f"Number of observations n={n} is less than or equal to the specified number of neighbors k={k}."
        )
    mat = squareform(pdist(flatten(embeddings))).astype(np.float64)
    sorted_dists = np.sort(mat, axis=1)
    crit = sorted_dists[:, k + 1]

    d = embeddings.shape[1]
    if radius_type == "naive":
        rho = (1 / math.sqrt(math.pi)) * ((2 * k * math.gamma(d / 2 + 1)) / (n)) ** (1 / d)
        pvals = np.where(crit > rho)[0]
    elif radius_type == "adaptive":
        # Use data adaptive cutoff as rho
        selection = int(max(n * percent, 1))
        pvals = np.argsort(crit)[::-1][:selection]
        rho = float(np.mean(np.sort(crit)[::-1][selection - 1 : selection + 1]))
    else:
        raise ValueError(f"{radius_type} is an invalid radius type. Expected 'adaptive' or 'naive'")
    return CoverageOutput(pvals, crit, rho)

from __future__ import annotations

__all__ = []

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform

from dataeval._interop import to_numpy
from dataeval._output import Output, set_metadata
from dataeval.utils._shared import flatten

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure


def _plot(images: NDArray[Any], num_images: int) -> Figure:
    """
    Creates a single plot of all of the provided images

    Parameters
    ----------
    images : NDArray
        Array containing only the desired images to plot

    Returns
    -------
    matplotlib.figure.Figure
        Plot of all provided images
    """
    import matplotlib.pyplot as plt

    num_images = min(num_images, len(images))

    if images.ndim == 4:
        images = np.moveaxis(images, 1, -1)
    elif images.ndim == 3:
        images = np.repeat(images[:, :, :, np.newaxis], 3, axis=-1)
    else:
        raise ValueError(
            f"Expected a (N,C,H,W) or a (N, H, W) set of images, but got a {images.ndim}-dimensional set of images."
        )

    rows = int(np.ceil(num_images / 3))
    fig, axs = plt.subplots(rows, 3, figsize=(9, 3 * rows))

    if rows == 1:
        for j in range(3):
            if j >= len(images):
                continue
            axs[j].imshow(images[j])
            axs[j].axis("off")
    else:
        for i in range(rows):
            for j in range(3):
                i_j = i * 3 + j
                if i_j >= len(images):
                    continue
                axs[i, j].imshow(images[i_j])
                axs[i, j].axis("off")

    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class CoverageOutput(Output):
    """
    Output class for :func:`.coverage` :term:`bias<Bias>` metric.

    Attributes
    ----------
    uncovered_indices : NDArray[np.intp]
        Array of uncovered indices
    critical_value_radii : NDArray[np.float64]
        Array of critical value radii
    coverage_radius : float
        Radius for :term:`coverage<Coverage>`
    """

    uncovered_indices: NDArray[np.intp]
    critical_value_radii: NDArray[np.float64]
    coverage_radius: float

    def plot(self, images: ArrayLike, top_k: int = 6) -> Figure:
        """
        Plot the top k images together for visualization

        Parameters
        ----------
        images : ArrayLike
            Original images (not embeddings) in (N, C, H, W) or (N, H, W) format
        top_k : int, default 6
            Number of images to plot (plotting assumes groups of 3)

        Returns
        -------
        matplotlib.figure.Figure
        """

        # Determine which images to plot
        highest_uncovered_indices = self.uncovered_indices[:top_k]

        # Grab the images
        images = to_numpy(images)
        selected_images = images[highest_uncovered_indices]

        # Plot the images
        fig = _plot(selected_images, top_k)

        return fig


@set_metadata
def coverage(
    embeddings: ArrayLike,
    radius_type: Literal["adaptive", "naive"] = "adaptive",
    num_observations: int = 20,
    percent: float = 0.01,
) -> CoverageOutput:
    """
    Class for evaluating :term:`coverage<Coverage>` and identifying images/samples that are in undercovered regions.

    Parameters
    ----------
    embeddings : ArrayLike, shape - (N, P)
        A dataset in an ArrayLike format.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimesionial space.
    radius_type : {"adaptive", "naive"}, default "adaptive"
        The function used to determine radius.
    num_observations : int, default 20
        Number of observations required in order to be covered.
        [1] suggests that a minimum of 20-50 samples is necessary.
    percent : float, default 0.01
        Percent of observations to be considered uncovered. Only applies to adaptive radius.

    Returns
    -------
    CoverageOutput
        Array of uncovered indices, critical value radii, and the radius for coverage

    Raises
    ------
    ValueError
        If embeddings are not on the unit interval [0-1]
    ValueError
        If length of :term:`embeddings<Embeddings>` is less than or equal to num_observations
    ValueError
        If radius_type is unknown

    Note
    ----
    Embeddings should be on the unit interval [0-1].

    Example
    -------
    >>> results = coverage(embeddings)
    >>> results.uncovered_indices
    array([447, 412,   8,  32,  63])
    >>> results.coverage_radius
    0.17592147193757596

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """

    # Calculate distance matrix, look at the (num_observations + 1)th farthest neighbor for each image.
    embeddings = to_numpy(embeddings)
    if np.min(embeddings) < 0 or np.max(embeddings) > 1:
        raise ValueError("Embeddings must be on the unit interval [0-1].")
    len_embeddings = len(embeddings)
    if len_embeddings <= num_observations:
        raise ValueError(
            f"Length of embeddings ({len_embeddings}) is less than or equal to the specified number of \
                observations ({num_observations})."
        )
    embeddings_matrix = squareform(pdist(flatten(embeddings))).astype(np.float64)
    sorted_dists = np.sort(embeddings_matrix, axis=1)
    critical_value_radii = sorted_dists[:, num_observations + 1]

    d = embeddings.shape[1]
    if radius_type == "naive":
        coverage_radius = (1 / math.sqrt(math.pi)) * (
            (2 * num_observations * math.gamma(d / 2 + 1)) / (len_embeddings)
        ) ** (1 / d)
        uncovered_indices = np.where(critical_value_radii > coverage_radius)[0]
    elif radius_type == "adaptive":
        # Use data adaptive cutoff as coverage_radius
        selection = int(max(len_embeddings * percent, 1))
        uncovered_indices = np.argsort(critical_value_radii)[::-1][:selection]
        coverage_radius = float(np.mean(np.sort(critical_value_radii)[::-1][selection - 1 : selection + 1]))
    else:
        raise ValueError(f"{radius_type} is an invalid radius type. Expected 'adaptive' or 'naive'")
    return CoverageOutput(uncovered_indices, critical_value_radii, coverage_radius)

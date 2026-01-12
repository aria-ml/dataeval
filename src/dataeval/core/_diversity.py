__all__ = []

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy

_logger = logging.getLogger(__name__)


def diversity_shannon(
    counts: NDArray[np.intp],
    num_bins: NDArray[np.intp],
) -> NDArray[np.double]:
    """
    Compute :term:`diversity<Diversity>` for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the Shannon entropy.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    counts : NDArray[np.intp]
        Array containing bin counts for each factor
    num_bins : NDArray[np.intp]
        Number of bins with values for each factor

    Returns
    -------
    NDArray[np.double]
        Diversity index per column of X

    See Also
    --------
    scipy.stats.entropy
    """
    _logger.debug("Computing Shannon diversity for %d factors", counts.shape[1] if counts.ndim > 1 else 1)
    raw_entropy = np.asarray(entropy(counts, axis=0))
    ent_norm = np.empty(raw_entropy.shape)
    ent_norm[num_bins != 1] = raw_entropy[num_bins != 1] / np.log(num_bins[num_bins != 1])
    ent_norm[num_bins == 1] = 0
    _logger.debug("Shannon diversity computed: mean=%.4f", np.mean(ent_norm))
    return ent_norm


def diversity_simpson(
    counts: NDArray[np.intp],
    num_bins: NDArray[np.intp],
) -> NDArray[np.double]:
    """
    Compute :term:`diversity<Diversity>` for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as the inverse Simpson diversity index linearly rescaled to the unit interval.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    counts : NDArray[np.intp]
        Array containing bin counts for each factor
    num_bins : NDArray[np.intp]
        Number of bins with values for each factor

    Notes
    -----
    If there is only one category, the diversity index takes a value of 0.

    Returns
    -------
    NDArray[np.double]
        Diversity index per column of X
    """
    _logger.debug("Computing Simpson diversity for %d factors", counts.shape[1])
    ev_index = np.empty(counts.shape[1])
    # loop over columns for convenience
    for col, cnts in enumerate(counts.T):
        # relative frequencies
        p_i = cnts / np.sum(cnts)
        # inverse Simpson index
        s_0 = 1 / np.sum(p_i**2)
        if num_bins[col] == 1:
            ev_index[col] = 0
        else:
            # normalized by number of bins
            ev_index[col] = (s_0 - 1) / (num_bins[col] - 1)
    _logger.debug("Simpson diversity computed: mean=%.4f", np.mean(ev_index))
    return ev_index

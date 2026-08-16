__all__ = []

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

from dataeval._log import get_logger
from dataeval.exceptions import ShapeMismatchError

_logger = get_logger(__name__)

CONTINUOUS_MIN_SAMPLE_SIZE = 20

# Share of a sample that must be repeated values before ties count as evidence that the
# support is discrete rather than finely recorded. See :func:`is_continuous`.
DUPLICATE_SUPPORT_FRACTION = 0.1

# Floor under the sqrt(n) level budget, so a small sample cannot push the budget below the
# cardinality of an ordinary categorical factor. See :func:`level_budget`.
MIN_LEVEL_BUDGET = 20


def get_counts(data: NDArray[np.intp], min_num_bins: int | None = None) -> NDArray[np.intp]:
    """
    Return columnwise unique counts for discrete data.

    Parameters
    ----------
    data : NDArray
        Array containing integer values for metadata factors
    min_num_bins : int | None, default None
        Minimum number of bins for bincount, helps force consistency across runs

    Returns
    -------
    NDArray[np.int]
        Bin counts per column of data.
    """
    max_value = data.max() + 1 if min_num_bins is None else min_num_bins
    cnt_array = np.zeros((max_value, data.shape[1]), dtype=np.intp)
    for idx in range(data.shape[1]):
        cnt_array[:, idx] = np.bincount(data[:, idx], minlength=max_value)

    return cnt_array


def _digitize_with_missing(data: NDArray[Any], bin_edges: Any) -> NDArray[np.intp]:
    """
    Digitize values, giving NaN a bin of its own above every other bin.

    A missing value is not a small value, a large value, or a value between two edges, so
    it cannot share a bin with observed data without distorting whatever reads the result.
    It gets the next index above the highest one the observed values reached, which keeps
    the codes contiguous — a gap would show up downstream as an empty category.
    """
    missing = np.isnan(data)
    if not missing.any():
        return np.digitize(data, bin_edges)

    # Digitize the missing entries against a placeholder, then overwrite their bin.
    binned = np.digitize(np.where(missing, 0.0, data), bin_edges)
    observed = binned[~missing]
    binned[missing] = observed.max() + 1 if observed.size else 0
    return binned


def digitize_data(data: list[Any] | NDArray[Any], bins: int | Iterable[float]) -> NDArray[np.intp]:
    """
    Digitizes a list of values into a given number of bins.

    NaN entries are placed in a bin of their own, above the bins holding observed values.

    Parameters
    ----------
    data : list | NDArray
        The values to be digitized.
    bins : int | Iterable[float]
        The number of bins or list of bin edges for the discrete values that data will be digitized into.

    Returns
    -------
    NDArray[np.intp]
        The digitized values
    """
    if not np.all([np.issubdtype(type(n), np.number) for n in data]):
        raise TypeError(
            "Encountered a data value with non-numeric type when digitizing a factor. "
            "Ensure all occurrences of continuous factors are numeric types.",
        )
    data = np.asarray(data)
    if isinstance(bins, int):
        # Edges describe where observed values fall, so they are derived from those alone.
        _, bin_edges = np.histogram(_observed(data), bins=bins)
        bin_edges[-1] = np.inf
        bin_edges[0] = -np.inf
    else:
        bin_edges = list(bins)
    return _digitize_with_missing(data, bin_edges)


def _observed(data: NDArray[Any]) -> NDArray[Any]:
    """Return the entries that can carry bin edges: finite, so neither missing nor infinite."""
    return data[np.isfinite(data)] if np.issubdtype(data.dtype, np.inexact) else data


def _uniform_edges(observed: NDArray[Any], bin_method: str, max_bins: int | None = None) -> NDArray[np.float64]:
    """
    Place bin edges by equal width, then move them to quantiles for ``uniform_count``.

    The starting count comes from NumPy's ``bins="auto"`` and is then reduced while any
    non-empty bin holds fewer than 10 entries, so the count is a function of the data
    rather than a setting. ``max_bins`` caps that count before the reduction runs, since
    the reduction walks down one bin at a time and gives up after twenty steps, which
    cannot bring a count chosen for a heavily peaked factor back within a budget set by
    the sample size. NumPy chooses the starting count from the data's spread, so a
    heavy-tailed factor can make ``bins="auto"`` itself expensive on NumPy 1.x, where the
    rule is uncapped; the cap here applies to the result, not to that call.

    The reduction never goes below two bins. One bin is a constant column: it carries no
    entropy, so every bias statistic reports it as zero and the factor disappears from the
    output rather than being reported coarsely.
    """
    counts, bin_edges = np.histogram(observed, bins="auto")
    n_bins = counts.size
    if max_bins is not None and n_bins > max_bins:
        n_bins = max(int(max_bins), 2)
        counts, bin_edges = np.histogram(observed, bins=n_bins)
    if counts[counts > 0].min() < 10:
        counter = 20
        while counts[counts > 0].min() < 10 and n_bins > 2 and counter > 0:
            counter -= 1
            n_bins -= 1
            counts, bin_edges = np.histogram(observed, bins=n_bins)

    if bin_method == "uniform_count":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.asarray(np.percentile(observed, quantiles))
    return np.asarray(bin_edges, dtype=np.float64)


def level_budget(n_samples: int) -> int:
    """
    Most distinct values a factor may carry into a contingency table over this many observations.

    Discreteness is a fact about a factor's support; it is not a promise that the support
    is small. An integer factor can be discrete and still take thousands of values --
    pixel areas, epoch seconds, identifiers -- and a factor like that defeats anything read
    off a contingency table: the table has more cells than the sample can fill, so the
    counts are dominated by which cells happened to be hit. Binning a factor that overruns
    this budget trades its exact values for cells the sample can actually support.

    The budget is ``sqrt(n_samples)``, the square-root rule used for histogram bin counts,
    floored at 20 so that a small sample cannot push it below the cardinality of an
    ordinary categorical factor. Both are rules of thumb rather than derived quantities.

    Parameters
    ----------
    n_samples : int
        Observations the factor was measured over, at the factor's own level.

    Returns
    -------
    int
        Level budget for a sample of this size.

    Examples
    --------
    A handful of categories over a large sample is comfortably within budget:

    >>> level_budget(10000)
    100

    An integer measurement taking hundreds of values is not:

    >>> level_budget(5717)
    75

    The floor keeps ordinary categorical factors intact when the sample is small:

    >>> level_budget(100)
    20
    """
    return int(max(MIN_LEVEL_BUDGET, np.sqrt(n_samples)))


def bin_data(data: NDArray[Any], bin_method: str, max_bins: int | None = None) -> NDArray[np.intp]:
    """
    Bins continuous data through either equal width bins, equal amounts in each bin, or by clusters.

    Bin edges are placed using the finite values only. The -inf outer edge absorbs
    negative infinity into the first bin, but digitizing is right-open, so positive
    infinity falls past the +inf outer edge into a bin above the highest finite one.
    NaN is then given a bin of its own above that.

    Parameters
    ----------
    data : NDArray[Any]
        Values to bin, one per entity.
    bin_method : {"uniform_width", "uniform_count", "clusters"}
        How the edges are placed.
    max_bins : int or None, default None
        Most bins the observed values may be placed into, or None for no cap. Every method
        here chooses its own count from the data, and a count chosen that way can still
        exceed what the sample supports; clusters that overrun the cap fall back to
        uniform edges, since merging clusters would place edges where the data says there
        is a boundary. The bins above the finite range — one for positive infinity, one
        for NaN — are added on top of this cap, not counted against it.

    Returns
    -------
    NDArray[np.intp]
        Bin index per entry, with missing values in a bin of their own above the rest.
    """
    data = np.asarray(data)
    observed = _observed(data)
    if observed.size == 0:
        # Nothing observed to place edges between, so every entry is the missing bin.
        return np.zeros(data.shape, dtype=np.intp)

    if bin_method == "clusters":
        bin_edges = _bin_by_clusters(observed)
        if max_bins is not None and bin_edges.size - 1 > max_bins:
            bin_edges = _uniform_edges(observed, "uniform_width", max_bins)
    else:
        bin_edges = _uniform_edges(observed, bin_method, max_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    return _digitize_with_missing(data, bin_edges)


def _gcd_ratio(data: NDArray[np.number[Any]], tol: float = 1e-9) -> float:
    """
    Measure how lattice-like the gaps between unique sorted values are.

    For discrete data on an integer or regular grid, the gaps between unique values are
    near-integer multiples of some base unit (the grid spacing). For continuous data, the
    gaps have no such structure. We estimate the smallest positive gap, then check what
    fraction of all gaps are near-integer multiples of it.

    Returns the fraction of gaps that are near-integer multiples of the smallest gap.
    Discrete-on-lattice data scores close to 1.0; continuous data scores around 0.1.
    """
    xu = np.sort(np.unique(data))
    if xu.size < 3:
        return 0.0

    gaps = np.diff(xu)
    positive = gaps[gaps > tol]
    if positive.size == 0:
        return 0.0

    min_gap = np.min(positive)
    ratios = positive / min_gap
    near_integer = np.abs(ratios - np.round(ratios)) < 0.05

    return float(np.mean(near_integer))


def _collapse_replicated(data: NDArray[np.number[Any]], groups: NDArray[Any]) -> NDArray[np.number[Any]]:
    """
    Reduce data to one entry per group, but only where the values are replicated.

    A value observed once and recorded once per group member carries no information in the
    repeats: they are one observation counted many times, and counting them multiplies the
    exact duplicates that :func:`is_continuous` reads as evidence of discreteness.

    Where the values vary within a group they are not replicates, so the sample is returned
    unchanged rather than silently thinned to one entry per group. The check is exact: a
    single entry out of place anywhere in the array takes the whole sample down the
    ungrouped path, which is logged because the two paths can reach opposite verdicts.
    """
    _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    representative = data[first]
    reconstructed = representative[inverse.reshape(-1)]
    if np.array_equal(reconstructed, data, equal_nan=np.issubdtype(data.dtype, np.inexact)):
        return representative

    _logger.debug(
        "Values are not constant within every group (%d groups over %d entries); "
        "scoring all entries rather than one per group.",
        first.size,
        data.shape[0],
    )
    return data


def is_continuous(data: NDArray[np.number[Any]], groups: NDArray[Any] | None = None) -> bool:  # noqa: C901
    """
    Determine whether a 1D sample was drawn from a continuous or a discrete distribution.

    The support of a distribution is not recoverable from a finite sample, so this is a
    heuristic decision rather than a hypothesis test: it reports no p-value and has no
    stated error rate. Three signals vote — the uniformity of the normalized near-neighbor
    distribution, the fraction of exact duplicates, and whether the distinct values lie on
    a lattice. See Notes for the construction and the decision rule.

    Parameters
    ----------
    data : NDArray[np.number]
        1D array of numeric values, one entry per observation. Non-finite entries are
        dropped before the test, since a missing value has no position on the line and an
        infinity no finite distance to its neighbors, so neither says anything about
        spacing. These are the same entries bin edges are placed on.
    groups : NDArray or None, default None
        Group identifier per entry of ``data``, of any comparable dtype. Entries sharing
        an identifier are repeated records of one underlying observation. Where the value
        is constant within every group it is counted once per group rather than once per
        entry, since replication is not evidence about the distribution — it multiplies
        exact duplicates and biases every signal below toward discreteness. Where the
        value varies within a group the sample is left at full length, so grouping cannot
        silently discard genuine variation. Defaults to no grouping, counting every entry.

    Returns
    -------
    bool
        True when the sample looks continuous, False when it looks discrete. Always False
        for fewer than 20 observations or fewer than 3 distinct values, neither of which
        supports the near-neighbor construction. Both counts are taken after grouping and
        after dropping non-finite entries, so either can put a sample under the
        20-observation floor.

    Raises
    ------
    ShapeMismatchError
        If ``groups`` is supplied and its length does not match ``data``.

    See Also
    --------
    scipy.stats.wasserstein_distance : Measures the departure of the NNN from uniform.

    Notes
    -----
    Consider the intervals between adjacent points of a 1D sample. Under a continuous
    distribution a point is equally likely to lie anywhere in the interval bounded by its
    two neighbors, and all such "between neighbor" locations can be put on a common 0 to 1
    scale by subtracting the smaller neighbor and dividing out the length of the interval.
    (Duplicates are either assigned to zero or ignored, depending on context.) These
    normalized locations are far more uniformly distributed for continuous data than for
    discrete, which is what makes them separable. Call this the Normalized Near Neighbor
    distribution (NNN), defined on [0, 1]. ``scipy.stats.wasserstein_distance`` then
    measures how far the NNN sits from uniform.

    Three signals are combined to make the decision:

    1. Adaptive WD threshold: the Wasserstein distance of the NNN from uniform shrinks as
       O(1/sqrt(n)) for truly continuous data. We use 0.5 / sqrt(n) as the primary
       threshold, which is equivalent to the previously used fixed threshold of 0.054 at
       n = 86.

    2. Duplicate fraction: truly continuous data drawn from a floating-point representation
       has probability zero of producing exact duplicates, so repeats point to discrete
       support and catch distributions with large support that would otherwise produce
       uniform-looking NNN values. The repeats have to account for a tenth of the sample
       before they count. Below that they are as consistent with a quantity recorded on a
       fine grid -- pixel counts, epoch seconds -- which collides now and then by luck
       while taking nearly as many distinct values as there are observations; treating
       those collisions as discrete support would send every integer-recorded measurement
       down the discrete path however finely it resolves.

    3. GCD lattice test: discrete data on an integer or regular grid has gaps between
       unique values that are near-integer multiples of a base unit. Continuous data does
       not. This catches lattice-structured discrete distributions (Poisson, Binomial,
       integer-valued) even before enough collisions accumulate for the duplicate test to
       trigger.

    A sample is called continuous when the Wasserstein distance clears the primary
    threshold and neither secondary signal fires. When exactly one secondary signal fires,
    the stricter 0.3 / sqrt(n) threshold decides; when both fire, the sample is called
    discrete regardless. All five constants are tuned values carrying no derivation, which
    is why the verdict is worth checking on data whose support is already known.

    Grouping addresses a specific failure of that construction: a value observed once but
    recorded k times appears as k - 1 exact duplicates. Those duplicates are an artifact of
    how the data was tabulated, not of the distribution, and they push all three signals
    toward discreteness — the duplicate fraction directly, the lattice test through the
    zero gaps, and the NNN through the zeros assigned to tied neighbors. Collapsing each
    group to a single entry restores the sample the question is actually about.

    Examples
    --------
    >>> rng = np.random.default_rng(0)

    A sample from a normal distribution is continuous:

    >>> is_continuous(rng.normal(size=200))
    True

    Integer counts are discrete, even spread over many distinct values:

    >>> is_continuous(rng.integers(0, 100, size=200))
    False

    So is a continuous quantity rounded to one decimal place — rounding puts the values on
    a lattice and introduces duplicates, and both secondary signals fire:

    >>> is_continuous(np.round(rng.normal(size=200), 1))
    False

    Below 20 samples the answer is always discrete:

    >>> is_continuous(rng.normal(size=15))
    False

    A continuous sample recorded three times per observation reads as discrete, because two
    thirds of the entries are exact duplicates:

    >>> observed = rng.normal(size=40)
    >>> groups = np.repeat(np.arange(40), 3)
    >>> recorded = observed[groups]
    >>> is_continuous(recorded)
    False

    Identifying the repeats scores the 40 distinct values instead:

    >>> is_continuous(recorded, groups=groups)
    True
    """
    data = np.asarray(data)

    # Count a value replicated across a group's rows once, not once per row
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        if groups.size != data.shape[0]:
            raise ShapeMismatchError(f"groups length {groups.size} does not match data length {data.shape[0]}.")
        data = _collapse_replicated(data, groups)

    # A missing value has no position on the line and an infinity has no finite distance
    # to its neighbors, so neither says anything about spacing. Left in, an infinity at
    # the low end makes both terms of the near-neighbor quotient infinite and the whole
    # Wasserstein statistic NaN, at which point the threshold comparison fails open and
    # the primary signal stops rejecting anything; one at the high end drives its windows
    # to a fabricated zero. `_observed` is what bin edges are placed on too, so the
    # verdict and the edges are read off the same values.
    data = _observed(data)

    n_examples = len(data)

    if n_examples < CONTINUOUS_MIN_SAMPLE_SIZE:
        _logger.warning(
            f"All samples look discrete with so few data points "
            f"({n_examples} < {CONTINUOUS_MIN_SAMPLE_SIZE}) — note this count is taken after grouping.",
        )
        return False

    # Require at least 3 unique values before bothering with NNN
    xu = np.unique(data)
    if xu.size < 3:
        return False

    xs = np.sort(data)

    x0, x1 = xs[0:-2], xs[2:]  # left and right neighbors

    dx = np.zeros(n_examples - 2)  # no dx at end points
    gtz = (x1 - x0) > 0  # check for dups; dx will be zero for them
    dx[np.logical_not(gtz)] = 0.0

    dx[gtz] = (xs[1:-1] - x0)[gtz] / (x1 - x0)[gtz]  # the core idea: dx is NNN samples.

    shift = wasserstein_distance(dx, np.linspace(0, 1, dx.size))  # how far is dx from uniform, for this feature?

    # Adaptive threshold: continuous WD shrinks as O(1/sqrt(n))
    wd_thresh = 0.5 / np.sqrt(n_examples)

    if shift >= wd_thresh:
        return False  # NNN is too far from uniform, even accounting for sample size

    # WD says continuous. Check for contradicting evidence from duplicates and lattice structure.
    #
    # Ties are evidence of discrete support only when they are structural rather than
    # incidental. A quantity recorded on a grid finer than its spread -- pixel areas,
    # epoch seconds, prices in cents -- collides occasionally by luck, and reading those
    # few collisions as discrete support forces every integer-recorded measurement down
    # the unbinned path no matter how many distinct values it takes. Requiring repeats to
    # account for a tenth of the sample separates the two: measurements of that kind sit
    # near 0.05 and below, while a genuinely small support puts the fraction near 1.
    dup_frac = 1.0 - xu.size / n_examples
    has_dups = dup_frac > DUPLICATE_SUPPORT_FRACTION

    on_lattice = _gcd_ratio(data) > 0.85

    if has_dups and on_lattice:
        return False  # two independent signals say discrete

    # If only one secondary signal fires, use a stricter WD threshold as tiebreaker
    if has_dups or on_lattice:
        strict_thresh = 0.3 / np.sqrt(n_examples)
        return bool(shift < strict_thresh)

    return True  # low WD, no dups, no lattice → continuous


def _bin_by_clusters(data: NDArray[np.number[Any]]) -> NDArray[np.float64]:  # noqa: C901
    """
    Bin continuous data by using the Clusterer to identify clusters.

    Incorporates outliers by adding them to the nearest bin.
    """
    # Delay load numba compiled functions
    from dataeval.core._clusterer import cluster

    # Create initial clusters
    c = cluster(data)

    # Create bins from clusters
    bin_edges = np.zeros(c["clusters"].max() + 2)
    for group in range(c["clusters"].max() + 1):
        points = np.nonzero(c["clusters"] == group)[0]
        bin_edges[group] = data[points].min()

    # Get the outliers
    outliers = np.nonzero(c["clusters"] == -1)[0]

    # Identify non-outlier neighbors
    nbrs = c["k_neighbors"][outliers]
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
                clusters = c["clusters"][nnbr]
                bin_edges[clusters] = min2add
    if extend_bins:
        bin_edges = np.concatenate([bin_edges, extend_bins])

    return np.sort(bin_edges)

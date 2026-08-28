"""Temporal structure in an ordered run of hashes.

A video's frames are a time series, not a bag. Consecutive frames of a static camera, a stalled
feed or a slow pan differ by a handful of bits, so a sequence can hold hundreds of frames and only
a few frames' worth of information. That is a form of redundancy still imagery does not have, and
it is the one that most distorts what a dataset appears to contain: every per-frame statistic over
such a sequence is weighted by dwell time rather than by content.

:func:`redundant_runs` measures it in a single linear pass.
"""

__all__ = []

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.core._hash_index import _popcount, _prepare

_logger = get_logger(__name__)


class RedundantRunResult(TypedDict):
    """
    Stretches of an ordered hash sequence whose neighbours carry no new information.

    Attributes
    ----------
    start : NDArray[np.intp]
        First position of each run, inclusive.
    end : NDArray[np.intp]
        Last position of each run, inclusive.
    length : NDArray[np.intp]
        Positions in each run, ``end - start + 1``.
    mean_distance : NDArray[np.float64]
        Mean Hamming distance between consecutive members of each run.
    redundant_fraction : float
        Share of the input that could be dropped without losing content: every run's members
        except one representative each, over the number of positions considered. ``0.0`` when
        nothing is redundant, approaching ``1.0`` for a sequence that never changes.
    """

    start: NDArray[np.intp]
    end: NDArray[np.intp]
    length: NDArray[np.intp]
    mean_distance: NDArray[np.float64]
    redundant_fraction: float


def _empty_runs() -> RedundantRunResult:
    """Return the well-typed answer for an input with nothing to report."""
    return RedundantRunResult(
        start=np.empty(0, dtype=np.intp),
        end=np.empty(0, dtype=np.intp),
        length=np.empty(0, dtype=np.intp),
        mean_distance=np.empty(0, dtype=np.float64),
        redundant_fraction=0.0,
    )


def _checked(
    codes: NDArray[np.uint64],
    radius: int,
    valid: NDArray[np.bool_] | None,
    min_length: int,
) -> tuple[NDArray[np.uint64], NDArray[np.bool_] | None]:
    """Validate the arguments and return them in the form the walk expects."""
    if min_length < 2:
        raise ValueError(f"redundant_runs: min_length must be at least 2; got {min_length}.")
    _, codes, _ = _prepare(codes, radius, None, None, "redundant_runs")
    if valid is None:
        return codes, None
    valid = np.asarray(valid, dtype=np.bool_)
    if valid.ndim != 1 or len(valid) != len(codes):
        raise ValueError(
            f"redundant_runs: valid must be a 1D array of length {len(codes)}; got shape {valid.shape}. "
            "It indexes the same hashes as codes and must agree."
        )
    return codes, valid


def redundant_runs(
    codes: NDArray[np.uint64],
    radius: int,
    *,
    valid: NDArray[np.bool_] | None = None,
    min_length: int = 2,
) -> RedundantRunResult:
    """
    Find stretches of an ordered hash sequence whose neighbours are within a Hamming radius.

    Parameters
    ----------
    codes : NDArray[np.uint64]
        Shape ``(N, W)`` packed hashes, as :func:`~dataeval.core.pack_hashes` returns, **in
        temporal order**. Order is the whole meaning of the result; an unordered input produces
        an answer about an ordering that does not exist.
    radius : int
        Maximum Hamming distance, in bits, for one position to be treated as carrying nothing new
        over the one before it. ``0`` links only identical hashes. Must not be negative.
    valid : NDArray[np.bool_] or None, default None
        Shape ``(N,)``. A position marked False belongs to no run and breaks any run spanning it:
        a hash that was never computed is not evidence that nothing changed.
    min_length : int, default 2
        Shortest run to report, in positions. Must be at least 2, since a run of one links
        nothing.

    Returns
    -------
    RedundantRunResult
        TypedDict containing:

        - **start**, **end** (*NDArray[np.intp]*) -- inclusive bounds of each run.
        - **length** (*NDArray[np.intp]*) -- positions per run.
        - **mean_distance** (*NDArray[np.float64]*) -- mean consecutive Hamming distance per run.
        - **redundant_fraction** (*float*) -- share of positions droppable without losing content.

    Raises
    ------
    ValueError
        If ``radius`` is negative, ``min_length`` is below 2, ``codes`` is not two-dimensional, or
        ``valid`` does not match ``codes``.

    See Also
    --------
    :func:`~dataeval.core.hash_groups` : Which hashes belong together, ignoring order
    :func:`~dataeval.core.pack_hashes` : Pack hex digests into the expected form

    Notes
    -----
    Each position is compared to its **predecessor**, not to a representative of the run so far.
    That is deliberate, and it is the difference between measuring redundancy and selecting key
    frames. A slow pan is one long run under a representative anchor and a series of short ones
    under a pairwise comparison, and the pairwise answer is the honest one: the content genuinely
    changes, just slowly. Anchoring on a representative is a *selection* rule, and belongs to a
    :class:`~dataeval.data.FrameSelector` -- see :class:`~dataeval.data.Redundancy`.

    A run is transitive by construction, so its first and last members may be far more than
    ``radius`` apart. That is the intended reading: nothing along the way carried new information.

    ``redundant_fraction`` counts only the runs actually reported, so raising ``min_length``
    lowers it.

    Examples
    --------
    >>> import numpy as np
    >>> from dataeval.core import pack_hashes, redundant_runs
    >>> codes, valid = pack_hashes(["ff00ff00ff00ff00", "ff00ff00ff00ff00", "ff00ff00ff00ff01", "0123456789abcdef"])
    >>> runs = redundant_runs(codes, radius=1, valid=valid)
    >>> runs["start"], runs["end"], runs["length"]
    (array([0]), array([2]), array([3]))
    >>> round(runs["redundant_fraction"], 3)
    0.5
    """
    codes, valid = _checked(codes, radius, valid, min_length)
    count = len(codes)
    if count < 2:
        return _empty_runs()

    distance = _popcount(codes[:-1], codes[1:])
    linked = distance <= radius
    if valid is not None:
        # A position with no hash links to nothing on either side, so a run cannot cross it.
        linked &= valid[:-1] & valid[1:]

    # A maximal span of `k` consecutive links is a run of `k + 1` positions.
    edges = np.diff(np.concatenate(([False], linked, [False])).astype(np.int8))
    start = np.flatnonzero(edges == 1).astype(np.intp)
    end = np.flatnonzero(edges == -1).astype(np.intp)
    length = end - start + 1

    keep = length >= min_length
    start, end, length = start[keep], end[keep], length[keep]
    if not len(start):
        return _empty_runs()

    # Mean over each run's own links, taken from a prefix sum so the runs are not walked.
    totals = np.concatenate(([0.0], np.cumsum(distance.astype(np.float64))))
    mean_distance = (totals[end] - totals[start]) / (end - start)

    fraction = float((length - 1).sum() / count)
    _logger.debug(
        "redundant_runs: %d run(s) over %d position(s) at radius %d, %.1f%% redundant",
        len(start),
        count,
        radius,
        fraction * 100,
    )
    return RedundantRunResult(
        start=start,
        end=end,
        length=length,
        mean_distance=mean_distance,
        redundant_fraction=fraction,
    )

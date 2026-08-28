"""Aligning one run of frames against another when the two do not advance in step.

:func:`~dataeval.core.match_segments` finds where two sequences run together by looking for a
*constant* offset — frame ``i`` of one matching frame ``i + k`` of the other, for a fixed ``k``.
That is the right model for a cut: an excerpt lifted out of a longer video keeps the source's
timing exactly, so its matches fall on one diagonal and the arithmetic is cheap.

It is the wrong model for a **speed edit**. Play a clip back at 1.5x and the offset grows by half
a frame every frame, so a shared minute arrives as a scatter of two- and three-frame fragments,
every one of them too short to report. The same happens to a frame-rate conversion, to a video
padded with a title card partway through, and to any edit that stretches time rather than moving
it.

Dynamic time warping absorbs exactly that. It aligns the two runs by *elastic* correspondence
rather than a fixed offset, so a stretch matches a stretch however unevenly the two advance
through it, and reports what the alignment cost. Being quadratic, it is a verifier rather than a
search: it runs on the pairs an earlier tier already nominated.
"""

__all__ = []

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._log import get_logger
from dataeval.core._hash_index import _popcount

_logger = get_logger(__name__)

#: Cells of the accumulated-cost matrix an unbanded alignment may build. Two 4 000-frame runs sit
#: just inside it. Past this the answer is not worth the wait, and ``band`` is the way through.
_MAX_CELLS = 16_000_000

_METRICS = ("hamming", "cosine", "euclidean")


class SubsequenceAlignment(TypedDict):
    """
    Where a query run best aligns inside a candidate run, and how well.

    Attributes
    ----------
    start : int
        Candidate index the aligned window begins at, inclusive.
    end : int
        Candidate index the aligned window ends at, inclusive.
    cost : float
        Total distance accumulated along the warping path.
    normalized_cost : float
        ``cost / path_length`` -- the mean distance per aligned step, and the only one of the two
        figures comparable between alignments. A longer alignment accumulates more cost simply by
        being longer.
    path_length : int
        Steps on the warping path. At least ``len(query)``, and larger wherever the path warped.
    """

    start: int
    end: int
    cost: float
    normalized_cost: float
    path_length: int


def _checked_policy(metric: str, band: int | None) -> None:
    """Reject an unknown metric or a negative band."""
    if metric not in _METRICS:
        raise ValueError(f"align_subsequence: metric must be one of {_METRICS}; got {metric!r}.")
    if band is not None and band < 0:
        raise ValueError(f"align_subsequence: band must be non-negative; got {band}.")


def _as_frames(run: ArrayLike, name: str) -> NDArray:
    """Return one run as a 2-D ``(frames, width)`` array, treating a bare 1-D run as width 1."""
    array = np.asarray(run)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"align_subsequence: {name} must be 1- or 2-D; got shape {array.shape}.")
    if not len(array):
        raise ValueError(f"align_subsequence: {name} must hold at least one frame.")
    return array


def _checked_runs(
    query: ArrayLike,
    candidate: ArrayLike,
    metric: str,
    band: int | None,
) -> tuple[NDArray, NDArray]:
    """Validate the two runs and return them as 2-D arrays in the dtype the metric expects."""
    _checked_policy(metric, band)
    left, right = _as_frames(query, "query"), _as_frames(candidate, "candidate")
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"align_subsequence: query frames are {left.shape[1]} wide and candidate frames "
            f"{right.shape[1]}; they must describe frames the same way."
        )
    if metric != "hamming":
        return left.astype(np.float64), right.astype(np.float64)
    if left.dtype != np.uint64 or right.dtype != np.uint64:
        raise ValueError(
            "align_subsequence: metric='hamming' consumes packed uint64 codes, as "
            f"dataeval.core.pack_hashes returns; got dtypes {left.dtype} and {right.dtype}. "
            "Use 'cosine' or 'euclidean' for descriptors."
        )
    return left, right


def _window(row: int, offset: int, band: int | None, n_candidate: int) -> tuple[int, int]:
    """Return the half-open candidate range row ``row`` may align to."""
    if band is None:
        return 0, n_candidate
    return row + offset - band, row + offset + band + 1


def _distances(frame: NDArray, candidate: NDArray, lo: int, hi: int, metric: str) -> NDArray[np.float64]:
    """Distance from one query frame to every candidate frame in ``[lo, hi)``, inf outside it."""
    costs = np.full(hi - lo, np.inf, dtype=np.float64)
    begin, end = max(lo, 0), min(hi, len(candidate))
    if begin >= end:
        return costs
    window = candidate[begin:end]
    if metric == "hamming":
        found = _popcount(np.broadcast_to(frame, window.shape), window).astype(np.float64)
    elif metric == "euclidean":
        found = np.linalg.norm(window - frame, axis=1)
    else:
        scale = np.linalg.norm(window, axis=1) * np.linalg.norm(frame)
        # A zero vector has no direction, so no angle to any other. Called maximally distant
        # rather than zero-distance, which is what dividing by the zero norm would imply.
        found = np.where(scale > 0, 1.0 - (window @ frame) / np.where(scale > 0, scale, 1.0), 1.0)
    costs[begin - lo : end - lo] = found
    return costs


def _shifted(row: NDArray[np.float64], by: int) -> NDArray[np.float64]:
    """Return ``result[c] = row[c + by]``, infinite where that falls outside the row."""
    if by == 0:
        return row
    out = np.full_like(row, np.inf)
    if by > 0:
        out[:-by] = row[by:]
    else:
        out[-by:] = row[:by]
    return out


def _accumulate(base: NDArray[np.float64], costs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Close a row under horizontal steps: ``D[c] = min(base[c], D[c - 1] + costs[c])``.

    Written as a scan rather than a loop. A path entering the row at column ``k`` and stepping
    horizontally to ``c`` costs ``base[k] + (S[c] - S[k])`` for the running total ``S``, so the
    best entry point is the running minimum of ``base - S`` and the whole row closes in two
    vectorized passes instead of one Python step per column.
    """
    finite = np.where(np.isfinite(costs), costs, 0.0)
    totals = np.cumsum(finite)
    # A cell the window excludes cannot be stepped through, so the chain restarts past it.
    reachable = np.where(np.isfinite(costs), base - totals, np.inf)
    return np.minimum(base, totals + np.minimum.accumulate(reachable))


def _origin(
    accumulated: list[NDArray[np.float64]],
    costs: list[NDArray[np.float64]],
    end: int,
    offset: int,
    band: int | None,
    n_candidate: int,
) -> tuple[int, int]:
    """Walk the best path back from the last row, returning where it started and how long it ran.

    Ties resolve toward the diagonal, then the vertical: among paths of equal cost that is the one
    that warped least, which is the honest reading of an ambiguous alignment.
    """
    row, column, length = len(accumulated) - 1, end, 1
    while row > 0:
        lo, _ = _window(row, offset, band, n_candidate)
        previous, delta = accumulated[row - 1], lo - _window(row - 1, offset, band, n_candidate)[0]
        target = accumulated[row][column] - costs[row][column]
        # Each move names the predecessor's cost and the cell it sits in: back a row and along
        # the diagonal, back a row straight up, or one column left in this row. `delta` re-bases
        # the column, because a banded row starts one candidate frame later than the one above it.
        moves = (
            (previous[column + delta - 1] if 0 <= column + delta - 1 < len(previous) else np.inf, -1, delta - 1),
            (previous[column + delta] if 0 <= column + delta < len(previous) else np.inf, -1, delta),
            (accumulated[row][column - 1] if column > 0 else np.inf, 0, -1),
        )
        for value, up, across in moves:
            if np.isclose(value, target):
                row, column, length = row + up, column + across, length + 1
                break
        else:  # pragma: no cover - the forward pass guarantees one of the three reached this cell
            break
    return _window(row, offset, band, n_candidate)[0] + column, length


def align_subsequence(
    query: ArrayLike,
    candidate: ArrayLike,
    *,
    metric: Literal["hamming", "cosine", "euclidean"] = "hamming",
    band: int | None = None,
    offset: int = 0,
    max_cells: int = _MAX_CELLS,
) -> SubsequenceAlignment:
    """
    Find where a run of frames best aligns inside a longer run, allowing for uneven timing.

    Subsequence dynamic time warping: the whole of ``query`` is aligned against the *best window*
    of ``candidate``, with no penalty for where that window starts or ends. The alignment is
    elastic, so one query frame may match several candidate frames and the reverse, which is what
    lets it follow a speed edit or a frame-rate conversion that a fixed offset cannot.

    Parameters
    ----------
    query : ArrayLike
        Shape ``(N,)`` or ``(N, W)``. The shorter run, aligned in its entirety. For
        ``metric="hamming"`` these are packed ``uint64`` codes as
        :func:`~dataeval.core.pack_hashes` returns; otherwise a float descriptor per frame.
    candidate : ArrayLike
        Shape ``(M,)`` or ``(M, W)``, described the same way as ``query``. The window is chosen
        from this run.
    metric : {"hamming", "cosine", "euclidean"}, default "hamming"
        How two frames are compared. ``"hamming"`` counts differing bits between packed hashes;
        the other two consume float descriptors, so an embedding aligns through the same call.
    band : int or None, default None
        Sakoe-Chiba constraint. ``None`` considers every window. An integer confines the path to
        within ``band`` frames of the diagonal through ``offset``, which both bounds the work to
        ``O(N * band)`` and rejects alignments that warp further than that -- usually the point,
        since an unconstrained warp will align nearly anything to nearly anything.
    offset : int, default 0
        Diagonal the band is centred on: query frame ``i`` sits opposite candidate frame
        ``i + offset``. :func:`~dataeval.core.match_segments` reports exactly this quantity, so a
        segment it found is verified by passing its ``offset`` through. Ignored when ``band`` is
        None.
    max_cells : int, default 16000000
        Refuse rather than build an accumulated-cost matrix larger than this. Guards the
        unbanded case, where the matrix is the product of the two lengths.

    Returns
    -------
    SubsequenceAlignment
        The aligned window and its cost. Compare ``normalized_cost``, not ``cost``.

    Raises
    ------
    ValueError
        If either run is empty, the two describe frames differently, ``metric`` is unknown,
        ``band`` is negative, ``metric="hamming"`` is given anything but packed ``uint64`` codes,
        or the matrix would exceed ``max_cells``.

    See Also
    --------
    :func:`~dataeval.core.match_segments` : Finds shared stretches at a fixed offset, far cheaper
    :func:`~dataeval.core.pack_hashes` : Produces the codes ``metric="hamming"`` consumes

    Notes
    -----
    **This is a verifier, not a search.** Cost is ``O(N * M)`` unbanded, against the near-linear
    grouping :func:`~dataeval.core.hash_groups` performs, so it belongs at the end of a cascade
    where an earlier tier has already reduced the corpus to a short list of candidate pairs.

    **An unconstrained warp aligns anything.** Given no ``band``, dynamic time warping will happily
    match one frame of the query against nine hundred of the candidate to save a few bits, and
    report a low cost for two videos with nothing to do with each other. ``band`` is what makes
    the result mean something, and ``normalized_cost`` -- which divides by a path length that
    warping inflates -- is what makes two alignments comparable.

    ``normalized_cost`` under ``metric="hamming"`` reads as the mean number of differing bits per
    aligned frame pair, on the same scale as the ``radius`` arguments elsewhere in this module.

    Examples
    --------
    A ten-frame clip, located inside a longer run that holds a copy of it:

    >>> import numpy as np
    >>> from dataeval.core import align_subsequence
    >>> clip = np.arange(10, dtype=np.float64)
    >>> source = np.concatenate([np.full(20, -1.0), clip, np.full(20, -1.0)])
    >>> found = align_subsequence(clip, source, metric="euclidean")
    >>> found["start"], found["end"], found["cost"]
    (20, 29, 0.0)

    The same clip where the copy plays back at half speed. A fixed offset finds nothing here,
    while the warping path absorbs the stretch at no cost and records it in its length:

    >>> slowed = np.concatenate([np.full(20, -1.0), np.repeat(clip, 2), np.full(20, -1.0)])
    >>> found = align_subsequence(clip, slowed, metric="euclidean")
    >>> found["cost"], found["path_length"]
    (0.0, 18)

    Against a run that does not hold the clip at all, where the cost says so:

    >>> unrelated = np.full(40, 99.0)
    >>> round(align_subsequence(clip, unrelated, metric="euclidean")["normalized_cost"], 1)
    94.5
    """
    left, right = _checked_runs(query, candidate, metric, band)
    n_query, n_candidate = len(left), len(right)
    width = 2 * band + 1 if band is not None else n_candidate
    if n_query * width > max_cells:
        raise ValueError(
            f"align_subsequence: aligning {n_query} against {n_candidate} frame(s) builds "
            f"{n_query * width} cells, past the {max_cells} allowed. Pass a band to bound the "
            "work, or align shorter runs."
        )

    costs: list[NDArray[np.float64]] = []
    accumulated: list[NDArray[np.float64]] = []
    for row in range(n_query):
        lo, hi = _window(row, offset, band, n_candidate)
        costs.append(_distances(left[row], right, lo, hi, metric))
        if row == 0:
            # The first row is free: the query may begin against any candidate frame, which is
            # what makes this a subsequence alignment rather than a whole-run one.
            accumulated.append(costs[0].copy())
            continue
        delta = lo - _window(row - 1, offset, band, n_candidate)[0]
        previous = accumulated[row - 1]
        base = costs[row] + np.minimum(_shifted(previous, delta), _shifted(previous, delta - 1))
        accumulated.append(_accumulate(base, costs[row]))

    last = accumulated[-1]
    if not np.isfinite(last).any():
        raise ValueError(
            f"align_subsequence: no path reaches the end of a {n_query} frame query within "
            f"band {band} of offset {offset} over {n_candidate} candidate frame(s)."
        )
    column = int(np.argmin(last))
    cost = float(last[column])
    end = _window(n_query - 1, offset, band, n_candidate)[0] + column
    start, path_length = _origin(accumulated, costs, column, offset, band, n_candidate)
    _logger.debug(
        "align_subsequence: %d against %d frame(s) -> [%d, %d] at %.3f per step",
        n_query,
        n_candidate,
        start,
        end,
        cost / path_length,
    )
    return SubsequenceAlignment(
        start=start,
        end=end,
        cost=cost,
        normalized_cost=cost / path_length,
        path_length=path_length,
    )

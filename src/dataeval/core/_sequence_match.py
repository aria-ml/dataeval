"""Matching one ordered run of hashes against another, in whole or in part.

Two videos rarely relate as *the same* or *different*. A clip is cut from a longer source; two
uploads share a middle section; one is a re-encode of the other end to end. Those are different
relations with different consequences — a clip of a training video appearing in a test split is
leakage, while two encodes of one collect are ordinary redundancy — and telling them apart needs
more than a similarity score.

The relation is also **directed**: a fifteen-second excerpt is entirely contained in its two-hour
source, while the source is barely contained in the excerpt. That asymmetry *is* the signal, and
it is what a symmetric group cannot express.

:func:`match_segments` finds where two sequences run together; :func:`sequence_containment` says
how much of each the other accounts for.
"""

__all__ = []

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.core._hash_index import pack_hashes

_logger = get_logger(__name__)


class SequenceFingerprint(TypedDict):
    r"""
    A sequence reduced to what is needed to match it against another.

    Attributes
    ----------
    exact : str
        Digest of the ordered frame hashes. Two sequences sharing it hold the same frames in the
        same order, which is the cheapest screen there is. Order-sensitive on purpose: the same
        frames in a different order are not the same video, and the segment matcher finds whatever
        they do share anyway.
    codes : NDArray[np.uint64]
        Shape ``(F, W)`` packed frame hashes, in order.
    valid : NDArray[np.bool\_]
        Shape ``(F,)``. False where a frame's hash was never computed.
    frame_indices : NDArray[np.intp]
        Shape ``(F,)``. Each row's frame index in the source video, which need not be its position
        here once frames have been sampled.
    times : NDArray[np.float64] or None
        Shape ``(F,)`` seconds per frame, or None unless *every* frame declared one. A partially
        populated timing is worse than none: every duration derived from it is silently wrong for
        the frames that lack it.
    """

    exact: str
    codes: NDArray[np.uint64]
    valid: NDArray[np.bool_]
    frame_indices: NDArray[np.intp]
    times: NDArray[np.float64] | None


class SegmentMatchResult(TypedDict):
    """
    Stretches over which two sequences run together.

    Every field is indexed by segment. Positions are into each sequence's own frames.

    Attributes
    ----------
    query_start, query_end : NDArray[np.intp]
        Inclusive bounds of the matched stretch in the query sequence.
    candidate_start, candidate_end : NDArray[np.intp]
        Inclusive bounds of the same stretch in the candidate sequence.
    offset : NDArray[np.intp]
        ``candidate_start - query_start``: how far the candidate lags the query. Constant within a
        segment, which is what a cut, a trim or an insertion produces.
    n_matched : NDArray[np.intp]
        Matched frame pairs inside the segment.
    mean_distance : NDArray[np.float64]
        Mean Hamming distance over those pairs. Near zero for a re-encode.
    density : NDArray[np.float64]
        ``n_matched`` over the segment's length in query frames. 1.0 when every frame matched;
        lower where frames were dropped, hashed differently, or bridged by ``max_gap``.
    """

    query_start: NDArray[np.intp]
    query_end: NDArray[np.intp]
    candidate_start: NDArray[np.intp]
    candidate_end: NDArray[np.intp]
    offset: NDArray[np.intp]
    n_matched: NDArray[np.intp]
    mean_distance: NDArray[np.float64]
    density: NDArray[np.float64]


def _empty_segments() -> SegmentMatchResult:
    """Return the well-typed answer for a pair of sequences that share nothing."""
    empty_i, empty_f = np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64)
    return SegmentMatchResult(
        query_start=empty_i,
        query_end=empty_i.copy(),
        candidate_start=empty_i.copy(),
        candidate_end=empty_i.copy(),
        offset=empty_i.copy(),
        n_matched=empty_i.copy(),
        mean_distance=empty_f,
        density=empty_f.copy(),
    )


def sequence_fingerprint(
    frame_hashes: Sequence[str],
    *,
    frame_indices: Sequence[int] | None = None,
    times: Sequence[float | None] | None = None,
) -> SequenceFingerprint:
    """
    Reduce one sequence's frame hashes to the form the matchers consume.

    Parameters
    ----------
    frame_hashes : Sequence[str]
        Hex digests, one per frame, **in temporal order**. The empty string marks a frame whose
        hash was never computed.
    frame_indices : Sequence[int] or None, default None
        Each frame's index in the source video. Defaults to its position here, which is what they
        are when no frames were sampled out.
    times : Sequence[float | None] or None, default None
        Each frame's time in seconds. Carried only when every frame declares one.

    Returns
    -------
    SequenceFingerprint
        TypedDict containing **exact**, **codes**, **valid**, **frame_indices** and **times**.

    Raises
    ------
    ValueError
        If ``frame_indices`` or ``times`` does not match ``frame_hashes`` in length, or if the
        digests are not all the same length.

    See Also
    --------
    :func:`~dataeval.core.match_segments` : Find where two fingerprints run together
    :func:`~dataeval.core.sequence_containment` : How much of each sequence the other accounts for

    Examples
    --------
    >>> from dataeval.core import sequence_fingerprint
    >>> a = sequence_fingerprint(["ff00ff00ff00ff00", "0123456789abcdef"])
    >>> b = sequence_fingerprint(["ff00ff00ff00ff00", "0123456789abcdef"])
    >>> a["exact"] == b["exact"]
    True
    >>> reordered = sequence_fingerprint(["0123456789abcdef", "ff00ff00ff00ff00"])
    >>> a["exact"] == reordered["exact"]
    False
    """
    import xxhash as xxh

    count = len(frame_hashes)
    for name, supplied in (("frame_indices", frame_indices), ("times", times)):
        if supplied is not None and len(supplied) != count:
            raise ValueError(
                f"sequence_fingerprint: {name} has length {len(supplied)} but frame_hashes has "
                f"{count}; they describe the same frames and must agree."
            )

    codes, valid = pack_hashes(frame_hashes)
    indices = np.arange(count, dtype=np.intp) if frame_indices is None else np.asarray(frame_indices, dtype=np.intp)

    carried: NDArray[np.float64] | None = None
    if times is not None:
        missing = sum(value is None for value in times)
        if missing:
            _logger.info(
                "sequence_fingerprint: %d of %d frame(s) declare no time, so no timings are "
                "carried; a partially populated timing makes every derived duration wrong for "
                "the frames that lack it.",
                missing,
                count,
            )
        else:
            carried = np.asarray(times, dtype=np.float64)

    return SequenceFingerprint(
        exact=xxh.xxh3_64_hexdigest("\x00".join(frame_hashes).encode()),
        codes=codes,
        valid=valid,
        frame_indices=indices,
        times=carried,
    )


def _merge_close(segments: SegmentMatchResult, offset_tolerance: int, max_gap: int) -> SegmentMatchResult:
    """Join segments on near-enough diagonals whose query spans run into one another.

    A small frame-rate difference makes one shared stretch drift across neighbouring offsets
    rather than sitting on a single one, so it arrives here as several short segments. Joining
    them reports the stretch that is actually shared.

    Each candidate is compared against the offset of the piece most recently joined, not against
    the offset the chain started on, so a steady drift chains the whole way. That is deliberate:
    a rate conversion drifts without bound, and refusing to follow it would report the first few
    seconds of a shared hour and call the rest unrelated. The reported ``offset`` is recomputed
    from the joined bounds rather than kept from the first piece.
    """
    order = np.argsort(segments["query_start"], kind="stable")
    merged: list[dict[str, float]] = []
    tails: list[int] = []
    for index in order:
        current = {key: float(segments[key][index]) for key in segments}  # type: ignore[literal-required]
        previous = merged[-1] if merged else None
        joins = (
            previous is not None
            and abs(int(current["offset"]) - tails[-1]) <= offset_tolerance
            and int(current["query_start"]) - int(previous["query_end"]) <= max_gap + 1
        )
        if previous is None or not joins:
            merged.append(current)
            tails.append(int(current["offset"]))
            continue
        total = previous["n_matched"] + current["n_matched"]
        previous["mean_distance"] = (
            previous["mean_distance"] * previous["n_matched"] + current["mean_distance"] * current["n_matched"]
        ) / total
        previous["n_matched"] = total
        previous["query_end"] = max(previous["query_end"], current["query_end"])
        previous["candidate_start"] = min(previous["candidate_start"], current["candidate_start"])
        previous["candidate_end"] = max(previous["candidate_end"], current["candidate_end"])
        tails[-1] = int(current["offset"])

    fields = {key: np.array([row[key] for row in merged]) for key in segments}
    length = fields["query_end"] - fields["query_start"] + 1
    query_start = fields["query_start"].astype(np.intp)
    candidate_start = fields["candidate_start"].astype(np.intp)
    return SegmentMatchResult(
        query_start=query_start,
        query_end=fields["query_end"].astype(np.intp),
        candidate_start=candidate_start,
        candidate_end=fields["candidate_end"].astype(np.intp),
        offset=(candidate_start - query_start).astype(np.intp),
        n_matched=fields["n_matched"].astype(np.intp),
        mean_distance=fields["mean_distance"].astype(np.float64),
        density=(fields["n_matched"] / length).astype(np.float64),
    )


def _at_least(segments: SegmentMatchResult, min_length: int) -> SegmentMatchResult:
    """Keep only the stretches spanning ``min_length`` query frames or more."""
    keep = (segments["query_end"] - segments["query_start"] + 1) >= min_length
    if not keep.any():
        return _empty_segments()
    return SegmentMatchResult(
        query_start=segments["query_start"][keep],
        query_end=segments["query_end"][keep],
        candidate_start=segments["candidate_start"][keep],
        candidate_end=segments["candidate_end"][keep],
        offset=segments["offset"][keep],
        n_matched=segments["n_matched"][keep],
        mean_distance=segments["mean_distance"][keep],
        density=segments["density"][keep],
    )


def _checked_matches(
    pairs: NDArray[np.intp],
    distances: NDArray[np.intp],
    min_length: int,
    max_gap: int,
    offset_tolerance: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Validate the matcher's arguments and return the two arrays in the dtypes it expects."""
    if min_length < 1:
        raise ValueError(f"match_segments: min_length must be at least 1; got {min_length}.")
    if max_gap < 0 or offset_tolerance < 0:
        raise ValueError(
            f"match_segments: max_gap and offset_tolerance must be non-negative; got {max_gap} and {offset_tolerance}."
        )
    pairs = np.asarray(pairs, dtype=np.intp)
    if pairs.ndim != 2 or (pairs.size and pairs.shape[1] != 2):
        raise ValueError(f"match_segments: pairs must be an (M, 2) array; got shape {pairs.shape}.")
    distances = np.asarray(distances, dtype=np.intp)
    if len(distances) != len(pairs):
        raise ValueError(
            f"match_segments: distances has length {len(distances)} but pairs has {len(pairs)}; "
            "they describe the same matches and must agree."
        )
    return pairs, distances


def match_segments(
    pairs: NDArray[np.intp],
    distances: NDArray[np.intp],
    *,
    min_length: int,
    max_gap: int = 0,
    offset_tolerance: int = 0,
) -> SegmentMatchResult:
    """
    Find the stretches over which two sequences run together, from their matched frames.

    Parameters
    ----------
    pairs : NDArray[np.intp]
        Shape ``(M, 2)``. Each row is one matched frame pair: column 0 a position in the query
        sequence, column 1 a position in the candidate sequence. Order does not matter, and a
        frame may appear in several pairs.
    distances : NDArray[np.intp]
        Shape ``(M,)``. Hamming distance of each pair, aligned with ``pairs``.
    min_length : int
        Shortest stretch to report, in query frames, measured *after* ``offset_tolerance`` has
        joined what it can. Below roughly a second's worth, shared intros, title cards and stock
        footage dominate the answer.
    max_gap : int, default 0
        Query frames a stretch may skip and still count as continuous. Bridges a dropped frame or
        one the hash missed; too large a value bridges a cut.
    offset_tolerance : int, default 0
        How far two stretches may differ in offset and still be joined. ``0`` requires one
        constant offset, which is what a cut, trim or insertion produces. Raise it where the two
        sequences were sampled at slightly different rates.

    Returns
    -------
    SegmentMatchResult
        TypedDict of per-segment bounds, offset, match count, mean distance and density.

    Raises
    ------
    ValueError
        If ``pairs`` is not ``(M, 2)``, ``distances`` does not match it in length, ``min_length``
        is below 1, or ``max_gap`` or ``offset_tolerance`` is negative.

    See Also
    --------
    :func:`~dataeval.core.sequence_containment` : How much of each sequence the other accounts for
    :func:`~dataeval.core.hash_neighbors` : Produce the matched pairs this consumes

    Notes
    -----
    **Diagonal offset voting.** Each matched pair votes for the offset ``candidate - query``. A
    genuinely shared stretch is a dense run of pairs on one offset; unrelated content scatters
    across offsets and forms no run. Grouping by offset and extracting runs of consecutive query
    positions therefore finds shared stretches directly, in time proportional to the *matched*
    pairs rather than to the frames.

    That is the same diagonal a trained similarity network learns to spot in a frame-to-frame
    similarity tensor, found by arithmetic instead.

    A constant offset covers cuts, trims, insertions and re-ordered compilations. It does not
    cover a speed change or a frame-rate conversion, where the offset drifts:
    ``offset_tolerance`` absorbs a slight drift, and anything more needs time-warped alignment.

    Examples
    --------
    A ten-frame excerpt of a longer video, starting at its frame 20:

    >>> import numpy as np
    >>> from dataeval.core import match_segments
    >>> pairs = np.array([[i, i + 20] for i in range(10)])
    >>> segments = match_segments(pairs, np.zeros(10, dtype=np.intp), min_length=5)
    >>> segments["query_start"], segments["query_end"], segments["offset"]
    (array([0]), array([9]), array([20]))
    """
    pairs, distances = _checked_matches(pairs, distances, min_length, max_gap, offset_tolerance)
    if not len(pairs):
        return _empty_segments()

    query, candidate = pairs[:, 0], pairs[:, 1]
    offset = candidate - query
    # Sorted by diagonal, then along it, so a run is a contiguous slice.
    order = np.lexsort((query, offset))
    query, offset = query[order], offset[order]
    ordered_distance = distances[order].astype(np.float64)

    opens = np.empty(len(order), dtype=np.bool_)
    opens[0] = True
    opens[1:] = (offset[1:] != offset[:-1]) | ((query[1:] - query[:-1]) > max_gap + 1)
    starts = np.flatnonzero(opens).astype(np.intp)
    ends = np.append(starts[1:], len(order)) - 1

    query_start, query_end = query[starts], query[ends]
    diagonal, length = offset[starts], query_end - query_start + 1
    matched = (ends - starts + 1).astype(np.intp)
    totals = np.concatenate(([0.0], np.cumsum(ordered_distance)))

    segments = SegmentMatchResult(
        query_start=query_start.astype(np.intp),
        query_end=query_end.astype(np.intp),
        candidate_start=(query_start + diagonal).astype(np.intp),
        candidate_end=(query_end + diagonal).astype(np.intp),
        offset=diagonal.astype(np.intp),
        n_matched=matched,
        mean_distance=(totals[ends + 1] - totals[starts]) / matched,
        density=(matched / length).astype(np.float64),
    )
    if offset_tolerance:
        # Joined before the length filter, never after. A drifted stretch arrives here as several
        # short pieces -- that is the whole reason `offset_tolerance` exists -- so filtering first
        # would discard exactly the pieces the join was going to make long enough, and the option
        # would do nothing on the inputs it was added for.
        segments = _merge_close(segments, offset_tolerance, max_gap)
    segments = _at_least(segments, min_length)
    _logger.debug("match_segments: %d segment(s) from %d matched pair(s)", len(segments["offset"]), len(pairs))
    return segments


def sequence_containment(pairs: NDArray[np.intp], n_query: int, n_candidate: int) -> tuple[float, float]:
    """
    Report how much of each sequence the other accounts for.

    Parameters
    ----------
    pairs : NDArray[np.intp]
        Shape ``(M, 2)`` matched frame pairs, as :func:`~dataeval.core.match_segments` consumes.
    n_query : int
        Frames in the query sequence. Must be positive.
    n_candidate : int
        Frames in the candidate sequence. Must be positive.

    Returns
    -------
    tuple[float, float]
        The share of the query's frames matched by the candidate, and the share of the
        candidate's matched by the query. Each in ``[0, 1]``.

    Raises
    ------
    ValueError
        If either count is not positive, or ``pairs`` is not ``(M, 2)``.

    See Also
    --------
    :func:`~dataeval.core.match_segments` : Where the two sequences run together

    Notes
    -----
    **The asymmetry is the signal.** Two high values mean each sequence is most of the other — a
    re-encode, a transcode, the same collect twice. One high and one low means the first is
    *contained in* the second: a clip cut from a longer source. That is the relation a symmetric
    similarity score cannot express and a transitive group erases, and it is the one that matters
    for train/test leakage, where a short test clip drawn from a long training video reads as
    nearly 1.0 one way and nearly 0.0 the other.

    Frames are counted once however many times they matched, so this measures coverage rather
    than the number of matches.

    Examples
    --------
    >>> import numpy as np
    >>> from dataeval.core import sequence_containment
    >>> excerpt = np.array([[i, i + 100] for i in range(10)])
    >>> query, candidate = sequence_containment(excerpt, n_query=10, n_candidate=1000)
    >>> query, round(candidate, 3)
    (1.0, 0.01)
    """
    if n_query < 1 or n_candidate < 1:
        raise ValueError(f"sequence_containment: frame counts must be positive; got {n_query} and {n_candidate}.")
    pairs = np.asarray(pairs, dtype=np.intp)
    if pairs.ndim != 2 or (pairs.size and pairs.shape[1] != 2):
        raise ValueError(f"sequence_containment: pairs must be an (M, 2) array; got shape {pairs.shape}.")
    if not len(pairs):
        return 0.0, 0.0
    return len(np.unique(pairs[:, 0])) / n_query, len(np.unique(pairs[:, 1])) / n_candidate

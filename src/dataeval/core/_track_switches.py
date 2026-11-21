"""Identity switches where two tracks meet, scored by whether their appearance fits better crossed than straight."""

__all__ = []

from typing import Any, NamedTuple, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._experimental import experimental
from dataeval.core._track_path_stats import _group, _read
from dataeval.core._track_steps import _centers, _diagonal
from dataeval.exceptions import ShapeMismatchError
from dataeval.types._target import whole_ids
from dataeval.utils._array import as_numpy

_GAP = 5  # meetings whose positions among the pair's shared frames differ by more than this are two encounters


class TrackSwitchesResult(TypedDict):
    """
    One row per encounter of two tracks, from :func:`.track_switches`.

    .. warning::
        Experimental: the columns and their meaning may change in a later release.

    Attributes
    ----------
    item_index : NDArray[np.intp]
        The video of the encounter, as ``sequence_indices`` gave it.
    label : NDArray[np.intp]
        The class both tracks share.
    track_a, track_b : NDArray[np.intp]
        The two track ids, ``track_a < track_b``.
    frame : NDArray[np.intp]
        The frame where the score is largest, or the first meeting where none could be scored.
    switch_score : NDArray[np.float64]
        How much better the two tracks' appearance fits crossed than straight at ``frame``; NaN where no meeting had
        enough rows on both sides.
    flagged : NDArray[bool]
        ``switch_score > 0``.
    row_a, row_b : NDArray[np.intp]
        The input row of each track at ``frame``.
    """

    item_index: NDArray[np.intp]
    label: NDArray[np.intp]
    track_a: NDArray[np.intp]
    track_b: NDArray[np.intp]
    frame: NDArray[np.intp]
    switch_score: NDArray[np.float64]
    flagged: NDArray[np.bool_]
    row_a: NDArray[np.intp]
    row_b: NDArray[np.intp]


def _check_window(window: Any) -> None:
    """Refuse a window that is not an integer of at least 2, naming the keyword."""
    if isinstance(window, bool) or not isinstance(window, int | np.integer) or window < 2:
        raise ValueError(f"window must be an integer of at least 2; got {window!r}. It counts rows on each side.")


def _read_boxes(boxes: ArrayLike, n: int) -> NDArray[np.float64]:
    """Return one finite box with positive width and height per row, refusing anything else."""
    box = as_numpy(boxes, dtype=np.float64)
    if box.shape != (n, 4):
        raise ShapeMismatchError(f"boxes must hold one x0, y0, x1, y1 box per row, shape ({n}, 4); got {box.shape}.")
    if not np.isfinite(box).all():
        raise ValueError("boxes must be finite, and some box holds NaN or infinity.")
    empty = np.flatnonzero((box[:, 2] <= box[:, 0]) | (box[:, 3] <= box[:, 1]))
    if len(empty):
        raise ValueError(f"boxes must have x1 > x0 and y1 > y0; row {empty[0]} is {box[empty[0]].tolist()}.")
    return box


def _read_labels(labels: ArrayLike, n: int) -> NDArray[np.intp]:
    """Return one whole class index per row."""
    label = whole_ids(labels, "labels")
    if len(label) != n:
        raise ValueError(f"labels has {len(label)} entries and embeddings has {n} rows. Give one per row.")
    return label


def _track_classes(labels: NDArray[np.intp], starts: NDArray[np.intp], ends: NDArray[np.intp]) -> NDArray[np.intp]:
    """Return each track's most common label, the smaller on a tie."""
    classes = np.empty(len(starts), dtype=np.intp)
    for i, (a, b) in enumerate(zip(starts, ends, strict=True)):
        values, counts = np.unique(labels[a:b], return_counts=True)
        classes[i] = values[np.argmax(counts)]
    return classes


def _close_pairs(
    group: NDArray[np.intp], centers: NDArray[np.float64], diagonal: NDArray[np.float64], tracks: NDArray[np.intp]
) -> NDArray[np.intp]:
    """Return the rows of one frame and class whose centers lie within the larger diagonal, lower track first."""
    gap = centers[group][:, None, :] - centers[group][None, :, :]
    reach = np.maximum(diagonal[group][:, None], diagonal[group][None, :])
    i, j = np.nonzero(np.triu(np.hypot(gap[..., 0], gap[..., 1]) <= reach, 1))
    a, b = group[i], group[j]
    first = tracks[a] < tracks[b]
    return np.column_stack([np.where(first, a, b), np.where(first, b, a)])


def _meetings(
    sequence: NDArray[np.intp],
    frame: NDArray[np.intp],
    classes: NDArray[np.intp],
    boxes: NDArray[np.float64],
    tracks: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Return every meeting as a pair of sorted rows, ``(m, 2)``, over all videos, frames and classes."""
    key = np.lexsort((classes, frame, sequence))
    s, f, c = sequence[key], frame[key], classes[key]
    cuts = np.flatnonzero((s[1:] != s[:-1]) | (f[1:] != f[:-1]) | (c[1:] != c[:-1])) + 1
    centers, diagonal = _centers(boxes), _diagonal(boxes)
    found = [_close_pairs(g, centers, diagonal, tracks) for g in np.split(key, cuts) if len(g) > 1]
    return np.concatenate(found) if found else np.empty((0, 2), dtype=np.intp)


def _encounters(
    meet: NDArray[np.intp],
    tracks: NDArray[np.intp],
    frame: NDArray[np.intp],
    starts: NDArray[np.intp],
    ends: NDArray[np.intp],
) -> list[NDArray[np.intp]]:
    """Split each pair's meetings, in frame order, where they lie more than ``_GAP`` shared frames apart."""
    ta, tb = tracks[meet[:, 0]], tracks[meet[:, 1]]
    order = np.lexsort((frame[meet[:, 0]], tb, ta))
    meet, ta, tb = meet[order], ta[order], tb[order]
    cuts = np.flatnonzero((ta[1:] != ta[:-1]) | (tb[1:] != tb[:-1])) + 1
    found: list[NDArray[np.intp]] = []
    for block in np.split(meet, cuts) if len(meet) else []:
        a, b = tracks[block[0, 0]], tracks[block[0, 1]]
        shared = np.intersect1d(frame[starts[a] : ends[a]], frame[starts[b] : ends[b]], assume_unique=True)
        where = np.searchsorted(shared, frame[block[:, 0]])
        found.extend(np.split(block, np.flatnonzero(np.diff(where) > _GAP) + 1))
    return found


class _Rows(NamedTuple):
    """The sorted unit rows, a running count of zero-length rows, and where each row's track starts and ends."""

    unit: NDArray[np.float64]
    zeros: NDArray[np.intp]  # cumulative count of zero-length rows, with a leading zero
    start: NDArray[np.intp]
    end: NDArray[np.intp]


def _running(
    unit: NDArray[np.float64], zero: NDArray[np.bool_], start: NDArray[np.intp], end: NDArray[np.intp]
) -> _Rows:
    """Return the rows with the running count that tells whether a window holds a zero-length row."""
    return _Rows(unit, np.concatenate([[0], np.cumsum(zero)]).astype(np.intp), start, end)


def _windows(
    rows: _Rows, at: NDArray[np.intp], window: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Return the means of the ``window`` rows before and from each row, and whether both fit with no zero row.

    Each mean sums only its own rows, so a score does not depend on any other track or video.
    """
    lo, hi = at - window, at + window
    fits = (lo >= rows.start[at]) & (hi <= rows.end[at])
    lo = np.where(fits, lo, at)
    fits &= rows.zeros[np.where(fits, hi, at)] == rows.zeros[lo]
    offsets = np.arange(window)
    before = np.where(fits[:, None], lo[:, None] + offsets, 0)
    after = np.where(fits[:, None], at[:, None] + offsets, 0)
    return rows.unit[before].mean(axis=1), rows.unit[after].mean(axis=1), fits


def _score_encounter(meetings: NDArray[np.intp], rows: _Rows, window: int) -> tuple[float, int]:
    """Return an encounter's largest score over its meetings and which meeting has it, the earliest on a tie."""
    b1, a1, fits1 = _windows(rows, meetings[:, 0], window)
    b2, a2, fits2 = _windows(rows, meetings[:, 1], window)

    def distance(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(np.sum((u - v) ** 2, axis=1))

    score = distance(b1, a1) + distance(b2, a2) - distance(b1, a2) - distance(b2, a1)
    score = np.where(fits1 & fits2, score, np.nan)
    if np.isnan(score).all():
        return float("nan"), 0
    best = int(np.nanargmax(score))
    return float(score[best]), best


def _sorted_result(columns: dict[str, NDArray[Any]]) -> TrackSwitchesResult:
    """Sort rows by score, highest first and NaN last, then by video, tracks and frame."""
    score = columns["switch_score"]
    missing = np.isnan(score)
    key = np.lexsort((
        columns["frame"],
        columns["track_b"],
        columns["track_a"],
        columns["item_index"],
        np.where(missing, 0.0, -score),
        missing,
    ))
    return TrackSwitchesResult(**{name: column[key] for name, column in columns.items()})  # type: ignore[typeddict-item]


@experimental
def track_switches(
    embeddings: ArrayLike,
    track_ids: ArrayLike,
    frame_indices: ArrayLike,
    sequence_indices: ArrayLike,
    *,
    boxes: ArrayLike,
    labels: ArrayLike,
    window: int = 10,
) -> TrackSwitchesResult:
    """Score every encounter of two tracks by whether their appearance fits better crossed than straight.

    An identity switch is an id that moves to another object partway through a track. Where two tracks of one class
    meet, a switch leaves a trace in their embeddings: after the meeting, each track looks like the other did before
    it. This function finds every meeting of two tracks and scores it by how much better the two tracks' embeddings fit
    crossed than straight. It reports places to look, not errors. The embeddings are of one detection each, such as
    the crops of a tracking dataset.

    .. versionadded:: 1.3

    Parameters
    ----------
    embeddings : ArrayLike or Embeddings
        One row per detection, shape ``(N, D)``. Pass the :class:`~dataeval.Embeddings` object itself when you have
        one.
    track_ids : ArrayLike
        The track of each row, shape ``(N,)``. A negative id marks a row no tracker linked, and the row is left out.
    frame_indices : ArrayLike
        The frame of each row, shape ``(N,)``. It orders the rows of a track and need not be consecutive. Two rows of
        one track cannot share a frame.
    sequence_indices : ArrayLike
        The video each row comes from, shape ``(N,)``, because a track id is unique only within one video. Over a
        :class:`~dataeval.data.SequenceFrames` view, ``frames.frame_map[crops.item_indices][:, 0]`` is the video of
        each crop.
    boxes : ArrayLike
        The box of each row, shape ``(N, 4)``, as ``x0, y0, x1, y1``, such as ``DetectionCrops.boxes``. Any unit
        works if it is the same on both axes: boxes normalized as ``x / width`` and ``y / height`` change which tracks
        meet unless the frame is square.
    labels : ArrayLike
        The class of each row, shape ``(N,)``, such as ``DetectionCrops.labels``. A track takes its most common class,
        the smaller on a tie, and only tracks of one class are paired. Give ``np.zeros(N)`` to pair every track.
    window : int, default 10
        The rows of each track, at least 2, averaged on each side of a meeting. A meeting without ``window`` rows on
        both sides of both tracks is not scored. The default is not tuned.

    Returns
    -------
    TrackSwitchesResult
        One row per encounter: ``item_index`` (the video), ``label``, ``track_a`` and ``track_b`` (``track_a <
        track_b``), ``frame``, ``switch_score``, ``flagged`` (``switch_score > 0``) and ``row_a`` and ``row_b``, the
        input row of each track at ``frame``, which ``DetectionCrops.source_indices`` turns into an address. Rows are
        sorted by ``switch_score``, highest first, then by video, tracks and frame; an encounter that could not be
        scored has a NaN score, is not flagged, and comes last at its first meeting.

    Raises
    ------
    ValueError
        If the inputs differ in length; ``embeddings`` is not two-dimensional or holds a value that is not finite;
        ``boxes`` is not ``(N, 4)``, holds a value that is not finite, or has a box with ``x1 <= x0`` or
        ``y1 <= y0``; an id or label is not a whole number; no row has a track id; no track has two rows; one track
        holds two rows at one frame; or ``window`` is not an integer of at least 2.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process.

    See Also
    --------
    track_steps : The step each detection takes from the previous one of its track, which shows a jump to a distant
        object.
    track_path_stats : How far, and in which direction, each track's embeddings travel.

    Notes
    -----
    **Meetings and encounters.** Two tracks of one class in one video meet at a frame where both have a row and their
    box centers lie within the larger of the two box diagonals. A pair's meetings join into one encounter unless two
    consecutive meetings lie more than 5 apart among the frames where both tracks have a row, so the rule does not
    depend on the frame rate. Each row of the result is one encounter.

    **The score.** Each row is scaled to unit length. At a meeting frame ``t``, ``b1`` and ``a1`` are the means of the
    first track's ``window`` rows before ``t`` and from ``t``, and ``b2`` and ``a2`` the same for the second track,
    counted in rows, not frames. The score is
    ``|b1 - a1| + |b2 - a2| - |b1 - a2| - |b2 - a1|``: the straight distances minus the crossed ones. It is above 0
    where each track's later rows look more like the other track's earlier rows. An encounter takes its largest score,
    at the earliest frame on a tie. A window that holds a row of zero length is not scored.

    **What a flag means.** A flag says the crossed matching fits better. The ids may have swapped; a box may have
    drifted onto the other track's object, as boxes interpolated between keyframes can during camera motion; or one
    object may pass behind an occluder, such as a sign or foliage, as the other comes out, so both tracks show the
    occluder. A row is about the pair: when one track takes the other's object and the other goes on to a third, the
    row does not say which one switched. Look at the frames of each flagged row.

    **What an empty result does not mean.** It does not mean the ids are right. A switch that lasts a few frames, a
    switch with no second track nearby, and tracks that never meet are not scored, and neither is a meeting too close
    to the start or end of a track. :func:`track_steps` shows a jump to a distant object as a large step. Windows are
    counted in rows, so a gap in a track widens them. Meetings are found among the rows passed, so a box left out, such
    as one ``DetectionCrops(min_size=...)`` dropped, cannot meet.

    **Cost.** Each frame compares every pair of tracks of one class present in it, so a frame with ``k`` such tracks
    costs ``k**2``.

    The meeting rule, the gap of 5, ``window`` and the flag at 0 are not tuned. The function is tested on constructed
    inputs only, not on real footage.

    Examples
    --------
    Two tracks whose boxes overlap on every frame, with their objects exchanged at frame 20:

    >>> from dataeval.core import track_switches
    >>> frames = np.r_[0:40, 0:40]
    >>> tracks = np.repeat([1, 2], 40)
    >>> objects = np.r_[np.repeat([0, 1], 20), np.repeat([1, 0], 20)]
    >>> rows = np.eye(4)[objects] + np.random.default_rng(0).normal(0.0, 0.02, (80, 4))
    >>> boxes = np.tile([0.0, 0.0, 10.0, 10.0], (80, 1))
    >>> result = track_switches(rows, tracks, frames, np.zeros(80, int), boxes=boxes, labels=np.zeros(80, int))
    >>> result["flagged"].tolist(), result["frame"].tolist()
    ([True], [20])
    """
    _check_window(window)
    x, track, frame, sequence = _read(embeddings, track_ids, frame_indices, sequence_indices)
    box, label = _read_boxes(boxes, len(x)), _read_labels(labels, len(x))
    order, starts, ends = _group(track, frame, sequence)
    tracks = np.repeat(np.arange(len(starts)), ends - starts)
    classes = _track_classes(label[order], starts, ends)
    meet = _meetings(sequence[order], frame[order], classes[tracks], box[order], tracks)
    encounters = _encounters(meet, tracks, frame[order], starts, ends)
    norms = np.linalg.norm(x[order], axis=1)
    unit = np.divide(x[order], norms[:, None], out=np.zeros((len(order), x.shape[1])), where=norms[:, None] > 0)
    rows = _running(unit, norms == 0, starts[tracks], ends[tracks])
    scored = [_score_encounter(e, rows, int(window)) for e in encounters]
    at = np.asarray([e[k] for e, (_, k) in zip(encounters, scored, strict=True)], dtype=np.intp).reshape(-1, 2)
    score = np.asarray([s for s, _ in scored], dtype=np.float64)
    pair = tracks[at[:, 0]]
    return _sorted_result({
        "item_index": sequence[order][at[:, 0]],
        "label": classes[pair],
        "track_a": track[order][at[:, 0]],
        "track_b": track[order][at[:, 1]],
        "frame": frame[order][at[:, 0]],
        "switch_score": score,
        "flagged": score > 0,
        "row_a": order[at[:, 0]],
        "row_b": order[at[:, 1]],
    })

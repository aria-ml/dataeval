"""Selecting frames from the annotations alone, so nothing is decoded."""

__all__ = []

from collections.abc import Iterable, Iterator
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.data._selectors import FrameCandidate, FrameInput, FrameSelector, FrameVerdict, SequenceInfo
from dataeval.types._target import instance_arrays, track_ids_of

_logger = get_logger(__name__)


def _read(target: Any) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32], NDArray[np.intp]]:
    """One frame's ``(labels, boxes, scores, track_ids)``."""
    labels, boxes, scores = instance_arrays(target)
    return labels, boxes, scores, track_ids_of(target, len(labels))


def _untracked_error(who: str, sequence: int, count: int, position: int, whole: bool) -> ValueError:
    """Build the refusal of untracked detections: ``count`` is the total offered, or ``whole=False`` one frame's.

    ``position`` counts the frames the selector was offered, which inside a ``Chain`` are those the
    stage before kept, and alone are the sequence's own, as ``Chain``'s messages count them.
    """
    where = (
        f"the frames it was offered from sequence {sequence} hold {count} detection(s) with no track id, "
        f"the first at position {position} of them"
        if whole
        else f"position {position} of the frames it was offered from sequence {sequence} holds {count} "
        "detection(s) with no track id"
    )
    return ValueError(f"{who}: {where}. Track the dataset, or pass on_missing_tracks to say what to do with them.")


def _check_policy(who: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{who}: on_missing_tracks must be one of {allowed}; got {value!r}.")


class TrackEvents(FrameSelector):
    """Keep the first frame, and each frame where the set of tracked objects changes.

    A frame is an event when a track appears or disappears. Between events nothing in the scene
    changes as far as the tracker can tell, so each kept frame stands for its run to the next.
    Decided from the annotations alone, so a view built on it is sized without decoding anything.

    Parameters
    ----------
    on_missing_tracks : {"error", "ignore", "label_counts"}, default "error"
        What to do with a detection no tracker linked (track id ``-1``, or no track id at all).
        ``"error"`` raises. ``"ignore"`` leaves it out of the comparison. ``"label_counts"``
        compares how many untracked detections of each label a frame holds.

    Raises
    ------
    ValueError
        If ``on_missing_tracks`` is not one of the three, or, at selection, under ``"error"``
        when a frame holds an untracked detection.

    Notes
    -----
    ``"label_counts"`` is a weaker signal than a track event: an object can leave and another of
    its class arrive without changing the counts.

    Examples
    --------
    >>> from dataeval.data import SequenceFrames, TrackEvents
    >>> frames = SequenceFrames(mot_dataset, TrackEvents())  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, on_missing_tracks: Literal["error", "ignore", "label_counts"] = "error") -> None:
        _check_policy("TrackEvents", on_missing_tracks, ("error", "ignore", "label_counts"))
        self.on_missing_tracks: str = on_missing_tracks

    def _key(self, target: Any) -> tuple[tuple[frozenset[int], tuple[tuple[int, int], ...]], int]:
        """Return a frame's key, and how many of its detections are untracked."""
        labels, _, _, ids = _read(target)
        counts: tuple[tuple[int, int], ...] = ()
        if self.on_missing_tracks == "label_counts":
            values, n = np.unique(labels[ids < 0], return_counts=True)
            counts = tuple(zip(values.tolist(), n.tolist(), strict=True))
        return (frozenset(ids[ids >= 0].tolist()), counts), int((ids < 0).sum())

    def plan(self, info: SequenceInfo) -> NDArray[np.intp] | None:
        """Return the first frame and each frame whose key differs from the one before, or None without a target."""
        if info.target is None:
            return None
        read = [self._key(t) for t in info.target.frame_tracks]
        total = sum(untracked for _, untracked in read)
        if self.on_missing_tracks == "error" and total:
            first = next(p for p, (_, untracked) in enumerate(read) if untracked)
            raise _untracked_error("TrackEvents", info.index, total, first, whole=True)
        keys = [key for key, _ in read]
        return np.array([p for p, key in enumerate(keys) if p == 0 or key != keys[p - 1]], dtype=np.intp)

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep the first frame and each frame whose key differs from the previous frame's."""
        previous = None
        for p, frame in enumerate(frames):
            key, untracked = self._key(frame.target)
            if self.on_missing_tracks == "error" and untracked:
                raise _untracked_error("TrackEvents", frame.sequence.index, untracked, frame.position, whole=False)
            if p == 0 or key != previous:
                yield FrameVerdict(frame.position)
            previous = key


class BestPerTrack(FrameSelector):
    """Keep, for each track, the one frame where it is largest or most confident.

    A filter, not a thinning: each kept frame stands only for itself. Ties go to the earliest
    frame. A track id belongs to its video, so the same id in two videos is two tracks.

    Parameters
    ----------
    by : {"area", "score"}, default "area"
        What to rank a track's detections by: the box area, ``(x1 - x0) * (y1 - y0)``, or the
        detection's own-class score.
    on_missing_tracks : {"error", "ignore"}, default "error"
        What to do with a detection no tracker linked. ``"error"`` raises, so an untracked dataset
        is refused on its first sequence with detections. ``"ignore"`` skips them.

    Raises
    ------
    ValueError
        If ``by`` or ``on_missing_tracks`` is not one of its choices. At selection, if a tracked
        detection's ranking value is not finite (no ``scores`` at all included), since leaving it
        out would assume it was worse, and a track with only such candidates has no defensible
        winner. Also under ``"error"`` when a detection is untracked.

    Notes
    -----
    A sequence with no tracked detection contributes nothing. Selection buffers each frame's
    target, never its pixels, and decides once its input ends. That is only valid where the whole
    sequence is offered first, which :class:`~dataeval.data.SequenceFrames` and
    :class:`~dataeval.data.Chain` always arrange by planning it.

    Examples
    --------
    >>> from dataeval.data import BestPerTrack, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, BestPerTrack(by="score"))  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.STRUCTURE
    excludes = True

    def __init__(
        self, by: Literal["area", "score"] = "area", on_missing_tracks: Literal["error", "ignore"] = "error"
    ) -> None:
        if by not in ("area", "score"):
            raise ValueError(f"BestPerTrack: by must be 'area' or 'score'; got {by!r}.")
        _check_policy("BestPerTrack", on_missing_tracks, ("error", "ignore"))
        self.by: str = by
        self.on_missing_tracks: str = on_missing_tracks

    def _ranked(self, sequence: int, position: int, target: Any) -> tuple[list[tuple[int, float]], int]:
        """Return a frame's tracked detections as ``(track id, ranking value)``, and its untracked count.

        Refuses a ranking value that is not finite.
        """
        _, boxes, scores, ids = _read(target)
        values = scores if self.by == "score" else (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ranked = [(t, v) for t, v in zip(ids.tolist(), values.tolist(), strict=True) if t >= 0]
        for track, value in ranked:
            if not np.isfinite(value):
                raise ValueError(
                    f"BestPerTrack: cannot rank track {track} of sequence {sequence} by {self.by}: "
                    f"position {position} of the frames it was offered gives {value}, which is not finite."
                )
        return ranked, int((ids < 0).sum())

    def _walk(
        self, sequence: int, frames: Iterable[tuple[int, Any]]
    ) -> tuple[dict[int, tuple[float, int]], list[tuple[int, int]]]:
        """Return each track's ``(best value, position)`` and the ``(position, count)`` of frames with untracked."""
        best: dict[int, tuple[float, int]] = {}
        missed: list[tuple[int, int]] = []
        for position, target in frames:
            ranked, missing = self._ranked(sequence, position, target)
            if missing:
                missed.append((position, missing))
            for track, value in ranked:
                if track not in best or value > best[track][0]:  # strict: ties go to the earliest frame
                    best[track] = (value, position)
        return best, missed

    def _best(self, sequence: int, frames: Iterable[tuple[int, Any]]) -> NDArray[np.intp]:
        best, missed = self._walk(sequence, frames)
        if missed and self.on_missing_tracks == "error":
            raise _untracked_error("BestPerTrack", sequence, sum(n for _, n in missed), missed[0][0], whole=True)
        if not best:
            _logger.info("BestPerTrack: sequence %d holds no tracked detection and contributes no frame.", sequence)
        return np.array(sorted({position for _, position in best.values()}), dtype=np.intp)

    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Return each track's best frame, from the sequence's target.

        Raises
        ------
        ValueError
            If ``info.target`` is None.
        """
        if info.target is None:
            raise ValueError("BestPerTrack: the sequence's target is needed to rank its tracks, and it is None.")
        return self._best(info.index, enumerate(info.target.frame_tracks))

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Buffer every frame's target, then yield each track's best frame once the input ends."""
        buffered = [(frame.sequence.index, frame.position, frame.target) for frame in frames]
        if buffered:
            for position in self._best(buffered[0][0], ((p, t) for _, p, t in buffered)).tolist():
                yield FrameVerdict(position)


class HasLabels(FrameSelector):
    """Keep the frames that hold enough detections of the given labels.

    A filter: a dropped frame is excluded rather than represented, so each kept frame stands only
    for itself. Decided from the annotations alone.

    Parameters
    ----------
    labels : Iterable[int]
        The labels to look for. One that never occurs is not an error.
    min_count : int, default 1
        Keep frames holding at least this many detections whose label is in ``labels``.

    Raises
    ------
    ValueError
        If ``labels`` is empty or ``min_count`` is less than 1.

    Examples
    --------
    >>> from dataeval.data import HasLabels, SequenceFrames
    >>> frames = SequenceFrames(mot_dataset, HasLabels([2, 3], min_count=2))  # doctest: +SKIP
    """

    needs: FrameInput = FrameInput.STRUCTURE
    excludes = True

    def __init__(self, labels: Iterable[int], min_count: int = 1) -> None:
        self.labels: frozenset[int] = frozenset(int(label) for label in labels)
        if not self.labels:
            raise ValueError("HasLabels: labels must name at least one label.")
        if min_count < 1:
            raise ValueError(f"HasLabels: min_count must be at least 1; got {min_count}.")
        self.min_count: int = int(min_count)
        self._wanted = np.array(sorted(self.labels), dtype=np.intp)

    def _holds(self, target: Any) -> bool:
        labels = instance_arrays(target)[0]
        return int(np.isin(labels, self._wanted).sum()) >= self.min_count

    def plan(self, info: SequenceInfo) -> NDArray[np.intp] | None:
        """Return the frames holding enough of the labels, or None without a target."""
        if info.target is None:
            return None
        return np.array([p for p, t in enumerate(info.target.frame_tracks) if self._holds(t)], dtype=np.intp)

    def select(self, frames: Iterator[FrameCandidate]) -> Iterator[FrameVerdict]:
        """Keep each frame holding enough of the labels."""
        for frame in frames:
            if self._holds(frame.target):
                yield FrameVerdict(frame.position)

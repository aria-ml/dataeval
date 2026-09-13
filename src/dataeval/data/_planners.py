"""Planners for video segmentation in :class:`~dataeval.data.VideoSegments`."""

__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.data._selectors import SequenceInfo
from dataeval.types import ReprMixin

_logger = get_logger(__name__)


class SegmentPlanner(ReprMixin, ABC):
    """Abstract base class for planning video segmentation boundaries.

    Subclasses define how :class:`~dataeval.data.VideoSegments` partitions
    source videos using metadata from :class:`~dataeval.data.SequenceInfo`.

    See Also
    --------
    :class:`Window` : Fixed-length windows, optionally sliding.
    :class:`Cuts` : Explicit frame boundaries per video.
    """

    @abstractmethod
    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Return segment boundaries for a video as ``(K, 2)`` rows of ``[start, end)``.

        Parameters
        ----------
        info : SequenceInfo
            Video sequence metadata.

        Returns
        -------
        NDArray[np.intp]
            Segment boundaries in ascending ``start`` order, where
            ``0 <= start < end <= info.n_frames``. If ``K == 0``, the video produces no segments.
        """


def validated_plan(plan: Any, planner: SegmentPlanner, info: SequenceInfo) -> NDArray[np.intp]:
    """Validate and return segment plan rows as a ``(K, 2)`` integer array.

    Raises
    ------
    ValueError
        If plan rows are invalid, out of range, or not in ascending order.
    """
    rows = np.asarray(plan)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.intp)
    problem = _plan_problem(rows, info.n_frames)
    if problem is not None:
        what, rule = problem
        raise ValueError(f"{planner!r} returned {what} for video {info.source_id!r} ({info.n_frames} frames); {rule}.")
    return rows.astype(np.intp)


def _plan_problem(rows: NDArray[Any], n_frames: int) -> tuple[str, str] | None:
    """Check plan rows for validation errors, returning a problem description or None if valid."""
    if rows.shape[1:] != (2,):
        return f"an array of shape {rows.shape}", "a plan is (K, 2) rows of [start, end)"
    if not np.issubdtype(rows.dtype, np.integer):
        return "non-integer rows", "frame positions are ints"
    starts, ends = rows[:, 0], rows[:, 1]
    bad = np.flatnonzero((starts < 0) | (starts >= ends) | (ends > n_frames))
    if bad.size:
        return f"row {rows[bad[0]].tolist()}", "every row must satisfy 0 <= start < end <= n_frames"
    if np.any(np.diff(starts) < 0):
        return "rows out of order", "rows must be in ascending start order"
    return None


class Window(SegmentPlanner):
    """Cut videos into fixed-length windows of ``size`` frames.

    Parameters
    ----------
    size : int
        Frames per window. Must be at least 1.
    stride : int or None, default None
        Frames between successive window starts. Defaults to ``size``
        (non-overlapping windows). Values smaller than ``size`` create overlapping
        windows; values larger than ``size`` leave gaps between windows.
    drop_remainder : bool, default False
        Whether to drop a trailing window that has fewer than ``size`` frames.

    Raises
    ------
    ValueError
        If ``size`` or ``stride`` is less than 1.

    See Also
    --------
    :class:`~dataeval.data.VideoSegments` : Segmented video dataset view.

    Notes
    -----
    Window starts begin at 0 and advance by ``stride``. A video shorter than
    ``size`` produces one truncated window, or no windows if ``drop_remainder=True``.

    Examples
    --------
    >>> from dataeval.data import SequenceInfo, Window
    >>> info = SequenceInfo(index=0, source_id="clip", n_frames=20, metadata={"id": "clip"})
    >>> Window(8).plan(info).tolist()
    [[0, 8], [8, 16], [16, 20]]
    >>> Window(8, drop_remainder=True).plan(info).tolist()
    [[0, 8], [8, 16]]
    >>> Window(8, stride=4).plan(info).tolist()
    [[0, 8], [4, 12], [8, 16], [12, 20], [16, 20]]
    """

    def __init__(self, size: int, stride: int | None = None, drop_remainder: bool = False) -> None:
        if size < 1:
            raise ValueError(f"Window: size must be at least 1; got {size}.")
        stride = size if stride is None else stride
        if stride < 1:
            raise ValueError(f"Window: stride must be at least 1; got {stride}.")
        self.size: int = int(size)
        self.stride: int = int(stride)
        self.drop_remainder: bool = bool(drop_remainder)

    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Compute window segment boundaries for a video."""
        starts = np.arange(0, info.n_frames, self.stride, dtype=np.intp)
        if self.drop_remainder:
            starts = starts[starts + self.size <= info.n_frames]
        if starts.size == 0:
            return np.empty((0, 2), dtype=np.intp)
        ends = np.minimum(starts + self.size, info.n_frames)
        return np.stack((starts, ends), axis=1)


class Cuts(SegmentPlanner):
    """Cut videos at explicit frame boundaries.

    Parameters
    ----------
    cuts : Mapping[int | str, Sequence[int]] or str
        Cut points per video keyed by datum ``id``, or the name of a datum metadata
        key containing the cut points.

    Raises
    ------
    ValueError
        If any cut point is not strictly within the video (``0 < cut < n_frames``).

    See Also
    --------
    :class:`~dataeval.data.VideoSegments` : Segmented video dataset view.

    Notes
    -----
    Cut points are sorted and deduplicated. Videos without cut points are kept as
    a single segment spanning the entire video.

    Examples
    --------
    >>> from dataeval.data import Cuts, SequenceInfo
    >>> info = SequenceInfo(index=0, source_id="clip-07", n_frames=400, metadata={"id": "clip-07"})
    >>> Cuts({"clip-07": [100, 250]}).plan(info).tolist()
    [[0, 100], [100, 250], [250, 400]]
    >>> info = SequenceInfo(0, "clip-07", 400, {"id": "clip-07", "shot_boundaries": [120]})
    >>> Cuts("shot_boundaries").plan(info).tolist()
    [[0, 120], [120, 400]]
    """

    def __init__(self, cuts: Mapping[int | str, Sequence[int]] | str) -> None:
        self.cuts: Mapping[int | str, Sequence[int]] | str = cuts

    def plan(self, info: SequenceInfo) -> NDArray[np.intp]:
        """Compute segment boundaries from cut points for a video."""
        if info.n_frames == 0:
            return np.empty((0, 2), dtype=np.intp)
        raw = self._cuts_for(info)
        points = sorted({int(cut) for cut in raw})
        bad = [cut for cut in points if not 0 < cut < info.n_frames]
        if bad:
            raise ValueError(
                f"Cuts: video {info.source_id!r} has {info.n_frames} frames, but cut point(s) {bad} "
                "are not strictly inside it; every cut must satisfy 0 < cut < n_frames."
            )
        bounds = [0, *points, info.n_frames]
        return np.array([[a, b] for a, b in zip(bounds, bounds[1:], strict=False)], dtype=np.intp)

    def _cuts_for(self, info: SequenceInfo) -> Sequence[int]:
        """Return cut points for a video from the mapping or metadata, or an empty sequence."""
        if isinstance(self.cuts, str):
            raw = info.metadata.get(self.cuts)
            where = f"metadata key {self.cuts!r}"
        else:
            raw = self.cuts.get(info.source_id)
            where = "the cuts mapping"
        if raw is None:
            _logger.info("Cuts: video %r has no cut points in %s and is kept as one segment.", info.source_id, where)
            return ()
        return raw

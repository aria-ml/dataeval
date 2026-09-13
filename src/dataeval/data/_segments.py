"""Present a multi-object-tracking dataset as a dataset of video segments."""

__all__ = []

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.data._planners import SegmentPlanner, validated_plan
from dataeval.data._selectors import SequenceInfo, sequence_infos
from dataeval.data._video_proxies import (
    _FrameTracksTarget,
    _LazyStream,
    _ShiftedFrame,
    checked_index,
    inherited_metadata,
    long_stream,
    short_stream,
    timing_of,
)
from dataeval.flags import ImageStats
from dataeval.protocols import (
    AnnotatedDataset,
    DatasetMetadata,
    DatumMetadata,
    MultiobjectTrackingDataset,
    MultiobjectTrackingDatum,
)
from dataeval.utils.data import requires_maite_dataset

_logger = get_logger(__name__)

# Sentinel indicating iterator exhaustion.
_EXHAUSTED: Any = object()


@dataclass
class _Cursor:
    """State of an open source video stream iterator."""

    frames: Iterator[Any]
    source: int
    position: int
    loaned: bool = False


TimestampPolicy = Literal["source", "local"]

INJECTED_KEYS: tuple[str, ...] = ("source_id", "segment_index", "start_frame", "end_frame")
"""Metadata keys added by :class:`VideoSegments`, replacing any matching source keys."""


def slice_per_frame(value: Any, n_frames: int, start: int, end: int) -> Any:
    """Slice per-frame values (sequences of length ``n_frames``) to ``[start:end]``, recursing into dicts."""
    if isinstance(value, dict):
        return {key: slice_per_frame(item, n_frames, start, end) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[start:end] if value.ndim >= 1 and len(value) == n_frames else value
    if isinstance(value, list | tuple) and len(value) == n_frames:
        return value[start:end]
    return value


def _plan_costs(plan: NDArray[np.intp], info: SequenceInfo) -> tuple[int, int]:
    """Calculate the number of dropped frames and re-decoded overlapping frames in a plan."""
    if len(plan) == 0:
        return info.n_frames, 0
    # Count covered frames in a single pass over sorted intervals.
    n_covered, reached = 0, 0
    for start, end in plan.tolist():
        n_covered += max(0, end - max(start, reached))
        reached = max(reached, end)
    # Excess over covered frames represents frames decoded multiple times during sequential access.
    return info.n_frames - n_covered, int((plan[:, 1] - plan[:, 0]).sum()) - n_covered


def _plan_dataset(
    dataset: MultiobjectTrackingDataset, planner: SegmentPlanner
) -> tuple[list[SequenceInfo], list[NDArray[np.intp]], set[int], int, int]:
    """Compute segment plans for each video in the dataset without decoding frames."""
    sequences: list[SequenceInfo] = []
    plans: list[NDArray[np.intp]] = []
    overlapping: set[int] = set()
    unplanned: list[Any] = []
    n_dropped = 0
    n_redecoded = 0
    for info, _ in sequence_infos(dataset):
        plan = validated_plan(planner.plan(info), planner, info)
        sequences.append(info)
        plans.append(plan)
        dropped, extra = _plan_costs(plan, info)
        if len(plan) == 0:
            unplanned.append(info.source_id)
        n_dropped += dropped
        n_redecoded += extra
        if extra:
            overlapping.add(info.index)
    if unplanned:
        # Log summary of videos with empty plans rather than individual entries.
        _logger.info(
            "VideoSegments: %r plans no segment for %d of %d video(s), starting with %r.",
            planner,
            len(unplanned),
            len(sequences),
            unplanned[0],
        )
    return sequences, plans, overlapping, n_dropped, n_redecoded


class VideoSegments(AnnotatedDataset[MultiobjectTrackingDatum]):
    """Present a multi-object-tracking dataset as a dataset of video segments.

    Each source video is partitioned by ``planner`` into one or more segments.
    Each segment is a datum consisting of a lazy frame stream, corresponding
    frame targets, and segment metadata. Frames are decoded lazily upon access.

    Parameters
    ----------
    dataset : MultiobjectTrackingDataset
        The source tracking dataset.
    planner : SegmentPlanner
        Planner determining where to cut each video. See :class:`~dataeval.data.Window`
        and :class:`~dataeval.data.Cuts`.
    timestamps : {"source", "local"}, default "source"
        Timestamp handling for ``time_s`` and ``pts``. ``"source"`` preserves the
        original timeline. ``"local"`` rebases timestamps to zero at the start
        of each segment. ``frame_index`` is always 0-indexed within the segment.

    Attributes
    ----------
    n_source_frames : int
        Total frames across all source videos.
    n_dropped_frames : int
        Total source frames omitted by the planner.

    Raises
    ------
    MaiteShapeError
        If ``dataset`` is not a multi-object-tracking dataset.
    TypeError
        If ``planner`` is not a :class:`~dataeval.data.SegmentPlanner`.
    ValueError
        If ``timestamps`` is invalid or the planner returns invalid segment bounds.

    See Also
    --------
    :class:`~dataeval.data.VideoStitch` : Combine video sequences.
    :class:`~dataeval.data.SequenceFrames` : Present tracking video frames as detection images.

    Notes
    -----
    Iterating segments sequentially decodes each source frame once by maintaining
    an open stream cursor. Non-sequential access or overlapping windows require
    reopening and seeking through the source stream.

    Each segment datum includes metadata with:

    - ``id`` (*int*) -- Segment index in this view.
    - ``source_id`` (*int | str*) -- Identifier of the source video datum.
    - ``segment_index`` (*int*) -- Index of this segment within its source video.
    - ``start_frame``, ``end_frame`` (*int*) -- Half-open frame range in the source video.
    - Source metadata attributes, with per-frame sequences sliced to the segment
      range. The ``size`` attribute is omitted.

    When segments cut across an ongoing track, track IDs are preserved within each
    segment.

    Examples
    --------
    >>> from dataeval.data import VideoSegments, Window
    >>> segments = VideoSegments(mot_dataset, Window(150))  # doctest: +SKIP
    """

    # Segmentation does not modify frame pixels.
    invalidates: ImageStats = ImageStats.NONE

    @requires_maite_dataset("dataset", expected="multiobject_tracking")
    def __init__(
        self,
        dataset: MultiobjectTrackingDataset,
        planner: SegmentPlanner,
        *,
        timestamps: TimestampPolicy = "source",
    ) -> None:
        if not isinstance(planner, SegmentPlanner):
            raise TypeError(f"planner must be a SegmentPlanner; got {type(planner).__name__}.")
        if timestamps not in ("source", "local"):
            raise ValueError(f"timestamps must be 'source' or 'local'; got {timestamps!r}.")
        self._dataset = dataset
        self._planner = planner
        self._timestamps: TimestampPolicy = timestamps

        # Read frame counts from targets without decoding video frames.
        sequences, plans, overlapping, n_dropped, n_redecoded = _plan_dataset(dataset, planner)
        self._sequences: list[SequenceInfo] = sequences
        self._overlapping: set[int] = overlapping
        self.n_dropped_frames: int = n_dropped
        self.n_source_frames: int = sum(info.n_frames for info in self._sequences)
        rows = [
            np.column_stack((np.full(len(plan), info.index, dtype=np.intp), plan))
            for info, plan in zip(self._sequences, plans, strict=True)
            if len(plan)
        ]
        self._segment_map: NDArray[np.intp] = np.concatenate(rows) if rows else np.empty((0, 3), dtype=np.intp)
        self._segment_map.flags.writeable = False
        if n_redecoded:
            _logger.warning(
                "VideoSegments: %r overlaps, so %d frame(s) will be decoded more than once under "
                "in-order access and appear as repeated unit rows in Metadata; Duplicates reports "
                "every repeat as an exact match.",
                planner,
                n_redecoded,
            )
        self._metadata = inherited_metadata(dataset, "segments")

        # Cache the most recently read source datum for consecutive segments.
        self._cache_index: int | None = None
        self._cache_datum: MultiobjectTrackingDatum | None = None
        self._cache_stream_unread: bool = False
        self._warned_keys: set[str] = set()

        # Stream cursor reused across consecutive in-order segment reads.
        self._cursor: _Cursor | None = None
        self._warned_skip: set[int] = set()

        _logger.debug(
            "VideoSegments: %d sequence(s), %d source frame(s), %d segment(s), planner=%r",
            len(self._sequences),
            self.n_source_frames,
            len(self._segment_map),
            planner,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    @property
    def source(self) -> MultiobjectTrackingDataset:
        """Underlying tracking dataset wrapped by this view."""
        return self._dataset

    @property
    def planner(self) -> SegmentPlanner:
        """The :class:`~dataeval.data.SegmentPlanner` used to partition videos."""
        return self._planner

    @property
    def timestamps(self) -> TimestampPolicy:
        """Timestamp policy (``'source'`` or ``'local'``)."""
        return self._timestamps

    @property
    def segment_map(self) -> NDArray[np.intp]:
        """Array of shape ``(K, 3)`` containing ``(source index, start, end)`` for each segment."""
        return self._segment_map

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata for the segmented view."""
        return self._metadata

    def __len__(self) -> int:
        return len(self._segment_map)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        """Return one segment as a tracking datum."""
        index = checked_index(index, len(self), "VideoSegments", "segment(s)")
        source_index, start, end = (int(value) for value in self._segment_map[index])
        _, target, source_metadata = self._source_datum(source_index)
        frame_tracks = _FrameTracksTarget(target.frame_tracks[start:end])
        stream = _LazyStream(lambda: self._iterate(source_index, start, end), f"segment {index}")
        return stream, frame_tracks, self._segment_metadata(index, source_index, start, end, source_metadata)

    def __iter__(self) -> Iterator[MultiobjectTrackingDatum]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"VideoSegments(dataset={self._dataset!r}, planner={self._planner!r}, timestamps={self._timestamps!r})"

    def __str__(self) -> str:
        title = "VideoSegments Dataset"
        sep = "-" * len(title)
        return (
            f"{title}\n{sep}\n    sequences: {len(self._sequences)}\n"
            f"    source frames: {self.n_source_frames}\n    segments: {len(self)}\n"
            f"    dropped frames: {self.n_dropped_frames}\n    planner: {self._planner!r}\n\n{self._dataset}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _source_datum(self, source_index: int) -> MultiobjectTrackingDatum:
        """Retrieve and cache a source datum, reusing it across consecutive segments."""
        if self._cache_index != source_index or self._cache_datum is None:
            self._cache_index, self._cache_datum = source_index, self._dataset[source_index]
            self._cache_stream_unread = True
        return self._cache_datum

    def _open_stream(self, source_index: int) -> Any:
        """Return the source video stream, reusing cached stream if unread."""
        if self._cache_index == source_index and self._cache_datum is not None and self._cache_stream_unread:
            self._cache_stream_unread = False
            return self._cache_datum[0]
        return self._dataset[source_index][0]

    def _segment_metadata(
        self, index: int, source_index: int, start: int, end: int, source: DatumMetadata
    ) -> DatumMetadata:
        """Construct segment metadata with sliced per-frame attributes and segment indices."""
        info = self._sequences[source_index]
        metadata: dict[str, Any] = {}
        for key, value in source.items():
            if key in ("id", "size"):
                continue
            if key in INJECTED_KEYS:
                if key not in self._warned_keys:
                    self._warned_keys.add(key)
                    _logger.warning(
                        "VideoSegments: source metadata key %r is displaced by the value VideoSegments "
                        "writes under that name.",
                        key,
                    )
                continue
            metadata[key] = slice_per_frame(value, info.n_frames, start, end)
        metadata.update(
            id=index,
            source_id=info.source_id,
            segment_index=index - int(np.searchsorted(self._segment_map[:, 0], source_index)),
            start_frame=start,
            end_frame=end,
        )
        return cast(DatumMetadata, metadata)

    def _iterate(self, source_index: int, start: int, end: int) -> Iterator[Any]:
        """Yield frames for a segment, reusing the stream cursor if positioned at ``start``."""
        info = self._sequences[source_index]
        cursor = self._cursor
        if cursor is not None and not cursor.loaned and cursor.source == source_index and cursor.position == start:
            frames = cursor.frames
        else:
            if start and source_index not in self._overlapping and source_index not in self._warned_skip:
                self._warned_skip.add(source_index)
                _logger.warning(
                    "VideoSegments: reaching frame %d of video %r decodes and discards the frames "
                    "before it. A VideoStream cannot seek; read segments in order to decode each "
                    "frame once.",
                    start,
                    info.source_id,
                )
            frames = iter(self._open_stream(source_index))
            self._skip(frames, start, info)
            cursor = _Cursor(frames, source_index, start)
            self._cursor = cursor
        cursor.loaned = True
        try:
            yield from self._emit(cursor, start, end, info)
        finally:
            cursor.loaned = False

    def _skip(self, frames: Iterator[Any], count: int, info: SequenceInfo) -> None:
        """Advance the stream by discarding ``count`` frames."""
        for position in range(count):
            if next(frames, _EXHAUSTED) is _EXHAUSTED:
                raise ValueError(self._short(info, position))

    def _emit(self, cursor: "_Cursor", start: int, end: int, info: SequenceInfo) -> Iterator[Any]:
        """Yield frames in range ``[start, end)`` with adjusted frame indices and timings."""
        time_shift, pts_shift = 0.0, 0
        for position in range(start, end):
            frame = next(cursor.frames, _EXHAUSTED)
            if frame is _EXHAUSTED:
                raise ValueError(self._short(info, position))
            cursor.position = position + 1
            if position == start:
                time_shift, pts_shift = self._shifts(frame)
            yield _ShiftedFrame(frame, position - start, time_shift, pts_shift)
        if end == info.n_frames:
            self._finish(cursor, info)

    def _finish(self, cursor: "_Cursor", info: SequenceInfo) -> None:
        """Verify stream completion at video end and release the cursor."""
        if next(cursor.frames, _EXHAUSTED) is not _EXHAUSTED:
            raise ValueError(long_stream("VideoSegments", info.source_id, info.n_frames))
        if self._cursor is cursor:
            self._cursor = None

    def _shifts(self, frame: Any) -> tuple[float, int]:
        """Compute timestamp shifts based on the active timestamp policy."""
        if self._timestamps == "source":
            return 0.0, 0
        time_s = timing_of(frame, "time_s")
        pts = timing_of(frame, "pts")
        return (0.0 if time_s is None else -time_s), (0 if pts is None else -int(pts))

    @staticmethod
    def _short(info: SequenceInfo, position: int) -> str:
        return short_stream("VideoSegments", info.source_id, info.n_frames, position)

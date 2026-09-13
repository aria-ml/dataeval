"""Combine videos of a multi-object-tracking dataset into stitched sequences."""

__all__ = []

import itertools
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.data._selectors import SequenceInfo, sequence_infos
from dataeval.data._video_proxies import (
    _FrameTracksTarget,
    _LazyStream,
    _OffsetFrameTarget,
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
from dataeval.types._target import detection_count, track_ids_of
from dataeval.types._track import frame_size
from dataeval.utils.data import requires_maite_dataset

_logger = get_logger(__name__)

# Sentinel indicating iterator exhaustion.
_EXHAUSTED: Any = object()

TrackIdPolicy = Literal["offset", "preserve", "error"]
StitchTimestamps = Literal["source", "offset"]
KeySpec: TypeAlias = str | Callable[[DatumMetadata], Any]

_GEOMETRY_KEYS: tuple[str, ...] = ("height", "width", "time_base")
_LINEAGE_KEYS: tuple[str, ...] = ("source_id", "start_frame", "end_frame")
# Metadata keys handled separately during merging: ``id`` is reindexed, ``size`` is summed.
_NOT_MERGED: tuple[str, ...] = ("id", "size")


def _key_of(spec: KeySpec, info: SequenceInfo, role: str) -> Any:
    """Extract grouping or ordering key from video metadata."""
    if isinstance(spec, str):
        metadata = cast(Mapping[str, Any], info.metadata)
        if spec not in metadata:
            raise KeyError(
                f"VideoStitch: {role} names metadata key {spec!r}, which video {info.source_id!r} "
                f"(item {info.index}) does not carry."
            )
        return metadata[spec]
    return spec(info.metadata)


def _track_id_summary(target: Any, collect: bool) -> tuple[int, set[int]]:
    """Return the maximum non-negative track ID, and optionally the set of all track IDs."""
    largest = -1
    seen: set[int] = set()
    for frame_target in target.frame_tracks:
        ids = track_ids_of(frame_target, detection_count(frame_target))
        tracked = ids[ids >= 0]
        if tracked.size:
            largest = max(largest, int(tracked.max()))
            if collect:
                seen.update(tracked.tolist())
    return largest, seen


def _read_videos(
    dataset: MultiobjectTrackingDataset, scan_ids: bool, collect_ids: bool = False
) -> tuple[list[SequenceInfo], list[int], list[set[int]]]:
    """Read sequence info and track ID summaries for all videos without decoding frames."""
    sequences: list[SequenceInfo] = []
    largest: list[int] = []
    id_sets: list[set[int]] = []
    for info, target in sequence_infos(dataset):
        sequences.append(info)
        big, seen = _track_id_summary(target, collect_ids) if scan_ids else (-1, set())
        largest.append(big)
        id_sets.append(seen)
    return sequences, largest, id_sets


def _log_repeated_ids(sequences: Sequence[SequenceInfo]) -> None:
    """Log an info message if any video datum IDs are duplicated."""
    ids = [info.source_id for info in sequences]
    repeats = len(ids) - len(set(ids))
    if repeats:
        _logger.info(
            "VideoStitch: %d video(s) share a datum id with another; grouping by 'source_id' would merge them.",
            repeats,
        )


def _group(
    sequences: Sequence[SequenceInfo], group_by: KeySpec | None, order_by: KeySpec | None
) -> dict[Any, list[int]]:
    """Group and sort video indices by the given key specifications."""
    groups: dict[Any, list[int]] = {}
    for info in sequences:
        key = None if group_by is None else _key_of(group_by, info, "group_by")
        groups.setdefault(key, []).append(info.index)
    if order_by is not None:
        for members in groups.values():
            members.sort(key=lambda i: _key_of(order_by, sequences[i], "order_by"))
    return groups


def _is_listlike(value: Any) -> bool:
    return isinstance(value, list | tuple) or (isinstance(value, np.ndarray) and value.ndim >= 1)


def _equal(left: Any, right: Any) -> bool:
    """Compare two metadata values for equality, handling array comparisons."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _merge_values(items: Sequence[Any], counts: Sequence[int]) -> tuple[bool, Any, str]:
    """Merge metadata values across constituents by concatenating per-frame sequences or asserting scalar equality."""
    if any(_is_listlike(item) for item in items):
        if all(_is_listlike(item) and len(item) == count for item, count in zip(items, counts, strict=True)):
            if all(isinstance(item, np.ndarray) for item in items):
                return True, np.concatenate(items), ""
            return True, list(itertools.chain.from_iterable(items)), ""
        return False, None, "it is a list whose length is not the video's frame count"
    if all(_equal(item, items[0]) for item in items[1:]):
        return True, items[0], ""
    return False, None, "the videos disagree on its value"


def _merge_constituents(
    values: Sequence[Mapping[str, Any]],
    counts: Sequence[int],
    omitted: Callable[[str, str], None],
    prefix: str = "",
) -> dict[str, Any]:
    """Merge constituent metadata mappings, concatenating per-frame lists and matching equal scalars."""
    merged: dict[str, Any] = {}
    for key in dict.fromkeys(key for value in values for key in value):
        name = f"{prefix}{key}"
        if not all(key in value for value in values):
            omitted(name, "not every video declares it")
            continue
        items = [value[key] for value in values]
        if all(isinstance(item, dict) for item in items):
            merged[key] = _merge_constituents(items, counts, omitted, f"{name}.")
            continue
        kept, value, reason = _merge_values(items, counts)
        if kept:
            merged[key] = value
        else:
            omitted(name, reason)
    return merged


class _Timeline:
    """Track and adjust a timing channel (``time_s`` or ``pts``) across stitched constituents."""

    def __init__(self, offset: bool, integer: bool) -> None:
        self._offset = offset
        self._integer = integer
        self.shift: float = 0
        self.backwards: bool = False
        self.unknown_interval: bool = False
        self._last_out: float | None = None
        self._first_out: float | None = None
        self._interval: float | None = None

    def begin(self, first: float | None) -> None:
        """Initialize timing shift for a constituent starting with value ``first``."""
        self._first_out = None
        self.backwards = False
        self.unknown_interval = False
        self.shift = 0
        if first is None or self._last_out is None:
            return
        if not self._offset:
            self.backwards = first < self._last_out
            return
        self.unknown_interval = self._interval is None
        shift = self._last_out + (self._interval or 0.0) - first
        self.shift = round(shift) if self._integer else shift

    def observe(self, value: float | None) -> None:
        """Update timeline state with a frame's raw timing value."""
        if value is None:
            return
        out = value + self.shift
        if self._first_out is None:
            self._first_out = out
        self._last_out = out

    def end(self, n_frames: int) -> None:
        """Finalize a constituent and calculate its average frame interval."""
        if n_frames >= 2 and self._first_out is not None and self._last_out is not None:
            self._interval = (self._last_out - self._first_out) / (n_frames - 1)


@dataclass
class _Chain:
    """State carried across constituents during stream playback."""

    time: _Timeline
    pts: _Timeline
    running: int = 0
    size: tuple[int | None, int | None] | None = None


class VideoStitch(AnnotatedDataset[MultiobjectTrackingDatum]):
    """Combine videos of a multi-object-tracking dataset into stitched sequences.

    Videos are grouped by ``group_by`` and sorted by ``order_by``. Each group becomes
    one tracking datum with sequentially played frames, concatenated frame targets,
    and track IDs resolved by ``track_ids``. Frames are decoded lazily upon access.

    Parameters
    ----------
    dataset : MultiobjectTrackingDataset
        The source dataset.
    group_by : str, callable, or None
        Metadata key or callable ``(DatumMetadata) -> Hashable`` defining video groups.
        If ``None``, all videos are stitched into a single sequence.
    order_by : str, callable, or None, default None
        Metadata key or callable ``(DatumMetadata) -> Any`` defining sequence order
        within each group. If ``None``, source dataset order is preserved.
    track_ids : {"offset", "preserve", "error"}, default "offset"
        Policy for resolving track IDs across constituent boundaries:

        - ``"offset"``: Shift each constituent's track IDs to avoid collisions.
        - ``"preserve"``: Retain original track IDs unchanged.
        - ``"error"``: Raise ``ValueError`` if track IDs overlap across constituents.
    timestamps : {"source", "offset"}, default "source"
        Policy for frame ``time_s`` and ``pts``:

        - ``"source"``: Preserve constituent timestamps.
        - ``"offset"``: Shift timestamps to create a continuous sequence based on
          the observed frame interval.

    Raises
    ------
    MaiteShapeError
        If ``dataset`` is not a multi-object-tracking dataset.
    KeyError
        If ``group_by`` or ``order_by`` specifies a metadata key missing from any datum.
    ValueError
        If policy options are invalid, constituents disagree on geometry metadata,
        segments overlap, or ``track_ids="error"`` encounters conflicting track IDs.

    See Also
    --------
    :class:`~dataeval.data.VideoSegments` : Partition videos into shorter segments.

    Notes
    -----
    Stitched datum metadata includes:

    - ``id`` (*int*) -- Stitched sequence index in this view.
    - ``n_sources`` (*int*) -- Number of constituent videos in this sequence.
    - Uniform scalar metadata across constituents (e.g. ``height``, ``width``, ``time_base``).
    - ``size`` summed across constituents if present in all.
    - Per-frame sequences concatenated across constituents.
    - Conflicting scalar metadata keys are omitted.

    Examples
    --------
    >>> from dataeval.data import VideoSegments, VideoStitch, Window
    >>> segments = VideoSegments(mot_dataset, Window(150))  # doctest: +SKIP
    >>> whole = VideoStitch(segments, group_by="source_id", track_ids="preserve")  # doctest: +SKIP
    """

    # Stitching does not modify frame pixels.
    invalidates: ImageStats = ImageStats.NONE

    @requires_maite_dataset("dataset", expected="multiobject_tracking")
    def __init__(
        self,
        dataset: MultiobjectTrackingDataset,
        *,
        group_by: KeySpec | None,
        order_by: KeySpec | None = None,
        track_ids: TrackIdPolicy = "offset",
        timestamps: StitchTimestamps = "source",
    ) -> None:
        _check_policies(track_ids, timestamps)
        self._dataset = dataset
        self._group_by = group_by
        self._order_by = order_by
        self._track_ids: TrackIdPolicy = track_ids
        self._timestamps: StitchTimestamps = timestamps

        # Inspect targets and metadata without decoding streams.
        sequences, largest, id_sets = _read_videos(
            dataset, scan_ids=track_ids != "preserve", collect_ids=track_ids == "error"
        )
        self._sequences: list[SequenceInfo] = sequences
        _log_repeated_ids(sequences)
        groups = _group(sequences, group_by, order_by)
        self._groups: tuple[tuple[int, ...], ...] = tuple(tuple(members) for members in groups.values())
        self._group_keys: tuple[Hashable, ...] = tuple(groups)

        self._logged_geometry: set[str] = set()
        for key, members in zip(self._group_keys, self._groups, strict=True):
            self._check_geometry(key, members)
            self._check_overlap(key, members)
            self._check_shared_ids(key, members, id_sets)

        self._frame_offsets: tuple[NDArray[np.intp], ...] = tuple(
            np.concatenate(([0], np.cumsum([sequences[i].n_frames for i in members]))).astype(np.intp)
            for members in self._groups
        )
        self._track_id_offsets: tuple[NDArray[np.intp], ...] = tuple(
            self._offsets_for(members, largest) for members in self._groups
        )
        for offsets in (*self._frame_offsets, *self._track_id_offsets):
            offsets.flags.writeable = False

        self._metadata = inherited_metadata(dataset, "stitched")
        self._logged_keys: set[str] = set()
        self._warned_backwards: set[int] = set()

        _logger.debug(
            "VideoStitch: %d video(s) into %d sequence(s), track_ids=%r, timestamps=%r",
            len(sequences),
            len(self._groups),
            track_ids,
            timestamps,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    @property
    def source(self) -> MultiobjectTrackingDataset:
        """Underlying tracking dataset wrapped by this view."""
        return self._dataset

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        """Source video indices for each stitched sequence."""
        return self._groups

    @property
    def group_keys(self) -> tuple[Hashable, ...]:
        """Grouping keys corresponding to each stitched sequence."""
        return self._group_keys

    @property
    def frame_offsets(self) -> tuple[NDArray[np.intp], ...]:
        """Frame boundary offsets of shape ``(M + 1,)`` for each stitched sequence."""
        return self._frame_offsets

    @property
    def track_id_offsets(self) -> tuple[NDArray[np.intp], ...]:
        """Track ID offsets of shape ``(M,)`` applied to each constituent."""
        return self._track_id_offsets

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata for the stitched view."""
        return self._metadata

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        """Return one stitched sequence as a tracking datum."""
        index = checked_index(index, len(self), "VideoStitch", "stitched video(s)")
        frame_tracks: list[Any] = []
        # Cache source streams during initial metadata/target access to avoid re-reading.
        held: list[Any] | None = []
        for offset, source_index in zip(self._track_id_offsets[index].tolist(), self._groups[index], strict=True):
            source_stream, target, _ = self._dataset[source_index]
            held.append(source_stream)
            tracks = target.frame_tracks
            frame_tracks.extend(tracks if offset == 0 else (_OffsetFrameTarget(track, offset) for track in tracks))

        def walk() -> Iterator[Any]:
            """Iterate streams, using cached streams for the first pass."""
            nonlocal held
            streams, held = held, None
            return self._iterate(index, streams)

        return _LazyStream(walk, f"stitched {index}"), _FrameTracksTarget(frame_tracks), self._stitched_metadata(index)

    def __iter__(self) -> Iterator[MultiobjectTrackingDatum]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return (
            f"VideoStitch(dataset={self._dataset!r}, group_by={self._group_by!r}, order_by={self._order_by!r}, "
            f"track_ids={self._track_ids!r}, timestamps={self._timestamps!r})"
        )

    def __str__(self) -> str:
        title = "VideoStitch Dataset"
        sep = "-" * len(title)
        return (
            f"{title}\n{sep}\n    videos: {len(self._sequences)}\n    stitched: {len(self)}\n"
            f"    track_ids: {self._track_ids}\n    timestamps: {self._timestamps}\n\n{self._dataset}"
        )

    # ------------------------------------------------------------------
    # Construction checks
    # ------------------------------------------------------------------

    def _check_geometry(self, key: Hashable, members: Sequence[int]) -> None:
        """Verify that constituents in a group have matching geometry metadata."""
        for name in _GEOMETRY_KEYS:
            declared = [
                (i, cast(Mapping[str, Any], self._sequences[i].metadata)[name])
                for i in members
                if name in self._sequences[i].metadata
            ]
            self._log_absent(name, len(declared), len(members))
            for i, value in declared[1:]:
                first_index, first = declared[0]
                if not _equal(value, first):
                    raise ValueError(
                        f"VideoStitch: group {key!r} stitches video {self._sequences[first_index].source_id!r} "
                        f"({name}={first!r}) with video {self._sequences[i].source_id!r} ({name}={value!r}); "
                        f"the videos of one sequence must agree on {name}."
                    )

    def _log_absent(self, name: str, declared: int, members: int) -> None:
        """Log a notice if any constituents omit a geometry key."""
        if declared < members and name not in self._logged_geometry:
            self._logged_geometry.add(name)
            _logger.info("VideoStitch: not every video declares %r, so it is not checked where it is absent.", name)

    def _lineage(self, members: Sequence[int]) -> list[Mapping[str, Any]] | None:
        """Return metadata for group members if they are segments of a single source video."""
        if len(members) < 2:
            return None
        metas = [cast(Mapping[str, Any], self._sequences[i].metadata) for i in members]
        if not all(all(name in meta for name in _LINEAGE_KEYS) for meta in metas):
            return None
        if len({meta["source_id"] for meta in metas}) != 1:
            return None
        return metas

    def _check_overlap(self, key: Hashable, members: Sequence[int]) -> None:
        """Verify segments of the same source video appear in order and do not overlap."""
        metas = self._lineage(members)
        if metas is None:
            return
        for previous, current in zip(metas, metas[1:], strict=False):
            self._check_adjacent(key, previous, current)

    @staticmethod
    def _check_adjacent(key: Hashable, previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
        """Validate order and non-overlap between two consecutive segments of the same source."""
        before = f"[{previous['start_frame']}, {previous['end_frame']})"
        after = f"[{current['start_frame']}, {current['end_frame']})"
        if current["start_frame"] < previous["start_frame"]:
            raise ValueError(
                f"VideoStitch: group {key!r} stitches segment {after} of video {current['source_id']!r} after "
                f"{before}, out of source order; pass order_by='start_frame' to play them as recorded."
            )
        if current["start_frame"] < previous["end_frame"]:
            raise ValueError(
                f"VideoStitch: group {key!r} stitches segments {before} and {after} of video "
                f"{current['source_id']!r}, which overlap; stitching them would repeat frames."
            )
        if current["start_frame"] > previous["end_frame"]:
            _logger.info(
                "VideoStitch: group %r leaves a gap of %d frame(s) between segments of video %r.",
                key,
                current["start_frame"] - previous["end_frame"],
                current["source_id"],
            )

    def _check_shared_ids(self, key: Hashable, members: Sequence[int], id_sets: Sequence[set[int]]) -> None:
        """Verify that constituents in a group do not share track IDs when track_ids='error'."""
        if self._track_ids != "error":
            return
        owner: dict[int, int] = {}
        for i in members:
            shared = sorted(id_sets[i] & owner.keys())
            if shared:
                raise ValueError(
                    f"VideoStitch: group {key!r} stitches videos {self._sequences[owner[shared[0]]].source_id!r} "
                    f"and {self._sequences[i].source_id!r}, which share track id(s) {shared[:5]}; pass "
                    "track_ids='offset' to keep them apart, or 'preserve' if they are the same tracks."
                )
            for track_id in id_sets[i]:
                owner.setdefault(track_id, i)

    def _offsets_for(self, members: Sequence[int], largest: Sequence[int]) -> NDArray[np.intp]:
        """Calculate cumulative track ID offsets for constituents in a group."""
        if self._track_ids != "offset":
            return np.zeros(len(members), dtype=np.intp)
        steps = [largest[i] + 1 for i in members]
        return np.concatenate(([0], np.cumsum(steps[:-1]))).astype(np.intp)

    # ------------------------------------------------------------------
    # Datum assembly
    # ------------------------------------------------------------------

    def _omitted(self, key: str, reason: str) -> None:
        """Log an omitted metadata key once per view."""
        if key == "id" or key in self._logged_keys:
            return
        self._logged_keys.add(key)
        _logger.info("VideoStitch: metadata key %r is omitted from stitched videos because %s.", key, reason)

    def _stitched_metadata(self, index: int) -> DatumMetadata:
        """Construct merged metadata for a stitched sequence."""
        infos = [self._sequences[i] for i in self._groups[index]]
        values = [
            {key: value for key, value in cast(Mapping[str, Any], info.metadata).items() if key not in _NOT_MERGED}
            for info in infos
        ]
        merged = _merge_constituents(values, [info.n_frames for info in infos], self._omitted)
        if all("size" in info.metadata for info in infos):
            merged["size"] = sum(int(cast(Mapping[str, Any], info.metadata)["size"]) for info in infos)
        merged["id"] = index
        merged["n_sources"] = len(infos)
        return cast(DatumMetadata, merged)

    def _iterate(self, index: int, streams: Sequence[Any] | None = None) -> Iterator[Any]:
        """Yield frames across all constituent videos with adjusted indices and timestamps."""
        offset = self._timestamps == "offset"
        chain = _Chain(time=_Timeline(offset, integer=False), pts=_Timeline(offset, integer=True))
        for position, source_index in enumerate(self._groups[index]):
            stream = self._dataset[source_index][0] if streams is None else streams[position]
            yield from self._play(index, self._sequences[source_index], chain, stream)

    def _play(self, index: int, info: SequenceInfo, chain: _Chain, stream: Any) -> Iterator[Any]:
        """Iterate frames of a single constituent within a stitched sequence."""
        frames = iter(stream)
        for position in range(info.n_frames):
            frame = next(frames, _EXHAUSTED)
            if frame is _EXHAUSTED:
                raise ValueError(short_stream("VideoStitch", info.source_id, info.n_frames, position))
            if position == 0:
                self._begin(index, info, frame, chain)
            chain.time.observe(timing_of(frame, "time_s"))
            chain.pts.observe(timing_of(frame, "pts"))
            yield _ShiftedFrame(frame, chain.running, chain.time.shift, int(chain.pts.shift))
            chain.running += 1
        if next(frames, _EXHAUSTED) is not _EXHAUSTED:
            raise ValueError(long_stream("VideoStitch", info.source_id, info.n_frames))
        chain.time.end(info.n_frames)
        chain.pts.end(info.n_frames)

    def _begin(self, index: int, info: SequenceInfo, frame: Any, chain: _Chain) -> None:
        """Initialize constituent frame size and timing shifts at its first frame."""
        chain.size = self._checked_size(chain.size, frame, info, self._group_keys[index])
        chain.time.begin(timing_of(frame, "time_s"))
        chain.pts.begin(timing_of(frame, "pts"))
        self._report_boundary(index, info, chain)

    @staticmethod
    def _checked_size(
        reference: tuple[int | None, int | None] | None, frame: Any, info: SequenceInfo, key: Hashable
    ) -> tuple[int | None, int | None]:
        """Verify that constituent frame dimensions match the reference sequence size."""
        size = frame_size(frame)
        if reference is None:
            return size
        if None not in size and None not in reference and size != reference:
            raise ValueError(
                f"VideoStitch: group {key!r} stitches video {info.source_id!r} with a frame size of "
                f"{size[0]}x{size[1]} after frames of {reference[0]}x{reference[1]}; the videos of one "
                "sequence must have one frame size."
            )
        return reference

    def _report_boundary(self, index: int, info: SequenceInfo, chain: _Chain) -> None:
        """Log warnings for non-monotonic timestamps or unestimated frame intervals."""
        channels = [name for name, line in (("time_s", chain.time), ("pts", chain.pts)) if line.backwards]
        if channels and index not in self._warned_backwards:
            self._warned_backwards.add(index)
            _logger.warning(
                "VideoStitch: %s steps backwards where video %r begins; the stitched videos do not "
                "share a timeline, and Metadata's temporal reductions order frames by time_s then pts. "
                "Pass timestamps='offset' for a continuous timeline.",
                " and ".join(channels),
                info.source_id,
            )
        if chain.time.unknown_interval or chain.pts.unknown_interval:
            _logger.info(
                "VideoStitch: no frame interval is known where video %r begins, so its first frame shares "
                "the previous frame's timestamp.",
                info.source_id,
            )


def _check_policies(track_ids: str, timestamps: str) -> None:
    """Validate policy parameter values."""
    if track_ids not in ("offset", "preserve", "error"):
        raise ValueError(f"track_ids must be 'offset', 'preserve' or 'error'; got {track_ids!r}.")
    if timestamps not in ("source", "offset"):
        raise ValueError(f"timestamps must be 'source' or 'offset'; got {timestamps!r}.")

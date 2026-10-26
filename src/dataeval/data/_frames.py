"""Presenting a multi-object-tracking dataset as an object-detection dataset of frames."""

__all__ = []

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.data._selectors import (
    AllFrames,
    FrameCandidate,
    FrameInput,
    FrameSelector,
    FrameVerdict,
    SequenceInfo,
    _Planned,
    _StageTrace,
    sequence_infos,
)
from dataeval.data._video_proxies import checked_index, inherited_metadata
from dataeval.flags import ImageStats
from dataeval.protocols import (
    AnnotatedDataset,
    ArrayLike,
    DatasetMetadata,
    DatumMetadata,
    MultiobjectTrackingDataset,
    MultiobjectTrackingTarget,
    ObjectDetectionDatum,
    ObjectDetectionTarget,
    SingleFrameObjectTrackingTarget,
)
from dataeval.types import FactorLevel, SourceIndex
from dataeval.types._target import instance_arrays
from dataeval.types._track import frame_number
from dataeval.utils._array import as_numpy
from dataeval.utils.data import requires_maite_dataset
from dataeval.utils.preprocessing import BoundingBox, normalize_image_shape

_logger = get_logger(__name__)

RESERVED_FACTORS: frozenset[str] = frozenset({
    "id",
    "source_id",
    "sequence",
    "frame",
    "time_s",
    "pts",
    "frames_represented",
    "seconds_represented",
    "sequence_position",
    "sequence_n_frames",
    *FactorLevel.__args__,  # type: ignore[attr-defined]
})
"""Datum-metadata names :class:`SequenceFrames` writes itself, plus the factor level names.

A selector naming one of these in :attr:`~dataeval.data.FrameVerdict.factors` is rejected rather
than silently overwritten -- the same rule the metadata structurers apply to a dataset key that
collides with a derived one.
"""

_Selected = tuple[
    int, Any, SingleFrameObjectTrackingTarget, float | None, int | None, Mapping[str, Any], float | None, bool
]
"""One kept frame: position, raw frame, target, timings, factors, any declared weight, and its unmeasured mark."""


def _same_selection(first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]) -> bool:
    """Whether two walks kept the same positions with the same metadata and marks (NaN equals NaN)."""

    def same(a: Any, b: Any) -> bool:
        return bool(a == b or (a != a and b != b))

    return len(first) == len(second) and all(
        a["position"] == b["position"]
        and a["unmeasured"] == b["unmeasured"]
        and a["metadata"].keys() == b["metadata"].keys()
        and all(same(a["metadata"][key], b["metadata"][key]) for key in a["metadata"])
        for a, b in zip(first, second, strict=True)
    )


@dataclass(frozen=True, eq=False)
class _SequenceRecord:
    """What the first complete walk of one sequence decided, so every later walk replays it.

    Attributes
    ----------
    entries : tuple[dict[str, Any], ...]
        One record per kept frame, in order: its sequence, position, metadata, and whether the
        selector kept it unmeasured (:attr:`~dataeval.data.FrameVerdict.unmeasured`). No pixels
        and no target.
    frame_numbers : NDArray[np.intp]
        The number of the frame at every position of the sequence, kept or not, which a later walk
        checks its own frames against.
    trace : tuple[_StageTrace, ...]
        What each stage of the selection decided, one per stage, for
        :meth:`SequenceFrames.selection_report`.
    """

    entries: tuple[dict[str, Any], ...]
    frame_numbers: NDArray[np.intp]
    trace: tuple[_StageTrace, ...]

    @cached_property
    def repeated(self) -> frozenset[int]:
        """Frame numbers held at more than one position, which no ``unit`` address names alone."""
        numbers, counts = np.unique(self.frame_numbers, return_counts=True)
        return frozenset(numbers[counts > 1].tolist())


class SelectionReport(NamedTuple):
    """What each stage of a frame selection did, per video and per class.

    Returned by :meth:`SequenceFrames.selection_report`. A selector is one stage, and a
    :class:`~dataeval.data.Chain` one per stage. This is an account of what each stage did, not a
    test of whether the selection is representative.

    Attributes
    ----------
    stages : pl.DataFrame
        One row per ``(sequence, stage)``. ``offered``, ``kept``, ``represented`` and ``excluded``
        count frames of the stage's own input, which are source frames only for stage 0: inside a
        :class:`~dataeval.data.Chain`, a frame a later stage drops may stand for several source
        frames. ``source_frames_excluded`` is in source frames.

        - ``sequence`` (*Int64*) -- the sequence's index in the source dataset.
        - ``source_id`` (*String*) -- the sequence datum's own ``id``, as a string.
        - ``stage`` (*Int64*) -- the stage's place in the selection, from 0.
        - ``selector`` (*String*) -- the stage's repr.
        - ``offered`` (*Int64*) -- the frames reaching the stage: every frame of the sequence for
          stage 0, and otherwise each the stage before kept, a frame it kept twice counting twice.
        - ``kept`` (*Int64*) -- the frames the stage kept, a frame it kept twice counting twice.
        - ``represented``, ``excluded`` (*Int64*) -- the frames offered that the stage never kept:
          ``excluded`` under a stage that excludes (:attr:`~dataeval.data.FrameSelector.excludes`),
          ``represented`` otherwise, and the other 0.
        - ``unmeasured`` (*Int64*) -- the frames the stage itself kept without measuring them
          (:attr:`~dataeval.data.FrameVerdict.unmeasured`). A presented frame is marked when any
          stage marked it (:attr:`SequenceFrames.unmeasured`), so a later stage's count here
          leaves out frames an earlier one marked.
        - ``source_frames_excluded`` (*Float64*) -- the source frames the stage excluded: the
          source frames the frames reaching it stood for (the sequence's frame count for stage 0)
          less those its kept frames stand for. A thinning stage that keeps any frame and declares
          no weight passes them all on, so reads 0. Per sequence these sum to the sequence's frame
          count less the ``frames_represented`` of the frames presented.

    labels : pl.DataFrame
        One row per ``(sequence, stage, label)``, for each label that some frame offered to the
        stage holds:

        - ``sequence``, ``source_id``, ``stage``, ``selector`` -- as in ``stages``.
        - ``label`` (*Int64*) -- the class label.
        - ``frames_offered``, ``frames_kept`` (*Int64*) -- the frames offered and kept holding at
          least one detection of the label, a frame offered or kept twice counting twice.
        - ``detections_offered``, ``detections_kept`` (*Int64*) -- the label's detections in them.

    Examples
    --------
    How much of class 7 survives each stage:

    >>> import polars as pl
    >>> from dataeval.data import Chain, Redundancy, SequenceFrames, Stride
    >>> frames = SequenceFrames(mot_dataset, Chain(Redundancy(), Stride(5)))  # doctest: +SKIP
    >>> frames.selection_report().labels.filter(pl.col("label") == 7)  # doctest: +SKIP
    """

    stages: pl.DataFrame
    labels: pl.DataFrame


def _label_counts(target: MultiobjectTrackingTarget) -> tuple[NDArray[np.intp], NDArray[np.int64]]:
    """Return the labels a sequence holds, and how many detections of each every frame holds."""
    labels = [instance_arrays(frame)[0] for frame in target.frame_tracks]
    classes, column = np.unique(np.concatenate([np.empty(0, dtype=np.intp), *labels]), return_inverse=True)
    counts = np.zeros((len(labels), len(classes)), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(len(labels)), [len(held) for held in labels]), column), 1)
    return classes, counts


class SequenceFrames(AnnotatedDataset[ObjectDetectionDatum]):
    """Present a multi-object-tracking dataset as an object-detection dataset of frames.

    One frame becomes one datum. :class:`~dataeval.data.DetectionCrops` does the same job one
    level down -- object detection presented as image classification -- and this is its analog
    one level up, so every per-image tool reaches video frames unchanged:
    :func:`~dataeval.core.image_stats` and everything built on it,
    :class:`~dataeval.Embeddings`, :class:`~dataeval.data.View`.

    A frame's target is its own :obj:`~dataeval.protocols.SingleFrameObjectTrackingTarget`, passed
    through rather than rebuilt: it already carries ``boxes``, ``labels`` and ``scores``, so it
    satisfies :obj:`~dataeval.protocols.ObjectDetectionTarget` as-is, and passing it whole keeps
    ``track_ids`` reachable instead of dropping them.

    Parameters
    ----------
    dataset : MultiobjectTrackingDataset
        The source dataset. Each datum is a MAITE ``(VideoStream, MultiobjectTrackingTarget,
        metadata)`` 3-tuple.
    selector : FrameSelector or None, default None
        Which frames take part, and what to record about each. None keeps every frame
        (:class:`~dataeval.data.AllFrames`). See :class:`~dataeval.data.FrameSelector`.

    Attributes
    ----------
    n_source_frames : int
        Total frames across every sequence, read from the targets and so known without decoding.
    n_dropped : int
        Frames the selector did not keep. Only known once every sequence has been walked; ``0``
        until then for a selector that cannot plan.

    Raises
    ------
    MaiteShapeError
        If ``dataset`` is not a multi-object-tracking dataset.
    ValueError
        If a selector names a reserved factor, if a sequence's stream disagrees with its target on
        frame count, comes back already spent on a later walk, or numbers its frames differently
        from the walk that recorded it, if a selector yields a verdict out of order without
        declaring ``two_pass``, or if a selector's weights for a sequence total more than the
        frames it holds.

    See Also
    --------
    :class:`~dataeval.data.FrameSelector` : The frame-selection extension point
    :class:`~dataeval.data.DetectionCrops` : The same reframing one level down

    Notes
    -----
    **Streaming is the contract.** A :obj:`~dataeval.protocols.VideoStream` is an *iterable* of
    frames, not an indexable one, so seeking to frame *k* means decoding *k* frames. :meth:`stream`
    walks each sequence exactly once and is what every bulk consumer should use.
    ``__getitem__`` exists so this genuinely satisfies :obj:`~dataeval.protocols.Dataset` -- it
    keeps a one-sequence cursor, making forward access amortized constant and backward access
    proportional to the frames before it.

    **The selection is fixed by the first walk.** The first complete walk of a sequence -- by
    :meth:`stream`, iteration, ``len()`` or ``__getitem__`` -- records which frames were kept and
    their metadata, and every later walk replays that record without consulting the selector
    again. Every consumer of one view therefore sees the same frames, even under a stateful
    selector, and a ``two_pass`` selector pays its second pass once. A replay checks each frame's
    number against the record and raises if the source changed in between. A walk abandoned
    partway records nothing for the sequence it left. Across separate views, reproducibility is
    the selector's own (a seed, say).

    Replaying needs each ``dataset[i]`` to return a stream that can be walked from its first
    frame, independently of any earlier stream for the same index. A walk that finds the
    stream already spent, by an earlier walk of this view or by another reader such as
    :class:`~dataeval.Metadata`, raises an error naming that contract.

    Each frame's metadata is a plain ``dict`` conforming to
    :obj:`~dataeval.protocols.DatumMetadata`, carrying:

    - ``id`` (*int*) -- the frame's position in this view.
    - ``source_id`` (*int | str*) -- the sequence datum's own ``id``, so a flagged frame resolves
      to the right video after the source has been filtered or reordered.
    - ``sequence`` (*int*) -- the sequence's index in the source dataset.
    - ``frame`` (*int*) -- the frame's own ``frame_index`` in the source video.
    - ``time_s``, ``pts`` -- the frame's own timings, where it declares them.
    - ``frames_represented`` (*float*) -- how many source frames this one stands for.
    - ``seconds_represented`` (*float*) -- where timings allow, the span of the timeline from this
      kept frame to the next, the first starting at the sequence's first timestamp and the last
      running to its final one. A span of the timeline, not a mass: it does not leave out the
      time an excluded frame covers.
    - ``sequence_position`` (*float*) -- position within the sequence, normalized to ``[0, 1]``.
    - ``sequence_n_frames`` (*int*) -- how many frames the source sequence holds.
    - anything a selector put in :attr:`~dataeval.data.FrameVerdict.factors`.

    Because this presents object detection, ``Metadata(SequenceFrames(ds))`` structures as
    ``unit`` = frame and ``instance`` = detection, so every one of those keys is a ``unit``-level
    factor through the ordinary path. ``sequence`` as a factor is what
    ``split_on="sequence"`` needs to keep every frame of one video on one side of a split.

    Frames of one video are not independent samples, so :class:`~dataeval.bias.Parity` clusters
    them by their source video, and :class:`~dataeval.bias.Balance` counts a video as one
    observation of any factor that is constant within it. The grouping holds through a
    :class:`~dataeval.data.View` and through :func:`~dataeval.data.merge_datasets`, where frame views
    built over the same dataset object share its videos.

    **Weights matter more than they look.** A frame kept out of 300 near-identical ones is not one
    observation. ``frames_represented`` records what it stands for so a reader can weight by it;
    nothing in DataEval consumes a sample weight yet, so an unweighted read of a thinned view
    describes *distinct content* rather than the source stream.

    ``frames_represented`` is exact source-frame mass. Every frame a selector drops is either
    *represented* by a kept frame, as a thinning selector's are, or *excluded*, as a filter's are
    (:attr:`~dataeval.data.FrameSelector.excludes`). Per sequence, the kept frames'
    ``frames_represented`` therefore sum to its frame count less the frames excluded. Under a
    thinning selector the first kept frame also stands for every frame before it, so a selection
    that skips a sequence's opening frames still accounts for them. A sequence whose weights total
    more than its frame count raises.

    Per-sequence aggregates are a group-by away and are deliberately not denormalized onto every
    row::

        md.dataframe.group_by("sequence").agg(pl.len().alias("sequence_n_selected"))

    Examples
    --------
    Frame-level statistics over a video dataset:

    >>> from dataeval.data import SequenceFrames, Stride
    >>> frames = SequenceFrames(mot_dataset, Stride(5))  # doctest: +SKIP
    >>> stats = image_stats(frames, stats=ImageStats.HASH)  # doctest: +SKIP
    """

    @requires_maite_dataset("dataset", expected="multiobject_tracking")
    def __init__(
        self,
        dataset: MultiobjectTrackingDataset,
        selector: FrameSelector | None = None,
    ) -> None:
        self._dataset = dataset
        self._selector = AllFrames() if selector is None else selector
        self._reads_pixels = bool(self._selector.needs & FrameInput.PIXELS)

        # Frame counts come from the targets, which cost no decode: MAITE requires one frame
        # target per frame, and the structurers already raise when a stream disagrees.
        self._sequences: list[SequenceInfo] = [info for info, _ in sequence_infos(dataset)]
        self.n_source_frames: int = sum(info.n_frames for info in self._sequences)

        # A planning selector is authoritative through `plan`, so the whole index map is known
        # here without decoding anything and `__len__` is free. Normalized to ascending, unique,
        # in-range positions -- which is what the replay walk emits -- so the length promised here
        # is the length the walk delivers even for a plan that repeats or overruns a sequence.
        # Planned once per sequence, through the private `_plan_weighted`, so any weights the
        # selector declares belong to these very positions even if it plans differently each time.
        planned = [self._selector._plan_weighted(info) for info in self._sequences]
        self._planned: list[NDArray[np.intp]] | None = None
        self._planned_weights: list[NDArray[np.float64] | None] = []
        # Kept so a planned view reports its selection without walking anything.
        self._planned_traces: list[tuple[_StageTrace, ...]] = []
        if all(plan is not None for plan in planned):
            normalized = [
                self._normalized_plan(cast(_Planned, plan), info)
                for plan, info in zip(planned, self._sequences, strict=True)
            ]
            self._planned = [positions for positions, _, _ in normalized]
            self._planned_weights = [weights for _, weights, _ in normalized]
            self._planned_traces = [trace for _, _, trace in normalized]
        # The execution record: each sequence's selection, fixed by its first complete walk.
        self._records: dict[int, _SequenceRecord] = {}
        self._realized: list[dict[str, Any]] | None = None
        # Per sequence, how many detections precede each frame position, read from the targets.
        self._detection_offsets: dict[int, NDArray[np.intp]] = {}
        self.n_dropped: int = 0
        if self._planned is not None:
            self.n_dropped = self.n_source_frames - sum(len(positions) for positions in self._planned)

        self._metadata = inherited_metadata(dataset, "frames")

        self._cursor_sequence: int | None = None
        self._cursor: Iterator[Any] | None = None
        self._cursor_position: int = -1
        self._cursor_frame: Any = None
        self._cursor_target: Any = None

        _logger.debug(
            "SequenceFrames: %d sequence(s), %d source frame(s), selector=%r, planned=%s",
            len(self._sequences),
            self.n_source_frames,
            self._selector,
            self._planned is not None,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    @property
    def source(self) -> MultiobjectTrackingDataset:
        """The tracking dataset this wraps -- one link up the chain.

        Named to match :attr:`~dataeval.data.View.source`, so a mixed wrapping chain is walkable
        through one public attribute and an invalidating operation inside it stays visible.
        """
        return self._dataset

    @property
    def selector(self) -> FrameSelector:
        """The :class:`~dataeval.data.FrameSelector` deciding which frames this view presents."""
        return self._selector

    @property
    def invalidates(self) -> ImageStats:
        """Statistics the selection makes describe itself rather than the data.

        Deferred to the selector rather than declared here: presenting a frame does not touch its
        pixels, so a frame view invalidates exactly what its selector says it does -- which is
        :attr:`~dataeval.flags.ImageStats.NONE` for every selector that only chooses.
        """
        return self._selector.invalidates

    @property
    def metadata(self) -> DatasetMetadata:
        """MAITE dataset metadata for the frame view (id and any inherited index2label)."""
        return self._metadata

    def __len__(self) -> int:
        """Return how many frames the view presents.

        Free for a selector that plans; otherwise this walks every sequence not yet recorded, and
        says so in the log. After a full :meth:`stream` it walks nothing.
        """
        if self._planned is not None:
            return sum(len(positions) for positions in self._planned)
        return len(self._realize())

    def __getitem__(self, index: int) -> ObjectDetectionDatum:
        """Return one frame as an object-detection datum."""
        entries = self._realize()
        index = checked_index(index, len(entries), "SequenceFrames", "frame(s)")
        entry = entries[index]
        frame, target = self._seek(entry["sequence"], entry["position"])
        pixels = normalize_image_shape(as_numpy(frame.pixels))
        return pixels, target, cast(DatumMetadata, {**entry["metadata"], "id": index})

    def __iter__(self) -> Iterator[ObjectDetectionDatum]:
        """Iterate frames in order, decoding each sequence once."""
        for position, (pixels, target, metadata) in enumerate(self.stream()):
            yield pixels, target, cast(DatumMetadata, {**metadata, "id": position})

    def __repr__(self) -> str:
        return f"SequenceFrames(dataset={self._dataset!r}, selector={self._selector!r})"

    def __str__(self) -> str:
        title = "SequenceFrames Dataset"
        sep = "-" * len(title)
        recorded = len(self._records) == len(self._sequences)
        length = len(self) if self._planned is not None or recorded else "unrealized"
        return (
            f"{title}\n{sep}\n    sequences: {len(self._sequences)}\n"
            f"    source frames: {self.n_source_frames}\n    selected: {length}\n"
            f"    selector: {self._selector!r}\n\n{self._dataset}"
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self) -> Iterator[tuple[NDArray[Any], SingleFrameObjectTrackingTarget, DatumMetadata]]:
        """Decode every selected frame of every sequence, once, in order.

        The primary interface. Splitting it with :func:`itertools.tee` gives
        :func:`~dataeval.core.image_stats` the pixel and box iterators it takes, the way
        :func:`~dataeval.data.unzip_dataset` does for object detection.

        Yields
        ------
        tuple[NDArray, SingleFrameObjectTrackingTarget, DatumMetadata]
            One selected frame's pixels in ``(C, H, W)``, its detections, and its metadata.
            ``id`` counts frames across the whole view.
        """
        position = 0
        for info in self._sequences:
            for entry, frame, target in self._walk(info):
                pixels = normalize_image_shape(as_numpy(frame.pixels))
                yield pixels, target, cast(DatumMetadata, {**entry["metadata"], "id": position})
                position += 1

    @staticmethod
    def boxes(frame: ArrayLike, target: ObjectDetectionTarget) -> list[BoundingBox]:
        """Convert one frame's detections into :class:`~dataeval.utils.preprocessing.BoundingBox`.

        The box iterator :func:`~dataeval.core.image_stats` takes beside its images, built from
        what :meth:`stream` yields.

        Parameters
        ----------
        frame : ArrayLike
            The frame's pixels, whose shape bounds the boxes.
        target : ObjectDetectionTarget
            The frame's detections. A frame's own tracking target satisfies this.

        Returns
        -------
        list[BoundingBox]
            One per detection, in absolute-pixel ``[x0, y0, x1, y1]``.

        Examples
        --------
        >>> from itertools import tee
        >>> pixels, targets = tee(frames.stream(), 2)  # doctest: +SKIP
        >>> stats = image_stats(  # doctest: +SKIP
        ...     (item[0] for item in pixels),
        ...     boxes=(SequenceFrames.boxes(item[0], item[1]) for item in targets),
        ... )
        """
        shape = as_numpy(frame).shape
        values = as_numpy(target.boxes).reshape(-1, 4).astype(np.float64)
        return [BoundingBox(box[0], box[1], box[2], box[3], image_shape=shape) for box in values]

    @property
    def _source_items(self) -> NDArray[np.intp]:
        """The source sequence behind each frame -- the private hook ``root_items`` walks."""
        return self.frame_map[:, 0]

    @property
    def frame_map(self) -> NDArray[np.intp]:
        """Shape ``(F, 2)`` -- the ``(sequence index, frame position)`` behind each view position."""
        located = self._located()
        if not located:
            return np.empty((0, 2), dtype=np.intp)
        return np.array(located, dtype=np.intp)

    @property
    def unmeasured(self) -> NDArray[np.bool_]:
        """Shape ``(F,)`` -- whether the frame behind each view position was kept without evidence.

        Aligned with :attr:`frame_map`. True where any stage of the selector kept the frame because
        it could not measure it (:attr:`~dataeval.data.FrameVerdict.unmeasured`), so it is
        cumulative across a :class:`~dataeval.data.Chain`. The ``unmeasured`` column of
        :meth:`selection_report` is per stage, counting the frames each stage itself marked.

        A selector that plans decides from structure alone and marks nothing, so a planned view
        answers without decoding. Otherwise this is read from the execution record, as
        :attr:`frame_map` is.

        Examples
        --------
        The frames a selection kept without judging them:

        >>> frames = SequenceFrames(mot_dataset, FrameRate(5.0))  # doctest: +SKIP
        >>> frames.frame_map[frames.unmeasured]  # doctest: +SKIP
        """
        if self._planned is not None:
            return np.zeros(sum(len(positions) for positions in self._planned), dtype=np.bool_)
        return np.array([entry["unmeasured"] for entry in self._realize()], dtype=np.bool_)

    @property
    def track_map(self) -> NDArray[np.intp]:
        """Shape ``(D, 2)`` -- the ``(view position, track id)`` behind each detection presented.

        Rows run in the order :func:`~dataeval.core.image_stats` measures detections: every
        detection of view position 0, then of view position 1, and so on. A detection MAITE marks
        as unlinked carries its ``-1`` through rather than being dropped, so a row here lines up
        with a per-target statistic row whether or not the detection belongs to a track.

        Read from the targets, which cost no decode -- the same source as
        :attr:`n_source_frames`.
        """
        located = self._located()
        if not located:
            return np.empty((0, 2), dtype=np.intp)
        rows: list[NDArray[np.intp]] = []
        held, tracks = -1, None
        for position, (sequence, frame) in enumerate(located):
            if sequence != held:
                # `located` runs sequence by sequence, so one sequence's targets are all that has
                # to be held at a time -- the same discipline the frame walk keeps.
                held, tracks = sequence, self._dataset[sequence][1].frame_tracks
            ids = as_numpy(cast(Any, tracks)[frame].track_ids).reshape(-1).astype(np.intp)
            rows.append(np.stack((np.full(len(ids), position, dtype=np.intp), ids), axis=1))
        return np.concatenate(rows) if rows else np.empty((0, 2), dtype=np.intp)

    @property
    def sequence_offsets(self) -> NDArray[np.intp]:
        """Shape ``(S + 1,)`` -- view-position boundaries, so one sequence's frames are a slice."""
        counts = np.zeros(len(self._sequences), dtype=np.intp)
        for sequence, _ in self._located():
            counts[sequence] += 1
        return np.concatenate(([0], np.cumsum(counts))).astype(np.intp)

    @property
    def source_indices(self) -> tuple[SourceIndex, ...]:
        """The source address of each frame this view presents.

        Returns
        -------
        tuple of SourceIndex
            One ``SourceIndex(sequence, frame_index, "unit")`` per view position, in view order,
            in the coordinates of :attr:`source`: the frame's own ``frame_index`` is the
            ``unit_index`` that ``Metadata(frames.source)`` names its frame rows by.

        See Also
        --------
        to_source : Translate any frame-view address, a detection's included.
        frame_map : The same frames as ``(sequence index, frame position)`` pairs.

        Notes
        -----
        The frame view's analog of :attr:`~dataeval.data.DetectionCrops.source_indices`. Read
        from the execution record, so it walks only the sequences no walk has recorded yet. Unlike
        :meth:`to_source` it does not refuse a frame number its sequence repeats;
        :class:`~dataeval.Metadata` refuses such an address when it is attached.

        Examples
        --------
        Attach what a selector recorded about each frame to the source dataset's frame rows:

        >>> frames = SequenceFrames(mot_dataset, Stride(5))  # doctest: +SKIP
        >>> represented = [meta["frames_represented"] for _, _, meta in frames.stream()]  # doctest: +SKIP
        >>> md = Metadata(frames.source)  # doctest: +SKIP
        >>> md.add_factors({"frames_represented": represented}, source_index=frames.source_indices)  # doctest: +SKIP
        """
        return tuple(SourceIndex(entry["sequence"], entry["metadata"]["frame"], "unit") for entry in self._realize())

    def to_source(self, addresses: Iterable[SourceIndex]) -> list[SourceIndex]:  # noqa: C901
        """Translate frame-view addresses to the source addresses of the same frames and detections.

        :func:`~dataeval.core.image_stats`, :class:`~dataeval.quality.Outliers` and every other
        tool run over this view address their results in its coordinates. This is the explicit
        step back to the coordinates of :attr:`source`, whose :class:`~dataeval.Metadata` the
        results belong on.

        Parameters
        ----------
        addresses : Iterable[SourceIndex]
            Frame-view addresses: ``SourceIndex(i)`` for the frame at view position ``i``, and
            ``SourceIndex(i, k)`` for that frame's ``k``-th detection.

        Returns
        -------
        list[SourceIndex]
            One address per input, in order, in the coordinates of :attr:`source`:

            - ``SourceIndex(i)`` becomes ``SourceIndex(sequence, frame_index, "unit")``.
            - ``SourceIndex(i, k)`` becomes ``SourceIndex(sequence, target_index, "instance")``,
              where ``target_index`` counts detections across the whole sequence, as
              :class:`~dataeval.Metadata` does.

        Raises
        ------
        ValueError
            If an address already names a level, if ``k`` is beyond the frame's detections, or if
            the frame's ``frame_index`` appears at another position of its sequence, which leaves
            its ``unit`` address naming more than one row.
        IndexError
            If ``i`` is out of range for this view.

        See Also
        --------
        source_indices : Every frame's source address at once.

        Notes
        -----
        Answered from the execution record and the targets, so nothing is decoded for a sequence
        some walk has recorded, and a sequence none has is walked once, with a log line. Under a
        selector that plans, only the sequences the addresses name are walked. Under one that
        cannot, every sequence is, because where a view position falls depends on how many frames
        each earlier sequence kept and its range on how many all of them did.

        Examples
        --------
        Attach per-frame statistics to the source dataset's frame and detection rows:

        >>> frames = SequenceFrames(mot_dataset, Stride(5))  # doctest: +SKIP
        >>> stats = image_stats(frames, stats=ImageStats.VISUAL_BRIGHTNESS)  # doctest: +SKIP
        >>> md = Metadata(frames.source)  # doctest: +SKIP
        >>> md.add_factors(stats, source_index=frames.to_source(stats["source_index"]))  # doctest: +SKIP
        """
        addresses = list(addresses)
        if not addresses:
            return []
        starts = self.sequence_offsets
        translated: list[SourceIndex] = []
        for address in addresses:
            if address.level is not None:
                raise ValueError(
                    f"SequenceFrames: {address!r} already names a level, so it is not a frame-view "
                    "address. to_source translates SourceIndex(i), the frame at view position i, and "
                    "SourceIndex(i, k), that frame's k-th detection."
                )
            if address.item < 0:
                raise IndexError(
                    f"SequenceFrames: {address!r} is negative. An address is never negative; "
                    "to_source takes the view positions a statistic was measured at."
                )
            index = checked_index(address.item, int(starts[-1]), "SequenceFrames", "frame(s)")
            sequence = int(np.searchsorted(starts, index, side="right")) - 1
            record = self._record_of(sequence)
            entry = record.entries[index - starts[sequence]]
            number = entry["metadata"]["frame"]
            if address.key is None:
                if number in record.repeated:
                    raise ValueError(
                        f"SequenceFrames: {address!r} is ambiguous: sequence {sequence} holds frame "
                        f"{number} at more than one position, so SourceIndex({sequence}, {number}, "
                        "'unit') names more than one row."
                    )
                translated.append(SourceIndex(sequence, number, "unit"))
                continue
            offsets = self._offsets_of(sequence)
            first, after = int(offsets[entry["position"]]), int(offsets[entry["position"] + 1])
            if not 0 <= address.key < after - first:
                raise ValueError(
                    f"SequenceFrames: {address!r} names detection {address.key} of view frame "
                    f"{index}, but that frame (frame {number} of sequence {sequence}) has "
                    f"{after - first} detection(s)."
                )
            translated.append(SourceIndex(sequence, first + address.key, "instance"))
        return translated

    def selection_report(self) -> SelectionReport:
        """Account for what each stage of the selection did, per video and per class.

        For each sequence and stage: the frames it was offered and kept, those it dropped as
        represented or excluded, and those it kept unmeasured; and per class label, the frames
        holding it and its detections, offered and kept. A selection that quietly removed a class,
        or kept most of a video without judging it, shows here. This is an account of what each
        stage did, not a test of whether the selection is representative.

        Returns
        -------
        SelectionReport
            The per-stage and per-label tables, described on
            :class:`~dataeval.data.SelectionReport`.

        Notes
        -----
        Built from the execution record and the targets, never from pixels. A view whose selector
        plans reports without walking anything. Otherwise each sequence no walk has recorded yet
        is walked once, with a log line, as for ``len()``; after a full :meth:`stream` nothing is.

        Examples
        --------
        >>> from dataeval.data import Chain, HasLabels, SequenceFrames, Stride
        >>> frames = SequenceFrames(mot_dataset, Chain(HasLabels([7]), Stride(5)))  # doctest: +SKIP
        >>> frames.selection_report().stages  # doctest: +SKIP
        """
        stage_rows: list[tuple[Any, ...]] = []
        label_rows: list[tuple[Any, ...]] = []
        for info in self._sequences:
            trace = self._planned_traces[info.index] if self._planned is not None else self._record_of(info.index).trace
            classes, counts = _label_counts(cast(MultiobjectTrackingTarget, info.target))
            offered, mass_in = np.arange(info.n_frames, dtype=np.intp), float(info.n_frames)
            for number, stage in enumerate(trace):
                head = (info.index, str(info.source_id), number, repr(stage.selector))
                dropped = len(offered) - len(np.unique(stage.ordinals))
                excluded = dropped if stage.selector.excludes else 0
                counted = (
                    len(offered),
                    len(stage.positions),
                    dropped - excluded,
                    excluded,
                    int(stage.unmeasured.sum()),
                    mass_in - stage.mass_out,
                )
                stage_rows.append((*head, *counted))
                # A frame offered or kept twice counts twice, as indexing by position repeats its row.
                into, out = counts[offered], counts[stage.positions]
                columns = np.stack(
                    ((into > 0).sum(axis=0), (out > 0).sum(axis=0), into.sum(axis=0), out.sum(axis=0)), 1
                )
                label_rows.extend((*head, int(classes[j]), *columns[j].tolist()) for j in np.flatnonzero(columns[:, 0]))
                offered, mass_in = stage.positions, stage.mass_out
        key_schema = {"sequence": pl.Int64, "source_id": pl.String, "stage": pl.Int64, "selector": pl.String}
        count_schema = dict.fromkeys(("offered", "kept", "represented", "excluded", "unmeasured"), pl.Int64)
        label_schema = dict.fromkeys(
            ("label", "frames_offered", "frames_kept", "detections_offered", "detections_kept"), pl.Int64
        )
        return SelectionReport(
            pl.DataFrame(
                stage_rows, schema={**key_schema, **count_schema, "source_frames_excluded": pl.Float64}, orient="row"
            ),
            pl.DataFrame(label_rows, schema={**key_schema, **label_schema}, orient="row"),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized(positions: NDArray[np.intp], info: SequenceInfo) -> NDArray[np.intp]:
        """Reduce a plan to the ascending, unique, in-range positions the replay walk will emit."""
        wanted = np.unique(np.asarray(positions, dtype=np.intp))
        return wanted[(wanted >= 0) & (wanted < info.n_frames)]

    def _normalized_plan(
        self, plan: _Planned, info: SequenceInfo
    ) -> tuple[NDArray[np.intp], NDArray[np.float64] | None, tuple[_StageTrace, ...]]:
        """Normalize a plan's positions, refusing a plan whose weights normalizing would misalign.

        A plan without a trace is a lone selector's: one stage over the source frames, so each
        position is also its ordinal, and decided from structure alone, so none goes unmeasured.
        """
        positions, weights, trace = plan
        normalized = self._normalized(positions, info)
        if weights is not None and (len(weights) != len(positions) or not np.array_equal(normalized, positions)):
            raise ValueError(
                f"SequenceFrames: {self._selector!r} declared weights for sequence {info.index} beside "
                "positions that are not ascending, unique and in range, or not one weight per position. "
                "Normalizing them would leave the weights describing other frames."
            )
        if trace is None:
            from dataeval.data._chain import _stage_masses  # here, as _chain imports this module

            # The mass `_record` will give its frames: as declared, or by the default rule.
            masses = _stage_masses(np.ones(info.n_frames), normalized, self._selector.excludes)
            mass = float((masses if weights is None else weights).sum())
            unmeasured = np.zeros(len(normalized), dtype=np.bool_)
            trace = (_StageTrace(self._selector, normalized, normalized, unmeasured, mass),)
        return normalized, weights, trace

    def _located(self) -> list[tuple[int, int]]:
        """Return the ``(sequence, position)`` behind each view position, in view order.

        Answered from the plan where there is one, so asking a planned view where its frames came
        from costs no decode; a planned sequence's record holds the same positions. Otherwise
        from the records, so only a sequence no walk has finished yet is walked.
        """
        if self._planned is not None:
            return [
                (info.index, int(position))
                for info, positions in zip(self._sequences, self._planned, strict=True)
                for position in positions
            ]
        return [(entry["sequence"], entry["position"]) for entry in self._realize()]

    def _realize(self) -> list[dict[str, Any]]:
        """Return every kept frame's record in view order, recording any sequence that has none."""
        if self._realized is None:
            self._realized = [entry for record in self._recorded() for entry in record.entries]
        return self._realized

    def _recorded(self) -> list[_SequenceRecord]:
        """Return every sequence's execution record, in order, walking those not yet recorded."""
        return [self._record_of(info.index) for info in self._sequences]

    def _record_of(self, sequence: int) -> _SequenceRecord:
        """Return one sequence's execution record, walking that sequence first if it has none."""
        if sequence not in self._records:
            _logger.info(
                "SequenceFrames: walking sequence %d to record which frames %r keeps. Use "
                "stream() to avoid this where the frames are only being read in order.",
                sequence,
                self._selector,
            )
            for _ in self._walk(self._sequences[sequence]):
                pass
        return self._records[sequence]

    def _offsets_of(self, sequence: int) -> NDArray[np.intp]:
        """Return how many detections precede each position of one sequence, then the total.

        Counted from each frame target's ``labels``, which establish the detection count for every
        reader, so nothing is decoded. These are the running ``target_index`` values the tracking
        structurer behind :class:`~dataeval.Metadata` assigns within a sequence.
        """
        if sequence not in self._detection_offsets:
            counts = [as_numpy(target.labels).size for target in self._dataset[sequence][1].frame_tracks]
            self._detection_offsets[sequence] = np.concatenate(([0], np.cumsum(counts, dtype=np.intp)))
        return self._detection_offsets[sequence]

    def _frames_of(self, info: SequenceInfo) -> Iterator[tuple[int, Any, SingleFrameObjectTrackingTarget]]:  # noqa: C901
        """Pair each decoded frame of one sequence with its target, in order.

        A frame count that disagrees with the target count is a dataset defect and is raised
        rather than absorbed: pairing up to the shorter of the two would either drop real
        detections or annotate frames with another frame's boxes, and would signal neither.

        A walk of a recorded sequence checks every frame's number against the record, so a source
        that changed under the view raises rather than presenting other frames under the recorded
        metadata.
        """
        stream, target, _ = self._dataset[info.index]
        targets = target.frame_tracks
        frames = iter(stream)
        record = self._records.get(info.index)
        if record is not None and len(targets) != len(record.frame_numbers):
            raise ValueError(
                f"SequenceFrames: sequence {info.index} changed between walks: its target now "
                f"declares {len(targets)} frame(s) where the walk that recorded it found "
                f"{len(record.frame_numbers)}. Build a new SequenceFrames over the changed source."
            )
        exhausted = object()
        for position in range(len(targets)):
            frame = next(frames, exhausted)
            if frame is exhausted:
                if position == 0:
                    raise ValueError(
                        f"SequenceFrames: sequence {info.index}'s video stream yielded no frames, "
                        f"though its target declares {len(targets)}: the stream was already consumed "
                        "by an earlier walk, this view's or another reader's such as Metadata. "
                        "dataset[i] must return a stream walkable from its first frame, "
                        "independently of any earlier stream for the same index."
                    )
                raise ValueError(
                    f"SequenceFrames: sequence {info.index} declares {len(targets)} frame "
                    f"target(s) but its video stream yielded only {position}."
                )
            if record is not None and (number := frame_number(frame, position)) != record.frame_numbers[position]:
                raise self._renumbered(info, position, number, int(record.frame_numbers[position]))
            yield position, frame, targets[position]
        if next(frames, exhausted) is not exhausted:
            raise ValueError(
                f"SequenceFrames: sequence {info.index}'s video stream yields more frames than "
                f"its {len(targets)} frame target(s)."
            )

    @staticmethod
    def _renumbered(info: SequenceInfo, position: int, number: int, earlier: int) -> ValueError:
        """Build the error for a source that numbers one position differently on two walks."""
        return ValueError(
            f"SequenceFrames: sequence {info.index} changed between walks: position {position} now "
            f"holds frame {number} where an earlier walk found frame {earlier}. Build a new "
            "SequenceFrames over the changed source."
        )

    def _candidates(
        self,
        info: SequenceInfo,
        held: list[Any],
        trailer: dict[str, Any],
    ) -> Iterator[FrameCandidate]:
        """Offer each frame of one sequence to the selector, holding only the current one."""
        for position, frame, target in self._observed(info, trailer):
            held.clear()
            held.append((position, frame, target))
            yield FrameCandidate(
                sequence=info,
                position=position,
                frame_index=frame_number(frame, position),
                time_s=getattr(frame, "time_s", None),
                pts=getattr(frame, "pts", None),
                target=target,
                _frame=frame,
                _allow_pixels=self._reads_pixels,
            )

    def _observed(
        self,
        info: SequenceInfo,
        trailer: dict[str, Any],
    ) -> Iterator[tuple[int, Any, SingleFrameObjectTrackingTarget]]:
        """Walk a sequence's frames, remembering the first and last timestamps and every frame's number.

        The first kept frame's span starts at the start of the sequence, and the final one's runs
        to its end rather than to a successor it does not have, so the walk has to carry both
        endpoints forward. Without the last one ``seconds_represented`` is null on one row per
        sequence — and a partially populated factor is dropped from factor analysis entirely,
        which loses it everywhere.

        The numbers are keyed by position, so the second walk of a two-pass selector checks them
        rather than repeating them: its first pass's verdicts must not be applied to other frames.
        """
        numbers = trailer.setdefault("numbers", {})
        for position, frame, target in self._frames_of(info):
            number = frame_number(frame, position)
            if numbers.setdefault(position, number) != number:
                raise self._renumbered(info, position, number, numbers[position])
            time_s = getattr(frame, "time_s", None)
            if time_s is not None:
                trailer.setdefault("first_time_s", float(time_s))
                trailer["time_s"] = float(time_s)
            yield position, frame, target

    def _verdicts(self, info: SequenceInfo, trailer: dict[str, Any]) -> Iterator[_Selected]:
        """Yield each kept frame of one sequence, in order, as the selector decides it."""
        if self._planned is not None:
            yield from self._replay(info, trailer)
        elif self._selector.two_pass:
            yield from self._two_pass(info, trailer)
        else:
            yield from self._single_pass(info, trailer)

    @staticmethod
    def _selected(
        position: int,
        frame: Any,
        target: Any,
        factors: Mapping[str, Any],
        weight: float | None = None,
        unmeasured: bool = False,
    ) -> _Selected:
        """Package one kept frame with the timings its stream declares."""
        return (
            position,
            frame,
            target,
            getattr(frame, "time_s", None),
            getattr(frame, "pts", None),
            factors,
            weight,
            unmeasured,
        )

    def _replay(self, info: SequenceInfo, trailer: dict[str, Any]) -> Iterator[_Selected]:
        """Emit the positions a planning selector named, without consulting `select`.

        Each carries the weight the selector declared for it when it planned, or None for
        :meth:`_record` to apply the default rule. A plan decides from structure alone, so no
        frame it names goes unmeasured. Nothing is planned again here.
        """
        planned = self._planned[info.index] if self._planned is not None else np.empty(0, dtype=np.intp)
        weights = self._planned_weights[info.index]
        trailer["trace"] = self._planned_traces[info.index]
        declared = [None] * len(planned) if weights is None else weights.tolist()
        wanted = dict(zip(planned.tolist(), declared, strict=True))
        for position, frame, target in self._observed(info, trailer):
            if position in wanted:
                yield self._selected(position, frame, target, {}, wanted[position])

    def _single_pass(
        self,
        info: SequenceInfo,
        trailer: dict[str, Any],
        select: Callable[[Iterator[FrameCandidate]], Iterable[FrameVerdict]] | None = None,
    ) -> Iterator[_Selected]:
        """Drive the selector, holding exactly one frame and refusing a verdict for a released one.

        ``select`` drives the walk in place of the selector's own, for :meth:`_two_pass`.
        """
        held: list[Any] = []
        candidates = self._candidates(info, held, trailer)
        for verdict in self._returned((self._selector.select if select is None else select)(candidates), trailer):
            if not held or verdict.position != held[0][0]:
                reached = held[0][0] if held else "no frame"
                raise ValueError(
                    f"SequenceFrames: {self._selector!r} yielded a verdict for position "
                    f"{verdict.position} of sequence {info.index} while the walk had reached "
                    f"{reached}. A selector that decides a frame only after seeing later ones "
                    "must declare two_pass = True, so each sequence is walked twice."
                )
            position, frame, target = held[0]
            yield self._selected(
                position, frame, target, self._checked_factors(verdict), verdict.weight, verdict.unmeasured
            )
        # A selector may stop reading before the sequence ends; the record numbers every frame.
        for _ in candidates:
            pass

    def _returned(self, verdicts: Iterable[FrameVerdict], trailer: dict[str, Any]) -> Iterator[FrameVerdict]:
        """Yield a selector's verdicts, then keep in ``trailer`` what a ``Chain``'s ``select`` generator returned.

        A private protocol whose only user is ``Chain``, read here because a ``for`` loop discards
        a generator's return value. Any other selector's return value is ignored, as a chain's
        stage run ignores its stages': a lone selector's trace is built from its recorded entries.
        A callable is the continuation :meth:`_two_pass` runs over the second walk. A tuple of
        ``_StageTrace``, one per stage, is the selection's trace, which :meth:`_walk` records; a
        continuation's own generator returns the trace in turn. A ``Chain`` always returns one of
        the two, so anything else is a bug in it, and is refused.
        """
        from dataeval.data._chain import Chain  # here, as _chain imports this module

        returned = yield from verdicts
        if not isinstance(self._selector, Chain):
            return
        if callable(returned):
            trailer["continuation"] = returned
        elif isinstance(returned, tuple) and all(isinstance(stage, _StageTrace) for stage in returned):
            trailer["trace"] = returned
        else:
            raise TypeError(
                f"SequenceFrames: {self._selector!r}'s select returned {returned!r}, which is neither a "
                "continuation nor a trace. A Chain's select always returns one of the two, so this is "
                "a bug in Chain."
            )

    def _two_pass(self, info: SequenceInfo, trailer: dict[str, Any]) -> Iterator[_Selected]:
        """Drive the selector over a whole sequence, then walk it again to emit what it chose.

        A private protocol whose only user is ``Chain``: a chain's ``select`` generator may return a
        callable, which then decides the second walk frame by frame, under the single-pass
        discipline :meth:`_single_pass` keeps, in place of emitting what the first walk chose.
        See :meth:`_returned`.
        """
        verdicts = self._returned(self._selector.select(self._candidates(info, [], trailer)), trailer)
        chosen = {verdict.position: verdict for verdict in verdicts}
        continuation = trailer.pop("continuation", None)
        if continuation is not None:
            yield from self._single_pass(info, trailer, continuation)
            return
        for position, frame, target in self._observed(info, trailer):
            verdict = chosen.get(position)
            if verdict is not None:
                yield self._selected(
                    position, frame, target, self._checked_factors(verdict), verdict.weight, verdict.unmeasured
                )

    def _checked_factors(self, verdict: FrameVerdict) -> Mapping[str, Any]:
        """Reject a selector factor that would displace one this view writes itself."""
        collisions = sorted(RESERVED_FACTORS & set(verdict.factors))
        if collisions:
            raise ValueError(
                f"SequenceFrames: {self._selector!r} named reserved factor(s) {collisions} in a "
                "FrameVerdict. These are written by SequenceFrames itself, or are factor level "
                "names; choose different names rather than displacing them."
            )
        return verdict.factors

    def _walk(self, info: SequenceInfo) -> Iterator[tuple[dict[str, Any], Any, SingleFrameObjectTrackingTarget]]:  # noqa: C901
        """Yield each kept frame of one sequence with its record and target, one frame behind.

        The lag is what makes the weights answerable: a frame's ``frames_represented`` is the gap
        to the *next* kept frame, so it cannot be written until that frame is known. Exactly one
        kept frame is held back, which is the same discipline the single-pass contract already
        requires.

        The first complete walk records the selection, and every later one replays it. The record
        is stored before the last kept frame is yielded, so a consumer that stops right after that
        frame still leaves the sequence recorded; a walk abandoned earlier records nothing.
        """
        record = self._records.get(info.index)
        if record is not None:
            yield from self._replayed(info, record)
            return
        previous: _Selected | None = None
        span = float(max(info.n_frames - 1, 1))
        trailer: dict[str, Any] = {}
        entries: list[dict[str, Any]] = []
        for current in self._verdicts(info, trailer):
            if previous is not None:
                entries.append(self._record(info, previous, current, span, trailer, first=not entries))
                yield entries[-1], previous[1], previous[2]
            previous = current
        if previous is not None:
            entries.append(self._record(info, previous, None, span, trailer, first=not entries))
        total = sum(entry["metadata"]["frames_represented"] for entry in entries)
        if total > info.n_frames * (1 + 1e-9):
            raise ValueError(
                f"SequenceFrames: {self._selector!r} gave the kept frames of sequence {info.index} "
                f"weights totaling {total:g} source frames, but the sequence holds {info.n_frames}. A "
                "FrameVerdict.weight counts the source frames a kept frame stands for, so a "
                "sequence's weights cannot total more than its frames."
            )
        numbers = np.fromiter(trailer.get("numbers", {}).values(), dtype=np.intp)
        trace = trailer.get("trace")
        if trace is None:
            # A lone selector is one stage over the source frames, so each position is also its
            # ordinal, and each frame's mark the one that stage set.
            positions = np.array([entry["position"] for entry in entries], dtype=np.intp)
            marks = np.array([entry["unmeasured"] for entry in entries], dtype=np.bool_)
            trace = (_StageTrace(self._selector, positions, positions, marks, float(total)),)
        # Another walk of this sequence may have finished while this one was paused; the first to
        # finish fixed the selection, so this one has to agree with it.
        recorded = self._records.setdefault(info.index, _SequenceRecord(tuple(entries), numbers, trace))
        if not _same_selection(recorded.entries, entries):
            raise ValueError(
                f"SequenceFrames: {self._selector!r} chose different frames on two walks of sequence "
                f"{info.index}, or the same frames with different weights or factors. The first "
                "walk to finish fixed this view's selection, and a walk started before it "
                "finished disagrees. Finish one walk of a sequence, such as a "
                "stream(), before starting another, such as len() or frame_map, or use a selector "
                "that gives the same answer each time it is asked."
            )
        if len(self._records) == len(self._sequences):
            self.n_dropped = self.n_source_frames - sum(len(kept.entries) for kept in self._records.values())
        if previous is not None:
            yield entries[-1], previous[1], previous[2]

    def _replayed(
        self, info: SequenceInfo, record: _SequenceRecord
    ) -> Iterator[tuple[dict[str, Any], Any, SingleFrameObjectTrackingTarget]]:
        """Walk a recorded sequence again, yielding the frames its record kept, with their entries as stored.

        The selector is not consulted, so a stateful one cannot change its answer, and a
        ``two_pass`` one is not paid for twice. :meth:`_frames_of` checks every frame's number
        against the record on the way.
        """
        kept = iter(record.entries)
        entry = next(kept, None)
        for position, frame, target in self._frames_of(info):
            # `while`, not `if`: a single-pass selector may keep one position more than once.
            while entry is not None and entry["position"] == position:
                yield entry, frame, target
                entry = next(kept, None)

    def _record(  # noqa: C901
        self,
        info: SequenceInfo,
        selected: _Selected,
        following: _Selected | None,
        span: float,
        trailer: dict[str, Any],
        first: bool,
    ) -> dict[str, Any]:
        """Build one kept frame's record, with the weight its successor determines.

        The weight follows the composition rule with every source frame of mass 1: a declared
        weight as declared; under a selector that excludes, the frame's own mass; otherwise the
        gap to the next kept frame, the first kept frame also taking every frame before it.
        """
        position, frame, _, time_s, pts, factors, weight, unmeasured = selected
        next_position = following[0] if following is not None else info.n_frames
        if weight is not None:
            # Converted here and nowhere sooner: a Chain declares a pending weight with its verdict
            # and settles it after, before this frame is recorded (the pending-weight protocol).
            represented = float(weight)
            if not 0 <= represented < np.inf:
                raise ValueError(
                    f"SequenceFrames: {self._selector!r} declared a weight of {represented:g} for "
                    f"position {position} of sequence {info.index}. A FrameVerdict.weight counts the "
                    "source frames a kept frame stands for, so it must be finite and not negative."
                )
        elif self._selector.excludes:
            represented = 1.0
        else:
            represented = float(next_position - (0 if first else position))
        metadata: dict[str, Any] = {
            "source_id": info.source_id,
            "sequence": info.index,
            "frame": frame_number(frame, position),
            "frames_represented": represented,
            "sequence_position": position / span,
            "sequence_n_frames": info.n_frames,
        }
        if time_s is not None:
            metadata["time_s"] = float(time_s)
            # A span of the timeline, not a mass: the first kept frame's starts at the sequence's
            # first timestamp, and the last one's runs to its end rather than to a successor.
            start = trailer["first_time_s"] if first else time_s
            next_time = following[3] if following is not None else trailer.get("time_s")
            if next_time is not None:
                metadata["seconds_represented"] = float(next_time) - float(start)
        if pts is not None:
            metadata["pts"] = int(pts)
        metadata.update(factors)
        # Beside the metadata, not in it: whether a frame was judged is no factor of the data.
        return {"sequence": info.index, "position": position, "metadata": metadata, "unmeasured": unmeasured}

    def _seek(self, sequence: int, position: int) -> tuple[Any, SingleFrameObjectTrackingTarget]:
        """Decode up to one frame of one sequence, reusing the cursor when moving forward.

        The frame the cursor currently rests on is held, so re-reading a position costs nothing.
        Re-reading is not a rare case: any consumer that probes ``dataset[0]`` before walking --
        :class:`~dataeval.Metadata` does, to pick a structurer -- asks for it twice.

        The frame's target is carried on the cursor beside it. The walk already pairs the two, and
        re-reading it through ``self._dataset[sequence]`` would index the source dataset once per
        frame -- for a video dataset that is a file opened per frame.
        """
        if self._cursor_sequence == sequence and self._cursor_position == position:
            return self._cursor_frame, cast(SingleFrameObjectTrackingTarget, self._cursor_target)
        if self._cursor_sequence != sequence or self._cursor_position > position:
            self._restart(sequence)

        cursor = cast(Iterator[Any], self._cursor)
        exhausted = object()
        while self._cursor_position < position:
            step = next(cursor, exhausted)
            if step is exhausted:
                raise IndexError(f"SequenceFrames: sequence {sequence} has no frame at position {position}.")
            self._cursor_position, self._cursor_frame, self._cursor_target = cast(tuple, step)
        return self._cursor_frame, cast(SingleFrameObjectTrackingTarget, self._cursor_target)

    def _restart(self, sequence: int) -> None:
        """Point the cursor at the start of one sequence, decoding it again from the beginning."""
        if self._cursor_sequence == sequence:
            _logger.warning(
                "SequenceFrames: backward access within sequence %d re-decodes it from the start. "
                "A VideoStream is not indexable; use stream() for in-order reads.",
                sequence,
            )
        self._cursor = self._frames_of(self._sequences[sequence])
        self._cursor_sequence = sequence
        self._cursor_position = -1
        self._cursor_frame = None
        self._cursor_target = None


def frame_numbers_at(
    frames: SequenceFrames,
    sequences: NDArray[np.intp],
    positions: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Return the ``frame_index`` of the frame at each ``(sequence, position)``.

    Read from each sequence's execution record, so only the sequences named here that no walk has
    recorded yet are walked, and none at all once the view has been measured.
    """
    numbers = np.empty(len(positions), dtype=np.intp)
    for sequence in np.unique(sequences):
        where = sequences == sequence
        numbers[where] = frames._record_of(int(sequence)).frame_numbers[positions[where]]  # pyright: ignore[reportPrivateUsage]
    return numbers

"""Multi-object tracking over video: the one task whose level graph is a diamond."""

__all__ = []

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval._metadata._structurers._accumulator import MOTAccumulator
from dataeval._metadata._structurers._block import RowBlock
from dataeval._metadata._structurers._data import StructuredData
from dataeval._metadata._structurers._dataset import DatasetStructurer
from dataeval._metadata._structurers._frames import FrameRows
from dataeval._metadata._structurers._instances import InstanceBuildingMixin
from dataeval._metadata._structurers._ordering import running_index
from dataeval._metadata._structurers._propagation import PropagationMixin
from dataeval._metadata._structurers._reporting import log_items_without_targets, without_displaced
from dataeval._metadata._structurers._reserved import reserved_block_columns
from dataeval.protocols import (
    AnnotatedDataset,
    DatumMetadata,
    ProgressCallback,
    SingleFrameObjectTrackingTarget,
)
from dataeval.types import FactorLevelSchema
from dataeval.types._track import frame_size

_logger = get_logger(__name__)

# Sentinel for "the iterator had nothing left", distinct from any value it could yield.
_EXHAUSTED: Any = object()


class MOTStructurer(InstanceBuildingMixin, PropagationMixin, DatasetStructurer):
    """Multi-object tracking over video: items are sequences, targets are instances.

    Four levels, and the only task whose graph is a diamond rather than a chain:
    ``sequence`` is the item level (one dataset item is one video), ``unit`` is a frame
    and ``track`` is one tracked object — siblings under the sequence — and ``instance``
    is the label level, one row per detection, which sits under *both*. A detection is one
    observation: of a track, in a frame.

    Because ``unit`` sits between the item level and the label level, an instance row
    needs its frame's key as well as its own to be uniquely identified: ``instance_index``
    counts within the frame, so ``(item_index, unit_index, instance_index)`` is the
    compound key, and ``(item_index, unit_index)`` joins instance rows to their frame's
    row. ``target_index`` keeps counting within the whole item, as it does for every task.

    A track is a level rather than a column so that metadata can be organized *by track*:
    a factor added at ``track`` is stored once per track and propagates down to every
    detection in it, and ``rows_at("track")`` reads it once per track rather than once per
    detection. Tracks are scoped to their sequence — the same ``track_id`` in two videos is
    two tracks — and ``track_index`` numbers them densely within each, in order of first
    appearance, because a dataset's own ids may be sparse or arbitrary.

    A detection that no tracker linked (``track_id == -1``) has a frame but **no track**.
    Its ``track_index`` is ``-1``, which propagates every track-level factor to it as null
    rather than inventing a singleton track for it. :class:`~dataeval.Metadata` keeps such
    a factor out of factor analysis at any view where some row is untracked, and
    ``md.at("track")`` still reads it in full — see :attr:`~dataeval.Metadata.factor_data`.

    Per-frame metadata is merged at the ``unit`` level, not the instance level. A video's
    list-valued metadata is per frame — one timestamp per frame, not one per detection —
    so expanding it across detections would be wrong even where the counts happen to
    match. Instance-level factors therefore come only from the target data itself.
    """

    task = "MOT"
    levels = FactorLevelSchema.of("sequence", "unit", "track", "instance")
    item_level = "sequence"
    label_level = "instance"
    multi_target = True
    unit_type = "frame"

    @classmethod
    def _frames_of(
        cls,
        video_stream: Any,
        frame_tracks: Sequence[SingleFrameObjectTrackingTarget],
    ) -> Iterator[FrameRows]:
        """Pair each decoded frame with its target, yielding one frame's rows at a time.

        Streams rather than materializing the frames: only each frame's keys and timings
        are retained, while a decoded :obj:`~dataeval.protocols.VideoFrame` holds a full
        pixel array.

        A frame count that disagrees with the target count is a dataset bug and is raised
        rather than absorbed. Pairing the two up to the shorter of them would either drop
        real detections or annotate frames with another frame's boxes, and would give no
        signal that either had happened.

        Yields
        ------
        FrameRows
            One frame's index, timings and detection arrays. ``time_s`` and ``pts`` are
            None for a frame that does not declare them.

        Raises
        ------
        ValueError
            When the stream and the target disagree on how many frames the item has.
        """
        frames = iter(video_stream)
        for position, frame_target in enumerate(frame_tracks):
            frame = next(frames, _EXHAUSTED)
            if frame is _EXHAUSTED:
                raise ValueError(
                    f"Tracking target declares {len(frame_tracks)} frame target(s) but the item's "
                    f"video stream yielded only {position}; frame_tracks must hold exactly one "
                    "target per frame.",
                )
            labels, boxes, scores = cls._instance_arrays(frame_target)
            # MAITE's VideoFrame declares frame_index, time_s and pts, but dispatch here
            # duck-types the target rather than requiring the full protocol, so each is
            # optional. frame_index falls back to decode order, which is what it means for
            # a conforming stream anyway; a timing has no such stand-in and stays None.
            # Taken here because the frame is released as soon as this yields.
            width, height = frame_size(frame)
            yield FrameRows(
                getattr(frame, "frame_index", position),
                getattr(frame, "time_s", None),
                getattr(frame, "pts", None),
                width,
                height,
                labels,
                boxes,
                scores,
                cls._track_ids(frame_target, len(labels)),
            )

        if next(frames, _EXHAUSTED) is not _EXHAUSTED:
            raise ValueError(
                f"Item's video stream yields more frames than the tracking target's "
                f"{len(frame_tracks)} frame target(s); frame_tracks must hold exactly one "
                "target per frame.",
            )

    @staticmethod
    def _frame_factors(rows: MOTAccumulator) -> dict[str, NDArray[Any]]:
        """Per-frame timings and dimensions, as ``unit``-level factors, when every frame has them.

        All-or-nothing rather than null-padded, one factor at a time. A partially null
        numeric factor cannot be binned — sorting it compares None against a float — so a
        factor present for only some frames would break factor analysis for the whole
        dataset rather than degrade gracefully. A conforming
        :obj:`~dataeval.protocols.VideoStream` declares all four, so the all-or-nothing
        case is the normal one; a duck-typed stream that omits one gets no factor for it
        and a log line saying so.

        ``width`` and ``height`` are recorded here because this walk is the only thing that
        ever holds a decoded frame. They are what
        :func:`~dataeval.core.track_stats` needs to decide whether a track enters or leaves
        at the frame border, and they carry the same names
        :func:`~dataeval.core.compute_stats` gives the same two quantities.
        """
        factors: dict[str, NDArray[Any]] = {}
        for name, values, dtype in (
            ("time_s", rows.frame_time_s, np.float64),
            ("pts", rows.frame_pts, np.intp),
            ("width", rows.frame_width, np.intp),
            ("height", rows.frame_height, np.intp),
        ):
            missing = sum(value is None for value in values)
            if missing:
                _logger.info(
                    "%d of %d frame(s) do not declare %r, so no %r factor is produced; a "
                    "partially populated factor cannot be binned.",
                    missing,
                    len(values),
                    name,
                    name,
                )
                continue
            factors[name] = np.asarray(values, dtype=dtype)
        return factors

    @staticmethod
    def _track_factors(rows: MOTAccumulator) -> dict[str, NDArray[Any]]:
        """Derive per-track factors: how long each track is, and how far it spans.

        ``track_length`` counts observations; ``frame_span`` counts frames from first to
        last inclusive. They differ exactly when a track has gaps, which makes the pair
        more informative than either alone. ``duration_s`` is the elapsed time over the
        same span, and follows the same all-or-nothing rule as the frame timings.

        These two are the same quantities :func:`~dataeval.core.track_stats` returns as
        ``n_appearances`` and ``track_duration``: ``track_length == n_appearances`` and
        ``frame_span == track_duration``. These are canonical for metadata, because they
        are accumulated during the structuring walk and so cost nothing;
        ``track_stats`` keeps its own names for standalone use. Attaching that result
        with ``add_factors(..., key="track_id")`` therefore brings two columns metadata
        already has, under other names.

        ``agg("instance", "track", pl.len())`` computes ``track_length`` a third way and
        is *not* the route to prefer, for the same reason: the walk has already counted
        it, and the aggregate re-derives it from the rows it was counted from.
        """
        factors: dict[str, NDArray[Any]] = {
            "track_length": np.asarray(rows.track_length, dtype=np.intp),
            "frame_span": np.asarray(rows.track_last_frame, dtype=np.intp)
            - np.asarray(rows.track_first_frame, dtype=np.intp)
            + 1,
        }
        if all(value is not None for value in (*rows.track_first_time, *rows.track_last_time)):
            factors["duration_s"] = np.asarray(rows.track_last_time, dtype=np.float64) - np.asarray(
                rows.track_first_time, dtype=np.float64
            )
        return factors

    @staticmethod
    def _stacked(
        labels: Sequence[NDArray[Any]],
        boxes: Sequence[NDArray[Any]],
        scores: Sequence[NDArray[Any]],
        track_ids: Sequence[NDArray[Any]],
    ) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32], NDArray[np.intp]]:
        """Concatenate the per-frame arrays into one array each, coarsest dtype preserved.

        Each is built explicitly when there is nothing to concatenate, because
        :func:`numpy.concatenate` rejects an empty sequence and because an empty result
        still has to carry the right dtype and, for boxes, the right width — a dataset
        with no detections at all must still produce a well-typed empty block.
        """
        return (
            np.concatenate(labels).astype(np.intp) if labels else np.empty(0, dtype=np.intp),
            np.concatenate(boxes).astype(np.float32) if boxes else np.empty((0, 4), dtype=np.float32),
            np.concatenate(scores).astype(np.float32) if scores else np.empty(0, dtype=np.float32),
            np.concatenate(track_ids).astype(np.intp) if track_ids else np.empty(0, dtype=np.intp),
        )

    def build(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> StructuredData:
        raw: list[Mapping[str, Any]] = []
        count = len(dataset)
        rows = MOTAccumulator()

        for i in range(count):
            video_stream, target, metadata = self._datum(dataset, i)
            raw.append(metadata)
            rows.add_item(i, self._frames_of(video_stream, target.frame_tracks))
            if progress_callback:
                progress_callback(i + 1, total=count)

        sequence_of_frame = np.asarray(rows.frame_sequence, dtype=np.intp)
        frame_own_index_arr = np.asarray(rows.frame_index, dtype=np.intp)
        n_frames = len(sequence_of_frame)

        sequence_of_track = np.asarray(rows.track_sequence, dtype=np.intp)
        n_tracks = len(sequence_of_track)

        class_labels, box_values, score_values, track_id_values = self._stacked(
            rows.instance_labels,
            rows.instance_boxes,
            rows.instance_scores,
            rows.instance_track_ids,
        )
        sequence_of_instance = np.asarray(rows.instance_sequence, dtype=np.intp)
        unit_pos_of_instance = np.asarray(rows.instance_unit_pos, dtype=np.intp)
        track_pos_of_instance = np.asarray(rows.instance_track_pos, dtype=np.intp)
        n_instances = len(sequence_of_instance)

        # Two distinct running indices, because an instance's direct parent (unit) and
        # its item (sequence) are no longer the same level, unlike object detection:
        # - instance_index: this level's own key, index within its own frame.
        # - target_index: the legacy public spelling, index within the whole item.
        # Instances were appended frame-by-frame within sequence-by-sequence order, so
        # both grouping arrays are already contiguous and running_index applies directly.
        instance_index = running_index(unit_pos_of_instance)
        instance_target_index = running_index(sequence_of_instance)
        # Derived from the finished counts rather than tracked during the walk: a sequence
        # contributes no instance rows exactly when none of its frames held a detection.
        instances_per_item = np.bincount(sequence_of_instance, minlength=count)
        undetected = np.flatnonzero(instances_per_item == 0).tolist()
        # The parent frame's own key, carried onto the instance row so that the compound
        # key is unique: instance_index alone repeats across the frames of one sequence.
        unit_index_of_instance = frame_own_index_arr[unit_pos_of_instance]
        # Dense within the sequence, in order of first observation. Tracks were opened
        # sequence-by-sequence, so the grouping array is already contiguous.
        track_index = running_index(sequence_of_track)
        # Gathered with the -1 markers preserved — clamped first, since a marker would
        # otherwise index from the end and report another track's number. Guarded for a
        # dataset in which nothing is tracked at all: there is then no index to gather from.
        track_index_of_instance = (
            np.where(track_pos_of_instance < 0, -1, track_index[np.maximum(track_pos_of_instance, 0)])
            if n_tracks
            else np.full(n_instances, -1, dtype=np.intp)
        )

        # Per-frame, not per-instance: a video's list-valued metadata has one value per
        # frame. Expanding it across detections would be wrong even when the counts
        # coincide, and dropping it — which is what expanding across detections does
        # whenever they disagree — loses the per-frame factors entirely.
        frames_per_item = np.bincount(sequence_of_frame, minlength=count).astype(int).tolist()
        unit_factors, dropped = self._merge_factors(raw, ignore_lists=False, targets_per_item=frames_per_item)
        sequence_factors, _ = self._merge_factors(raw, ignore_lists=True)
        # Same rule as the image-based tasks: a name both merges produced is item metadata
        # replicated onto the frame rows, so keep it once at the sequence level and let
        # propagation do the replicating.
        unit_factors = {name: values for name, values in unit_factors.items() if name not in sequence_factors}

        track_factors = self._track_factors(rows)
        frame_factors = self._frame_factors(rows)
        # A factor can only be declared at one level, so the structurer's own derived
        # values displace a metadata key of the same name rather than colliding with it:
        # these are read off the frames and the targets themselves, which outranks a
        # per-item dictionary that happens to reuse the spelling.
        derived = {*track_factors, *frame_factors}
        sequence_factors = without_displaced(sequence_factors, derived, "sequence")
        unit_factors = {**without_displaced(unit_factors, derived, "unit"), **frame_factors}

        sequence_block = RowBlock(
            "sequence",
            count,
            reserved_block_columns("sequence", count, item_index=list(range(count)), sequence_index=list(range(count))),
            {"sequence": self._own_positions(count)},
        )
        unit_block = RowBlock(
            "unit",
            n_frames,
            reserved_block_columns("unit", n_frames, item_index=sequence_of_frame, unit_index=frame_own_index_arr),
            {**self._inherit(sequence_block.ancestor_pos, sequence_of_frame), "unit": self._own_positions(n_frames)},
        )
        track_block = RowBlock(
            "track",
            n_tracks,
            reserved_block_columns(
                "track",
                n_tracks,
                item_index=sequence_of_track,
                track_index=track_index,
                track_id=np.asarray(rows.track_id, dtype=np.intp),
            ),
            {**self._inherit(sequence_block.ancestor_pos, sequence_of_track), "track": self._own_positions(n_tracks)},
        )
        instance_block = RowBlock(
            "instance",
            n_instances,
            reserved_block_columns(
                "instance",
                n_instances,
                item_index=sequence_of_instance,
                target_index=instance_target_index,
                class_label=class_labels,
                score=score_values,
                box=box_values,
                instance_index=instance_index,
                unit_index=unit_index_of_instance,
                track_index=track_index_of_instance,
                track_id=track_id_values,
            ),
            # The diamond: two parents, so two inherited maps. The ``unit`` branch supplies
            # ``sequence`` and is spread last, because the track branch would supply it too
            # and an untracked row's track position is a null marker rather than an index.
            # ``track`` is taken from the accumulator directly, negatives intact — the track
            # block's own positions are the identity, so gathering through them would only
            # destroy the markers.
            {
                "track": track_pos_of_instance,
                **self._inherit(unit_block.ancestor_pos, unit_pos_of_instance),
                "instance": self._own_positions(n_instances),
            },
        )

        log_items_without_targets(undetected, "instance", count)
        untracked = int(np.count_nonzero(track_pos_of_instance < 0))
        if untracked:
            _logger.info(
                "%d of %d detection(s) carry no track id and contribute no %r rows. Track-level "
                "factors read as null on them, and are excluded from factor analysis at any view "
                "where that happens; Metadata.at('track') reads them in full.",
                untracked,
                n_instances,
                "track",
            )
        _logger.info(
            "MOT dataset: %d sequences, %d frames, %d tracks, %d classes, %d detections",
            count,
            n_frames,
            n_tracks,
            len(np.unique(class_labels)),
            n_instances,
        )
        return StructuredData(
            [sequence_block, unit_block, track_block, instance_block],
            {
                "sequence": sequence_factors,
                "unit": unit_factors,
                "track": track_factors,
                "instance": {},
            },
            dropped,
            raw,
            class_labels,
            sequence_of_instance,
        )

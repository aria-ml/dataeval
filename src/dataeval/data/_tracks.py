from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast, overload

import numpy as np

from dataeval.protocols import Dataset, MultiobjectTrackingDataset, MultiobjectTrackingTarget
from dataeval.types import Track


def _build_tracks(tracking_target: MultiobjectTrackingTarget) -> Mapping[int, Track]:
    """Reorganize a `MultiobjectTrackingTarget` from frame-indexed to track-indexed."""
    _boxes: defaultdict[int, list[list[float]]] = defaultdict(list)
    _frames: defaultdict[int, list[int]] = defaultdict(list)
    _scores: defaultdict[int, list[float]] = defaultdict(list)
    _labels: defaultdict[int, list[int]] = defaultdict(list)

    for frame_idx, frame_target in enumerate(tracking_target.frame_tracks):
        track_ids = np.asarray(frame_target.track_ids)
        if track_ids.size == 0:
            continue
        boxes = np.asarray(frame_target.boxes)
        scores = np.asarray(frame_target.scores)
        labels = np.asarray(frame_target.labels)
        for det_idx, tid in enumerate(track_ids.tolist()):
            _boxes[tid].append(boxes[det_idx].tolist())
            _frames[tid].append(frame_idx)
            _scores[tid].append(float(scores[det_idx]))
            _labels[tid].append(int(labels[det_idx]))

    return {
        tid: Track(
            track_id=tid,
            boxes=np.array(_boxes[tid], dtype=np.float32),
            frame_indices=np.array(_frames[tid], dtype=np.int64),
            scores=np.array(_scores[tid], dtype=np.float32),
            labels=np.array(_labels[tid], dtype=np.int64),
        )
        for tid in sorted(_boxes)
    }


@overload
def build_tracks(source: MultiobjectTrackingDataset) -> Mapping[str, Mapping[int, Track]]: ...
@overload
def build_tracks(source: MultiobjectTrackingTarget) -> Mapping[int, Track]: ...


def build_tracks(
    source: MultiobjectTrackingDataset | MultiobjectTrackingTarget,
) -> Mapping[str, Mapping[int, Track]] | Mapping[int, Track]:
    """Build track dicts for a single target or an entire dataset.

    Parameters
    ----------
    source : MultiobjectTrackingDataset | MultiobjectTrackingTarget
        A single target or a dataset containing multiple targets.

    Returns
    -------
    Mapping[int, Track] or Mapping[str, Mapping[int, Track]]
        If source is a target, returns mapping of track ID to Track.
        If source is a dataset, returns mapping of sequence ID to track mappings.

    Notes
    -----
    MultiobjectTrackingTarget stores detections grouped by frame.  This
    function inverts that structure so that each unique track ID maps to all
    of its observations across the sequence, in frame order.
    """
    # MultiobjectTrackingTarget is not runtime_checkable
    if isinstance(getattr(source, "frame_tracks", None), Sequence):
        return _build_tracks(cast(MultiobjectTrackingTarget, source))

    if isinstance(source, Dataset):
        result: dict[str, Mapping[int, Track]] = {}
        for i in range(len(source)):
            _, video_target, datum_metadata = source[i]
            seq_id = str(datum_metadata["id"])
            result[seq_id] = _build_tracks(video_target)
        return result

    raise TypeError("'source' must be a MultiobjectTrackingDataset or a MultiObjectTrackingTarget.")

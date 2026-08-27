__all__ = []

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast, overload

import numpy as np

from dataeval._log import get_logger
from dataeval.protocols import Dataset, MultiobjectTrackingDataset, MultiobjectTrackingTarget
from dataeval.types import Track

_logger = get_logger(__name__)


def _log_untracked(untracked: int) -> None:
    """Report detections that belong to no track, when there were any."""
    if untracked:
        _logger.info("%d detection(s) carry no track id and are not part of any track.", untracked)


def _build_tracks(tracking_target: MultiobjectTrackingTarget) -> Mapping[int, Track]:
    """Reorganize a `MultiobjectTrackingTarget` from frame-indexed to track-indexed.

    A detection carrying a negative id belongs to no track and is left out. Collecting
    them under a shared id would invent one track that jumps between unrelated objects,
    and every per-track statistic computed from it — speed, straightness, duration —
    would describe that fiction rather than anything in the data. The structuring walk
    behind :class:`~dataeval.Metadata` applies the same rule when it builds track rows,
    so the two agree on which detections belong to a track.
    """
    _boxes: defaultdict[int, list[list[float]]] = defaultdict(list)
    _frames: defaultdict[int, list[int]] = defaultdict(list)
    _scores: defaultdict[int, list[float]] = defaultdict(list)
    _labels: defaultdict[int, list[int]] = defaultdict(list)

    untracked = 0
    for frame_idx, frame_target in enumerate(tracking_target.frame_tracks):
        # A frame holds as many detections as it has labels, and one holding none is
        # allowed to carry no ``track_ids`` at all — which is what the structuring walk
        # behind :class:`~dataeval.Metadata` allows, so refusing it here would reject a
        # target that metadata accepts.
        count = len(np.asarray(frame_target.labels).reshape(-1))
        track_ids = np.asarray(getattr(frame_target, "track_ids", ())).reshape(-1)[:count] if count else np.empty(0)
        # Selected up front rather than skipped inside the loop, so the walk below sees
        # only detections that belong to a track.
        tracked = np.flatnonzero(track_ids >= 0)
        untracked += track_ids.size - tracked.size
        if tracked.size == 0:
            continue
        boxes = np.asarray(frame_target.boxes)
        scores = np.asarray(frame_target.scores)
        labels = np.asarray(frame_target.labels)
        for det_idx in tracked.tolist():
            tid = int(track_ids[det_idx])
            _boxes[tid].append(boxes[det_idx].tolist())
            _frames[tid].append(frame_idx)
            _scores[tid].append(float(scores[det_idx]))
            _labels[tid].append(int(labels[det_idx]))

    _log_untracked(untracked)

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

    A detection whose track ID is negative belongs to no track and appears in no
    entry of the result.  Gathering them under the sentinel would produce a track
    that is really a bag of unrelated objects, and per-track statistics computed
    over it would describe nothing.  This matches how
    :class:`~dataeval.Metadata` builds its ``track`` rows, so a result attached
    with ``add_factors(..., key="track_id")`` names exactly the tracks metadata
    holds.
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

"""Row accumulators for the tracking walk, which fills three levels at once."""

__all__ = []

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from numpy.typing import NDArray

from dataeval._metadata._structurers._frames import FrameRows


@dataclass
class MOTAccumulator:
    """Row accumulators for one pass over a tracking dataset.

    A class rather than a wall of locals in ``build``, because the walk fills three levels
    at once — frames, tracks and instances — and has to keep a per-sequence track registry
    alive while it does. Threading that many parallel lists through helpers is what makes
    the alternative unreadable.

    Tracks are discovered rather than declared: a sequence's track rows are created on each
    ``track_id``'s first appearance, so they end up densely numbered in order of first
    observation whatever ids the dataset used. The registry is per sequence, which is what
    keeps the same id in two videos two separate tracks.
    """

    frame_sequence: list[int] = field(default_factory=list)
    frame_index: list[int] = field(default_factory=list)
    frame_time_s: list[float | None] = field(default_factory=list)
    frame_pts: list[int | None] = field(default_factory=list)

    track_sequence: list[int] = field(default_factory=list)
    track_id: list[int] = field(default_factory=list)
    track_length: list[int] = field(default_factory=list)
    track_first_frame: list[int] = field(default_factory=list)
    track_last_frame: list[int] = field(default_factory=list)
    track_first_time: list[float | None] = field(default_factory=list)
    track_last_time: list[float | None] = field(default_factory=list)

    instance_labels: list[NDArray[Any]] = field(default_factory=list)
    instance_boxes: list[NDArray[Any]] = field(default_factory=list)
    instance_scores: list[NDArray[Any]] = field(default_factory=list)
    instance_track_ids: list[NDArray[Any]] = field(default_factory=list)
    instance_sequence: list[int] = field(default_factory=list)
    instance_unit_pos: list[int] = field(default_factory=list)
    instance_track_pos: list[int] = field(default_factory=list)

    def add_item(self, item: int, frames: Iterable[FrameRows]) -> None:
        """Absorb one dataset item — one video — and everything inside it."""
        registry: dict[int, int] = {}
        for rows in frames:
            position = len(self.frame_sequence)
            self.frame_sequence.append(item)
            self.frame_index.append(rows.frame_index)
            self.frame_time_s.append(rows.time_s)
            self.frame_pts.append(rows.pts)

            if len(rows.labels):
                self.instance_labels.append(rows.labels)
                self.instance_boxes.append(rows.boxes)
                self.instance_scores.append(rows.scores)
                self.instance_track_ids.append(rows.track_ids)
                self.instance_sequence.extend([item] * len(rows.labels))
                self.instance_unit_pos.extend([position] * len(rows.labels))
                self._add_tracks(item, rows, registry)

    def _add_tracks(self, item: int, rows: FrameRows, registry: dict[int, int]) -> None:
        """Attach one frame's detections to their tracks, opening any not yet seen.

        A detection with a negative id belongs to no track, and records ``-1`` as its track
        position: the layout's marker for "no ancestor at that level". Nothing is invented
        for it — a singleton track would be a track the data says does not exist, and would
        skew every per-track statistic toward length one.
        """
        for track_id in rows.track_ids.tolist():
            if track_id < 0:
                self.instance_track_pos.append(-1)
                continue

            position = registry.get(track_id)
            if position is None:
                position = registry[track_id] = len(self.track_sequence)
                self.track_sequence.append(item)
                self.track_id.append(track_id)
                self.track_length.append(0)
                self.track_first_frame.append(rows.frame_index)
                self.track_last_frame.append(rows.frame_index)
                self.track_first_time.append(rows.time_s)
                self.track_last_time.append(rows.time_s)
            else:
                # min/max rather than "the latest wins": frame_index comes off the stream
                # and a duck-typed frame is not obliged to number its frames in order.
                self.track_first_frame[position] = min(self.track_first_frame[position], rows.frame_index)
                self.track_last_frame[position] = max(self.track_last_frame[position], rows.frame_index)
                self.track_first_time[position] = _min_or_none(self.track_first_time[position], rows.time_s)
                self.track_last_time[position] = _max_or_none(self.track_last_time[position], rows.time_s)

            self.track_length[position] += 1
            self.instance_track_pos.append(position)


def _min_or_none(current: float | None, candidate: float | None) -> float | None:
    """Smaller of two optional times, None when either is missing."""
    return None if current is None or candidate is None else min(current, candidate)


def _max_or_none(current: float | None, candidate: float | None) -> float | None:
    """Larger of two optional times, None when either is missing."""
    return None if current is None or candidate is None else max(current, candidate)

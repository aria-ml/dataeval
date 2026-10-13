from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from dataeval.protocols import DatumMetadata


class MockICDataset:
    """Minimal image-classification dataset: one-hot targets + index2label."""

    def __init__(self, labels: Sequence[int], index2label: Mapping[int, str]) -> None:
        self._labels = list(labels)
        self._n_classes = len(index2label)  # fixed at init so metadata edits don't break __getitem__
        self.metadata = {"id": "mock-ic", "index2label": dict(index2label)}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        onehot = np.zeros(self._n_classes, dtype=np.float32)
        onehot[self._labels[index]] = 1.0
        return np.zeros((3, 2, 2), dtype=np.float32), onehot, {"id": index}


class _ODTarget:
    """One image's detections, scored in either layout MAITE permits.

    ``classes`` is None for one confidence per box, or the vocabulary size for the
    ``(N, CLASSES)`` layout every ground-truth dataset in ``maite_datasets`` emits.
    """

    def __init__(self, labels: Sequence[int], classes: int | None = None, confidence: float = 1.0) -> None:
        box = [0.0, 0.0, 1.0, 1.0]
        self.labels = np.asarray(labels, dtype=np.intp)
        self.boxes = np.asarray([box] * len(labels), dtype=np.float32)
        if classes is None:
            self.scores = np.full(len(labels), confidence, dtype=np.float32)
        else:
            self.scores = np.zeros((len(labels), classes), dtype=np.float32)
            self.scores[np.arange(len(labels)), self.labels] = confidence


class MockODDataset:
    """Minimal object-detection dataset: per-image detection label lists.

    ``per_class`` scores every class of ``index2label`` rather than every box, the layout
    a vocabulary-sized score array arrives in.
    """

    def __init__(
        self,
        detections: Sequence[Sequence[int]],
        index2label: Mapping[int, str],
        *,
        per_class: bool = False,
        confidence: float = 1.0,
        dataset_id: str = "mock-od",
    ) -> None:
        self._detections = [list(d) for d in detections]
        self._classes = len(index2label) if per_class else None
        self._confidence = confidence
        self.metadata = {"id": dataset_id, "index2label": dict(index2label)}

    def __len__(self) -> int:
        return len(self._detections)

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        target = _ODTarget(self._detections[index], self._classes, self._confidence)
        return np.zeros((3, 4, 4), dtype=np.float32), target, {"id": index}


@pytest.fixture
def ic_dataset():
    return MockICDataset


@pytest.fixture
def od_dataset():
    return MockODDataset


class FactorODDataset:
    """Object detection with image-level factors and per-detection lists, which structure as instance factors."""

    def __init__(
        self,
        labels: Sequence[Sequence[int]],
        metadata: Sequence[Mapping[str, Any]],
        *,
        per_class: bool = False,
        index2label: Mapping[int, str] | None = None,
    ) -> None:
        self._labels = [list(item) for item in labels]
        self._metadata = [dict(item) for item in metadata]
        vocabulary = dict(index2label or {0: "car", 1: "person", 2: "boat"})
        self._classes = len(vocabulary) if per_class else None
        self.metadata = {"id": "factor-od", "index2label": vocabulary}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        target = _ODTarget(self._labels[index], self._classes)
        return (
            np.zeros((3, 4, 4), dtype=np.float32),
            target,
            cast(DatumMetadata, {"id": index, **self._metadata[index]}),
        )


class FactorICDataset:
    """Image classification with image-level factors."""

    def __init__(
        self, labels: Sequence[int], metadata: Sequence[Mapping[str, Any]], index2label: Mapping[int, str] | None = None
    ) -> None:
        self._labels = list(labels)
        self._metadata = [dict(item) for item in metadata]
        vocabulary = dict(index2label or {0: "car", 1: "person", 2: "boat"})
        self._n_classes = len(vocabulary)
        self.metadata = {"id": "factor-ic", "index2label": vocabulary}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        onehot = np.zeros(self._n_classes, dtype=np.float32)
        onehot[self._labels[index]] = 1.0
        return (
            np.zeros((3, 2, 2), dtype=np.float32),
            onehot,
            cast(DatumMetadata, {"id": index, **self._metadata[index]}),
        )


@dataclass
class _VideoFrame:
    frame_index: int
    pixels: np.ndarray


@dataclass
class _TrackedFrame:
    labels: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray


@dataclass
class _VideoTarget:
    frame_tracks: list[Any]


def _tracked_frame(detections: Sequence[tuple[int, int]]) -> _TrackedFrame:
    """One frame's detections, each a ``(label, track_id)`` pair, ``-1`` for one no tracker linked."""
    n = len(detections)
    return _TrackedFrame(
        labels=np.array([label for label, _ in detections], dtype=np.intp),
        boxes=np.tile(np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32), (n, 1)),
        scores=np.ones(n, dtype=np.float32),
        track_ids=np.array([track for _, track in detections], dtype=np.intp),
    )


class FactorVideos:
    """Multi-object tracking: sequence-level factors, and per-frame lists, which structure as frame factors."""

    def __init__(
        self, sequences: Sequence[Sequence[Sequence[tuple[int, int]]]], metadata: Sequence[Mapping[str, Any]]
    ) -> None:
        self._sequences = sequences
        self._metadata = [dict(item) for item in metadata]
        self.metadata = {"id": "factor-videos", "index2label": {0: "car", 1: "person"}}

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        frames = self._sequences[index]
        stream = [_VideoFrame(i, np.zeros((3, 8, 8), dtype=np.uint8)) for i in range(len(frames))]
        target = _VideoTarget([_tracked_frame(detections) for detections in frames])
        return stream, target, cast(DatumMetadata, {"id": f"v{index}", **self._metadata[index]})


class TimedVideos(FactorVideos):
    """:class:`FactorVideos` whose frames declare ``time_s`` and ``pts``, except frame 1 of sequence 0, which declares
    no ``time_s``."""

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        stream, target, metadata = super().__getitem__(index)
        frames = [
            SimpleNamespace(
                frame_index=i,
                pixels=frame.pixels,
                pts=i * 3000,
                **({} if (index, i) == (0, 1) else {"time_s": i * 0.5}),
            )
            for i, frame in enumerate(stream)
        ]
        return frames, target, metadata


# Image 3 is a night image with no detections; image 4 declares no altitude.
OD_LABELS = [[0, 1], [1], [2, 2], [], [0], [1, 2]]
OD_FACTORS = [
    {"tod": "night", "alt": 100.0, "occl": [0.1, 0.9]},
    {"tod": "day", "alt": 200.0, "occl": [0.5]},
    {"tod": "night", "alt": 300.0, "occl": [0.2, 0.3]},
    {"tod": "night", "alt": 400.0, "occl": []},
    {"tod": "dusk", "occl": [0.7]},
    {"tod": "night", "alt": 600.0, "occl": [0.4, 0.8]},
]
IC_LABELS = [0, 1, 2, 1, 0]
IC_FACTORS = [{"tod": "night"}, {"tod": "day"}, {"tod": "night"}, {"tod": "night"}, {"tod": "day"}]
# Sequence 0: track 1 (car) in frames 0-1, track 2 (person) in frames 0 and 2, an untracked car in frame 2.
# Sequence 1: track 7 (car) in both frames. Sequence 2: track 3 flickers car -> person -> car.
# track_length counts observations: tracks 1, 2 and 7 have 2, track 3 has 3.
VIDEOS = [
    [[(0, 1), (1, 2)], [(0, 1)], [(1, 2), (0, -1)]],
    [[(0, 7)], [(0, 7)]],
    [[(0, 3)], [(1, 3)], [(0, 3)]],
]
VIDEO_FACTORS = [
    {"weather": "rain", "tod": ["day", "night", "night"]},
    {"weather": "sun", "tod": ["day", "day"]},
    {"weather": "rain", "tod": ["day", "day", "day"]},
]


@pytest.fixture
def factor_od():
    return FactorODDataset(OD_LABELS, OD_FACTORS)


@pytest.fixture
def factor_ic():
    return FactorICDataset(IC_LABELS, IC_FACTORS)


@pytest.fixture
def factor_videos():
    return FactorVideos(VIDEOS, VIDEO_FACTORS)


@pytest.fixture
def timed_videos():
    return TimedVideos(VIDEOS, VIDEO_FACTORS)

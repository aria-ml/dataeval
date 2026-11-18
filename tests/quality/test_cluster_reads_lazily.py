"""Clustering across several datasets extracts features in batches as it reads, never holding every image at once."""

from typing import Any

import numpy as np
import pytest

from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers


class _Recording:
    """Indexable images that log each read to a shared event list. No ``__iter__``, so iteration indexes."""

    def __init__(self, name: str, images: np.ndarray, events: list[tuple[Any, ...]]) -> None:
        self._name, self._images, self._events = name, images, events

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> np.ndarray:
        if index >= len(self._images):
            raise IndexError(index)
        self._events.append(("read", self._name, index))
        return self._images[index]


class _Flatten:
    """Flattens each image, logging how many images each call is given."""

    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self._events = events

    def __call__(self, data: Any) -> np.ndarray:
        images = [np.asarray(image, dtype=np.float32).ravel() for image in data]
        self._events.append(("extract", len(images)))
        return np.stack(images)


def _datasets(events: list[tuple[Any, ...]]) -> tuple[_Recording, _Recording]:
    rng = np.random.default_rng(0)
    return (
        _Recording("a", rng.random((12, 3, 8, 8)), events),
        _Recording("b", rng.random((12, 3, 8, 8)), events),
    )


def _assert_streamed(events: list[tuple[Any, ...]], batch_size: int) -> None:
    # Item 0 of each dataset is also probed for its shape, before and after the walk, so the walk is the rest.
    walk = [i for i, event in enumerate(events) if event[0] == "read" and event[2] > 0]
    extracts = [event for event in events if event[0] == "extract"]
    first_extract = next(i for i, event in enumerate(events) if event[0] == "extract")
    assert first_extract < walk[-1], "every image was read before any was extracted"
    assert max(n for _, n in extracts) <= batch_size
    walked = [event[1:] for event in events if event[0] == "read" and event[2] > 0]
    assert len(walked) == len(set(walked)) == 22, "an image after the first was read twice, or not at all"


@pytest.mark.required
class TestMultiDatasetClusterReads:
    def test_duplicates_extracts_in_batches_while_reading(self):
        events: list[tuple[Any, ...]] = []
        detector = Duplicates(flags=ImageStats.NONE, extractor=_Flatten(events), cluster_sensitivity=1.0, batch_size=4)
        detector.evaluate(*_datasets(events))
        _assert_streamed(events, batch_size=4)

    def test_outliers_extracts_in_batches_while_reading(self):
        events: list[tuple[Any, ...]] = []
        detector = Outliers(flags=ImageStats.NONE, extractor=_Flatten(events), batch_size=4)
        detector.evaluate(*_datasets(events))
        _assert_streamed(events, batch_size=4)

from typing import cast

import numpy as np
import pytest

from dataeval.data import unzip_dataset
from dataeval.data._unzip import _unzip_annotations
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.utils._annotations import Annotations, build_annotations
from dataeval.utils.preprocessing import BoundingBox

pytestmark = pytest.mark.required

SHAPE = (3, 20, 40)
NAN = float("nan")


class _Target:
    """An object-detection target that counts how often its labels are read."""

    def __init__(self, boxes, labels) -> None:
        self.boxes = boxes
        self._labels = labels
        self.label_reads = 0
        self.scores = np.ones((len(labels), 1), dtype=np.float32)

    @property
    def labels(self):
        self.label_reads += 1
        return self._labels


class _Dataset:
    metadata = DatasetMetadata(id="unzip_annotations")

    def __init__(self, items) -> None:
        self.targets = [_Target(np.asarray(boxes, dtype=np.float32), np.asarray(labels)) for boxes, labels in items]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i):
        return np.zeros(SHAPE, dtype=np.float32), self.targets[i], DatumMetadata(id=i)

    @property
    def label_reads(self) -> int:
        return sum(t.label_reads for t in self.targets)


def annotations_of(dataset, **kwargs) -> list[Annotations]:
    images, annotations, _ = _unzip_annotations(dataset, per_target=True, **kwargs)
    assert annotations is not None
    list(images)
    return cast("list[Annotations]", list(annotations))


def test_dataset_raw_matches_the_explicit_path_including_a_reversed_box():
    rows = [(0, 0, NAN, 10), (30, 0, 10, 10)]
    (ann,) = annotations_of(_Dataset([(rows, [0, 1])]))

    expected = build_annotations(np.asarray(rows, dtype=np.float32), SHAPE, 0)
    assert isinstance(ann, Annotations)
    np.testing.assert_array_equal(ann.raw, expected.raw)
    assert ann.raw[1].tolist() == [30, 0, 10, 10]  # not re-derived from the swapped box
    assert [b.xyxy for b in ann.boxes] == [b.xyxy for b in expected.boxes]


@pytest.mark.parametrize("bad", [np.zeros((2, 5)), np.array([1, 2, 3, 4])], ids=["five_wide", "one_dimensional"])
def test_five_wide_and_one_dimensional_boxes_raise_naming_the_image(bad):
    dataset = _Dataset([([(1, 1, 5, 5)], [0]), ([(1, 1, 5, 5)], [0])])
    dataset.targets[1].boxes = bad

    with pytest.raises(ValueError, match=r"image 1, box 0.*expected 4 values"):
        annotations_of(dataset)


@pytest.mark.parametrize("empty", [np.empty((0, 4)), np.empty(0)])
def test_empty_boxes_are_accepted(empty):
    dataset = _Dataset([([(1, 1, 5, 5)], [0])])
    dataset.targets[0].boxes = empty

    (ann,) = annotations_of(dataset)
    assert ann.raw.shape == (0, 4)
    assert ann.boxes == []


def test_labels_are_read_only_when_asked():
    rows = [(1, 1, 5, 5), (2, 2, 9, 9)]
    without, asked = _Dataset([(rows, [3, 4])] * 2), _Dataset([(rows, [3, 4])] * 2)

    assert [a.labels for a in annotations_of(without)] == [None, None]
    labeled = [a.labels for a in annotations_of(asked, with_labels=True)]

    assert all(labels is not None for labels in labeled)
    assert [labels.tolist() for labels in labeled if labels is not None] == [[3, 4], [3, 4]]
    assert labeled[0] is not None
    assert labeled[0].dtype == np.int64
    # Runtime protocol checks may read the attribute too on some Pythons; asking for labels adds exactly one read each.
    assert asked.label_reads - without.label_reads == len(asked)


def test_unzip_dataset_still_returns_bounding_boxes_and_none_for_images_only():
    dataset = _Dataset([([(1, 1, 5, 5), (30, 0, 10, 10)], [0, 1])])
    images, targets = unzip_dataset(dataset, per_target=True)

    assert targets is not None
    (boxes,) = list(targets)
    assert boxes is not None
    assert all(isinstance(b, BoundingBox) for b in boxes)
    assert [b.xyxy for b in boxes] == [(1, 1, 5, 5), (10, 0, 30, 10)]
    assert len(list(images)) == 1

    _, no_targets = unzip_dataset(dataset, per_target=False)
    assert no_targets is None

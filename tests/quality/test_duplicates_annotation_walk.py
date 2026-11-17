"""The annotation walk of `Duplicates` reads a dataset only when it can carry annotations.

It decodes every image to look for boxes, so a classification or image-only dataset, which has none, paid a whole extra
pass to learn that. An object-detection dataset still pays it, once per item, and gets exactly the groups it did.
"""

from collections import Counter
from typing import Any, cast

import numpy as np
import pytest

from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Duplicates
from dataeval.quality._duplicates import _annotation_sources

pytestmark = pytest.mark.required

N = 30
PROBES = 12  # the repeated reads of item 0 that validation makes, which this change does not touch


class _Target:
    """An object-detection target."""

    def __init__(self, boxes: Any, labels: Any) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones((len(self.labels), 3), dtype=np.float32)


class _Dataset:
    """Items 20 to 29 repeat items 0 to 9. `kind` picks what each item is, and every read is counted."""

    metadata = DatasetMetadata(id="walk", index2label={0: "a", 1: "b", 2: "c"})

    def __init__(self, kind: str, size: int = N) -> None:
        self.kind, self.size = kind, size
        self.reads: Counter[int] = Counter()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Any:
        self.reads[index] += 1
        image = np.random.default_rng(index % 20).random((3, 16, 16), dtype=np.float32)
        if self.kind == "bare":
            return image
        metadata = cast(DatumMetadata, {"id": index})
        if self.kind == "ic":
            return image, np.eye(3, dtype=np.float32)[index % 3], metadata
        if self.kind == "od_first_empty" and index == 0:
            return image, _Target([], []), metadata
        return image, _Target([[1, 1, 8, 8], [2, 2, 12, 12]], [index % 3, (index + 1) % 3]), metadata

    @property
    def total(self) -> int:
        return sum(self.reads.values())


class TestAnnotationSources:
    @pytest.mark.parametrize("kind", ["ic", "bare"])
    def test_a_dataset_without_detection_targets_is_probed_once_and_not_walked(self, kind):
        dataset = _Dataset(kind)
        assert _annotation_sources(dataset) == {}
        assert dataset.total == 1

    def test_an_image_array_is_probed_once_and_not_walked(self):
        assert _annotation_sources(np.zeros((N, 3, 8, 8), dtype=np.float32)) == {}

    def test_a_detection_dataset_is_walked_once_per_item_with_no_extra_read(self):
        dataset = _Dataset("od")
        sources = _annotation_sources(dataset)
        assert len(sources) == N
        assert dataset.reads == Counter(dict.fromkeys(range(N), 1))

    def test_an_empty_first_target_is_still_a_detection_target_so_the_walk_runs(self):
        dataset = _Dataset("od_first_empty")
        sources = _annotation_sources(dataset)
        assert set(sources) == set(range(N))  # an empty detection target is still an annotation, of no boxes
        assert dataset.reads == Counter(dict.fromkeys(range(N), 1))

    def test_an_empty_dataset_reads_nothing(self):
        dataset = _Dataset("od", size=0)
        assert _annotation_sources(dataset) == {}
        assert dataset.total == 0

    def test_a_list_of_datasets_is_decided_for_each_one(self):
        detections, classes = _Dataset("od", size=10), _Dataset("ic", size=7)
        sources = _annotation_sources([detections, classes, _Dataset("od", size=5)])
        assert set(sources) == set(range(10)) | set(range(17, 22))
        assert detections.total == 10
        assert classes.total == 1


class TestDuplicatesEndToEnd:
    def test_a_classification_dataset_is_read_about_once_per_item(self):
        dataset = _Dataset("ic")
        result = Duplicates(hash_radius=0).evaluate(dataset)
        assert dataset.total <= N + PROBES
        assert sorted(sorted(group) for group in result.exact) == [[i, i + 20] for i in range(10)]

    def test_a_detection_dataset_still_pays_for_the_annotation_walk(self):
        classification, detection = _Dataset("ic"), _Dataset("od")
        Duplicates(hash_radius=0).evaluate(classification)
        Duplicates(hash_radius=0).evaluate(detection)
        assert detection.total <= 2 * N + PROBES
        assert detection.total - classification.total >= N - 2

    @pytest.mark.parametrize("kind", ["od", "od_first_empty"])
    def test_a_detection_dataset_still_reports_images_whose_annotations_disagree(self, kind):
        result = Duplicates(hash_radius=0).evaluate(_Dataset(kind))
        assert len(result.divergent().data()) > 0

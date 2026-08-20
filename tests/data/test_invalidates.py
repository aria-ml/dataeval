"""Tests for the transform-invalidation declaration and its walk over a wrapping chain."""

import numpy as np
import pytest

from dataeval.data import DetectionCrops, Relabel, View
from dataeval.data._invalidates import invalidated_stats
from dataeval.data._view import Operation
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, ObjectDetectionDatum


class Nop(Operation):
    """Operation that declares no invalidation."""

    def apply(self, view: View) -> None:
        pass


class Resizer(Operation):
    """Operation whose invalidation depends on constructor args, declared as a property."""

    def __init__(self, size: tuple[int, int] | None) -> None:
        self.size = size

    @property
    def invalidates(self) -> ImageStats:
        if self.size is None:
            return ImageStats.NONE
        return ImageStats.DIMENSION | ImageStats.VISUAL_SHARPNESS

    def apply(self, view: View) -> None:
        pass


class Images:
    """Minimal image-only dataset."""

    metadata = {"id": "images"}

    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        return np.full((3, 8, 8), index / 10, dtype=np.float32)


class _ODTarget:
    def __init__(self, boxes, labels) -> None:
        self._boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self._labels = np.asarray(labels, dtype=np.intp)
        self._scores = np.ones(len(self._labels), dtype=np.float32)

    @property
    def boxes(self) -> np.ndarray:
        return self._boxes

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def scores(self) -> np.ndarray:
        return self._scores


class _ODDataset:
    """Minimal object-detection dataset, one box per image."""

    def __init__(self, n: int = 3) -> None:
        self._n = n
        self.metadata = DatasetMetadata(id="toy-od", index2label={0: "a"})

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> ObjectDetectionDatum:
        image = np.full((3, 16, 16), index / 10, dtype=np.float32)
        return image, _ODTarget([[1, 1, 9, 9]], [0]), {"id": index}


@pytest.mark.required
class TestOperationInvalidatesDeclaration:
    def test_base_operation_defaults_to_none(self):
        assert Nop().invalidates is ImageStats.NONE

    def test_existing_operations_default_to_none(self):
        assert Relabel({"0": "a"}).invalidates is ImageStats.NONE

    def test_property_override_reads_constructor_args(self):
        assert Resizer(None).invalidates is ImageStats.NONE
        assert Resizer((2, 2)).invalidates == ImageStats.DIMENSION | ImageStats.VISUAL_SHARPNESS


@pytest.mark.required
class TestInvalidatedStats:
    def test_plain_dataset_returns_none(self):
        assert invalidated_stats(Images()) is ImageStats.NONE

    def test_bare_image_list_returns_none(self):
        assert invalidated_stats([np.zeros((3, 8, 8), dtype=np.float32)]) is ImageStats.NONE

    def test_empty_view_returns_none(self):
        assert invalidated_stats(View(Images())) is ImageStats.NONE

    def test_view_of_non_invalidating_op_returns_none(self):
        assert invalidated_stats(View(Images(), [Nop()])) is ImageStats.NONE

    def test_view_surfaces_operation_invalidation(self):
        view = View(Images(), [Resizer((2, 2))])
        assert invalidated_stats(view) == ImageStats.DIMENSION | ImageStats.VISUAL_SHARPNESS

    def test_nested_views_union_their_operations(self):
        inner = View(Images(), [Resizer((2, 2))])
        outer = View(inner, [Nop()])
        assert invalidated_stats(outer) == ImageStats.DIMENSION | ImageStats.VISUAL_SHARPNESS


@pytest.mark.required
class TestInvalidatedStatsCrossesNonViewBarrier:
    def test_crops_declares_dimension_except_channels(self):
        expected = ImageStats.DIMENSION & ~ImageStats.DIMENSION_CHANNELS
        assert DetectionCrops.invalidates == expected
        assert invalidated_stats(DetectionCrops(_ODDataset())) == expected

    def test_crops_does_not_invalidate_channels(self):
        assert not invalidated_stats(DetectionCrops(_ODDataset())) & ImageStats.DIMENSION_CHANNELS

    def test_walk_descends_a_wrapper_exposing_only_the_public_source(self):
        # The wrapper contract is `source`, not a private `_dataset` probe: a third-party
        # wrapper that names its parent the documented way is walked through.
        class ThirdPartyWrapper:
            metadata = {"id": "third-party"}

            def __init__(self, dataset) -> None:
                self.source = dataset

            def __len__(self) -> int:
                return len(self.source)

            def __getitem__(self, index: int):
                return self.source[index]

        wrapped = ThirdPartyWrapper(View(Images(), [Resizer((2, 2))]))
        assert invalidated_stats(wrapped) == ImageStats.DIMENSION | ImageStats.VISUAL_SHARPNESS

    def test_wrapping_view_is_not_credited_with_the_crop_invalidation(self):
        view = View(DetectionCrops(_ODDataset()))
        assert getattr(view, "invalidates", ImageStats.NONE) is ImageStats.NONE

    def test_walk_descends_through_a_non_view_wrapper(self):
        # View(DetectionCrops(View(base, [op]))) — operation_groups alone stops at the
        # DetectionCrops barrier and would silently miss the inner operation.
        inner = View(_ODDataset(), [Resizer((2, 2))])
        outer = View(DetectionCrops(inner))
        result = invalidated_stats(outer)
        assert result & ImageStats.VISUAL_SHARPNESS
        assert result & ImageStats.DIMENSION_WIDTH

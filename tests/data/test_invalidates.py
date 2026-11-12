"""Tests for the transform-invalidation declaration and its walk over a wrapping chain."""

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._image_stats import _unmeasurable_flags
from dataeval.data import (
    ClassFilter,
    Crop,
    DetectionCrops,
    Filter,
    Indices,
    Relabel,
    Resize,
    Reverse,
    SelectChannels,
    TorchvisionTransform,
    View,
)
from dataeval.data._invalidates import invalidated_stats
from dataeval.data._view import Operation
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, ObjectDetectionDatum
from dataeval.types import SourceIndex


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
        assert Reverse().invalidates is ImageStats.NONE

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


@pytest.mark.required
class TestBlurIsInvalidated:
    """`ALL` holds the blur flags, so a declaration of everything covers them. An operation that names its flags
    one by one, because it does not declare `ALL`, needs the blur entry of its own.

    Resizing changes how wide an edge is, and a gray mix changes which picture is read.
    """

    def test_a_declaration_of_all_covers_blur(self):
        class Everything(Operation):
            @property
            def invalidates(self) -> ImageStats:
                return ImageStats.ALL

            def apply(self, view: View) -> None:
                pass

        assert ImageStats.BLUR in Everything().invalidates
        assert ImageStats.BLUR in invalidated_stats(View(Images(), [Everything()]))

    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    def test_resize_invalidates_blur(self, mode):
        assert ImageStats.BLUR in Resize((8, 8), mode=mode).invalidates

    def test_gray_selection_invalidates_blur(self):
        assert ImageStats.BLUR in SelectChannels("gray").invalidates

    def test_other_selections_leave_blur_alone(self):
        assert not SelectChannels([0, 1]).invalidates & ImageStats.BLUR
        assert not SelectChannels("rgb").invalidates & ImageStats.BLUR

    def test_torchvision_default_is_all_and_so_covers_blur(self):
        assert TorchvisionTransform(object()).invalidates == ImageStats.ALL
        assert ImageStats.BLUR in TorchvisionTransform(object()).invalidates
        assert ImageStats.SPATIAL in TorchvisionTransform(object()).invalidates

    def test_every_audit_point_covers_blur(self):
        """One place for the hand-kept list: a new transform or a new check that skips blur fails here.

        `wide_band_names`, the other hand-kept list, is a local of `image_stats` and cannot be called alone. It is
        guarded by `test_wide_cube_warns_and_names_blur` in `tests/core/test_blur_stats.py`.
        """
        assert all(ImageStats.BLUR in Resize((8, 8), mode=mode).invalidates for mode in ("stretch", "pad", "crop"))
        assert ImageStats.BLUR in SelectChannels("gray").invalidates
        assert ImageStats.BLUR in TorchvisionTransform(object()).invalidates
        assert ImageStats.BLUR in _unmeasurable_flags(ImageStats.BLUR, None, False)
        view = View(Images(), [Resize((8, 8)), SelectChannels("gray")])
        assert ImageStats.BLUR in invalidated_stats(view)


class _EdgeDataset:
    """One 16x16 image whose one box, (8, 0, 24, 8), lies half outside it."""

    def __init__(self) -> None:
        self.metadata = DatasetMetadata(id="edge-od", index2label={0: "a"})

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> ObjectDetectionDatum:
        return np.zeros((3, 16, 16), dtype=np.float32), _ODTarget([[8, 0, 24, 8]], [0]), {"id": index}


SPATIAL = ImageStats.SPATIAL


@pytest.mark.required
@pytest.mark.filterwarnings("ignore")
class TestSpatialDeclarations:
    """What changes a box or removes its peers makes the SPATIAL statistics describe the operation."""

    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    def test_resize_declares_spatial_in_every_mode(self, mode):
        assert Resize((8, 8), mode=mode).invalidates & SPATIAL == SPATIAL

    def test_crop_declares_spatial(self):
        assert Crop((0, 0, 8, 8)).invalidates & SPATIAL == SPATIAL

    @pytest.mark.parametrize(
        "operation",
        [Filter(classes=[0]), Filter(classes=[0], drop="detections"), ClassFilter([0])],
        ids=["filter_both", "filter_detections", "class_filter"],
    )
    def test_a_filter_that_drops_detections_declares_spatial(self, operation):
        assert operation.invalidates & SPATIAL == SPATIAL

    @pytest.mark.parametrize(
        "operation",
        [Filter(classes=[0], drop="items"), ClassFilter([0], filter_detections=False)],
        ids=["filter_items", "class_filter_keeping_detections"],
    )
    def test_a_filter_that_keeps_every_detection_of_the_items_it_keeps_does_not(self, operation):
        assert not operation.invalidates & SPATIAL

    def test_indices_that_remove_rows_inside_items_declare_spatial(self):
        assert Indices([SourceIndex(0, 1)], exclude=True).invalidates & SPATIAL == SPATIAL

    @pytest.mark.parametrize(
        "operation",
        [Indices([0, 1], exclude=True), Indices([0, 1]), Indices([SourceIndex(0)], exclude=True)],
        ids=["whole_items_removed", "items_kept", "item_address_removed"],
    )
    def test_indices_that_keep_every_row_of_the_items_they_keep_do_not(self, operation):
        assert not operation.invalidates & SPATIAL

    def test_relabel_declares_the_group_when_it_drops_detections_and_the_class_statistics_always(self):
        classes = ImageStats.SPATIAL_MAX_IOU_SAME_CLASS | ImageStats.SPATIAL_MAX_IOU_OTHER_CLASS

        assert Relabel({"0": "a"}).invalidates & SPATIAL == SPATIAL
        # A relabel changes which boxes share a class, so these two describe another vocabulary even when no box is
        # dropped; the geometry of a box and of its peers is unchanged.
        assert Relabel({"0": "a"}, on_unmatched="raise").invalidates == classes

    def test_select_channels_does_not(self):
        assert not SelectChannels("gray").invalidates & SPATIAL

    def test_the_torchvision_default_covers_spatial_and_all(self):
        declared = TorchvisionTransform(lambda image: image).invalidates
        assert declared & SPATIAL == SPATIAL
        assert declared & ImageStats.ALL == ImageStats.ALL

    def test_a_resize_clips_the_box_so_the_fraction_reads_one_afterwards(self):
        def fraction(data):
            result = image_stats(data, stats=ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION, per_image=False)
            return result["stats"]["box_in_image_fraction"].tolist()

        assert fraction(_EdgeDataset()) == [0.5]
        assert fraction(View(_EdgeDataset(), [Resize((8, 8))])) == [1.0]
        assert fraction(View(_EdgeDataset(), [Crop((4, 0, 12, 8))])) == [1.0]

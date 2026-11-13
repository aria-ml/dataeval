"""View operations that move a box's pixels or its peers declare CONTEXT invalidated, and the others leave it alone."""

import pytest

from dataeval.data import (
    ClassFilter,
    Crop,
    DetectionCrops,
    Filter,
    Indices,
    Relabel,
    Resize,
    SelectChannels,
    TorchvisionTransform,
)
from dataeval.flags import ImageStats
from dataeval.types import SourceIndex

pytestmark = pytest.mark.required

CONTEXT = ImageStats.CONTEXT


class TestDeclarations:
    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    def test_resize_declares_it_in_every_mode(self, mode):
        # A resize moves every pixel of the ring and clips the boxes to the new canvas, and a letterbox adds bars to it.
        assert Resize((8, 8), mode=mode).invalidates & CONTEXT == CONTEXT  # pyright: ignore[reportArgumentType]

    def test_crop_declares_it(self):
        assert Crop((0, 0, 8, 8)).invalidates & CONTEXT == CONTEXT

    @pytest.mark.parametrize(
        "operation",
        [Filter(classes=[0]), Filter(classes=[0], drop="detections"), ClassFilter([0])],
        ids=["filter_both", "filter_detections", "class_filter"],
    )
    def test_a_filter_that_drops_detections_declares_it_because_the_peers_it_removes_leave_the_surround(
        self, operation
    ):
        assert operation.invalidates & CONTEXT == CONTEXT

    @pytest.mark.parametrize(
        "operation",
        [Filter(classes=[0], drop="items"), ClassFilter([0], filter_detections=False)],
        ids=["filter_items", "class_filter_keeping_detections"],
    )
    def test_a_filter_that_keeps_every_detection_of_the_items_it_keeps_does_not(self, operation):
        assert not operation.invalidates & CONTEXT

    def test_indices_that_remove_rows_inside_items_declare_it(self):
        assert Indices([SourceIndex(0, 1)], exclude=True).invalidates & CONTEXT == CONTEXT

    @pytest.mark.parametrize(
        "operation",
        [Indices([0, 1], exclude=True), Indices([0, 1]), Indices([SourceIndex(0)], exclude=True)],
        ids=["whole_items_removed", "items_kept", "item_address_removed"],
    )
    def test_indices_that_keep_every_row_of_the_items_they_keep_do_not(self, operation):
        assert not operation.invalidates & CONTEXT

    def test_relabel_declares_it_only_when_it_drops_detections(self):
        # Dropping removes peers. Changing a class moves no pixel and no box, and the surround reads no class.
        assert Relabel({"0": "a"}).invalidates & CONTEXT == CONTEXT
        assert not Relabel({"0": "a"}, on_unmatched="raise").invalidates & CONTEXT

    def test_gray_declares_it_because_a_mix_of_channels_is_not_the_channel_mean_it_averages(self):
        assert SelectChannels("gray").invalidates & CONTEXT == CONTEXT

    @pytest.mark.parametrize("channels", [[0], [0, 1], "rgb"])
    def test_other_selections_keep_the_surviving_channels_values_so_they_do_not(self, channels):
        assert not SelectChannels(channels).invalidates & CONTEXT

    def test_detection_crops_do_not_because_a_crop_has_no_boxes_of_its_own_to_measure(self):
        assert not DetectionCrops.invalidates & CONTEXT

    def test_the_torchvision_default_covers_it_through_all(self):
        assert TorchvisionTransform(object()).invalidates & CONTEXT == CONTEXT

    def test_an_override_replaces_the_computed_declaration(self):
        assert not Resize((8, 8), invalidates=ImageStats.DIMENSION).invalidates & CONTEXT

    def test_the_default_request_meets_no_declaration_so_outliers_default_raises_nothing_new(self):
        default = ImageStats.STANDARD | ImageStats.HASH
        assert not Resize((8, 8)).invalidates & default & CONTEXT
        assert not Crop((0, 0, 8, 8)).invalidates & default & CONTEXT

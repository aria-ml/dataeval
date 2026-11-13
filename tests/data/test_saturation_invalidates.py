"""View operations that move values declare SATURATION invalidated, and the others leave it alone."""

import numpy as np
import pytest

from dataeval.data import Crop, DetectionCrops, Indices, Resize, SelectChannels, TorchvisionTransform, View
from dataeval.exceptions import StatsInvalidatedWarning
from dataeval.flags import ImageStats
from dataeval.quality._shared import checked_compute_stats

pytestmark = pytest.mark.required

SATURATION = ImageStats.SATURATION


class TestDeclarations:
    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    def test_resize_declares_it_in_every_mode(self, mode: str):
        # It returns floats, so the dtype endpoint no longer applies, and interpolation moves values at a clip's edge.
        assert Resize((8, 8), mode=mode).invalidates & SATURATION == SATURATION  # pyright: ignore[reportArgumentType]

    def test_gray_declares_it_because_a_mix_of_channels_is_not_at_the_endpoint_where_one_channel_was(self):
        assert SelectChannels("gray").invalidates & SATURATION == SATURATION

    @pytest.mark.parametrize("channels", [[0], [0, 1], "rgb"])
    def test_other_selections_keep_the_surviving_channels_values_so_they_do_not(self, channels: object):
        assert not SelectChannels(channels).invalidates & SATURATION  # pyright: ignore[reportArgumentType]

    def test_detection_crops_declare_it(self):
        assert DetectionCrops.invalidates & SATURATION == SATURATION

    def test_a_crop_and_a_frame_selection_leave_the_values_alone_so_they_do_not(self):
        assert not Crop((0, 0, 8, 8)).invalidates & SATURATION
        assert not Indices([0]).invalidates & SATURATION

    def test_the_torchvision_default_covers_it_through_all(self):
        assert TorchvisionTransform(object()).invalidates & SATURATION == SATURATION

    def test_an_override_replaces_the_computed_declaration(self):
        assert not Resize((8, 8), invalidates=ImageStats.DIMENSION).invalidates & SATURATION


class _Images:
    metadata = {"id": "images"}

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> np.ndarray:
        return np.full((3, 8, 8), 255, dtype=np.uint8)


class TestTheWarning:
    def test_a_saturation_request_through_a_resize_warns_and_names_the_statistic(self):
        view = View(_Images(), [Resize((4, 4))])  # pyright: ignore[reportArgumentType]
        with pytest.warns(StatsInvalidatedWarning, match="saturation") as caught:
            checked_compute_stats([view], stats=SATURATION, caller="Outliers")
        assert len(caught) == 1

    def test_the_same_request_on_a_plain_dataset_does_not_warn(self, recwarn: pytest.WarningsRecorder):
        checked_compute_stats([_Images()], stats=SATURATION, caller="Outliers")
        assert not [w for w in recwarn if issubclass(w.category, StatsInvalidatedWarning)]

    def test_a_pixel_request_through_a_resize_warns_for_saturation_alone(self):
        # Saturation is in `PIXEL`, so a request for the group meets the declaration, and only for that statistic.
        view = View(_Images(), [Resize((4, 4), mode="stretch")])  # pyright: ignore[reportArgumentType]
        with pytest.warns(StatsInvalidatedWarning, match=r"requested by Outliers: saturation\.") as caught:
            checked_compute_stats([view], stats=ImageStats.PIXEL, caller="Outliers")
        assert len(caught) == 1

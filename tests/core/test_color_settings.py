"""The color keywords: three plain settings of `image_stats`, checked before any image is read."""

import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._image_stats import _settings
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required

IMAGES = [np.random.default_rng(0).integers(0, 256, (3, 16, 16)).astype(np.uint8) for _ in range(2)]


def settings(**kw: Any):
    given = {"color_rgb_channels": None, "color_min_pixels": 32, "color_chroma_threshold": 12.0} | kw
    return _settings(0.0, 0.25, 32, 1 / 255, **given)


@pytest.mark.parametrize(
    "bad", [(0, 1), (0, 1, 1), (-1, 0, 1), (0, 1, 2.5), "rgb", True, (0, 1, 2, 3), (0, True, 2), 5]
)
def test_a_bad_declaration_raises_naming_the_keyword(bad: Any):
    with pytest.raises(ValueError, match="color_rgb_channels"):
        image_stats(IMAGES, stats=ImageStats.PIXEL_MEAN, color_rgb_channels=bad)


@pytest.mark.parametrize(
    "spelling", [[0, 1, 2], (0, 1, 2), np.array([0, 1, 2]), [np.int64(0), np.int64(1), np.int64(2)]]
)
def test_list_tuple_and_array_read_alike(spelling: Any):
    channels = settings(color_rgb_channels=spelling).color.rgb_channels
    assert channels == (0, 1, 2)
    assert all(type(c) is int for c in channels or ())


def test_the_order_is_kept():
    assert settings(color_rgb_channels=(2, 1, 0)).color.rgb_channels == (2, 1, 0)


def test_no_declaration_is_none_and_the_defaults_hold():
    color = settings().color
    assert color.rgb_channels is None
    assert (color.min_pixels, color.min_valid_fraction, color.chroma_threshold) == (32, 0.90, 12.0)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, float("nan"), "32"])
def test_the_floor_is_a_whole_number_of_one_or_more(bad: Any):
    with pytest.raises(ValueError, match="color_min_pixels"):
        image_stats(IMAGES, stats=ImageStats.PIXEL_MEAN, color_min_pixels=bad)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "12"])
def test_the_threshold_is_a_finite_number_above_zero(bad: Any):
    with pytest.raises(ValueError, match="color_chroma_threshold"):
        image_stats(IMAGES, stats=ImageStats.PIXEL_MEAN, color_chroma_threshold=bad)


def test_keywords_given_with_no_color_flag_requested_warn_once_in_one_message():
    with pytest.warns(UserWarning, match=r"color_min_pixels and color_chroma_threshold") as caught:
        image_stats(IMAGES, stats=ImageStats.PIXEL_MEAN, color_chroma_threshold=5.0, color_min_pixels=10)
    assert len([w for w in caught if "color_" in str(w.message)]) == 1


def test_the_defaults_given_explicitly_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image_stats(IMAGES, stats=ImageStats.PIXEL_MEAN, color_min_pixels=32, color_chroma_threshold=12.0)

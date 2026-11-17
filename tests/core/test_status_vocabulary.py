"""One status vocabulary for the v1.3 families, one meaning of `invalid_box`, and `ok` always with a value."""

import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import BoundingBox, BoundingBoxFormat

pytestmark = pytest.mark.required

VOCABULARY = {
    "ok",
    "not_applicable",
    "invalid_box",
    "reversed_box",
    "missing_channel",
    "unknown_range",
    "empty_surround",
    "insufficient_support",
}
GOOD = [10.0, 10.0, 40.0, 40.0]
REVERSED = [70.0, 50.0, 50.0, 30.0]  # its twin (50, 30, 70, 50) lies inside the image
REVERSED_OUTSIDE = [500.0, 500.0, 300.0, 300.0]  # its twin lies outside the image
OUTSIDE = [300.0, 300.0, 320.0, 320.0]
ZERO = [20.0, 20.0, 20.0, 30.0]
NAN = [np.nan, 5.0, 10.0, 10.0]


def _image() -> np.ndarray:
    return np.random.default_rng(3).integers(0, 256, (3, 96, 128), dtype=np.uint8)


def run(boxes: list[list[float]], stats: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return image_stats([_image()], boxes=[boxes], labels=[[0] * len(boxes)], stats=stats, **kwargs)


def statuses(result: Any, name: str) -> list[str]:
    return [str(s) for s in result["details"][name]]


class TestBoxRule:
    @pytest.mark.parametrize("family", ["spatial", "context"])
    def test_a_reversed_box_with_a_valid_twin_is_reversed_box_where_geometry_is_read(self, family):
        stats = (
            ImageStats.SPATIAL_COVERED_FRACTION if family == "spatial" else ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN
        )
        result = run([GOOD, REVERSED], stats, per_image=False)
        assert statuses(result, f"{family}_status") == ["ok", "reversed_box"]
        assert np.isnan(next(iter(result["stats"].values()))[1])

    @pytest.mark.parametrize("family", ["spatial", "context"])
    def test_a_reversed_box_whose_twin_is_invalid_is_invalid_box(self, family):
        stats = (
            ImageStats.SPATIAL_COVERED_FRACTION if family == "spatial" else ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN
        )
        result = run([GOOD, REVERSED_OUTSIDE], stats, per_image=False)
        assert statuses(result, f"{family}_status")[1] == "invalid_box"

    @pytest.mark.parametrize(
        ("flag", "status"),
        [
            (ImageStats.SPATIAL_COVERED_FRACTION, "spatial_status"),
            (ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN, "context_status"),
            (ImageStats.PIXEL_SATURATED_FRACTION, "saturated_status"),
            (ImageStats.VISUAL_COLOR_LIGHTNESS_MEAN, "color_status"),
            (ImageStats.PIXEL_NOISE_STD, "noise_status"),
        ],
    )
    def test_outside_zero_area_and_nan_boxes_are_invalid_box_in_every_new_family(self, flag, status):
        result = run([OUTSIDE, ZERO, NAN], flag, per_image=False, color_rgb_channels=(0, 1, 2))
        assert statuses(result, status) == ["invalid_box"] * 3
        assert np.isnan(next(iter(result["stats"].values()))).all()

    @pytest.mark.parametrize(
        ("flag", "column", "status"),
        [
            (ImageStats.PIXEL_SATURATED_FRACTION, "saturated_fraction", "saturated_status"),
            (ImageStats.VISUAL_COLOR_LIGHTNESS_MEAN, "color_lightness_mean", "color_status"),
            (ImageStats.PIXEL_NOISE_STD, "noise_std", "noise_status"),
        ],
    )
    def test_pixel_families_measure_a_reversed_box_as_its_twin(self, flag, column, status):
        twin = [REVERSED[2], REVERSED[3], REVERSED[0], REVERSED[1]]
        reversed_ = run([REVERSED], flag, per_image=False, color_rgb_channels=(0, 1, 2))
        swapped = run([twin], flag, per_image=False, color_rgb_channels=(0, 1, 2))
        assert statuses(reversed_, status) == ["ok"]
        assert reversed_["stats"][column][0] == swapped["stats"][column][0]

    def test_no_status_contradicts_the_invalid_box_column(self):
        negative_width = BoundingBox(40, 40, -15, -15, bbox_format=BoundingBoxFormat.XYWH, image_shape=(3, 96, 128))
        boxes: list[Any] = [GOOD, REVERSED, REVERSED_OUTSIDE, OUTSIDE, ZERO, NAN, negative_width]
        result = run(
            boxes,
            ImageStats.DIMENSION_INVALID_BOX
            | ImageStats.SPATIAL
            | ImageStats.CONTEXT
            | ImageStats.PIXEL_SATURATED_FRACTION
            | ImageStats.NOISE,
            per_image=False,
        )
        column = [bool(v) for v in result["stats"]["invalid_box"]]
        for name in ("spatial_status", "context_status", "saturated_status", "noise_status"):
            for flagged, status in zip(column, statuses(result, name), strict=True):
                assert (status == "invalid_box") == flagged, (name, status, flagged)


class TestIncompleteGeometry:
    @pytest.mark.parametrize("family", ["spatial", "context"])
    def test_a_box_beside_a_reversed_one_is_ok_with_a_value_and_flagged(self, family):
        stats = (
            ImageStats.SPATIAL_COVERED_FRACTION if family == "spatial" else ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN
        )
        result = run([GOOD, REVERSED], stats, per_image=False)
        assert statuses(result, f"{family}_status")[0] == "ok"
        assert not np.isnan(next(iter(result["stats"].values()))[0])
        assert result["details"][f"{family}_incomplete_geometry"].tolist() == [True, False]

    @pytest.mark.parametrize("family", ["spatial", "context"])
    def test_a_complete_frame_is_not_flagged_and_an_image_row_is_not_applicable(self, family):
        stats = (
            ImageStats.SPATIAL_COVERED_FRACTION if family == "spatial" else ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN
        )
        result = run([GOOD], stats)
        assert statuses(result, f"{family}_status") == ["not_applicable", "ok"]
        assert result["details"][f"{family}_incomplete_geometry"].tolist() == [False, False]


class TestFamilyStatuses:
    def test_context_a_box_that_fills_its_window_is_empty_surround(self):
        result = run([[0.0, 0.0, 128.0, 96.0]], ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN, per_image=False)
        assert statuses(result, "context_status") == ["empty_surround"]

    @pytest.mark.parametrize(
        ("flag", "status"),
        [
            (ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN, "far_context_status"),
            (ImageStats.PIXEL_SATURATED_FRACTION, "far_saturated_status"),
            (ImageStats.PIXEL_NOISE_STD, "far_noise_status"),
        ],
    )
    def test_a_channel_group_the_datum_cannot_supply_is_missing_channel(self, flag, status):
        result = run([GOOD], {None: ImageStats.PIXEL_MEAN, "far": flag}, per_image=False, channels={"far": [7]})
        assert statuses(result, status) == ["missing_channel"]

    def test_color_a_missing_declared_channel_is_missing_channel(self):
        result = run([GOOD], ImageStats.VISUAL_COLOR_LIGHTNESS_MEAN, per_image=False, color_rgb_channels=(0, 1, 5))
        assert statuses(result, "color_status") == ["missing_channel"]

    def test_saturation_without_an_endpoint_is_unknown_range(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = image_stats([np.full((3, 8, 8), 1, dtype=np.uint16)], stats=ImageStats.PIXEL_SATURATED_FRACTION)
        assert statuses(result, "saturated_status") == ["unknown_range"]

    def test_saturation_of_an_all_nan_view_or_a_covered_background_is_insufficient_support(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nan = image_stats(
                [np.full((3, 8, 8), np.nan)], stats=ImageStats.PIXEL_SATURATED_FRACTION, value_range=(0, 1)
            )
        assert statuses(nan, "saturated_status") == ["insufficient_support"]
        covered = run([[0.0, 0.0, 128.0, 96.0]], ImageStats.PIXEL_SATURATED_FRACTION, per_background=True)
        assert statuses(covered, "background_saturated_status")[0] == "insufficient_support"

    @pytest.mark.parametrize(
        ("flag", "status"),
        [
            (ImageStats.PIXEL_SATURATED_FRACTION, "background_saturated_status"),
            (ImageStats.VISUAL_COLOR_LIGHTNESS_MEAN, "background_color_status"),
            (ImageStats.PIXEL_NOISE_STD, "background_noise_status"),
        ],
    )
    def test_a_status_on_a_row_its_view_does_not_define_is_not_applicable(self, flag, status):
        result = run([GOOD], flag, per_background=True, color_rgb_channels=(0, 1, 2))
        assert statuses(result, status) == ["ok", "not_applicable"]


class TestEveryFamily:
    @pytest.mark.parametrize("data", ["uint8", "float_unknown", "uint16"])
    def test_statuses_come_from_the_vocabulary_and_ok_always_has_a_value(self, data):
        image = {"uint8": _image(), "float_unknown": _image() * 3.0, "uint16": _image().astype(np.uint16) * 16}[data]
        boxes = [GOOD, REVERSED, REVERSED_OUTSIDE, OUTSIDE, ZERO, NAN, [0.0, 0.0, 128.0, 96.0]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result: Any = image_stats(
                [image],
                boxes=[boxes],
                labels=[[0] * len(boxes)],
                stats=ImageStats.ALL,
                per_background=True,
                color_rgb_channels=(0, 1, 2),
                channels={"far": [7], "rg": [0, 1]},
            )
        families = {
            "spatial_status": ["covered_fraction"],
            "context_status": ["surround_intensity_mean"],
            "saturated_status": ["saturated_fraction"],
            "color_status": ["color_lightness_mean"],
            "noise_status": ["noise_std", "noise_var"],
        }
        for status_name, columns in families.items():
            for prefix in ("", "background_", "rg_", "far_"):
                name = f"{prefix}{status_name}"
                if name not in result["details"]:
                    continue
                values = set(statuses(result, name))
                assert values <= VOCABULARY, (name, values - VOCABULARY)
                for column in columns:
                    full = f"{prefix}{column}"
                    if full not in result["stats"]:
                        continue
                    for status, value in zip(statuses(result, name), result["stats"][full], strict=True):
                        assert (status == "ok") == (not np.isnan(value)), (full, status, value)


@pytest.mark.parametrize(
    ("stats", "status"),
    [
        (ImageStats.PIXEL_SATURATED_FRACTION, "g_saturated_status"),
        (ImageStats.VISUAL_BLUR_EDGE_WIDTH, "g_blur_status"),
        (ImageStats.PIXEL_NOISE_STD, "g_noise_status"),
    ],
)
def test_a_channel_group_on_a_one_dimensional_datum_is_missing_channel(stats, status):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = image_stats([np.arange(50, dtype=np.uint8)], stats={"g": stats}, channels={"g": [0]})
    assert statuses(result, status) == ["missing_channel"]

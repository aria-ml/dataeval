"""The color statistics: values, statuses, details and views, read from the calculator and through `image_stats`."""

import math
import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._calculators import _colorstats
from dataeval.core._calculators._base import ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._colorstats import ColorStatCalculator
from dataeval.flags import ImageStats
from dataeval.types._stats_config import ColorConfig, ImageStatsConfig
from dataeval.utils.preprocessing import BoundingBox, get_value_range

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

COLOR = ImageStats.COLOR
STATS = [
    "color_lightness_mean",
    "color_chroma_mean",
    "color_achromatic_fraction",
    "color_hue_cos_mean",
    "color_hue_sin_mean",
]
DETAILS = ["color_status", "color_valid_fraction", "color_valid_pixels", "color_excluded_pixels", "color_hue_reason"]


def details_of(result: Any) -> Any:
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def solid(rgb: tuple[int, int, int], size: int = 8, dtype: Any = np.uint8) -> np.ndarray:
    return np.broadcast_to(np.array(rgb, dtype=dtype)[:, None, None], (3, size, size)).copy()


def measure(
    datum: np.ndarray,
    *,
    channels: tuple[int, int, int] | None = (0, 1, 2),
    declared: tuple[float, float] | None = (0, 255),
    min_pixels: int = 1,
    threshold: float = 12.0,
    box: tuple[int, int, int, int] | None = None,
    exclude: np.ndarray | None = None,
    view: ViewKind = ViewKind.WHOLE,
) -> dict[str, Any]:
    """One row's statistics and details, read from the calculator itself, as one flat mapping of scalars."""
    cache = CalculatorCache(
        datum,
        None if box is None else BoundingBox(*box, image_shape=datum.shape),
        exclude=exclude,
        value_range=None if declared is None else get_value_range(datum, declared=declared),
        settings=ImageStatsConfig(
            color=ColorConfig(rgb_channels=channels, min_pixels=min_pixels, chroma_threshold=threshold)
        ),
    )
    calculator = ColorStatCalculator(datum, cache)
    flat = {**calculator.compute(COLOR, view), **calculator.details(COLOR, view)}
    return {name: values[0] for name, values in flat.items()}


class TestValues:
    def test_a_red_image_reads_the_reference_triple(self):
        got = measure(solid((255, 0, 0)))
        assert got["color_status"] == "ok"
        assert got["color_lightness_mean"] == pytest.approx(53.2408, abs=0.01)
        assert got["color_chroma_mean"] == pytest.approx(104.5518, abs=0.01)
        assert got["color_hue_cos_mean"] == pytest.approx(0.76606, abs=1e-4)
        assert got["color_hue_sin_mean"] == pytest.approx(0.64277, abs=1e-4)
        assert got["color_achromatic_fraction"] == 0.0
        assert got["color_hue_reason"] == ""

    def test_a_gray_image_is_achromatic_with_no_hue(self):
        got = measure(solid((128, 128, 128)))
        assert got["color_status"] == "ok"
        assert got["color_lightness_mean"] == pytest.approx(53.585, abs=0.01)
        assert got["color_chroma_mean"] < 1e-3
        assert got["color_achromatic_fraction"] == 1.0
        assert math.isnan(got["color_hue_cos_mean"])
        assert math.isnan(got["color_hue_sin_mean"])
        assert got["color_hue_reason"] == "achromatic"

    def test_half_gray_half_red_reads_hue_from_the_red_half_only(self):
        image = solid((128, 128, 128))
        image[:, :, 4:] = np.array([255, 0, 0], dtype=np.uint8)[:, None, None]
        got = measure(image)
        assert got["color_achromatic_fraction"] == 0.5
        assert got["color_hue_cos_mean"] == pytest.approx(0.76606, abs=1e-4)

    def test_the_threshold_moves_what_is_achromatic(self):
        faint = solid((140, 128, 128))  # a few levels off a gray: chroma of about 4
        assert measure(faint, threshold=12.0)["color_achromatic_fraction"] == 1.0
        assert measure(faint, threshold=2.0)["color_achromatic_fraction"] == 0.0

    def test_alpha_is_ignored_and_the_declared_order_is_read(self):
        rgb = np.random.default_rng(0).integers(0, 256, (3, 8, 8)).astype(np.uint8)
        rgba = np.concatenate([rgb, np.full((1, 8, 8), 255, dtype=np.uint8)])
        bgr = rgb[::-1].copy()
        want = measure(rgb)
        for datum, channels in ((rgba, (0, 1, 2)), (bgr, (2, 1, 0))):
            got = measure(datum, channels=channels)
            for name in STATS:
                assert got[name] == want[name], name
        assert measure(bgr, channels=(0, 1, 2))["color_chroma_mean"] != want["color_chroma_mean"]


class TestStatuses:
    @pytest.mark.parametrize(
        "datum",
        [
            np.zeros((1, 8, 8), dtype=np.uint8),
            np.zeros((2, 8, 8), dtype=np.uint8),
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros(8, dtype=np.uint8),
        ],
        ids=["gray", "two_channels", "two_dimensional", "one_dimensional"],
    )
    def test_a_datum_without_the_declared_channels_is_missing_band(self, datum: np.ndarray):
        got = measure(datum)
        assert got["color_status"] == "missing_band"
        assert all(math.isnan(got[name]) for name in STATS)
        assert got["color_valid_pixels"] == -1
        assert got["color_hue_reason"] == ""

    def test_an_index_the_datum_does_not_carry_is_missing_band(self):
        assert measure(solid((1, 2, 3)), channels=(0, 1, 5))["color_status"] == "missing_band"

    def test_an_undeclared_range_on_anything_but_uint8_is_unknown_range(self):
        got = measure(np.full((3, 8, 8), 0.5, dtype=np.float32), declared=None)
        assert got["color_status"] == "unknown_range"
        assert math.isnan(got["color_chroma_mean"])
        assert got["color_valid_pixels"] == -1

    def test_nan_infinity_and_values_outside_the_range_are_not_valid_and_never_clipped(self):
        image = np.full((3, 4, 4), 0.5, dtype=np.float32)
        image[0, 0, 0] = np.nan
        image[1, 0, 1] = np.inf
        image[2, 0, 2] = 1.5  # outside the declared (0, 1)
        got = measure(image, declared=(0, 1))
        assert got["color_valid_pixels"] == 13
        assert got["color_valid_fraction"] == pytest.approx(13 / 16)
        assert got["color_status"] == "insufficient_support"  # 13/16 is under the fixed 0.90

    def test_a_value_above_a_declared_twelve_bit_range_is_not_valid(self):
        image = np.full((3, 8, 8), 2000, dtype=np.uint16)
        image[:, 0, :] = 4096  # a whole row beyond (0, 4095)
        got = measure(image, declared=(0, 4095))
        assert got["color_valid_pixels"] == 56

    def test_the_floor_and_the_fixed_valid_fraction(self):
        thirty = solid((200, 10, 10), size=6)[:, :5, :]  # 30 pixels
        assert measure(thirty, min_pixels=32)["color_status"] == "insufficient_support"
        assert measure(thirty, min_pixels=30)["color_status"] == "ok"
        holes = np.full((3, 10, 10), 0.5, dtype=np.float64)
        holes[:, :2, :] = np.nan  # 20% not valid: a fraction of 0.80, under the fixed 0.90
        assert measure(holes, declared=(0, 1))["color_status"] == "insufficient_support"
        holes[:, 1, :] = 0.5  # 10% not valid: exactly the floor
        assert measure(holes, declared=(0, 1))["color_status"] == "ok"

    def test_an_insufficient_view_reports_its_counts_and_nan_statistics(self):
        got = measure(solid((255, 0, 0), size=4), min_pixels=32)
        assert got["color_status"] == "insufficient_support"
        assert got["color_valid_pixels"] == 16
        assert all(math.isnan(got[name]) for name in STATS)


class TestViews:
    def test_a_box_reads_its_own_window(self):
        image = solid((128, 128, 128), size=10)
        image[:, 2:6, 2:6] = np.array([255, 0, 0], dtype=np.uint8)[:, None, None]
        inside = measure(image, box=(2, 2, 6, 6))
        assert inside["color_valid_pixels"] == 16
        assert inside["color_excluded_pixels"] == 0
        assert inside["color_achromatic_fraction"] == 0.0

    def test_a_box_partly_outside_the_image_is_measured_over_the_part_inside(self):
        # 17 of the box's 20 columns are inside the image, and every pixel there is valid: nothing is missing.
        got = measure(solid((255, 0, 0), size=20), box=(-3, 0, 17, 20))
        assert got["color_valid_pixels"] == 17 * 20
        assert got["color_valid_fraction"] == 1.0
        assert got["color_status"] == "ok"

    def test_a_box_wholly_outside_the_image_has_nothing_to_measure(self):
        got = measure(solid((255, 0, 0), size=20), box=(30, 30, 40, 40))
        assert got["color_status"] == "insufficient_support"
        assert got["color_valid_pixels"] == 0
        assert math.isnan(got["color_valid_fraction"])

    def test_a_truncated_box_with_invalid_pixels_inside_is_still_judged_on_the_inside(self):
        image = np.full((3, 20, 20), 0.5, dtype=np.float64)
        image[:, :4, :] = np.nan  # a fifth of the image is not valid, and the box covers only the top half
        got = measure(image, declared=(0, 1), box=(-5, 0, 15, 10))
        assert got["color_valid_pixels"] == 15 * 6
        assert got["color_valid_fraction"] == pytest.approx(90 / 150)
        assert got["color_status"] == "insufficient_support"

    def test_the_background_counts_what_the_mask_removes_and_divides_by_the_rest(self):
        image = solid((128, 128, 128), size=10)
        exclude = np.zeros((10, 10), dtype=bool)
        exclude[2:6, 2:6] = True
        got = measure(image, exclude=exclude, view=ViewKind.MASK)
        assert got["color_excluded_pixels"] == 16
        assert got["color_valid_pixels"] == 84
        assert got["color_valid_fraction"] == 1.0
        assert got["color_status"] == "ok"

    def test_a_background_with_nothing_left_is_insufficient_and_still_reports_the_exclusion(self):
        got = measure(solid((128, 128, 128), size=4), exclude=np.ones((4, 4), dtype=bool), view=ViewKind.MASK)
        assert got["color_status"] == "insufficient_support"
        assert got["color_excluded_pixels"] == 16
        assert got["color_valid_pixels"] == 0

    def test_a_missing_band_still_reports_the_exclusion(self):
        exclude = np.zeros((8, 8), dtype=bool)
        exclude[:2] = True
        got = measure(np.zeros((1, 8, 8), dtype=np.uint8), exclude=exclude, view=ViewKind.MASK)
        assert got["color_status"] == "missing_band"
        assert got["color_excluded_pixels"] == 16


class TestThroughImageStats:
    def test_the_columns_and_details_of_the_image_a_box_and_the_background(self):
        image = solid((200, 40, 40), size=16)
        result = image_stats(
            [image],
            boxes=[[(2, 2, 10, 10)]],
            stats=COLOR,
            per_background=True,
            color_rgb_channels=(0, 1, 2),
            value_range=(0, 255),
        )
        names = set(result["stats"])
        assert set(STATS) <= names
        assert {f"background_{name}" for name in STATS} <= names
        assert set(DETAILS) <= set(details_of(result))
        assert {f"background_{name}" for name in DETAILS} <= set(details_of(result))
        assert set(details_of(result)["color_status"].tolist()) == {"ok"}

    def test_a_mixed_gray_and_rgb_dataset_reads_each_row_by_its_own_channels(self):
        gray, rgb = np.full((1, 8, 8), 100, dtype=np.uint8), solid((255, 0, 0))
        result = image_stats([gray, rgb], stats=COLOR, color_rgb_channels=(0, 1, 2), value_range=(0, 255))
        assert details_of(result)["color_status"].tolist() == ["missing_band", "ok"]
        assert np.isnan(result["stats"]["color_chroma_mean"][0])
        assert result["stats"]["color_chroma_mean"][1] == pytest.approx(104.55, abs=0.01)

    def test_a_band_group_asked_for_color_produces_no_column(self):
        with pytest.warns(UserWarning, match="do not vary with a band subset"):
            result = image_stats(
                [solid((255, 0, 0))],
                stats={None: ImageStats.PIXEL_MEAN, "rgb": COLOR},
                channels={"rgb": [0, 1, 2]},
                color_rgb_channels=(0, 1, 2),
                value_range=(0, 255),
            )
        assert not [name for name in result["stats"] if "color_" in name]

    def test_reordering_the_images_permutes_the_rows_and_changes_nothing_else(self):
        images = [np.random.default_rng(i).integers(0, 256, (3, 8, 8)).astype(np.uint8) for i in range(4)]
        kw: dict[str, Any] = {"stats": COLOR, "color_rgb_channels": (0, 1, 2), "value_range": (0, 255)}
        forward, backward = image_stats(images, **kw), image_stats(images[::-1], **kw)
        for name in STATS:
            np.testing.assert_array_equal(forward["stats"][name], backward["stats"][name][::-1])

    def test_blocks_do_not_change_a_value(self, monkeypatch: pytest.MonkeyPatch):
        image = np.random.default_rng(3).integers(0, 256, (3, 40, 13)).astype(np.uint8)
        whole = measure(image)
        monkeypatch.setattr(_colorstats, "_BLOCK_PIXELS", 7 * 13)
        blocks = measure(image)
        for name in STATS:
            assert blocks[name] == pytest.approx(whole[name], rel=1e-12), name
        assert blocks["color_valid_pixels"] == whole["color_valid_pixels"] == 520


def test_a_call_that_declares_nothing_warns_nothing_new():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        image_stats([solid((255, 0, 0))], stats=ImageStats.PIXEL_MEAN)

"""What the calculator cache knows that SATURATION needs: where a range came from, and whether a view holds pixels."""

import numpy as np
import pytest

from dataeval.core._calculators._cache import CalculatorCache
from dataeval.types._stats_config import ImageStatsConfig, SaturationConfig
from dataeval.utils.preprocessing import BoundingBox, get_value_range

pytestmark = pytest.mark.required

RGB = np.zeros((3, 10, 12), dtype=np.uint8)


def box(x0: int, y0: int, x1: int, y1: int, shape: tuple[int, ...] = RGB.shape) -> BoundingBox:
    return BoundingBox(x0, y0, x1, y1, image_shape=shape)


class TestHasDeclaredRange:
    """A declared interval is told from one decoded off the data by its depth: declared has none."""

    def test_a_declared_range_gives_a_known_interval_with_no_depth(self):
        # The inference the property rests on: if `get_value_range` ever gave a declared range a depth, this fails.
        declared = get_value_range(np.zeros((3, 4, 4), dtype=np.uint16), declared=(0, 4095))
        assert declared.is_known
        assert np.isnan(declared.depth)

    def test_a_declared_range_passed_to_the_cache_is_declared(self):
        data = np.zeros((3, 4, 4), dtype=np.uint16)
        cache = CalculatorCache(data, value_range=get_value_range(data, declared=(0, 4095)))
        assert cache.has_declared_range

    @pytest.mark.parametrize(
        "data",
        [
            np.array([[0, 100]], dtype=np.uint8),
            np.array([[0, 3000]], dtype=np.uint16),
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[0.0, 200.0]], dtype=np.float32),
        ],
    )
    def test_a_range_decoded_from_the_data_is_not_declared(self, data: np.ndarray):
        assert not CalculatorCache(data).has_declared_range
        assert not CalculatorCache(data, value_range=get_value_range(data)).has_declared_range

    def test_an_unknown_range_is_not_declared(self):
        # Float data outside both conventional spellings has no interval to be declared or decoded.
        assert not CalculatorCache(np.array([[-50.0, 50.0]])).has_declared_range


class TestHasPixels:
    """A view holds pixels when some position in it comes from the datum and has not been excluded."""

    def test_a_full_image_an_interior_box_and_a_box_partly_outside_hold_pixels(self):
        assert CalculatorCache(RGB).has_pixels
        assert CalculatorCache(RGB, box(2, 2, 6, 6)).has_pixels
        assert CalculatorCache(RGB, box(-3, -3, 4, 4)).has_pixels

    def test_a_box_wholly_outside_the_image_holds_none(self):
        assert not CalculatorCache(RGB, box(50, 50, 60, 60)).has_pixels

    def test_a_zero_area_box_holds_none(self):
        # The cache widens it to one all-fill pixel, which is not a pixel of the datum.
        assert not CalculatorCache(RGB, box(4, 4, 4, 4)).has_pixels

    def test_an_inverted_box_reads_the_window_it_describes(self):
        assert CalculatorCache(RGB, box(8, 8, 2, 2)).has_pixels

    def test_a_window_the_mask_excludes_entirely_holds_none(self):
        assert not CalculatorCache(RGB, exclude=np.ones((10, 12), dtype=bool)).has_pixels

    def test_a_partial_mask_leaves_pixels(self):
        mask = np.zeros((10, 12), dtype=bool)
        mask[:5] = True
        assert CalculatorCache(RGB, exclude=mask).has_pixels

    def test_a_mask_covering_the_only_pixels_inside_a_box_that_leaves_the_image_holds_none(self):
        # The window's fill is not part of the datum and not excluded either, so it must not count as a pixel.
        mask = np.zeros((10, 12), dtype=bool)
        mask[:4, :4] = True
        assert not CalculatorCache(RGB, box(-3, -3, 4, 4), exclude=mask).has_pixels

    def test_a_band_group_the_datum_cannot_supply_holds_none(self):
        assert CalculatorCache(RGB, bands=(1,)).has_pixels
        assert not CalculatorCache(RGB, bands=(5,)).has_pixels
        assert not CalculatorCache(RGB, bands=(0, 3)).has_pixels

    def test_a_two_dimensional_datum_cannot_supply_a_band_group_without_a_box(self):
        # `image` answers NaN for it, so the view holds no pixels.
        flat = np.zeros((10, 12))
        assert not CalculatorCache(flat, bands=(0,)).has_pixels
        assert CalculatorCache(flat, box(0, 0, 4, 4, flat.shape), bands=(0,)).has_pixels

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (np.array(5.0), True),
            (np.zeros(7), True),
            (np.zeros((10, 12)), True),
            (np.zeros((2, 3, 10, 12)), True),
            (np.zeros((3, 0, 0)), False),
            (np.zeros(0), False),
        ],
        ids=["0-d", "1-D", "2-D", "4-D", "zero-size 3-D", "empty 1-D"],
    )
    def test_degenerate_shapes_answer_without_raising(self, data: np.ndarray, expected: bool):
        assert CalculatorCache(data).has_pixels is expected


class TestSettings:
    def test_the_cache_carries_the_settings_it_is_given_and_defaults_to_the_documented_ones(self):
        assert CalculatorCache(RGB).settings == ImageStatsConfig()
        custom = ImageStatsConfig(saturation=SaturationConfig(atol=1))
        assert CalculatorCache(RGB, settings=custom).settings is custom

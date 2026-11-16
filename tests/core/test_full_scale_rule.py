"""CONTEXT and COLOR share one full-scale rule: declared, else uint8, else the two float conventions, else unknown."""

import logging
import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import ChannelGroup

pytestmark = pytest.mark.required

BOX = [[[20.0, 20.0, 40.0, 40.0]]]


def _frame(surround: float, peak: float, dtype) -> np.ndarray:
    """A one-band frame whose box's surround holds `surround` and whose corner holds `peak`."""
    frame = np.full((1, 64, 64), surround, dtype=np.float64)
    frame[0, 25:35, 25:35] = surround / 2
    frame[0, :4, :4] = peak
    return frame.astype(dtype)


def _context(images, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = image_stats(
            images,
            boxes=BOX * len(images),
            labels=[[0]] * len(images),
            stats=ImageStats.CONTEXT,
            per_image=False,
            **kwargs,
        )
    return result["stats"]["surround_intensity_mean"], result["details"]["context_status"]


def _rgb(dtype, scale: float) -> np.ndarray:
    rng = np.random.default_rng(2)
    return (rng.integers(0, 256, (3, 32, 32)) * scale).astype(dtype)


def _color(images, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = image_stats(images, stats=ImageStats.COLOR, color_rgb_channels=(0, 1, 2), **kwargs)
    return result["stats"]["color_lightness_mean"], result["details"]["color_status"]


class TestContext:
    def test_integer_data_above_eight_bits_needs_a_declaration(self, caplog):
        frames = [_frame(100, 180, np.uint16), _frame(100, 4000, np.uint16)]
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            values, status = _context(frames)
        assert np.isnan(values).all()
        assert list(status) == ["unknown_range", "unknown_range"]
        assert len([r for r in caplog.records if "unknown_range" in r.getMessage()]) == 1
        declared, status = _context(frames, value_range=(0, 4095))
        assert list(status) == ["ok", "ok"]
        assert declared[0] == pytest.approx(declared[1])
        assert declared[0] == pytest.approx(100 / 4095, rel=1e-6)

    @pytest.mark.parametrize(("dtype", "scale"), [(np.float32, 1 / 255), (np.float32, 1.0)])
    def test_the_float_conventions_read_as_the_uint8_original(self, dtype, scale):
        original, _ = _context([_frame(100, 200, np.uint8)])
        converted, status = _context([(_frame(100, 200, np.uint8) * scale).astype(dtype)])
        assert list(status) == ["ok"]
        assert converted[0] == pytest.approx(original[0], rel=1e-6)

    @pytest.mark.parametrize("dtype", [np.int16, np.bool_])
    def test_other_integers_and_bool_need_a_declaration(self, dtype):
        _, status = _context([_frame(1, 1, dtype)])
        assert list(status) == ["unknown_range"]

    def test_uint8_reads_against_its_dtype_whatever_its_maximum(self):
        values, status = _context([_frame(1, 1, np.uint8)])
        assert list(status) == ["ok"]
        assert values[0] == pytest.approx(1 / 255, rel=1e-6)

    def test_a_dark_float_image_in_the_eight_bit_convention_reads_as_zero_to_one(self):
        # Accepted: every value at or below 1 is read as the (0, 1) convention.
        values, status = _context([_frame(0.8, 0.9, np.float32)])
        assert list(status) == ["ok"]
        assert values[0] == pytest.approx(0.8, rel=1e-6)

    def test_a_channel_groups_own_declared_range_applies(self):
        frame = np.concatenate([_frame(100, 4000, np.uint16), _frame(100, 4000, np.uint16)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result: Any = image_stats(
                [frame],
                boxes=BOX,
                labels=[[0]],
                per_image=False,
                stats={"nir": ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN},
                channels={"nir": ChannelGroup([1], value_range=(0, 4095))},
            )
        assert result["stats"]["surround_nir_intensity_mean"][0] == pytest.approx(100 / 4095, rel=1e-6)


class TestColor:
    def test_the_float_conventions_gain_values(self):
        original, _ = _color([_rgb(np.uint8, 1.0)])
        for scale in (1 / 255, 1.0):
            converted, status = _color([(_rgb(np.uint8, 1.0) * scale).astype(np.float32)])
            assert list(status) == ["ok"]
            assert converted[0] == pytest.approx(original[0], rel=1e-5)

    def test_integers_other_than_uint8_need_a_declaration(self):
        _, status = _color([_rgb(np.uint16, 16.0)])
        assert list(status) == ["unknown_range"]
        _, status = _color([_rgb(np.uint16, 16.0)], value_range=(0, 4095))
        assert list(status) == ["ok"]

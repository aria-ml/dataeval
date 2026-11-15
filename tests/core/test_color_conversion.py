"""The sRGB to CIELAB conversion (D65, 2 degree observer) and the additive sums the color statistics are made of."""

import numpy as np
import pytest

from dataeval.core._color import ColorSums, _sums_from_lab, block_sums, srgb_to_lab

pytestmark = pytest.mark.required

# CIE reference values for the sRGB primaries; the issue states red and white.
REFERENCE = {
    "red": ((1.0, 0.0, 0.0), (53.2408, 80.0925, 67.2032)),
    "green": ((0.0, 1.0, 0.0), (87.7347, -86.1827, 83.1793)),
    "blue": ((0.0, 0.0, 1.0), (32.2970, 79.1875, -107.8602)),
    "white": ((1.0, 1.0, 1.0), (100.0, 0.0, 0.0)),
}


@pytest.mark.parametrize("name", REFERENCE)
def test_reference_values(name: str):
    rgb, lab = REFERENCE[name]
    np.testing.assert_allclose(srgb_to_lab(np.array(rgb)), lab, atol=0.01)


def test_black_is_exactly_zero_and_mid_gray_is_neutral():
    assert (srgb_to_lab(np.zeros(3)) == 0).all()
    lab = srgb_to_lab(np.full(3, 128 / 255))
    assert lab[0] == pytest.approx(53.5850, abs=0.01)
    assert abs(lab[1]) < 1e-4
    assert abs(lab[2]) < 1e-4


def test_gray_ramp_stays_neutral_and_lightness_rises():
    levels = np.repeat(np.linspace(0, 1, 256)[:, None], 3, axis=1)
    lab = srgb_to_lab(levels)
    assert np.abs(lab[:, 1:]).max() < 1e-4
    assert (np.diff(lab[:, 0]) > 0).all()


def test_a_trailing_axis_of_three_is_all_that_is_read():
    image = np.random.default_rng(0).random((4, 5, 3))
    assert srgb_to_lab(image).shape == (4, 5, 3)
    np.testing.assert_array_equal(srgb_to_lab(image)[2, 3], srgb_to_lab(image[2, 3]))


@pytest.mark.parametrize(
    ("channel", "levels", "chroma"),
    [(0, 16, 6.48), (1, 16, 11.12), (2, 16, 9.22), (0, 30, 12.32), (1, 30, 20.56), (2, 30, 17.28)],
)
def test_cast_arithmetic(channel: int, levels: int, chroma: float):
    # One channel `levels` of 255 above a mid gray: the arithmetic behind the documented sentence about the threshold.
    rgb = np.full(3, 128 / 255)
    rgb[channel] = (128 + levels) / 255
    lab = srgb_to_lab(rgb)
    assert np.hypot(lab[1], lab[2]) == pytest.approx(chroma, abs=0.01)


def _hue(sums: ColorSums) -> tuple[float, float]:
    return sums.a_chromatic / sums.chroma_chromatic, sums.b_chromatic / sums.chroma_chromatic


def test_hue_identities_on_random_chromatic_pixels():
    rgb = np.random.default_rng(1).random((5000, 3))
    cos, sin = _hue(block_sums(rgb, chroma_threshold=12.0))
    assert -1 <= cos <= 1
    assert -1 <= sin <= 1
    assert cos**2 + sin**2 <= 1 + 1e-12


def test_one_hue_gives_a_unit_vector_and_pure_red_reads_its_angle():
    sums = block_sums(np.array([[1.0, 0.0, 0.0]] * 10), chroma_threshold=12.0)
    cos, sin = _hue(sums)
    assert cos == pytest.approx(0.76606, abs=1e-5)
    assert sin == pytest.approx(0.64277, abs=1e-5)
    assert cos**2 + sin**2 == pytest.approx(1.0, abs=1e-12)


def test_equal_chroma_opposite_hues_cancel():
    lab = np.array([[60.0, 25.98, -16.74], [60.0, -25.98, 16.74]])
    sums = _sums_from_lab(lab, chroma_threshold=12.0)
    assert sums.chromatic == 2
    cos, sin = _hue(sums)
    assert abs(cos) < 1e-12
    assert abs(sin) < 1e-12


def test_sums_count_each_class_of_pixel():
    lab = np.array([[50.0, 3.0, 4.0], [70.0, 30.0, 40.0]])  # chroma 5 and 50
    sums = _sums_from_lab(lab, chroma_threshold=12.0)
    assert (sums.pixels, sums.achromatic, sums.chromatic) == (2, 1, 1)
    assert sums.lightness == pytest.approx(120.0)
    assert sums.chroma == pytest.approx(55.0)
    assert (sums.a_chromatic, sums.b_chromatic, sums.chroma_chromatic) == pytest.approx((30.0, 40.0, 50.0))


def test_no_chromatic_pixel():
    sums = block_sums(np.full((6, 3), 0.5), chroma_threshold=12.0)
    assert sums.chromatic == 0
    assert sums.achromatic == sums.pixels == 6


def test_blocks_add_to_the_whole():
    rgb = np.random.default_rng(2).random((1000, 3))
    whole = block_sums(rgb, chroma_threshold=12.0)
    parts = block_sums(rgb[:300], 12.0) + block_sums(rgb[300:], 12.0)
    assert parts.pixels == whole.pixels
    assert parts.chromatic == whole.chromatic
    assert parts.achromatic == whole.achromatic
    np.testing.assert_allclose(list(parts), list(whole), rtol=1e-12)

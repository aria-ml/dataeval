"""The surround's geometry and pixel reading, without `image_stats`: hand-worked fixtures F2 and F4 of the design."""

from collections.abc import Sequence

import numpy as np
import pytest

from dataeval.core._calculators._contextstats import Regions, read_surround, surround_regions
from dataeval.utils.preprocessing import boxes_to_mask

pytestmark = pytest.mark.required

# F2: a 10 x 10 image, pixel (y, x) = (10 * y + x) / 99, so every pixel has its own value in [0, 1].
F2_BOXES = [(3, 3, 6, 6), (5, 5, 9, 9)]
# The 13 pixels of the window (2, 2, 7, 7), less the focal box (3, 3, 6, 6), less the peer (5, 5, 9, 9), as (y, x).
F2_RING = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 2), (3, 6), (4, 2), (4, 6), (5, 2), (6, 2), (6, 3), (6, 4)]
# The 16 pixels of the same window less the focal box alone.
FOCAL_RING = [(y, x) for y in range(2, 7) for x in range(2, 7) if not (3 <= y < 6 and 3 <= x < 6)]


def f2_image() -> np.ndarray:
    return np.arange(100).reshape(1, 10, 10) / 99


def retained(
    extent: tuple[int, int], boxes: Sequence[tuple[float, ...]], window: tuple[int, int, int, int]
) -> np.ndarray:
    """The window-shaped mask of pixels no box covers, from the library's own mask of the boxes."""
    x0, y0, x1, y1 = window
    return ~boxes_to_mask(extent, boxes)[y0:y1, x0:x1]


def read_focal(image: np.ndarray, *, low: float = 0.0, high: float = 1.0, peers: list | None = None):
    """Read the surround of the focal box (3, 3, 6, 6) of a 10 x 10 image."""
    focal = (3, 3, 6, 6)
    regions = surround_regions(np.array(focal, dtype=np.float64), (10, 10), 0.25)
    keep = retained((10, 10), [focal, *(peers or [])], regions.window)
    return read_surround(image, range(image.shape[0]), regions, keep, low, high)


@pytest.mark.parametrize(
    ("row", "extent", "window", "target"),
    [
        ((3, 3, 6, 6), (10, 10), (2, 2, 7, 7), (3, 3, 6, 6)),
        ((0, 0, 4, 4), (8, 8), (0, 0, 5, 5), (0, 0, 4, 4)),
        ((2, 2, 6, 6), (8, 8), (1, 1, 7, 7), (2, 2, 6, 6)),
        ((2.5, 2.5, 5.5, 5.5), (8, 8), (1, 1, 7, 7), (2, 2, 6, 6)),
        # Half outside: the width and height are the box's own (8 and 4), not the clipped ones, so the padding holds.
        ((-4, 2, 4, 6), (8, 8), (0, 1, 6, 7), (0, 2, 4, 6)),
    ],
)
def test_regions_of_the_hand_worked_boxes(row, extent, window, target):
    assert surround_regions(np.array(row, dtype=np.float64), extent, 0.25) == Regions(window, target)


def test_a_box_whose_width_overflows_does_not_raise():
    # `x1 - x0` is infinity in float64, and `math.floor(inf)` raises, so the window is clipped before it is rounded.
    regions = surround_regions(np.array([-1e308, 0, 1e308, 5]), (8, 8), 0.25)
    assert regions == Regions((0, 0, 8, 7), (0, 0, 8, 5))


def test_the_window_always_contains_the_target():
    rng = np.random.default_rng(1605)
    extent = (37, 53)
    checked = 0
    while checked < 200:
        x0, y0 = rng.uniform(-10, 60), rng.uniform(-10, 45)
        row = np.array([x0, y0, x0 + rng.uniform(0.5, 30), y0 + rng.uniform(0.5, 30)])
        if row[2] <= 0 or row[3] <= 0 or row[0] >= extent[1] or row[1] >= extent[0]:
            continue  # not a VALID box: outside the image
        (wx0, wy0, wx1, wy1), (tx0, ty0, tx1, ty1) = surround_regions(row, extent, 0.25)
        assert tx1 > tx0
        assert ty1 > ty0
        assert wx0 <= tx0
        assert wy0 <= ty0
        assert tx1 <= wx1
        assert ty1 <= wy1
        assert wx0 >= 0
        assert wy0 >= 0
        assert wx1 <= extent[1]
        assert wy1 <= extent[0]
        checked += 1


def test_read_surround_on_f2():
    regions = surround_regions(np.array(F2_BOXES[0], dtype=np.float64), (10, 10), 0.25)
    reading = read_surround(f2_image(), [0], regions, retained((10, 10), F2_BOXES, regions.window), 0.0, 1.0)
    expected = np.array([(10 * y + x) / 99 for y, x in F2_RING])
    assert reading.measurable == 13
    assert reading.mean == pytest.approx(expected.mean(), abs=1e-12)
    assert reading.std == pytest.approx(expected.std(), abs=1e-12)  # population: ddof=0
    assert reading.target_valid_fraction == 1.0


def test_the_intensity_is_a_mean_over_channels():
    image = np.stack([np.full((10, 10), 0.2), np.full((10, 10), 0.6)])
    assert read_focal(image).mean == pytest.approx(0.4, abs=1e-12)


def test_a_pixel_out_of_range_in_one_channel_is_not_measurable_in_any():
    image = np.stack([np.full((10, 10), 0.2), np.full((10, 10), 0.6)])
    image[0, 2, 2] = 0.9  # fine in channel 0, which must not count while channel 1 is out of range there
    image[1, 2, 2] = 1.5
    reading = read_focal(image)
    assert reading.measurable == len(FOCAL_RING) - 1
    assert reading.mean == pytest.approx(0.4, abs=1e-12)


def test_nan_and_infinity_are_not_measurable_and_nothing_is_clipped():
    image = np.full((1, 10, 10), 0.5)
    image[0, 2, 2], image[0, 2, 3], image[0, 2, 4] = np.nan, np.inf, 1.5
    reading = read_focal(image)
    assert reading.measurable == len(FOCAL_RING) - 3
    assert reading.mean == 0.5  # a clipped 1.5 would have moved it


def test_a_constant_ring_has_zero_spread():
    image = np.full((1, 10, 10), 0.2)
    image[0, 3:6, 3:6] = 0.8  # the target, distinct from the ring
    reading = read_focal(image)
    assert reading.mean == pytest.approx(0.2, abs=1e-12)
    assert reading.std <= 1e-12


def test_two_measurable_pixels_have_std_half_their_difference():
    image = np.full((1, 10, 10), np.nan)
    image[0, 2, 2], image[0, 2, 3] = 0.2, 0.7
    image[0, 3:6, 3:6] = 0.5
    reading = read_focal(image)
    assert reading.measurable == 2
    assert reading.mean == pytest.approx(0.45, abs=1e-12)
    assert reading.std == pytest.approx(0.25, abs=1e-12)


def test_nothing_measurable_gives_nan_and_a_zero_count():
    reading = read_focal(np.full((1, 10, 10), np.nan))
    assert reading.measurable == 0
    assert np.isnan(reading.mean)
    assert np.isnan(reading.std)
    assert reading.target_valid_fraction == 0.0


@pytest.mark.parametrize(("dtype", "value", "high"), [(np.uint8, 250, 255), (np.uint16, 4000, 4095)])
def test_integers_are_not_clipped_by_their_dtype(dtype, value, high):
    image = np.full((1, 10, 10), value, dtype=dtype)
    image[0, 3:6, 3:6] = 0
    reading = read_focal(image, high=float(high))
    assert reading.measurable == len(FOCAL_RING)
    assert reading.mean == pytest.approx(value / high, abs=1e-12)  # a uint8 subtraction would have wrapped


def test_target_valid_fraction_counts_the_target_pixels_that_measure():
    image = np.full((1, 10, 10), 0.5)
    image[0, 4, 4] = np.nan  # one of the nine target pixels
    assert read_focal(image).target_valid_fraction == pytest.approx(8 / 9, abs=1e-12)

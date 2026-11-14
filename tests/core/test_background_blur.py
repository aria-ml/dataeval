"""Tests for `background_blur`: the blur value over the scene outside the boxes, under `per_background=True`.

Fixtures come from `tests.core.test_blur.textured` (a seeded 256x256 image of hard-edged blobs on a coarse texture,
which reads a finite blur at 8 bits) and `tests.core.test_blur_stats.picture`. `scene` is the one built here: that
picture sharp inside a box and a 5 px ring around it, and Gaussian-blurred everywhere else, which is the defect the
validation study pastes.
"""

import gc
import logging
import warnings
import weakref

import numpy as np
import pytest
from scipy import ndimage

import dataeval.core._calculators._blurstats as blurstats_module
from dataeval.config import use_max_processes
from dataeval.core import image_stats
from dataeval.core._blur import background_blur, grow_region, measure_edges
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import boxes_to_mask
from tests.core.test_blur import gaussian, textured
from tests.core.test_blur_stats import gray_of, picture, to_uint8

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")

BOX = (64, 64, 192, 192)


def stats_of(image, boxes=None, stats=ImageStats.BLUR, **kwargs):
    """`image_stats` over one image with `per_background=True`, as the dict of columns."""
    boxes = None if boxes is None else [boxes]
    return image_stats([image], boxes=boxes, stats=stats, per_background=True, **kwargs)["stats"]


def scene(background_sigma: float) -> np.ndarray:
    """A 3 x 256 x 256 uint8 picture: `textured` sharp inside `BOX` and a 5 px ring, blurred by sigma outside it."""
    sharp = textured()
    img = gaussian(sharp, background_sigma).copy()
    img[59:197, 59:197] = sharp[59:197, 59:197]
    return to_uint8(np.stack([img] * 3))


def reference(image, boxes, margin=9):
    """The value computed straight from the core functions, for a gray-band-mean of `image`."""
    mask = boxes_to_mask(image.shape, boxes)
    return background_blur(measure_edges(gray_of(image)), grow_region(mask, margin))


def test_no_boxes_equals_blur():
    """An image with nothing annotated is all background, so the two values are one value, to the bit."""
    for stats in (stats_of(picture()), stats_of(picture(), [])):
        assert np.isfinite(stats["blur"][0])
        assert stats["background_blur"][0] == stats["blur"][0]


def test_one_column_is_added_and_nothing_else_moves():
    """`background_blur` is the only column the blur flag adds under `per_background`, and no other value changes."""
    image, boxes = picture(), [(20, 20, 90, 90), (120, 60, 200, 180)]
    with_blur = stats_of(image, boxes, ImageStats.BLUR | ImageStats.VISUAL_SHARPNESS | ImageStats.PIXEL_MEAN)
    without = stats_of(image, boxes, ImageStats.VISUAL_SHARPNESS | ImageStats.PIXEL_MEAN)

    added = {"blur", "blur_directionality", "background_blur"}
    assert set(with_blur) - set(without) == added
    assert [name for name in with_blur if name not in added] == list(without), "the other columns keep their order"
    for name in without:
        assert np.array_equal(with_blur[name], without[name], equal_nan=True), name


def test_a_blurred_background_raises_the_value_and_the_frame_value_stays():
    """The sharp box keeps supplying the frame's sharpest edges, so only the background value sees the blur.

    Measured on this fixture with the core functions before the test was written: the background value rises by 0.88
    in log, the whole-frame value by 0.0.
    """
    sharp, blurred = stats_of(scene(0.0), [BOX]), stats_of(scene(3.0), [BOX])

    assert np.log(blurred["background_blur"][0] / sharp["background_blur"][0]) >= 0.7
    assert np.log(blurred["blur"][0] / sharp["blur"][0]) < 0.1


def test_blurring_the_boxes_barely_moves_it():
    """Replacing the content of the box with its own blurred copy leaves the background value within 2% (log 0.02)."""
    sharp = textured()
    inside = sharp.copy()
    inside[64:192, 64:192] = gaussian(sharp, 3.0)[64:192, 64:192]
    before = stats_of(to_uint8(np.stack([sharp] * 3)), [BOX])["background_blur"][0]
    after = stats_of(to_uint8(np.stack([inside] * 3)), [BOX])["background_blur"][0]

    assert abs(np.log(after / before)) < 0.02


def test_the_shipped_margin_is_nine():
    """The column is the pooling at the 9 px margin, and a sharp ring just outside the box would count at margin 0."""
    image = scene(3.0)
    shipped = stats_of(image, [BOX])["background_blur"][0]

    assert shipped == pytest.approx(reference(image, [BOX], 9), rel=1e-6)  # the column is float32
    assert shipped > 1.2 * reference(image, [BOX], 0)


def test_boxes_covering_the_datum(caplog):
    with caplog.at_level(logging.WARNING, logger="dataeval.core"):
        stats = stats_of(picture(), [(0, 0, 256, 256)])

    assert "Boxes cover the entire datum" in caplog.text
    assert np.isnan(stats["background_blur"][0])
    assert np.isfinite(stats["blur"][0]), "the whole-image value is untouched"


def test_too_few_background_edges_is_nan():
    """A flat background has no edge points outside the box; the picture's own texture there makes it finite."""
    flat = np.full((256, 256), 128.0, dtype=np.float32)
    flat[64:192, 64:192] = textured()[64:192, 64:192]

    assert np.isnan(stats_of(to_uint8(np.stack([flat] * 3)), [BOX])["background_blur"][0])
    assert np.isfinite(stats_of(picture(), [BOX])["background_blur"][0])


def test_a_nan_pixel_is_nan():
    """One NaN pixel spreads through the smoothing and no edge is found, as for `blur`."""
    holed = picture().astype(np.float64)
    holed[:, 100, 100] = np.nan

    stats = stats_of(holed, [(20, 20, 60, 60)], normalize_pixel_values=False)

    assert np.isnan(stats["blur"][0])
    assert np.isnan(stats["background_blur"][0])


def test_a_nan_pixel_inside_a_box_is_nan_too():
    """The edges are found on the whole image, so a NaN pixel anywhere withholds the value, even inside a box."""
    holed = picture().astype(np.float64)
    holed[:, 50, 50] = np.nan
    stats = stats_of(
        holed, [(20, 20, 90, 90)], stats=ImageStats.BLUR | ImageStats.VISUAL_BRIGHTNESS, normalize_pixel_values=False
    )

    assert np.isnan(stats["background_blur"][0])
    assert np.isfinite(stats["background_brightness"][0]), "a statistic that reads the masked picture is unaffected"


def test_a_box_past_the_border_is_clipped_and_is_not_a_nan_case():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the box is reported as invalid, which is not what is tested here
        stats = stats_of(picture(), [(-30, -30, 60, 60)])

    assert np.isfinite(stats["background_blur"][0])


def test_unknown_value_range_is_nan_and_declaring_one_is_finite():
    signed = ((textured().astype(np.float64) - 128.0) * 4.0)[np.newaxis]

    undeclared = stats_of(signed, [(20, 20, 60, 60)], normalize_pixel_values=False)
    declared = stats_of(signed, [(20, 20, 60, 60)], normalize_pixel_values=False, value_range=(-512.0, 512.0))

    assert np.isnan(undeclared["background_blur"][0])
    assert np.isfinite(declared["background_blur"][0])


def test_overlapping_boxes_equal_their_union():
    a, b = (20, 20, 120, 120), (80, 80, 180, 180)
    once = stats_of(picture(), [a, b])["background_blur"][0]

    assert stats_of(picture(), [a, b, a, b])["background_blur"][0] == once, "a box listed twice counts once"
    halves = stats_of(picture(), [(20, 20, 120, 70), (20, 70, 120, 120)])["background_blur"][0]
    whole = stats_of(picture(), [(20, 20, 120, 120)])["background_blur"][0]
    assert halves == whole, "two boxes that tile a rectangle are that rectangle"


@pytest.mark.parametrize(
    "box",
    [
        (100, 100, 101, 101),
        (50, 50, 50, 80),
        (300, 300, 350, 350),
        (0, 0, 40, 40),
        (200, 200, 300, 300),
        (-30, -30, 60, 60),
    ],
    ids=["1x1", "zero_area", "off_image", "touches_corner", "past_far_border", "past_near_border"],
)
def test_degenerate_and_border_boxes_follow_the_mask(box):
    """Each is painted as `boxes_to_mask` paints it (clipped, or nothing), and none raises."""
    image = picture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an invalid box is reported, which is not what is tested here
        value = stats_of(image, [box])["background_blur"][0]

    np.testing.assert_allclose(value, reference(image, [box]), rtol=1e-6)  # the column is float32; NaN equals NaN


def test_float_image_equals_uint8_and_1d_has_no_background_columns():
    image = picture()
    wide = stats_of(image.astype(np.float64), [BOX], normalize_pixel_values=False, value_range=(0.0, 255.0))
    narrow = stats_of(image, [BOX])

    assert wide["background_blur"][0] == pytest.approx(narrow["background_blur"][0], rel=1e-6)
    line = np.random.default_rng(0).random(10000)
    same = ImageStats.VISUAL_SHARPNESS
    assert {n for n in stats_of(line, None, ImageStats.BLUR) if n.startswith("background_")} == {
        n for n in stats_of(line, None, same) if n.startswith("background_")
    }


def test_band_groups_pool_over_their_own_gray():
    """`background_nir_blur` equals a separate pooling: the group's band mean, a scipy dilation, a percentile."""
    t = textured()
    cube = to_uint8(np.stack([np.roll(t, 3 * band, axis=1) for band in range(6)]))
    groups = {"vis": [0, 1, 2], "nir": [3, 4, 5]}
    box = (20, 20, 90, 90)

    stats = stats_of(cube, [box], channels=groups)

    mask = np.zeros(cube.shape[-2:], dtype=bool)
    mask[box[1] : box[3], box[0] : box[2]] = True
    region = np.asarray(ndimage.binary_dilation(mask, structure=np.ones((19, 19), dtype=bool)), dtype=bool)
    for name, bands in (("background_vis_blur", slice(0, 3)), ("background_nir_blur", slice(3, 6))):
        edges = measure_edges(gray_of(cube[bands]))
        pool = edges.widths[~region[edges.points[:, 0], edges.points[:, 1]]]
        assert stats[name][0] == pytest.approx(np.percentile(pool, 5), rel=1e-6), name  # the column is float32
    assert stats["background_vis_blur"][0] != stats["background_nir_blur"][0], "the groups differ, so the check bites"


def test_a_band_group_the_datum_cannot_supply_is_nan_with_the_warning(caplog):
    cube = to_uint8(np.stack([textured()] * 6))

    with caplog.at_level(logging.WARNING, logger="dataeval.core"):
        stats = stats_of(cube, [(20, 20, 90, 90)], channels={"bad": [5, 7]})

    assert "names band 7" in caplog.text
    assert np.isnan(stats["background_bad_blur"][0])


def test_box_order_and_image_order_change_nothing():
    boxes = [(20, 20, 90, 90), (120, 60, 200, 180), (10, 150, 60, 240)]
    forward = stats_of(picture(), boxes)
    backward = stats_of(picture(), boxes[::-1])
    assert backward["background_blur"][0] == forward["background_blur"][0]

    images = [scene(0.0), scene(3.0)]
    kwargs = {"boxes": [[BOX]] * 2, "stats": ImageStats.BLUR, "per_background": True, "per_target": False}
    both = image_stats(images, **kwargs)["stats"]
    reversed_ = image_stats(images[::-1], **kwargs)["stats"]
    assert np.array_equal(reversed_["background_blur"], both["background_blur"][::-1])
    assert both["background_blur"][0] < both["background_blur"][1], "the rows differ, so the reversal shows"


@pytest.fixture
def edge_passes(monkeypatch):
    """The edge measurements `BlurStatCalculator` makes, as the list of picture shapes, in order."""
    calls: list[tuple[int, ...]] = []

    def counting(gray):
        calls.append(np.shape(gray))
        return measure_edges(gray)

    monkeypatch.setattr(blurstats_module, "measure_edges", counting)
    return calls


def test_one_edge_pass_per_datum_and_view(edge_passes):
    """The whole-image value and the background value are read off one measurement of the image's edges."""
    kwargs = {"stats": ImageStats.BLUR, "per_background": True}
    with use_max_processes(1):
        image_stats([picture()], boxes=[[(20, 20, 90, 90)]], per_target=False, **kwargs)
        assert len(edge_passes) == 1, "one image, no box rows"

        edge_passes.clear()
        boxes = [[(20, 20, 90, 90), (100, 100, 150, 150), (160, 160, 240, 240)]]
        image_stats([picture()], boxes=boxes, per_target=True, **kwargs)
        assert len(edge_passes) == 1 + 3, "the image once and each box crop once"

        edge_passes.clear()
        cube = to_uint8(np.stack([np.roll(textured(), 3 * band, axis=1) for band in range(6)]))
        groups = {"vis": [0, 1, 2], "nir": [3, 4, 5]}
        image_stats([cube], boxes=[[BOX]], per_target=False, channels=groups, **kwargs)
        assert len(edge_passes) == 3, "the unnamed view and each group once"


def test_the_shared_value_is_the_one_computed_on_its_own():
    """Whichever statistics ride along, the column is the pooling of that image's edges, from the core functions."""
    image, boxes = picture(), [(20, 20, 90, 90), (120, 60, 200, 180)]
    expected = reference(image, boxes)

    for stats in (ImageStats.BLUR, ImageStats.ALL, ImageStats.BLUR | ImageStats.VISUAL_SHARPNESS):
        got = stats_of(image, boxes, stats)["background_blur"][0]
        assert got == pytest.approx(expected, rel=1e-6), stats


def test_without_the_image_row_the_value_is_the_same():
    """`per_image=False` leaves nothing to share, so the masked pass measures for itself, and gets the same number."""
    image, boxes = picture(), [(20, 20, 90, 90), (120, 60, 200, 180)]
    shared = stats_of(image, boxes)
    alone = image_stats([image], boxes=[boxes], stats=ImageStats.BLUR, per_background=True, per_image=False)["stats"]

    assert alone["background_blur"][0] == shared["background_blur"][0]


def test_the_edges_are_released_before_the_box_rows(monkeypatch):
    """The image's edges can run to tens of megabytes, so they are dropped once its masked pass has read them."""
    refs: list[weakref.ref] = []
    alive_when_a_box_row_starts: list[int] = []

    def counting(gray):
        gc.collect()
        alive_when_a_box_row_starts.append(sum(ref() is not None for ref in refs))
        edges = measure_edges(gray)
        refs.append(weakref.ref(edges.widths))
        return edges

    monkeypatch.setattr(blurstats_module, "measure_edges", counting)
    with use_max_processes(1):
        stats_of(picture(), [(20, 20, 90, 90), (120, 60, 200, 180)])

    assert len(alive_when_a_box_row_starts) == 3, "the image and two boxes"
    assert alive_when_a_box_row_starts[1:] == [0, 0]


def test_directionality_alone_still_warns_and_the_warning_names_the_way_out():
    with pytest.warns(
        UserWarning,
        match=r"none of the requested statistics apply to a background region.*blur directionality is not "
        r"measured.*ImageStats\.VISUAL_BLUR for it",
    ):
        image_stats([picture()], boxes=[[BOX]], stats=ImageStats.VISUAL_BLUR_DIRECTIONALITY, per_background=True)


def test_blur_for_the_background_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image_stats([picture()], boxes=[[BOX]], stats=ImageStats.VISUAL_BLUR, per_background=True)

    assert not [w for w in caught if "background region" in str(w.message)]

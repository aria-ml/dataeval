"""Blur under `image_stats(mask=)`: the edge threshold and the edge widths leave the mask out (#1593).

Under a mask the edges are still found on the whole picture, but the threshold an edge point must reach is the mean
gradient outside the mask grown by the 9 px pooling margin, and the gradient magnitude the widths run along is zeroed
on the mask grown by 1 px, so that every width profile ends at the mask. Directionality reads NaN, with the reason
``masked``.

The fixtures are built from `tests.core.test_blur.textured`, a seeded 256x256 picture of hard-edged blobs on a coarse
texture that reads a finite blur at 8 bits. The mask is a hood across the bottom 56 rows. Each test changes only the
pixels under it. Every test fails on a RuntimeWarning, so no empty-slice or zero-sum warning escapes.
"""

from typing import Any

import numpy as np
import pytest

import dataeval.core._calculators._blurstats as blurstats_module
from dataeval.config import use_max_processes
from dataeval.core import image_stats
from dataeval.core._blur import _peak_mask, background_blur, edge_widths, grow_region, measure_edges
from dataeval.flags import ImageStats
from tests.core.test_blur import gaussian, textured
from tests.core.test_blur_stats import gray_of, to_uint8
from tests.core.test_image_stats_mask import assert_same

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
    pytest.mark.filterwarnings("error::RuntimeWarning"),
]

SIZE = 256
HOOD = 200  # the first masked row
MASK = np.zeros((SIZE, SIZE), dtype=bool)
MASK[HOOD:] = True

# Box rows against the hood, as (x0, y0, x1, y1).
STRADDLE = (20.0, 150.0, 120.0, 230.0)
TOUCH = (130.0, 100.0, 230.0, 201.0)  # its last row is the hood's first
NINE_PX = (20.0, 40.0, 120.0, 191.0)  # its last row is 190, 9 rows above the hood
UNDER = (140.0, 210.0, 250.0, 250.0)
BOXES = [STRADDLE, TOUCH, NINE_PX, UNDER]


def rgb(gray: np.ndarray) -> np.ndarray:
    """A 3-band 8-bit image of a 0-255 gray picture."""
    return to_uint8(np.stack([gray] * 3))


def painted(image: np.ndarray, part: Any) -> np.ndarray:
    """`image` with only the pixels under `MASK` replaced, by a value or by the same pixels of an array."""
    out = image.copy()
    out[:, MASK] = part if np.isscalar(part) else part[:, MASK]
    return out


def blurred_frame() -> np.ndarray:
    """The issue's frame: the scene out of focus (sigma 3)."""
    return rgb(gaussian(textured(), 3.0))


def sharp_part() -> np.ndarray:
    """The issue's part: a sharp texture, which sets the frame's sharpest edges when nothing masks it."""
    return rgb(np.roll(textured(), 40, axis=1))


def wide_edge() -> np.ndarray:
    """A step from 40 to 200 at row 188, 12 px above the hood, blurred with sigma 5, so its gradient runs on into it."""
    step = np.full((SIZE, SIZE), 40.0, dtype=np.float32)
    step[HOOD - 12 :] = 200.0
    return rgb(gaussian(step, 5.0))


def stats(images: list[np.ndarray], boxes: list[list[Any]] | None = None, **kwargs: Any) -> Any:
    kwargs.setdefault("stats", ImageStats.BLUR)
    if boxes is not None:
        kwargs.setdefault("labels", [[0] * len(b) for b in boxes])
    with use_max_processes(1):
        return image_stats(images, boxes=boxes, **kwargs)


def pooled(image: np.ndarray, **kwargs: Any) -> float:
    """The image row's value from the core functions, pooled outside the grown mask."""
    return background_blur(measure_edges(gray_of(image), **kwargs), grow_region(MASK))


REGION = ~grow_region(MASK)
CUT = grow_region(MASK, 1)


class TestMaskedPixelsDoNotMatter:
    """AT1, the blur parametrizations: changing only the masked pixels moves no blur statistic or detail."""

    def test_a_sharp_part_over_a_blurred_frame(self):
        frame = blurred_frame()
        sharp, flat = painted(frame, sharp_part()), painted(frame, 0)
        kwargs: dict[str, Any] = {"per_background": True, "channels": {"rg": [0, 1]}}
        one, two = stats([sharp], mask=MASK, **kwargs), stats([flat], mask=MASK, **kwargs)
        assert_same(one, two)
        assert np.isfinite(one["stats"]["background_blur_edge_width"][0])
        # The value is the core functions' with the threshold outside the mask grown by 9 px and the cut at 1 px.
        expected = pooled(sharp, threshold_region=REGION, cut=CUT)
        assert np.isfinite(expected)
        assert one["stats"]["blur_edge_width"][0] == pytest.approx(expected, rel=1e-6)

        # Not vacuous: without the mask the sharp part sets the value, and a threshold over the whole picture lets it
        # choose which scene edges count, even with the widths cut at the mask.
        loose = [stats([image])["stats"]["blur_edge_width"][0] for image in (sharp, flat)]
        assert loose[0] < loose[1]
        assert pooled(sharp, cut=CUT) != pooled(flat, cut=CUT)

    def test_a_wide_edge_whose_profile_reaches_the_mask(self):
        # A part that continues the scene lets the edge's gradient tail run on under the mask, and a dark part ends it
        # with a step of its own.
        frame = wide_edge()
        dark = painted(frame, 0)
        one, two = stats([frame], mask=MASK), stats([dark], mask=MASK)
        assert_same(one, two)
        assert one["stats"]["blur_edge_width"][0] > 3.0, "the wide edge's width, not a sharp step's"

        # Not vacuous: with the threshold outside the margin but no cut, the profiles read the masked pixels.
        assert pooled(frame, threshold_region=REGION) != pooled(dark, threshold_region=REGION)

    @pytest.mark.parametrize("part", [np.nan, 1e6, 0.0])
    def test_a_float_part_written_as_nan_or_out_of_range(self, part):
        frame = gaussian(textured(), 1.0)[None].repeat(3, axis=0).astype(np.float64) / 255.0
        frame = np.clip(frame, 0.0, 1.0)
        one = stats([painted(frame, part)], [[STRADDLE]], mask=MASK, per_background=True)
        two = stats([painted(frame, 1.0)], [[STRADDLE]], mask=MASK, per_background=True)
        assert_same(one, two)
        assert np.isfinite(one["stats"]["blur_edge_width"]).all()
        assert np.isfinite(one["stats"]["background_blur_edge_width"][0])

    def test_box_rows_straddling_touching_near_and_under_the_mask(self):
        frame = rgb(gaussian(textured(), 1.0))
        one = stats([painted(frame, sharp_part())], [BOXES], mask=MASK, per_image=False)
        two = stats([painted(frame, 0)], [BOXES], mask=MASK, per_image=False)
        assert_same(one, two)

        width = one["stats"]["blur_edge_width"]
        assert np.isfinite(width[:3]).all(), "straddling, touching and 9 px from the mask"
        assert np.isnan(width[3])
        assert one["details"]["blur_status"][3] == "insufficient_support"
        # A box the mask does not reach reads as it does without the mask, to the bit.
        loose = stats([painted(frame, 0)], [BOXES], per_image=False)
        assert width[2] == loose["stats"]["blur_edge_width"][2]


class TestWhatStaysWhole:
    """AT2, task 2 parts: directionality under a mask, and the background value beside every statistic."""

    def test_directionality_is_nan_and_masked_on_every_row(self, monkeypatch):
        def refuse(_: Any) -> Any:
            raise AssertionError("the whole picture's values were measured under a mask")

        monkeypatch.setattr(blurstats_module, "blur_values", refuse)
        outside = (300.0, 300.0, 320.0, 320.0)  # not a valid box, so its status is invalid_box
        result = stats(
            [blurred_frame()],
            [[*BOXES, outside]],
            mask=MASK,
            per_background=True,
            channels={"rg": [0, 1]},
        )
        assert len(result["source_index"]) == 6
        for prefix in ("", "rg_"):
            assert np.isnan(result["stats"][f"{prefix}blur_edge_directionality"]).all()
            assert list(result["details"][f"{prefix}blur_edge_directionality_reason"]) == ["masked"] * 6
        assert "invalid_box" in list(result["details"]["blur_status"])

    def test_the_background_value_beside_every_statistic_is_finite_and_does_not_move(self):
        frame = blurred_frame()
        kwargs: dict[str, Any] = {"stats": ImageStats.ALL, "per_background": True, "color_rgb_channels": (0, 1, 2)}
        values = [
            stats([painted(frame, part)], [[STRADDLE]], mask=MASK, **kwargs)["stats"]["background_blur_edge_width"][0]
            for part in (sharp_part(), 0)
        ]
        assert np.isfinite(values[0])
        assert values[0] == values[1]


class TestSharedEdges:
    """AT13, task 2 part, and the one edge set the image row and its background share."""

    def test_the_background_value_is_the_same_with_and_without_the_image_row(self):
        image = painted(blurred_frame(), sharp_part())
        kwargs: dict[str, Any] = {"mask": MASK, "per_background": True, "channels": {"rg": [0, 1]}}
        shared = stats([image], [[STRADDLE]], per_target=False, **kwargs)
        alone = stats([image], [[STRADDLE]], per_target=False, per_image=False, **kwargs)
        for name in ("background_blur_edge_width", "background_rg_blur_edge_width"):
            assert np.isfinite(shared["stats"][name][0])
            assert alone["stats"][name][0] == shared["stats"][name][0], name

    def test_one_edge_pass_per_datum_and_view_under_a_mask(self, monkeypatch):
        shapes: list[tuple[int, ...]] = []

        def counting(gray: np.ndarray, **kwargs: Any) -> Any:
            shapes.append(np.shape(gray))
            return measure_edges(gray, **kwargs)

        monkeypatch.setattr(blurstats_module, "measure_edges", counting)
        stats([blurred_frame()], [[STRADDLE]], mask=MASK, per_target=False, per_background=True)
        assert shapes == [(SIZE, SIZE)]
        shapes.clear()
        stats([blurred_frame()], [[STRADDLE]], mask=MASK, per_target=False, channels={"rg": [0, 1]})
        assert len(shapes) == 2, "the unnamed view and the group once each"


class TestEdgeCases:
    """The empty-region guard, a NaN outside the mask, and the steps with nothing to leave out (AT11)."""

    def test_an_empty_threshold_region_finds_no_edges(self):
        gray = gaussian(textured(), 1.0)
        none = np.zeros(gray.shape, dtype=bool)
        edges = measure_edges(gray, threshold_region=none)
        assert len(edges.widths) == len(edges.points) == 0
        assert not _peak_mask(gray, gray, 1, region=none).any()

    def test_a_box_the_margin_covers_reads_insufficient_support(self):
        just_above = (20.0, 195.0, 120.0, 205.0)  # the mask's 9 px margin covers all of it, the mask does not
        result = stats([blurred_frame()], [[just_above]], mask=MASK, per_image=False)
        assert np.isnan(result["stats"]["blur_edge_width"][0])
        assert result["details"]["blur_status"][0] == "insufficient_support"

    def test_a_nan_outside_the_mask_reads_as_it_does_without_one(self):
        frame = gaussian(textured(), 1.0)[None].repeat(3, axis=0).astype(np.float64) / 255.0
        frame[:, SIZE // 2, SIZE // 2] = np.nan
        for kwargs in ({}, {"mask": MASK}):
            result = stats([painted(frame, 0.5)], [[STRADDLE]], per_target=False, per_background=True, **kwargs)
            assert np.isnan(result["stats"]["blur_edge_width"][0])
            assert np.isnan(result["stats"]["background_blur_edge_width"][0])
            assert result["details"]["blur_status"][0] == "insufficient_support"

    @pytest.mark.parametrize("sigma", [0.0, 1.0, 3.0])
    def test_a_cut_of_nothing_is_no_cut(self, sigma):
        gray = gaussian(textured(), sigma)
        points = measure_edges(gray).points
        nothing = np.zeros(gray.shape, dtype=bool)
        for got, want in zip(edge_widths(gray, points, cut=nothing), edge_widths(gray, points), strict=True):
            np.testing.assert_array_equal(got, want)

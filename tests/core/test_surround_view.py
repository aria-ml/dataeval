"""The surround view: the color statistics over each box's surround, on that box's row (#1616)."""

import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._calculators._contextstats import surround_regions
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import boxes_to_mask

pytestmark = pytest.mark.required

RGB = (0, 1, 2)
SURROUND_STATS = [
    "surround_color_lightness_mean",
    "surround_color_chroma_mean",
    "surround_color_achromatic_fraction",
    "surround_color_hue_cos_mean",
    "surround_color_hue_sin_mean",
]
SURROUND_DETAILS = [
    "surround_color_status",
    "surround_color_valid_fraction",
    "surround_color_valid_pixels",
    "surround_color_excluded_pixels",
    "surround_color_hue_reason",
]
GOOD = [40.0, 30.0, 60.0, 50.0]


def _image(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (3, 96, 128), dtype=np.uint8)


def run(images: list[np.ndarray], boxes: list[list[Any]], **kwargs: Any) -> Any:
    kwargs.setdefault("stats", ImageStats.COLOR)
    kwargs.setdefault("color_rgb_channels", RGB)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return image_stats(images, boxes=boxes, labels=[[0] * len(b) for b in boxes], **kwargs)


def column(result: Any, name: str) -> list[Any]:
    values = result["stats"][name] if name in result["stats"] else result["details"][name]
    return [v.item() if hasattr(v, "item") else v for v in values]


def window_pixels(box: list[float], extent: tuple[int, int], padding: float = 0.25) -> int:
    x0, y0, x1, y1 = surround_regions(np.asarray(box, float), extent, padding).window
    return (x1 - x0) * (y1 - y0)


class TestValues:
    def test_a_uniform_surround_reads_its_own_color_and_the_box_keeps_its_columns(self):
        image = np.empty((3, 96, 128), np.uint8)
        image[:] = np.array([200, 40, 30], np.uint8)[:, None, None]
        image[:, 30:50, 40:60] = np.array([20, 90, 210], np.uint8)[:, None, None]
        solid = np.broadcast_to(np.array([200, 40, 30], np.uint8)[:, None, None], (3, 16, 16)).copy()
        expected = run([solid], [[]], per_target=False)
        with_view = run([image], [[GOOD]], per_image=False, per_surround=True)
        without = run([image], [[GOOD]], per_image=False)
        for name in SURROUND_STATS:
            assert column(with_view, name)[0] == pytest.approx(
                column(expected, name.removeprefix("surround_"))[0], abs=1e-6
            )
        for name in without["stats"]:
            np.testing.assert_array_equal(with_view["stats"][name], without["stats"][name])
        assert column(with_view, "surround_color_status") == ["ok"]

    def test_crowding_leaves_ok_while_the_uncovered_pixels_meet_the_rule(self):
        neighbor = [30.0, 20.0, 70.0, 30.0]  # covers the top band of GOOD's window
        image = _image()
        result = run([image], [[GOOD, neighbor]], per_image=False, per_surround=True)
        mask = boxes_to_mask(image.shape, [GOOD, neighbor])
        x0, y0, x1, y1 = surround_regions(np.asarray(GOOD), (96, 128), 0.25).window
        uncovered = int((~mask[y0:y1, x0:x1]).sum())
        assert column(result, "surround_color_valid_pixels")[0] == uncovered
        assert column(result, "surround_color_status")[0] == "ok"
        short = run([image], [[GOOD, neighbor]], per_image=False, per_surround=True, color_min_pixels=uncovered + 1)
        assert column(short, "surround_color_status")[0] == "insufficient_support"
        assert np.isnan(column(short, "surround_color_lightness_mean")[0])

    def test_a_box_that_fills_its_window_has_too_few_pixels(self):
        result = run([_image()], [[[0.0, 0.0, 128.0, 96.0]]], per_image=False, per_surround=True)
        assert column(result, "surround_color_status") == ["insufficient_support"]
        assert column(result, "surround_color_valid_pixels") == [0]


class TestRegion:
    def test_the_counts_agree_with_the_context_statistics_on_the_same_box(self):
        boxes = [GOOD, [55.0, 45.0, 75.0, 70.0], [5.5, 5.2, 20.7, 18.9], [100.0, 60.0, 128.0, 96.0]]
        result = run(
            [_image()],
            [boxes],
            stats=ImageStats.COLOR | ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN,
            per_image=False,
            per_surround=True,
        )
        assert column(result, "surround_color_valid_pixels") == column(result, "surround_measurable_pixels")
        excluded = [
            window_pixels(b, (96, 128)) - r
            for b, r in zip(boxes, column(result, "surround_retained_pixels"), strict=True)
        ]
        assert column(result, "surround_color_excluded_pixels") == excluded


class TestBoxes:
    def test_a_reversed_box_reads_the_surround_of_its_twin(self):
        reversed_ = [60.0, 50.0, 40.0, 30.0]
        a = run([_image()], [[reversed_]], per_image=False, per_surround=True)
        b = run([_image()], [[GOOD]], per_image=False, per_surround=True)
        assert column(a, "surround_color_status") == ["ok"]
        for name in [*SURROUND_STATS, *SURROUND_DETAILS]:
            assert column(a, name) == column(b, name)

    def test_a_reversed_neighbor_is_masked_as_its_twin(self):
        a = run([_image()], [[GOOD, [70.0, 45.0, 55.0, 35.0]]], per_image=False, per_surround=True)
        b = run([_image()], [[GOOD, [55.0, 35.0, 70.0, 45.0]]], per_image=False, per_surround=True)
        assert column(a, "surround_color_excluded_pixels") == column(b, "surround_color_excluded_pixels")

    def test_boxes_the_column_calls_invalid_read_invalid_box_and_no_other_box_does(self):
        boxes = [
            GOOD,
            [300.0, 300.0, 320.0, 320.0],  # outside
            [20.0, 20.0, 20.0, 40.0],  # zero area
            [np.nan, 5.0, 10.0, 10.0],  # not finite
            [130.0, 10.0, 150.0, 30.0],  # just past the right edge, its window reaching in
            [60.0, 50.0, 40.0, 30.0],  # reversed, valid as its twin
        ]
        result = run(
            [_image()],
            [boxes],
            stats=ImageStats.COLOR | ImageStats.DIMENSION_INVALID_BOX,
            per_image=False,
            per_surround=True,
        )
        flagged = [bool(v) for v in column(result, "invalid_box")]
        statuses = column(result, "surround_color_status")
        assert flagged == [False, True, True, True, True, False]
        assert [s == "invalid_box" for s in statuses] == flagged
        for status, value in zip(statuses, column(result, "surround_color_lightness_mean"), strict=True):
            assert (status == "ok") == (not np.isnan(value))


class TestAlignment:
    def test_every_row_of_every_image_carries_the_names(self):
        images = [_image(1), _image(2), _image(3)]
        boxes = [[], [[300.0, 300.0, 320.0, 320.0]], [GOOD]]
        result = run(images, boxes, per_surround=True)
        rows = len(result["source_index"])
        assert rows == 5  # three image rows and two box rows
        for name in SURROUND_STATS:
            assert len(result["stats"][name]) == rows
        for name in SURROUND_DETAILS:
            assert len(result["details"][name]) == rows
        status = column(result, "surround_color_status")
        keys = [s.key for s in result["source_index"]]
        assert [st for st, k in zip(status, keys, strict=True) if k is None] == ["not_applicable"] * 3
        assert [st for st, k in zip(status, keys, strict=True) if k is not None] == ["invalid_box", "ok"]

    def test_an_image_with_no_box_alone_still_carries_the_names(self):
        result = run([_image()], [[]], per_surround=True)
        assert column(result, "surround_color_status") == ["not_applicable"]
        assert np.isnan(column(result, "surround_color_lightness_mean")).all()


class TestNothingElseMoves:
    @pytest.mark.parametrize("per_background", [False, True])
    def test_the_view_adds_columns_and_changes_none(self, per_background):
        boxes = [[GOOD, [55.0, 45.0, 75.0, 70.0], [300.0, 300.0, 320.0, 320.0]], []]
        kwargs: dict[str, Any] = {"stats": ImageStats.ALL, "per_background": per_background}
        base = run([_image(1), _image(2)], boxes, **kwargs)
        off = run([_image(1), _image(2)], boxes, per_surround=False, **kwargs)
        on = run([_image(1), _image(2)], boxes, per_surround=True, **kwargs)
        for part in ("stats", "details"):
            assert set(off[part]) == set(base[part])
            for name in base[part]:
                np.testing.assert_array_equal(off[part][name], base[part][name])
                np.testing.assert_array_equal(on[part][name], base[part][name])
        assert set(on["stats"]) - set(base["stats"]) == set(SURROUND_STATS)
        assert set(on["details"]) - set(base["details"]) == set(SURROUND_DETAILS)


def test_a_datum_with_no_image_plane_carries_the_names_without_raising():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: Any = image_stats(
            [np.arange(50, dtype=np.uint8)], stats=ImageStats.COLOR, color_rgb_channels=RGB, per_surround=True
        )
    assert column(result, "surround_color_status") == ["not_applicable"]


class TestSpecs:
    def test_the_view_records_its_region_and_padding(self):
        result = run([_image()], [[GOOD]], per_surround=True, surround_padding=0.5)
        surround = result["measurement_specs"]["surround_color_lightness_mean"]
        own = result["measurement_specs"]["color_lightness_mean"]
        assert surround.region == "surround"
        assert surround.config["surround_padding"] == 0.5
        assert own.region == "pixels"
        assert "surround_padding" not in own.config

    def test_results_measured_with_different_paddings_do_not_combine(self):
        from dataeval.core import combine_stats_results

        a = run([_image()], [[GOOD]], per_surround=True)
        b = run([_image()], [[GOOD]], per_surround=True, surround_padding=0.5)
        with pytest.raises(ValueError, match="surround_padding"):
            combine_stats_results([a, b])
        _, source_index, _ = combine_stats_results([a, a])
        assert len(source_index) == 4


def _user_warnings(**kwargs: Any) -> list[str]:
    kwargs.setdefault("stats", ImageStats.COLOR)
    kwargs.setdefault("color_rgb_channels", RGB)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image_stats(**kwargs)
    return [str(w.message) for w in caught if w.category is UserWarning]


class TestWarnings:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"data": [np.zeros((3, 32, 32), np.uint8)]},  # no boxes at all
            {"data": [np.zeros((3, 32, 32), np.uint8)], "boxes": [[[2, 2, 9, 9]]], "per_target": False},
            {"data": [np.zeros((3, 32, 32), np.uint8)] * 2, "boxes": [[], []]},  # the data holds no box
        ],
    )
    def test_a_call_with_no_box_rows_warns_once(self, kwargs):
        said = [m for m in _user_warnings(per_surround=True, **kwargs) if "per_surround" in m]
        assert len(said) == 1
        assert "box" in said[0]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stats": ImageStats.ALL, "color_rgb_channels": None},
            {"stats": ImageStats.PIXEL_MEAN},
            {"stats": {"g": ImageStats.COLOR}, "channels": {"g": [0, 1, 2]}},
        ],
    )
    def test_a_request_nothing_in_answers_warns_once(self, kwargs):
        said = _user_warnings(data=[_image()], boxes=[[GOOD]], per_surround=True, **kwargs)
        assert len([m for m in said if "per_surround" in m]) == 1

    def test_the_padding_is_read_by_the_view_and_the_support_floor_is_not(self):
        padding = _user_warnings(data=[_image()], boxes=[[GOOD]], per_surround=True, surround_padding=0.5)
        assert not [m for m in padding if "ignored" in m]
        floor = _user_warnings(data=[_image()], boxes=[[GOOD]], per_surround=True, surround_min_pixels=8)
        assert [m for m in floor if "surround_min_pixels is ignored" in m]
        alone = _user_warnings(data=[_image()], boxes=[[GOOD]], surround_padding=0.5)
        assert [m for m in alone if "surround_padding is ignored" in m]

    def test_a_measured_call_says_nothing(self):
        assert _user_warnings(data=[_image()], boxes=[[GOOD]], labels=[[0]], per_surround=True) == []


class TestCollisions:
    def test_a_channel_group_named_color_with_the_context_statistics_raises(self):
        with pytest.raises(ValueError, match="surround_color_valid_fraction"):
            run(
                [_image()],
                [[GOOD]],
                stats={None: ImageStats.COLOR, "color": ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN},
                channels={"color": [0]},
                per_surround=True,
            )

    def test_a_group_named_surround_measures_beside_the_view(self):
        result = run(
            [_image()],
            [[GOOD]],
            stats={None: ImageStats.COLOR, "surround": ImageStats.PIXEL_MEAN},
            channels={"surround": [0]},
            per_surround=True,
        )
        assert {"surround_mean", "surround_color_lightness_mean"} <= set(result["stats"])


class TestReviewFindings:
    def test_an_invalid_box_has_no_surround_to_exclude_from(self):
        boxes = [GOOD, [20.5, 5.0, 20.5, 40.0], [300.0, 300.0, 320.0, 320.0]]
        result = run([_image()], [boxes], per_image=False, per_surround=True)
        assert column(result, "surround_color_status")[1:] == ["invalid_box", "invalid_box"]
        # As on the box's own row, where nothing is masked: none of the box's own footprint is counted.
        assert column(result, "surround_color_excluded_pixels")[1:] == column(result, "color_excluded_pixels")[1:]

    def test_a_per_row_color_warning_names_each_box_row_once(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="dataeval"):
            run(
                [np.zeros((3, 20, 20), np.uint16)],
                [[[2, 2, 8, 8], [10, 10, 15, 15]]],
                per_surround=True,
            )
        said = [r.getMessage() for r in caplog.records if "unknown_range" in r.getMessage()]
        assert said
        assert all(m.count("0[0]") == 1 and m.count("0[1]") == 1 for m in said), said

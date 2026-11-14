"""The CONTEXT surround statistics: flags, values, statuses, details and the rules for a request that includes them."""

import math
import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._calculators._base import ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._contextstats import ContextStatCalculator
from dataeval.flags import ImageStats
from dataeval.types._stats_config import ImageStatsConfig, SurroundConfig
from dataeval.utils._annotations import AnnotationContext, build_annotations
from dataeval.utils.preprocessing import get_value_range

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

CONTEXT = ImageStats.CONTEXT
NAN = float("nan")
FOCAL = (3, 3, 6, 6)
F2_BOXES = [FOCAL, (5, 5, 9, 9)]
# The 13 surround pixels of F2 as (y, x), and the 16 of the focal box alone, for hand-computed expectations.
F2_RING = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 2), (3, 6), (4, 2), (4, 6), (5, 2), (6, 2), (6, 3), (6, 4)]
F2_MEAN = float(np.mean([(10 * y + x) / 99 for y, x in F2_RING]))
F2_STD = float(np.std([(10 * y + x) / 99 for y, x in F2_RING]))


def details_of(result: Any) -> Any:
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def f2_image() -> np.ndarray:
    return np.arange(100).reshape(1, 10, 10) / 99


def measure(
    datum: np.ndarray,
    boxes: list[Any],
    focal: int | None,
    *,
    min_pixels: int = 1,
    padding: float = 0.25,
    declared: tuple[float, float] | None = None,
    bands: tuple[int, ...] | None = None,
    view: ViewKind = ViewKind.WHOLE,
) -> dict[str, Any]:
    """One row's statistics and details, read from the calculator itself, as one flat mapping of scalars."""
    annotations = build_annotations(boxes, datum.shape, 0)
    context = AnnotationContext(annotations, (datum.shape[-2], datum.shape[-1]))
    cache = CalculatorCache(
        datum,
        None if focal is None else annotations.boxes[focal],
        value_range=get_value_range(datum, declared=declared),
        bands=bands,
        context=context,
        focal=focal,
        settings=ImageStatsConfig(surround=SurroundConfig(padding=padding, min_pixels=min_pixels)),
    )
    calculator = ContextStatCalculator(datum, cache)
    flat = {**calculator.compute(CONTEXT, view), **calculator.details(CONTEXT, view)}
    return {name: values[0] for name, values in flat.items()}


class TestFlags:
    def test_the_names_split_into_family_and_statistic(self):
        # `quality._shared._stat_names` and dataeval-flow's `stat_names_for` both read a flag as <FAMILY>_<STATISTIC>.
        names = [(m.name or "").split("_", 1)[1].lower() for m in CONTEXT]
        assert names == ["surround_mean", "surround_std", "target_surround_contrast", "signal_clutter_ratio"]

    def test_all_holds_the_group_and_no_other_group_does(self):
        assert CONTEXT in ImageStats.ALL
        for group in ("STANDARD", "NO_HASH", "BASIC", "PIXEL", "VISUAL", "DIMENSION", "HASH", "BLUR", "SPATIAL"):
            assert not CONTEXT & getattr(ImageStats, group), group

    def test_the_v12_all_is_still_the_v12_column_set(self):
        v12 = ImageStats.ALL & ~ImageStats.BLUR & ~ImageStats.SPATIAL & ~ImageStats.PIXEL_SATURATION & ~CONTEXT
        assert v12 == (ImageStats.STANDARD | ImageStats.HASH) & ~ImageStats.PIXEL_SATURATION

    def test_no_existing_bit_moves(self):
        assert ImageStats.CONTEXT_SURROUND_MEAN.value > ImageStats.PIXEL_SATURATION.value
        assert ImageStats.CONTEXT_SURROUND_STD.value > ImageStats.CONTEXT_SURROUND_MEAN.value
        assert ImageStats.CONTEXT_TARGET_SURROUND_CONTRAST.value > ImageStats.CONTEXT_SURROUND_STD.value
        assert ImageStats.CONTEXT_SIGNAL_CLUTTER_RATIO.value > ImageStats.CONTEXT_TARGET_SURROUND_CONTRAST.value


class TestValues:
    def test_a_constant_ring_around_a_distinct_target(self):
        image = np.full((1, 10, 10), 0.2)
        image[0, 3:6, 3:6] = 0.8  # the target, so a region read wrongly reads wrong
        row = measure(image, [FOCAL], 0)
        assert row["surround_mean"] == pytest.approx(0.2, abs=1e-12)
        assert row["surround_mean"] != pytest.approx(0.8)
        assert row["surround_std"] <= 1e-12
        assert (row["context_status"], row["surround_candidate_pixels"]) == ("ok", 16)

    def test_f2_overlap(self):
        row = measure(f2_image(), F2_BOXES, 0)
        assert row["surround_mean"] == pytest.approx(F2_MEAN, abs=1e-12)
        assert row["surround_std"] == pytest.approx(F2_STD, abs=1e-12)
        assert row["context_status"] == "ok"
        assert row["surround_candidate_pixels"] == 16
        assert row["surround_retained_pixels"] == 13
        assert row["ring_peer_excluded_fraction"] == 0.1875
        assert row["surround_measurable_pixels"] == 13
        assert row["surround_valid_fraction"] == 1.0
        assert row["target_valid_fraction"] == 1.0
        assert (row["context_range_low"], row["context_range_high"]) == (0.0, 1.0)

    def test_a_twin_of_the_focal_box_changes_nothing(self):
        assert measure(f2_image(), [*F2_BOXES, FOCAL], 0) == measure(f2_image(), F2_BOXES, 0)

    @pytest.mark.parametrize(
        "peer",
        [(4, 2, 4, 6), (20, 20, 30, 30)],
        ids=["zero_area", "outside"],
    )
    def test_a_peer_that_is_skipped_is_not_removed_from_the_ring_and_leaves_the_status(self, peer):
        row = measure(f2_image(), [*F2_BOXES, peer], 0)
        assert row == measure(f2_image(), F2_BOXES, 0)

    @pytest.mark.parametrize(
        "peer",
        [(NAN, 0, 5, 5), (9, 9, 6, 6)],
        ids=["nonfinite", "reversed"],
    )
    def test_a_corrupt_peer_keeps_the_value_and_says_so(self, peer):
        # The statistic removes valid boxes, so a box that is not valid is not removed, and the value stays correct.
        row = measure(f2_image(), [*F2_BOXES, peer], 0)
        expected = measure(f2_image(), F2_BOXES, 0)
        assert row["context_status"] == "incomplete_geometry"
        assert {k: v for k, v in row.items() if k != "context_status"} == {
            k: v for k, v in expected.items() if k != "context_status"
        }

    def test_f3_a_ring_mostly_covered_stays_finite_and_says_so(self):
        row = measure(f2_image(), [FOCAL, (0, 0, 10, 6)], 0, min_pixels=5)
        assert row["context_status"] == "ok"
        assert row["surround_retained_pixels"] == 5  # the row y = 6, x = 2..6
        assert row["ring_peer_excluded_fraction"] == 0.6875
        assert row["surround_mean"] == pytest.approx(0.6464646464646464, abs=1e-12)
        assert row["surround_std"] == pytest.approx(0.014284985478516093, abs=1e-12)

    def test_f3_a_ring_wholly_covered_is_insufficient_support(self):
        # A peer that contains the whole window, as a parent box such as an ignored region does.
        row = measure(f2_image(), [FOCAL, (0, 0, 10, 10)], 0)
        assert row["context_status"] == "insufficient_support"
        assert row["surround_retained_pixels"] == 0
        assert math.isnan(row["surround_valid_fraction"])
        assert math.isnan(row["surround_mean"])

    @pytest.mark.parametrize(
        ("box", "candidate"),
        [((0, 0, 4, 4), 9), ((2, 2, 6, 6), 20), ((2.5, 2.5, 5.5, 5.5), 20), ((-4, 2, 4, 6), 20)],
        ids=["corner", "interior", "fractional", "half_outside"],
    )
    def test_f4_edges_and_fractions(self, box, candidate):
        # The half-outside box takes its width and height from its raw coordinates: a clipped width would give 14.
        assert measure(np.zeros((1, 8, 8)), [box], 0)["surround_candidate_pixels"] == candidate


class TestEncodings:
    def test_uint8_declared_uint16_and_float_agree_to_float64_precision(self):
        # Extremes of 10 and 200, so a range read off the view or the box would not give the encoding's 255 or 1.
        y, x = np.mgrid[0:60, 0:80]
        u8 = np.stack([10 + (3 * y + 7 * x + 11 * c) % 191 for c in range(3)]).astype(np.uint8)
        box = [(10, 10, 40, 40)]
        reference = measure(u8, box, 0, min_pixels=32)
        assert reference["context_range_high"] == 255.0
        for other in (
            measure(u8.astype(np.uint16), box, 0, min_pixels=32, declared=(0, 255)),
            measure(u8 / 255.0, box, 0, min_pixels=32),
        ):
            assert other["surround_mean"] == pytest.approx(reference["surround_mean"], abs=1e-12)
            assert other["surround_std"] == pytest.approx(reference["surround_std"], abs=1e-12)


class TestStatuses:
    def test_invalid_box(self):
        row = measure(f2_image(), [(NAN, 0, 5, 5), (1, 1, 3, 3)], 0)
        assert row["context_status"] == "invalid_box"
        assert math.isnan(row["surround_mean"])
        assert math.isnan(row["surround_std"])
        assert (row["surround_candidate_pixels"], row["surround_retained_pixels"]) == (-1, -1)
        assert row["surround_measurable_pixels"] == -1
        assert math.isnan(row["ring_peer_excluded_fraction"])
        assert math.isnan(row["surround_valid_fraction"])
        assert math.isnan(row["target_valid_fraction"])
        assert (row["context_range_low"], row["context_range_high"]) == (0.0, 1.0)

    def test_invalid_box_wins_over_an_incomplete_frame(self):
        # The NaN box is the focal one and sits in a frame that also holds a reversed box.
        assert measure(f2_image(), [(NAN, 0, 5, 5), (9, 9, 6, 6)], 0)["context_status"] == "invalid_box"

    def test_unknown_range(self):
        image = np.full((1, 10, 10), 0.5)
        image[0, 8, 8] = np.inf  # a float image with an infinite value decodes to no interval
        row = measure(image, [FOCAL], 0)
        assert row["context_status"] == "unknown_range"
        assert math.isnan(row["surround_mean"])
        assert (row["surround_candidate_pixels"], row["surround_retained_pixels"]) == (16, 16)
        assert row["ring_peer_excluded_fraction"] == 0.0
        assert row["surround_measurable_pixels"] == -1
        assert math.isnan(row["context_range_low"])
        assert math.isnan(row["context_range_high"])

    @pytest.mark.parametrize(
        "declared", [(-1.7e308, 1.7e308), (-1e308, 1e308)], ids=["overflowing", "also_overflowing"]
    )
    def test_a_declared_range_whose_span_is_not_finite_is_unknown_and_not_a_wrong_value(self, declared):
        # `high - low` is infinity, so every intensity would read 0 or NaN under the status `ok`.
        image = np.random.default_rng(0).random((1, 10, 10))
        row = measure(image, [FOCAL], 0, declared=declared)
        assert row["context_status"] == "unknown_range"
        assert math.isnan(row["surround_mean"])
        assert math.isnan(row["surround_std"])

    @pytest.mark.parametrize("box", [(0, 0, 10, 10), None], ids=["filling_box", "full_image_entry"])
    def test_empty_ring(self, box):
        # A box that fills its window leaves no surround. A `None` entry is the full-image box.
        row = measure(f2_image(), [box], 0)
        assert row["context_status"] == "empty_ring"
        assert math.isnan(row["surround_mean"])
        assert (row["surround_candidate_pixels"], row["surround_retained_pixels"]) == (0, 0)
        assert math.isnan(row["ring_peer_excluded_fraction"])
        assert row["surround_measurable_pixels"] == 0
        assert math.isnan(row["surround_valid_fraction"])
        assert row["target_valid_fraction"] == 1.0
        assert (row["context_range_low"], row["context_range_high"]) == (0.0, 1.0)

    def test_insufficient_support(self):
        row = measure(f2_image(), F2_BOXES, 0, min_pixels=14)  # F2 has 13
        assert row["context_status"] == "insufficient_support"
        assert math.isnan(row["surround_mean"])
        assert math.isnan(row["surround_std"])
        assert (row["surround_candidate_pixels"], row["surround_retained_pixels"]) == (16, 13)
        assert row["surround_measurable_pixels"] == 13
        assert row["surround_valid_fraction"] == 1.0

    def test_the_outer_box_keeps_its_surround_and_the_box_inside_it_has_none(self):
        # An ignored region or a scene-level box does not lose its own surround: the boxes inside it do.
        image = np.zeros((1, 200, 200))
        outer, inner = (20, 20, 120, 120), (50, 50, 70, 70)
        assert measure(image, [outer, inner], 0, min_pixels=32)["context_status"] == "ok"
        inside = measure(image, [outer, inner], 1, min_pixels=32)
        assert inside["context_status"] == "insufficient_support"
        assert inside["surround_retained_pixels"] == 0

    def test_the_image_row_is_not_applicable(self):
        row = measure(f2_image(), F2_BOXES, None)
        assert row["context_status"] == "not_applicable"
        assert math.isnan(row["surround_mean"])
        assert row["surround_candidate_pixels"] == -1


def picture(seed: int = 1605) -> np.ndarray:
    return np.random.default_rng(seed).random((3, 60, 80))


BIG_BOXES = [(10, 10, 40, 40), (45, 15, 75, 45)]


class TestThroughImageStats:
    def test_the_defaults_measure_big_boxes_and_the_image_row_is_not_applicable(self):
        # Review Focus 3: an image row and box rows together keep their alignment and the integer dtype of the counts.
        result = image_stats([picture()], boxes=[BIG_BOXES], stats=CONTEXT)
        details = details_of(result)
        assert details["context_status"].tolist() == ["not_applicable", "ok", "ok"]
        assert np.isnan(result["stats"]["surround_mean"][0])
        assert details["surround_candidate_pixels"].dtype.kind == "i"
        assert details["surround_candidate_pixels"].tolist() == [-1, 1216, 1078]
        assert not np.isnan(result["stats"]["surround_mean"][1:]).any()

    def test_per_background_adds_no_context_column(self):
        result = image_stats([picture()], boxes=[BIG_BOXES], stats=ImageStats.PIXEL_MEAN | CONTEXT, per_background=True)
        assert "surround_mean" in result["stats"]
        assert not [name for name in result["stats"] if name.startswith("background_surround")]


def images(n: int = 2) -> list[np.ndarray]:
    return [picture(i) for i in range(n)]


def caught_for_peers(stats: ImageStats) -> list[str]:
    """The messages of the warnings that say a statistic needs box rows, for a call with no box rows."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image_stats(images(), stats=stats, per_target=False)
    return [str(w.message) for w in caught if "box" in str(w.message) and "peers" in str(w.message)]


class TestRequestRules:
    def test_only_context_flags_without_box_rows_raises(self):
        with pytest.raises(ValueError, match="need box rows"):
            image_stats(images(), stats=CONTEXT, per_target=False)

    def test_a_mixed_request_without_box_rows_warns_and_names_the_flags_and_what_they_read(self):
        with pytest.warns(UserWarning, match="CONTEXT_SURROUND_MEAN, CONTEXT_SURROUND_STD.*read the pixels around"):
            image_stats(images(), stats=ImageStats.PIXEL | CONTEXT, per_target=False)

    def test_all_is_silent_and_returns_nan_columns(self):
        assert caught_for_peers(ImageStats.ALL) == []
        result = image_stats(images(), stats=ImageStats.ALL, per_target=False)
        assert np.isnan(result["stats"]["surround_mean"]).all()

    @pytest.mark.parametrize("extra", [ImageStats.SPATIAL_BOX_IN_IMAGE_FRACTION, ImageStats.CONTEXT_SURROUND_MEAN])
    def test_everything_but_the_box_needing_statistics_plus_one_is_silent(self, extra):
        # `asks_for_all` names what needs boxes, so the next box-needing family cannot miss it.
        assert caught_for_peers((ImageStats.ALL & ~(ImageStats.SPATIAL | CONTEXT)) | extra) == []

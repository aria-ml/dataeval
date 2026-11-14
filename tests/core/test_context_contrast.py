"""The CONTEXT target-surround contrast and signal-to-clutter ratio: values, floors, support, details and specs.

Every fixture has a 10 x 10 target box ``(10, 10, 20, 20)`` in a 30 x 30 image. Its window under the default padding is
``(7, 7, 23, 23)``, so the ring holds 156 pixels, which a test sets one by one, and no other box takes part of it.
"""

import json
import math
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._calculators._base import ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._contextstats import ContextStatCalculator
from dataeval.flags import ImageStats
from dataeval.types import MeasurementSpec
from dataeval.types._stats_config import ImageStatsConfig, SurroundConfig
from dataeval.utils._annotations import AnnotationContext, build_annotations
from dataeval.utils.preprocessing import get_value_range

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

CONTEXT = ImageStats.CONTEXT
CONTRAST = ImageStats.CONTEXT_TARGET_SURROUND_CONTRAST
CLUTTER = ImageStats.CONTEXT_SIGNAL_CLUTTER_RATIO
NAN = float("nan")
BOX = (10, 10, 20, 20)
RING = 156
DEFAULT_FLOOR = 1 / 255
RATIOS = ("target_surround_contrast", "signal_clutter_ratio")
NEW_DETAILS = ("contrast_reason", "target_surround_contrast_floored", "signal_clutter_ratio_floored")


def two_valued(low: float, high: float) -> np.ndarray:
    """The ring's values: half of its pixels at `low` and half at `high`, so the mean is their midpoint."""
    return np.array([low] * (RING // 2) + [high] * (RING // 2))


def ring_image(target: float | np.ndarray, ring: np.ndarray | float) -> np.ndarray:
    """A 1 x 30 x 30 image whose target is `target` (a scalar or 10 x 10) and whose 156 ring pixels are `ring`."""
    window = np.zeros((16, 16))
    mask = np.ones((16, 16), dtype=bool)
    mask[3:13, 3:13] = False  # the target sits at (3, 3) inside the window, which starts at (7, 7)
    window[mask] = ring  # row-major, a scalar or one value for each of the 156 pixels
    window[3:13, 3:13] = target
    image = np.zeros((1, 30, 30))
    image[0, 7:23, 7:23] = window
    return image


def measure(
    datum: np.ndarray,
    boxes: list[Any] | None = None,
    focal: int | None = 0,
    *,
    flags: ImageStats = CONTEXT,
    min_pixels: int = 32,
    floor: float = DEFAULT_FLOOR,
    minimum: float = 0.90,
    declared: tuple[float, float] | None = (0.0, 1.0),
    bands: tuple[int, ...] | None = None,
    view: ViewKind = ViewKind.WHOLE,
) -> dict[str, Any]:
    """One row's statistics and details, read from the calculator itself, as one flat mapping of scalars."""
    boxes = [BOX] if boxes is None else boxes
    annotations = build_annotations(boxes, datum.shape, 0)
    context = AnnotationContext(annotations, (datum.shape[-2], datum.shape[-1]))
    cache = CalculatorCache(
        datum,
        None if focal is None else annotations.boxes[focal],
        value_range=get_value_range(datum, declared=declared),
        bands=bands,
        context=context,
        focal=focal,
        settings=ImageStatsConfig(
            surround=SurroundConfig(min_pixels=min_pixels, intensity_floor=floor, min_target_valid_fraction=minimum)
        ),
    )
    calculator = ContextStatCalculator(datum, cache)
    flat = {**calculator.compute(flags, view), **calculator.details(flags, view)}
    return {name: values[0] for name, values in flat.items()}


def details_of(result: Any) -> Any:
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def assert_ratios_are_nan(row: dict[str, Any]) -> None:
    for name in RATIOS:
        assert math.isnan(row[name]), name


def with_infinity() -> np.ndarray:
    """A ring image with an infinite pixel outside the window: a float image like that decodes to no interval."""
    image = ring_image(0.6, 0.2)
    image[0, 29, 29] = np.inf
    return image


class TestValues:
    def test_f6_gives_2_and_4(self):
        # target 0.6; ring half 0.1 and half 0.3, so the mean is 0.2 and the population standard deviation 0.1
        row = measure(ring_image(0.6, two_valued(0.1, 0.3)))
        assert row["context_status"] == "ok"
        assert row["target_surround_contrast"] == pytest.approx(2.0, rel=1e-6)
        assert row["signal_clutter_ratio"] == pytest.approx(4.0, rel=1e-6)
        assert (row["target_surround_contrast_floored"], row["signal_clutter_ratio_floored"]) == (False, False)
        assert row["contrast_reason"] == ""

    def test_the_sign_is_kept(self):
        # a darker target: (0.1 - 0.6) over 0.6, and over the standard deviation 0.1
        row = measure(ring_image(0.1, two_valued(0.5, 0.7)))
        assert row["target_surround_contrast"] == pytest.approx(-0.5 / 0.6, rel=1e-6)
        assert row["signal_clutter_ratio"] == pytest.approx(-5.0, rel=1e-6)

    def test_a_target_equal_to_the_surround_mean_reads_zero(self):
        row = measure(ring_image(0.2, two_valued(0.1, 0.3)))
        assert row["target_surround_contrast"] == pytest.approx(0.0, abs=1e-12)
        assert row["signal_clutter_ratio"] == pytest.approx(0.0, abs=1e-12)

    def test_the_target_mean_covers_the_whole_box_including_a_peer(self):
        # The target's left half is 0.2 and its right half 1.0, so its mean is 0.6. A peer box covers the left half.
        # Its pixels leave the ring, not the target: a mean without them would read 1.0, and the contrast 4 and not 2.
        target = np.full((10, 10), 1.0)
        target[:, :5] = 0.2
        row = measure(ring_image(target, two_valued(0.1, 0.3)), [BOX, (10, 10, 15, 20)])
        assert row["context_status"] == "ok"
        assert row["surround_mean"] == pytest.approx(0.2, abs=1e-12)  # the peer lies inside the target, not the ring
        assert row["target_surround_contrast"] == pytest.approx(2.0, rel=1e-6)
        assert row["signal_clutter_ratio"] == pytest.approx(4.0, rel=1e-6)

    @pytest.mark.parametrize("dtype", ["uint8", "float32"])
    def test_other_dtypes_read_the_same_ratios(self, dtype):
        # uint8 decodes to [0, 255]: 51, 153 and 204 are 0.2, 0.6 and 0.8, so the ring has mean 0.4 and spread 0.2.
        ring, target, scale = two_valued(51, 153), 204, 255
        image = ring_image(target, ring)
        image = image.astype(dtype) if dtype == "uint8" else (image / scale).astype(dtype)
        result = image_stats([image], boxes=[[BOX]], stats=CONTEXT, per_image=False)
        assert result["stats"]["target_surround_contrast"][0] == pytest.approx(1.0, rel=1e-5)
        assert result["stats"]["signal_clutter_ratio"][0] == pytest.approx(2.0, rel=1e-5)


class TestFloors:
    @pytest.mark.parametrize(
        ("name", "target", "ring", "contrast_floored", "clutter_floored"),
        [
            ("one_bright_pixel", 0.6, np.r_[0.1, np.zeros(RING - 1)], True, False),
            ("constant_half", 0.6, np.full(RING, 0.5), False, True),
            ("constant_point_six", 0.8, np.full(RING, 0.6), False, True),  # its std is rounding residue, not 0
        ],
    )
    def test_a_denominator_under_the_floor_is_replaced_and_recorded(
        self, name, target, ring, contrast_floored, clutter_floored
    ):
        row = measure(ring_image(target, ring))
        mean, std = float(ring.mean()), float(ring.std())
        assert row["target_surround_contrast_floored"] is contrast_floored, name
        assert row["signal_clutter_ratio_floored"] is clutter_floored, name
        for key, denominator in (("target_surround_contrast", mean), ("signal_clutter_ratio", std)):
            value = row[key]
            assert math.isfinite(value)
            assert abs(value) <= (1 / DEFAULT_FLOOR) * (1 + 1e-9)  # |mu_t - mu_s| <= 1, and the denominator >= floor
            assert value == pytest.approx((target - mean) / max(denominator, DEFAULT_FLOOR), rel=1e-6)

    def test_a_mean_equal_to_the_floor_is_not_floored_and_one_under_it_is(self):
        ring = np.full(RING, 0.5)
        at = measure(ring_image(0.8, ring), floor=0.5)
        assert at["target_surround_contrast_floored"] is False
        assert at["target_surround_contrast"] == pytest.approx(0.3 / 0.5, rel=1e-6)
        under = measure(ring_image(0.8, ring), floor=0.6)
        assert under["target_surround_contrast_floored"] is True
        assert under["target_surround_contrast"] == pytest.approx(0.3 / 0.6, rel=1e-6)

    def test_a_spread_equal_to_the_floor_is_not_floored(self):
        # 0.5 and 0.5 + 2**-7 are exact, so the ring's spread is exactly 2**-8, which is the floor.
        ring = two_valued(0.5, 0.5 + 2**-7)
        at = measure(ring_image(0.8, ring), floor=2**-8)
        assert at["signal_clutter_ratio_floored"] is False
        assert at["signal_clutter_ratio"] == pytest.approx((0.8 - 0.50390625) / 2**-8, rel=1e-6)
        under = measure(ring_image(0.8, ring), floor=2**-8 * 1.01)
        assert under["signal_clutter_ratio_floored"] is True

    def test_a_floor_of_one_bounds_every_ratio_by_one(self):
        row = measure(ring_image(1.0, np.zeros(RING)), floor=1.0)
        assert row["target_surround_contrast"] == pytest.approx(1.0)
        assert row["signal_clutter_ratio"] == pytest.approx(1.0)
        assert (row["target_surround_contrast_floored"], row["signal_clutter_ratio_floored"]) == (True, True)


class TestSupport:
    @staticmethod
    def with_invalid(count: int, value: float = NAN) -> np.ndarray:
        target = np.full((10, 10), 0.6)
        target.flat[:count] = value
        return ring_image(target, two_valued(0.1, 0.3))

    def test_half_a_target_keeps_the_status_and_the_surround_and_drops_the_ratios(self):
        row = measure(self.with_invalid(50))
        assert row["context_status"] == "ok"
        assert row["surround_mean"] == pytest.approx(0.2, abs=1e-12)
        assert row["surround_std"] == pytest.approx(0.1, abs=1e-12)
        assert row["target_valid_fraction"] == 0.5
        assert_ratios_are_nan(row)
        assert row["contrast_reason"] == "insufficient_target_support"
        assert (row["target_surround_contrast_floored"], row["signal_clutter_ratio_floored"]) == (False, False)

    def test_a_fraction_of_exactly_the_minimum_passes_and_just_under_it_does_not(self):
        at, under = measure(self.with_invalid(10)), measure(self.with_invalid(11))
        assert at["target_valid_fraction"] == 0.9
        assert at["target_surround_contrast"] == pytest.approx(2.0, rel=1e-6)
        assert at["contrast_reason"] == ""
        assert math.isnan(under["target_surround_contrast"])
        assert under["contrast_reason"] == "insufficient_target_support"

    def test_a_target_with_no_valid_pixel_is_not_a_measurement(self):
        row = measure(self.with_invalid(100))
        assert row["target_valid_fraction"] == 0.0
        assert_ratios_are_nan(row)
        assert row["contrast_reason"] == "insufficient_target_support"

    def test_a_target_with_no_valid_pixel_is_refused_even_where_the_minimum_is_zero(self):
        # The share is 0.0, which a minimum of 0 accepts, but there is no mean to divide by.
        row = measure(self.with_invalid(100), minimum=0.0)
        assert_ratios_are_nan(row)
        assert row["contrast_reason"] == "insufficient_target_support"

    def test_the_mean_is_over_the_valid_pixels_only(self):
        # 10 pixels outside the interval are not clipped to 1, and not counted: the mean is that of the other 90
        out_of_range = measure(self.with_invalid(10, 7.0))
        assert out_of_range["target_valid_fraction"] == 0.9
        assert out_of_range["target_surround_contrast"] == pytest.approx(2.0, rel=1e-6)

    @pytest.mark.parametrize(
        ("datum", "boxes", "focal", "kw", "status"),
        [
            (ring_image(0.6, 0.2), [(NAN, 0, 5, 5)], 0, {}, "invalid_box"),
            (with_infinity(), None, 0, {"declared": None}, "unknown_range"),
            (ring_image(0.6, 0.2), [(0, 0, 30, 30)], 0, {}, "empty_ring"),
            (ring_image(0.6, 0.2), None, 0, {"min_pixels": 200}, "insufficient_support"),
            (ring_image(0.6, 0.2), None, None, {}, "not_applicable"),
            (ring_image(0.6, 0.2), None, 0, {"bands": (5,), "view": ViewKind.BAND}, "insufficient_support"),
        ],
        ids=["invalid_box", "unknown_range", "empty_ring", "insufficient_support", "not_applicable", "missing_band"],
    )
    def test_a_surround_that_is_not_measured_gives_nan_and_no_reason(self, datum, boxes, focal, kw, status):
        # Nothing wrong with the target is claimed where the surround is the problem. A band group that names a band
        # the datum lacks reads a target fraction of 0.0, which must not be mistaken for one.
        row = measure(datum, boxes, focal, **kw)
        assert row["context_status"] == status
        assert_ratios_are_nan(row)
        assert row["contrast_reason"] == ""
        assert (row["target_surround_contrast_floored"], row["signal_clutter_ratio_floored"]) == (False, False)

    def test_incomplete_geometry_keeps_both_ratios(self):
        datum = ring_image(0.6, two_valued(0.1, 0.3))
        row = measure(datum, [BOX, (29, 29, 26, 26)])  # a reversed peer: it may lie in the ring, and the status says so
        assert row["context_status"] == "incomplete_geometry"
        assert row["target_surround_contrast"] == pytest.approx(2.0, rel=1e-6)
        assert row["signal_clutter_ratio"] == pytest.approx(4.0, rel=1e-6)


class TestDetails:
    def test_details_follow_the_flags_that_read_them(self):
        datum = ring_image(0.6, two_valued(0.1, 0.3))
        for flags, expected in (
            (ImageStats.CONTEXT_SURROUND_MEAN, set()),
            (ImageStats.CONTEXT_SURROUND_STD, set()),
            (CONTRAST, {"contrast_reason", "target_surround_contrast_floored"}),
            (CLUTTER, {"contrast_reason", "signal_clutter_ratio_floored"}),
            (CONTRAST | CLUTTER, set(NEW_DETAILS)),
            (CONTEXT, set(NEW_DETAILS)),
        ):
            assert {name for name in measure(datum, flags=flags) if name in NEW_DETAILS} == expected, flags

    def test_only_the_requested_statistics_are_computed(self):
        row = measure(ring_image(0.6, two_valued(0.1, 0.3)), flags=CLUTTER)
        assert "signal_clutter_ratio" in row
        assert "target_surround_contrast" not in row

    def test_a_row_with_no_measurement_keeps_the_dtypes(self):
        # With no per_image the item's row carries the background statistics alone, so no CONTEXT calculator runs for
        # it and `get_empty_values` supplies its details: without it the padding writes NaN, and a Boolean column
        # becomes an object column.
        image = ring_image(0.6, np.r_[0.1, np.zeros(RING - 1)])  # the contrast is floored on the box's row
        result = image_stats(
            [image],
            boxes=[[BOX]],
            stats=CONTEXT | ImageStats.PIXEL_MEAN,
            per_image=False,
            per_background=True,
            value_range=(0.0, 1.0),
        )
        details = details_of(result)
        assert details["target_surround_contrast_floored"].dtype == bool
        assert details["signal_clutter_ratio_floored"].dtype == bool
        assert details["contrast_reason"].dtype.kind == "U"
        assert details["target_surround_contrast_floored"].tolist().count(True) == 1  # the box's row alone
        assert set(details["contrast_reason"].tolist()) == {""}
        assert len(details["contrast_reason"]) == 2  # the item's background-only row, and the box

    def test_a_request_for_one_ratio_adds_only_its_own_columns(self):
        result = image_stats([ring_image(0.6, two_valued(0.1, 0.3))], boxes=[[BOX]], stats=CONTRAST, per_image=False)
        assert set(result["stats"]) == {"target_surround_contrast"}
        assert {n for n in details_of(result) if n in NEW_DETAILS} == {
            "contrast_reason",
            "target_surround_contrast_floored",
        }


class TestChannelGroups:
    def test_a_group_has_its_own_columns_read_from_its_own_band(self):
        green = ring_image(0.1, two_valued(0.5, 0.7))
        datum = np.concatenate([ring_image(0.6, two_valued(0.1, 0.3)), green, np.zeros_like(green)])
        result = image_stats(
            [datum], boxes=[[BOX]], stats=CONTEXT, per_image=False, channels={"g": 1}, value_range=(0.0, 1.0)
        )
        stats, details = result["stats"], details_of(result)
        assert stats["g_target_surround_contrast"][0] == pytest.approx(-0.5 / 0.6, rel=1e-6)
        assert stats["g_signal_clutter_ratio"][0] == pytest.approx(-5.0, rel=1e-6)
        assert details["g_contrast_reason"].tolist() == [""]
        assert details["g_target_surround_contrast_floored"].tolist() == [False]
        assert details["g_signal_clutter_ratio_floored"].tolist() == [False]
        # the unnamed view reads all three channels, so its values are not the band's
        assert stats["target_surround_contrast"][0] != pytest.approx(stats["g_target_surround_contrast"][0])


class TestNothingElseMoves:
    """A request with neither ratio returns what it did before they existed, and the ratios move no older column."""

    OLD_DETAILS = {
        "context_status",
        "surround_candidate_pixels",
        "surround_retained_pixels",
        "ring_peer_excluded_fraction",
        "surround_measurable_pixels",
        "surround_valid_fraction",
        "target_valid_fraction",
        "context_range_low",
        "context_range_high",
    }
    SURROUND = ImageStats.CONTEXT_SURROUND_MEAN | ImageStats.CONTEXT_SURROUND_STD

    @staticmethod
    def run(stats: ImageStats) -> Any:
        image = np.random.default_rng(1605).random((3, 60, 80))
        boxes = [(10, 10, 40, 40), (30, 20, 60, 50), (5, 5, 20, 18)]
        return image_stats([image], boxes=[boxes], stats=stats, per_image=False)

    def test_the_surround_statistics_alone_return_the_columns_details_and_specs_they_always_did(self):
        result = self.run(ImageStats.CONTEXT_SURROUND_MEAN)
        assert set(result["stats"]) == {"surround_mean"}
        assert set(result["details"]) == self.OLD_DETAILS
        assert set(result["measurement_specs"]) == {"surround_mean"}
        assert dict(result["measurement_specs"]["surround_mean"].config) == {"padding": 0.25, "min_pixels": 32}

    @pytest.mark.parametrize("other", [CONTEXT, CONTRAST, ImageStats.ALL], ids=["context", "contrast", "all"])
    def test_every_older_column_is_equal_whether_or_not_the_ratios_are_requested(self, other):
        plain, more = self.run(self.SURROUND), self.run(other | self.SURROUND)
        assert more["stats"]["surround_mean"].dtype == plain["stats"]["surround_mean"].dtype
        for group, names in (("stats", {"surround_mean", "surround_std"}), ("details", self.OLD_DETAILS)):
            for name in names:
                np.testing.assert_array_equal(plain[group][name], more[group][name], err_msg=f"{group} {name}")


class TestTheFloorKeyword:
    @staticmethod
    def run(floor: float | None = None, stats: ImageStats = CONTEXT) -> Any:
        image = ring_image(0.8, np.full(RING, 0.5))  # a constant ring, so the spread is 0 and the floor divides
        floor = DEFAULT_FLOOR if floor is None else floor
        return image_stats(
            [image], boxes=[[BOX]], stats=stats, per_image=False, value_range=(0.0, 1.0), surround_intensity_floor=floor
        )

    def test_the_floor_reaches_the_ratio_and_its_flag(self):
        default, wide = self.run(), self.run(0.5)
        assert default["stats"]["signal_clutter_ratio"][0] == pytest.approx(0.3 * 255, rel=1e-6)
        assert wide["stats"]["signal_clutter_ratio"][0] == pytest.approx(0.3 / 0.5, rel=1e-6)
        assert default["details"]["signal_clutter_ratio_floored"].tolist() == [True]
        assert wide["details"]["signal_clutter_ratio_floored"].tolist() == [True]

    def test_the_floor_is_recorded_with_the_ratios_and_not_the_surround_statistics(self):
        specs = self.run(0.5)["measurement_specs"]
        assert specs["target_surround_contrast"].config["intensity_floor"] == 0.5
        assert specs["signal_clutter_ratio"].config["intensity_floor"] == 0.5
        assert "intensity_floor" not in specs["surround_mean"].config

    def test_the_floor_changes_no_surround_column(self):
        default, wide = self.run(), self.run(0.5)
        for name in ("surround_mean", "surround_std"):
            np.testing.assert_array_equal(default["stats"][name], wide["stats"][name], err_msg=name)


class TestSpecs:
    @staticmethod
    def specs(stats: ImageStats = CONTEXT, **kw: Any) -> dict[str, MeasurementSpec]:
        image = ring_image(0.6, two_valued(0.1, 0.3))
        result = image_stats([image], boxes=[[BOX]], stats=stats, per_image=False, **kw)
        return dict(result.get("measurement_specs", {}))

    def test_the_four_statistics_record_their_specs(self):
        assert set(self.specs()) == {"surround_mean", "surround_std", *RATIOS}

    def test_the_surround_spec_is_the_one_main_recorded(self):
        old = MeasurementSpec(
            algorithm="surround_intensity",
            version="1",
            units="normalized intensity",
            region="surround",
            normalization="[0, 1]",
            config={"padding": 0.25, "min_pixels": 32},
            range_policy="inferred",
            declared_range=None,
        )
        specs = self.specs()
        assert specs["surround_mean"] == specs["surround_std"] == old

    @pytest.mark.parametrize("name", RATIOS)
    def test_a_ratio_records_the_settings_that_decide_its_value(self, name):
        spec = self.specs()[name]
        assert (spec.algorithm, spec.version, spec.units) == (name, "1", "ratio")
        assert (spec.region, spec.normalization) == ("target and surround", "[0, 1]")
        assert dict(spec.config) == {
            "padding": 0.25,
            "min_pixels": 32,
            "min_target_valid_fraction": 0.9,
            "intensity_floor": DEFAULT_FLOOR,
        }
        assert (spec.range_policy, spec.declared_range) == ("inferred", None)
        assert MeasurementSpec(**json.loads(json.dumps(spec.to_dict()))) == spec

    def test_a_declared_range_is_recorded_on_all_four(self):
        specs = self.specs(value_range=(0.0, 1.0))
        assert {s.range_policy for s in specs.values()} == {"declared"}
        assert {s.declared_range for s in specs.values()} == {(0.0, 1.0)}

    def test_a_request_records_only_the_specs_of_the_statistics_it_holds(self):
        assert set(self.specs(ImageStats.CONTEXT_SURROUND_MEAN)) == {"surround_mean"}
        assert set(self.specs(CONTRAST)) == {"target_surround_contrast"}
        assert set(self.specs(CLUTTER)) == {"signal_clutter_ratio"}

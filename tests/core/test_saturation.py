"""ImageStats.PIXEL_SATURATION: the share of finite raw values sitting at a stated clipping endpoint."""

import tracemalloc
import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.core._calculators._base import ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._pixelstats import PixelStatCalculator
from dataeval.flags import ImageStats
from dataeval.types._stats_config import ImageStatsConfig, SaturationConfig
from dataeval.utils.preprocessing import BoundingBox, ChannelGroup, get_value_range

# Every single-bit member of ImageStats as of v1.3's SPATIAL flags, by name. Flags are persisted as integers, so a new
# member is appended and never inserted: none of these may move.
_EXISTING_VALUES = {
    "PIXEL_MEAN": 1, "PIXEL_STD": 2, "PIXEL_VAR": 4, "PIXEL_SKEW": 8, "PIXEL_KURTOSIS": 16, "PIXEL_ENTROPY": 32,
    "PIXEL_MISSING": 64, "PIXEL_ZEROS": 128, "PIXEL_HISTOGRAM": 256, "VISUAL_BRIGHTNESS": 512,
    "VISUAL_CONTRAST": 1024, "VISUAL_DARKNESS": 2048, "VISUAL_SHARPNESS": 4096, "VISUAL_PERCENTILES": 8192,
    "DIMENSION_OFFSET_X": 16384, "DIMENSION_OFFSET_Y": 32768, "DIMENSION_WIDTH": 65536, "DIMENSION_HEIGHT": 131072,
    "DIMENSION_CHANNELS": 262144, "DIMENSION_SIZE": 524288, "DIMENSION_ASPECT_RATIO": 1048576,
    "DIMENSION_DEPTH": 2097152, "DIMENSION_CENTER": 4194304, "DIMENSION_DISTANCE_CENTER": 8388608,
    "DIMENSION_DISTANCE_EDGE": 16777216, "DIMENSION_INVALID_BOX": 33554432, "DIMENSION_REL_OFFSET_X": 67108864,
    "DIMENSION_REL_OFFSET_Y": 134217728, "DIMENSION_REL_WIDTH": 268435456, "DIMENSION_REL_HEIGHT": 536870912,
    "DIMENSION_REL_SIZE": 1073741824, "DIMENSION_REL_DISTANCE_CENTER": 2147483648,
    "DIMENSION_REL_DISTANCE_EDGE": 4294967296, "HASH_XXHASH": 8589934592, "HASH_PHASH": 17179869184,
    "HASH_PHASH_D4": 34359738368, "HASH_DHASH": 68719476736, "HASH_DHASH_D4": 137438953472,
    "VISUAL_BLUR": 274877906944, "VISUAL_BLUR_DIRECTIONALITY": 549755813888,
    "SPATIAL_NN_DISTANCE_NORM": 1099511627776, "SPATIAL_BOX_IN_IMAGE_FRACTION": 2199023255552,
    "SPATIAL_COVERED_FRACTION": 4398046511104, "SPATIAL_MAX_IOU_SAME_CLASS": 8796093022208,
    "SPATIAL_MAX_IOU_OTHER_CLASS": 17592186044416,
}  # fmt: skip


@pytest.mark.required
class TestFlag:
    """The flag is one bit, in `PIXEL` and so in everything made of it, and named the way every consumer expects."""

    def test_the_issue_spelling_is_an_alias_of_the_family_named_bit(self):
        assert ImageStats.PIXEL_SATURATION is ImageStats.PIXEL_SATURATION
        assert ImageStats.PIXEL_SATURATION.name == "PIXEL_SATURATION"

    def test_no_existing_flag_moved_and_the_new_one_is_appended(self):
        # The CONTEXT flags were appended after it by a later change, and have a test of their own.
        members = {
            m.name: m.value
            for m in ImageStats
            if m.name != "PIXEL_SATURATION" and not (m.name or "").startswith("CONTEXT_")
        }
        assert members == _EXISTING_VALUES
        assert ImageStats.PIXEL_SATURATION.value == 2 * max(_EXISTING_VALUES.values())

    def test_all_holds_it_and_the_complement_of_all_is_none(self):
        assert ImageStats.PIXEL_SATURATION in ImageStats.ALL
        assert ~ImageStats.ALL == ImageStats.NONE

    @pytest.mark.parametrize("group", ["PIXEL", "STANDARD", "NO_HASH", "ALL"])
    def test_the_pixel_family_and_what_is_made_of_it_hold_it(self, group: str):
        assert ImageStats.PIXEL_SATURATION in getattr(ImageStats, group)

    @pytest.mark.parametrize(
        "group", ["PIXEL_BASIC", "PIXEL_DISTRIBUTION", "VISUAL", "BASIC", "DIMENSION", "HASH", "BLUR", "SPATIAL"]
    )
    def test_the_other_families_and_the_curated_subsets_do_not(self, group: str):
        assert not ImageStats.PIXEL_SATURATION & getattr(ImageStats, group)

    def test_the_default_request_holds_it(self):
        assert ImageStats.PIXEL_SATURATION in ImageStats.STANDARD | ImageStats.HASH

    def test_the_v12_column_set_is_the_default_request_without_it(self):
        v12 = (
            ImageStats.ALL & ~ImageStats.BLUR & ~ImageStats.SPATIAL & ~ImageStats.PIXEL_SATURATION & ~ImageStats.CONTEXT
        )
        assert v12 == (ImageStats.STANDARD | ImageStats.HASH) & ~ImageStats.PIXEL_SATURATION

    def test_the_name_splits_into_a_family_and_the_calculators_statistic_name(self):
        # `quality._shared._stat_names` and dataeval-flow's `stat_names_for` both read a flag as <FAMILY>_<STATISTIC>.
        names = [(m.name or "").split("_", 1)[1].lower() for m in ImageStats.PIXEL_SATURATION]
        assert names == ["saturation"]


DETAILS = {
    "saturation_status",
    "saturation_valid_fraction",
    "saturation_above_range_fraction",
    "saturation_upper",
    "saturation_upper_rule",
}


def measure(
    datum: np.ndarray,
    *,
    upper: float | None = None,
    atol: float = 0.0,
    declared: tuple[float, float] | None = None,
    box: BoundingBox | None = None,
    exclude: np.ndarray | None = None,
    bands: tuple[int, ...] | None = None,
    view: ViewKind = ViewKind.WHOLE,
) -> dict[str, Any]:
    """One row's SATURATION reading and details, from the calculator itself, as one flat mapping.

    `upper` is shorthand for declaring the range ``(0, upper)``, which is the only way an endpoint is stated.
    """
    settings = ImageStatsConfig(saturation=SaturationConfig(atol=atol))
    declared = (0.0, upper) if upper is not None else declared
    value_range = None if declared is None else get_value_range(datum, declared=declared)
    cache = CalculatorCache(datum, box, exclude=exclude, value_range=value_range, bands=bands, settings=settings)
    calculator = PixelStatCalculator(datum, cache)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = calculator.compute(ImageStats.PIXEL_SATURATION, view)
        details = calculator.details(ImageStats.PIXEL_SATURATION, view)
    assert set(stats) == {"saturation"}
    assert set(details) == DETAILS
    return {name: values[0] for name, values in {**stats, **details}.items()}


def box_at(shape: tuple[int, ...], x0: int, y0: int, x1: int, y1: int) -> BoundingBox:
    return BoundingBox(x0, y0, x1, y1, image_shape=shape)


@pytest.mark.required
class TestEndpointRule:
    """The endpoint comes from a declared range or the dtype, in that order, and never from the data."""

    def test_the_upper_bound_of_a_declared_range_is_the_endpoint_wherever_the_data_sits(self):
        row = measure(np.full((1, 4, 4), 200, dtype=np.uint8), declared=(0, 200))
        assert (row["saturation"], row["saturation_upper"], row["saturation_upper_rule"]) == (1.0, 200.0, "declared")

    def test_a_declared_range_wins_over_the_dtype(self):
        row = measure(np.full((1, 4, 4), 4095, dtype=np.uint16), declared=(0, 4095))
        assert (row["saturation"], row["saturation_upper"], row["saturation_upper_rule"]) == (1.0, 4095.0, "declared")

    def test_the_dtype_maximum_is_the_endpoint_when_nothing_else_names_one(self):
        row = measure(np.full((1, 4, 4), 255, dtype=np.uint8))
        assert (row["saturation"], row["saturation_upper"], row["saturation_upper_rule"]) == (1.0, 255.0, "dtype")

    def test_twelve_bit_data_in_uint16_measures_the_storage_rail_not_the_sensors_maximum(self):
        # The values reach 4095 and `get_value_range` decodes that as 12 bits, but the endpoint is not read off them.
        data = np.full((1, 4, 4), 4095, dtype=np.uint16)
        assert get_value_range(data).pmax == 4095.0
        row = measure(data)
        assert (row["saturation"], row["saturation_upper"], row["saturation_upper_rule"]) == (0.0, 65535.0, "dtype")
        assert measure(data, upper=4095)["saturation"] == 1.0

    @pytest.mark.parametrize(
        ("dtype", "endpoint"),
        [(np.uint8, 255.0), (np.uint16, 65535.0), (np.uint32, 4294967295.0), (np.bool_, 1.0)],
    )
    def test_an_unsigned_integer_of_32_bits_or_fewer_and_a_bool_have_a_dtype_endpoint(
        self, dtype: Any, endpoint: float
    ):
        top = np.iinfo(dtype).max if dtype is not np.bool_ else True
        row = measure(np.full((1, 3, 3), top, dtype=dtype))
        assert (row["saturation"], row["saturation_upper"], row["saturation_upper_rule"]) == (1.0, endpoint, "dtype")

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int64, np.uint64, np.float16, np.float32, np.float64])
    def test_a_signed_integer_a_uint64_and_a_float_have_no_dtype_endpoint(self, dtype: Any):
        row = measure(np.full((1, 3, 3), 1, dtype=dtype))
        assert row["saturation_status"] == "unknown_upper"
        assert np.isnan(row["saturation"])
        assert np.isnan(row["saturation_upper"])
        assert row["saturation_upper_rule"] == ""
        assert row["saturation_valid_fraction"] == 1.0
        assert np.isnan(row["saturation_above_range_fraction"])

    def test_floats_in_zero_to_one_need_a_stated_endpoint(self):
        data = np.array([[[0.0, 0.5], [1.0, 1.0]]], dtype=np.float32)
        assert measure(data)["saturation_status"] == "unknown_upper"
        assert measure(data, upper=1)["saturation"] == 0.5
        assert measure(data, declared=(0, 1))["saturation"] == 0.5

    def test_uint8_is_exact_at_255_and_cannot_wrap(self):
        data = np.array([[[3, 5, 254, 255]]], dtype=np.uint8)
        assert measure(data)["saturation"] == 0.25
        assert measure(data, atol=1)["saturation"] == 0.5


@pytest.mark.required
class TestCounting:
    """Channel values at the endpoint, over the finite ones; values above it are kept and reported, not counted."""

    def test_one_clipped_channel_of_three_reads_one_third(self):
        data = np.zeros((3, 4, 4), dtype=np.uint8)
        data[0] = 255
        assert measure(data)["saturation"] == pytest.approx(1 / 3)

    def test_nan_is_in_neither_count_and_the_valid_fraction_shows_it(self):
        data = np.array([[1.0, np.nan], [0.5, 1.0]], dtype=np.float32)
        row = measure(data, declared=(0, 1))
        assert row["saturation"] == pytest.approx(2 / 3)
        assert row["saturation_valid_fraction"] == 0.75

    def test_a_value_above_the_range_stays_in_the_denominator_and_is_reported(self):
        data = np.array([[1.0, 1.5, 0.5], [2.0, 1.0, 0.0]], dtype=np.float64)
        row = measure(data, declared=(0, 1))
        assert row["saturation"] == pytest.approx(2 / 6)
        assert row["saturation_above_range_fraction"] == pytest.approx(2 / 6)

    def test_an_infinity_is_not_finite_so_it_is_neither_counted_nor_above_the_range(self):
        data = np.array([[1.0, np.inf, np.nan, 0.0]], dtype=np.float64)
        row = measure(data, declared=(0, 1))
        assert row["saturation"] == 0.5
        assert row["saturation_valid_fraction"] == 0.5
        assert row["saturation_above_range_fraction"] == 0.0

    def test_atol_widens_the_band_on_integers_and_on_floats(self):
        ints = np.array([[250, 253, 255, 255, 100]], dtype=np.uint8)
        assert measure(ints)["saturation"] == pytest.approx(2 / 5)
        assert measure(ints, atol=2)["saturation"] == pytest.approx(3 / 5)
        floats = np.array([[0.97, 0.99, 1.0, 0.5]])
        assert measure(floats, upper=1.0)["saturation"] == pytest.approx(1 / 4)
        assert measure(floats, upper=1.0, atol=0.05)["saturation"] == pytest.approx(3 / 4)

    def test_values_above_the_band_count_as_above_range_even_with_a_tolerance(self):
        row = measure(np.array([[255, 255, 254]], dtype=np.uint8), upper=254, atol=0)
        assert row["saturation"] == pytest.approx(1 / 3)
        assert row["saturation_above_range_fraction"] == pytest.approx(2 / 3)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_a_float_endpoint_that_the_dtype_cannot_hold_counts_the_same_on_a_2d_and_a_3d_datum(self, dtype: Any):
        # A 3-D datum reaches the calculator promoted to float64 and a 2-D one in its own dtype, so an endpoint
        # compared as given would match one and miss the other.
        value = dtype(0.9)
        flat = np.full((4, 4), value, dtype=dtype)
        cube = np.full((1, 4, 4), value, dtype=dtype)
        assert measure(flat, upper=0.9)["saturation"] == 1.0
        assert measure(cube, upper=0.9)["saturation"] == 1.0

    def test_a_float32_view_and_its_float64_promotion_agree_at_the_edge_of_the_band(self):
        values = np.array([0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float32)
        flat = values.reshape(1, 5)
        cube = values.reshape(1, 1, 5)
        for atol in (0.0, 0.05, 0.1):
            assert (
                measure(flat, upper=0.9, atol=atol)["saturation"] == measure(cube, upper=0.9, atol=atol)["saturation"]
            )

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.uint8, np.uint16])
    def test_a_very_large_tolerance_counts_every_finite_value_and_does_not_warn(self, dtype: Any):
        data = np.array([[[1, 2, 3]]], dtype=dtype)
        assert measure(data, upper=2, atol=1e300)["saturation"] == 1.0  # `measure` turns any warning into an error

    def test_a_large_plain_integer_window_is_counted_without_a_float_copy(self):
        data = np.random.default_rng(0).integers(0, 4096, size=(3, 1000, 1000), dtype=np.uint16)
        tracemalloc.start()
        try:
            measure(data, upper=4095)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # A float64 copy is 8x the 6 MB array, and `abs(x - upper) <= atol` peaks at 8x as well.
        assert peak < 5 * data.nbytes


@pytest.mark.required
class TestStatus:
    """Each status, and the values every detail takes under it."""

    def test_a_view_with_no_finite_value_is_no_finite_values_but_still_names_its_endpoint(self):
        row = measure(np.full((1, 3, 3), np.nan), declared=(0, 1))
        assert row["saturation_status"] == "no_finite_values"
        assert np.isnan(row["saturation"])
        assert np.isnan(row["saturation_above_range_fraction"])
        assert row["saturation_valid_fraction"] == 0.0
        assert (row["saturation_upper"], row["saturation_upper_rule"]) == (1.0, "declared")

    def test_a_view_of_only_infinities_has_no_finite_values_either(self):
        row = measure(np.full((1, 3, 3), np.inf), upper=1)
        assert row["saturation_status"] == "no_finite_values"
        assert row["saturation_valid_fraction"] == 0.0

    def test_a_box_wholly_outside_the_image_is_an_empty_region_with_none_of_it_valid(self):
        data = np.zeros((3, 10, 12), dtype=np.uint8)
        row = measure(data, box=box_at(data.shape, 50, 50, 60, 60))
        assert row["saturation_status"] == "empty_region"
        assert np.isnan(row["saturation"])
        assert row["saturation_valid_fraction"] == 0.0
        assert (row["saturation_upper"], row["saturation_upper_rule"]) == (255.0, "dtype")

    def test_a_zero_area_box_is_an_empty_region(self):
        data = np.zeros((3, 10, 12), dtype=np.uint8)
        assert measure(data, box=box_at(data.shape, 4, 4, 4, 4))["saturation_status"] == "empty_region"

    def test_a_window_the_exclusion_covers_entirely_is_an_empty_region_with_no_valid_fraction(self):
        data = np.zeros((3, 10, 12), dtype=np.uint8)
        row = measure(data, exclude=np.ones((10, 12), dtype=bool))
        assert row["saturation_status"] == "empty_region"
        assert np.isnan(row["saturation_valid_fraction"])

    def test_a_band_group_the_datum_cannot_supply_is_an_empty_region(self):
        data = np.zeros((3, 10, 12), dtype=np.uint16)
        row = measure(data, bands=(5,), view=ViewKind.BAND)
        assert row["saturation_status"] == "empty_region"
        assert row["saturation_valid_fraction"] == 0.0

    def test_an_empty_region_outranks_an_unknown_endpoint(self):
        data = np.zeros((3, 10, 12), dtype=np.float32)
        row = measure(data, box=box_at(data.shape, 50, 50, 60, 60))
        assert row["saturation_status"] == "empty_region"
        assert np.isnan(row["saturation_upper"])
        assert row["saturation_upper_rule"] == ""

    def test_no_finite_values_outranks_an_unknown_endpoint(self):
        assert measure(np.full((1, 3, 3), np.nan))["saturation_status"] == "no_finite_values"

    def test_an_inverted_box_reads_the_window_it_describes(self):
        data = np.full((3, 10, 12), 255, dtype=np.uint8)
        row = measure(data, box=box_at(data.shape, 8, 8, 2, 2))
        assert (row["saturation_status"], row["saturation"]) == ("ok", 1.0)

    def test_an_excluded_pixel_is_in_neither_half_and_the_valid_fraction_is_over_the_rest(self):
        data = np.full((1, 4, 4), 255, dtype=np.uint8)
        data[:, 2:] = 0
        mask = np.zeros((4, 4), dtype=bool)
        mask[2:] = True
        row = measure(data, exclude=mask)
        assert (row["saturation"], row["saturation_valid_fraction"]) == (1.0, 1.0)

    def test_the_valid_fraction_counts_nan_infinity_and_fill_against_the_positions_of_the_view(self):
        data = np.array([[[1.0, np.nan, np.inf, 0.5]]], dtype=np.float64)
        row = measure(data, upper=1)
        assert row["saturation_valid_fraction"] == 0.5
        assert row["saturation"] == 0.5

    @pytest.mark.parametrize(
        "shape",
        [(), (7,), (10, 12), (2, 3, 10, 12), (3, 0, 0), (0,)],
        ids=["0-d", "1-D", "2-D", "4-D", "zero-size 3-D", "empty 1-D"],
    )
    def test_degenerate_shapes_answer_without_raising(self, shape: tuple[int, ...]):
        row = measure(np.zeros(shape, dtype=np.uint8))
        assert row["saturation_status"] in {"ok", "empty_region"}


def run(images: list[np.ndarray], **kw: Any) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = image_stats(images, stats=kw.pop("stats", ImageStats.PIXEL_SATURATION), **kw)
    return dict(result["stats"]), dict(result.get("details", {})), list(result["source_index"])


@pytest.mark.required
class TestThroughImageStats:
    """The statistic in the views `image_stats` measures: the image, a box, the background and a band group."""

    def test_an_image_and_its_boxes_each_get_a_row(self):
        image = np.zeros((3, 10, 10), dtype=np.uint8)
        image[:, :5, :5] = 255
        stats, details, index = run([image], boxes=[[(0, 0, 5, 5), (5, 5, 10, 10)]])
        assert [s.target for s in index] == [None, 0, 1]
        assert stats["saturation"].tolist() == pytest.approx([0.25, 1.0, 0.0])
        assert details["saturation_status"].tolist() == ["ok", "ok", "ok"]
        assert set(details) == DETAILS

    def test_the_background_excludes_the_boxes_and_a_box_row_has_no_background(self):
        image = np.zeros((3, 10, 10), dtype=np.uint8)
        image[:, :5, :5] = 255
        stats, details, index = run([image], boxes=[[(0, 0, 5, 5)]], per_background=True)
        assert [s.target for s in index] == [None, 0]
        assert stats["background_saturation"][0] == 0.0  # the clipped box is outside the background
        assert np.isnan(stats["background_saturation"][1])
        assert details["background_saturation_status"].tolist() == ["ok", ""]
        assert details["background_saturation_upper_rule"].tolist() == ["dtype", ""]
        assert np.isnan(details["background_saturation_upper"][1])

    def test_a_band_group_has_its_own_range_prefixed_columns_and_details(self):
        image = np.zeros((4, 8, 8), dtype=np.uint16)
        image[3] = 4095
        stats, details, _ = run(
            [image], value_range=(0, 65535), channels={"nir": ChannelGroup(3, value_range=(0, 4095))}
        )
        assert stats["saturation"][0] == 0.0  # against the call's 65535
        assert stats["nir_saturation"][0] == 1.0  # against the group's own 4095
        assert details["saturation_upper"].tolist() == [65535.0]
        assert details["nir_saturation_upper"].tolist() == [4095.0]
        assert details["nir_saturation_upper_rule"].tolist() == ["declared"]
        assert details["nir_saturation_status"].tolist() == ["ok"]

    def test_a_band_group_the_image_cannot_supply_reports_an_empty_region_beside_a_measured_image(self):
        image = np.full((3, 8, 8), 255, dtype=np.uint8)
        stats, details, _ = run([image], channels={"nir": 5})
        assert stats["saturation"][0] == 1.0
        assert details["nir_saturation_status"].tolist() == ["empty_region"]

    def test_normalize_pixel_values_changes_nothing(self):
        image = np.random.default_rng(0).integers(0, 256, size=(3, 8, 8), dtype=np.uint8)
        image[0, :4] = 255
        plain = run([image])
        normalized = run([image], normalize_pixel_values=True)
        np.testing.assert_array_equal(plain[0]["saturation"], normalized[0]["saturation"])

    def test_a_request_for_a_pixel_statistic_alone_returns_no_details(self):
        _, details, _ = run([np.zeros((3, 4, 4), dtype=np.uint8)], stats=ImageStats.PIXEL_MEAN)
        assert details == {}

    def test_a_request_for_pixel_returns_the_column_and_its_details(self):
        stats, details, _ = run([np.zeros((3, 4, 4), dtype=np.uint8)], stats=ImageStats.PIXEL)
        assert "saturation" in stats
        assert set(details) >= DETAILS

    def test_a_request_for_all_returns_the_column_and_its_details(self):
        image = np.random.default_rng(1).integers(0, 256, size=(3, 16, 16), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = image_stats([image], stats=ImageStats.ALL)
        assert "saturation" in result["stats"]
        assert set(result.get("details", {})) >= DETAILS

    def test_the_default_request_holds_the_column_and_its_details(self):
        result = image_stats([np.zeros((3, 4, 4), dtype=np.uint8)])
        assert "saturation" in result["stats"]
        assert set(result.get("details", {})) >= DETAILS


@pytest.mark.required
class TestKeywords:
    """`saturation_atol` is a plain keyword, checked before any image is read. The endpoint is not a keyword."""

    def test_the_twelve_bit_remedy_gives_the_analytic_fraction(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 4096, size=(3, 40, 40), dtype=np.uint16)
        image[:, :4] = 4095
        expected = float(np.mean(image == 4095))
        assert expected > 0.0
        default, details, _ = run([image])
        assert default["saturation"][0] == 0.0
        assert details["saturation_upper_rule"].tolist() == ["dtype"]
        fixed, details, _ = run([image], value_range=(0, 4095))
        assert fixed["saturation"][0] == pytest.approx(expected)
        assert (details["saturation_upper"].tolist(), details["saturation_upper_rule"].tolist()) == (
            [4095.0],
            ["declared"],
        )

    def test_the_endpoint_is_not_a_keyword(self):
        with pytest.raises(TypeError, match="saturation_upper"):
            image_stats([np.zeros((3, 4, 4), dtype=np.uint8)], stats=ImageStats.PIXEL_SATURATION, saturation_upper=255)  # pyright: ignore[reportCallIssue]

    def test_the_tolerance_reaches_the_statistic(self):
        image = np.array([[[250, 253, 255, 255, 100]]], dtype=np.uint8)
        assert run([image], saturation_atol=2)[0]["saturation"][0] == pytest.approx(3 / 5)

    def test_one_endpoint_serves_every_view(self):
        image = np.full((4, 10, 10), 4095, dtype=np.uint16)
        stats, details, _ = run(
            [image], boxes=[[(0, 0, 5, 5)]], per_background=True, channels={"nir": 3}, value_range=(0, 4095)
        )
        assert stats["saturation"].tolist() == [1.0, 1.0]
        assert stats["nir_saturation"].tolist() == [1.0, 1.0]
        assert stats["background_saturation"][0] == 1.0
        for column in ("saturation", "nir_saturation", "background_saturation"):
            assert details[f"{column}_upper_rule"][0] == "declared"

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("saturation_atol", -1.0),
            ("saturation_atol", True),
            ("saturation_atol", None),
            ("saturation_atol", float("nan")),
            ("saturation_atol", float("inf")),
        ],
    )
    @pytest.mark.parametrize("stats", [ImageStats.PIXEL_SATURATION, ImageStats.PIXEL_MEAN])
    def test_a_bad_value_raises_naming_the_keyword_whether_or_not_the_flag_is_requested(
        self, name: str, value: Any, stats: ImageStats
    ):
        with pytest.raises(ValueError, match=name):
            image_stats([np.zeros((3, 4, 4), dtype=np.uint8)], stats=stats, **{name: value})

    def test_a_numpy_scalar_is_accepted(self):
        image = np.full((1, 4, 4), 4095, dtype=np.uint16)
        stats, _, _ = run([image], value_range=(0, 4095), saturation_atol=np.float32(0.5))
        assert stats["saturation"][0] == 1.0

    def test_a_setting_without_the_flag_warns_once_and_names_the_flag(self):
        with pytest.warns(UserWarning, match=r"ImageStats\.PIXEL_SATURATION") as caught:
            image_stats([np.zeros((3, 4, 4), dtype=np.uint8)], stats=ImageStats.PIXEL_MEAN, saturation_atol=0.5)
        assert len([w for w in caught if "SATURATION" in str(w.message)]) == 1

    def test_an_explicit_default_tolerance_does_not_warn(self):
        run(
            [np.zeros((3, 4, 4), dtype=np.uint8)], stats=ImageStats.PIXEL_MEAN, saturation_atol=0.0
        )  # `run` errors on a warning

    def test_a_group_that_alone_asks_for_the_flag_uses_the_setting_and_does_not_warn(self):
        image = np.full((4, 6, 6), 4095, dtype=np.uint16)
        stats, _, _ = run(
            [image],
            stats={None: ImageStats.PIXEL_MEAN, "nir": ImageStats.PIXEL_SATURATION},
            channels={"nir": 3},
            value_range=(0, 4095),
        )
        assert stats["nir_saturation"][0] == 1.0
        assert "saturation" not in stats

    def test_results_are_the_same_in_parallel_and_the_input_order_permutes_the_rows(self):
        from dataeval.config import use_max_processes

        rng = np.random.default_rng(5)
        images = [rng.integers(0, 4096, size=(3, 12, 12), dtype=np.uint16) for _ in range(6)]
        for k, image in enumerate(images):
            image[:, :k] = 4095
        with use_max_processes(1):
            serial, _, _ = run(images, value_range=(0, 4095))
        with use_max_processes(2):
            parallel, _, _ = run(images, value_range=(0, 4095))
        np.testing.assert_array_equal(serial["saturation"], parallel["saturation"])
        perm = [3, 5, 0, 4, 1, 2]
        shuffled, _, _ = run([images[p] for p in perm], value_range=(0, 4095))
        np.testing.assert_array_equal(shuffled["saturation"], serial["saturation"][perm])

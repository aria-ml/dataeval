"""End-to-end import of real ``compute_stats`` output into ``Metadata``.

Every test here starts from an actual :func:`~dataeval.core.compute_stats` call on a mock
dataset and finishes by checking where the values landed. That combination is the point:
unit tests that hand-build a factor array agree with whatever convention the code under
test currently implements, so they keep passing when the convention and the producer drift
apart. Only real output pins the two together.

The matrix is (IC, OD) x (per_image, per_target, both) x (add_factors, from_factors).
"""

from typing import Any

import numpy as np
import pytest

from dataeval import Metadata
from dataeval._metadata import _is_stats_result
from dataeval.core import compute_stats
from dataeval.flags import ImageStats
from dataeval.types import SourceIndex
from tests.conftest import _get_mock_ic_dataset, _get_mock_od_dataset

# 2/1/3/0/2 detections: uneven counts, and an item whose target is empty, which contributes
# an image row but no instance rows.
OD_LABELS = [[0, 1], [1], [0, 1, 0], [], [1, 2]]
IC_LABELS = [0, 1, 0, 2, 1]

SCALAR_STATS = ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS


def _images(count: int) -> list[np.ndarray]:
    """Random pixels, so no two crops share a mean by accident.

    Uniform images make every row-alignment bug invisible: a misplaced value equals the
    value it displaced, and the assertions pass on a permuted column.
    """
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (3, 24, 24), dtype=np.uint8) for _ in range(count)]


def _boxes(labels: list[list[int]]) -> list[list[list[int]]]:
    """A distinct crop per detection, so sibling detections have distinct statistics too."""
    return [[[j, j, 10 + j, 12 + j] for j, _ in enumerate(image_labels)] for image_labels in labels]


def _od_dataset(labels: list[list[int]], boxes: list[list[list[int]]] | None = None) -> Any:
    """Build a mock object detection dataset, typed as loosely as the fixtures are."""
    return _get_mock_od_dataset(_images(len(labels)), labels, _boxes(labels) if boxes is None else boxes)


@pytest.fixture
def od_dataset():
    return _od_dataset(OD_LABELS)


@pytest.fixture
def ic_dataset():
    return _get_mock_ic_dataset(_images(len(IC_LABELS)), IC_LABELS)


def _stats(dataset, **kwargs):
    return compute_stats(dataset, stats=SCALAR_STATS, normalize_pixel_values=False, **kwargs)


def _distinct(values) -> bool:
    """Whether every value differs, so a permutation cannot pass unnoticed."""
    values = np.asarray(values, dtype=np.float64)
    return len(np.unique(np.round(values, 9))) == len(values)


def _expected_by_label(result) -> dict[SourceIndex, float]:
    """Map each source-index label to the mean it describes."""
    return dict(zip(result["source_index"], result["stats"]["mean"], strict=True))


def _actual_by_label(md: Metadata, column: str) -> dict[SourceIndex, float]:
    """Map each row's (item, target) identity to the value stored in ``column``."""
    frame = md.dataframe.select("level", "item_index", "target_index", column).drop_nulls(column)
    return {
        SourceIndex(int(item), None if level == md._item_level else int(target)): value
        for level, item, target, value in frame.rows()
    }


class TestAddFactorsFromComputeStats:
    """``Metadata(dataset).add_factors(...)`` over real stats output."""

    @pytest.mark.parametrize("kwargs", [{"per_target": False}, {"per_image": False}, {}])
    def test_source_index_places_every_value_on_the_row_it_describes(self, od_dataset, kwargs):
        result = _stats(od_dataset, **kwargs)
        assert _distinct(result["stats"]["mean"])

        md = Metadata(od_dataset)
        md.add_factors(dict(result["stats"]), source_index=result["source_index"])

        # A two-level array splits into one factor per level; a single-level one keeps its name.
        columns = [c for c in md.dataframe.columns if c.endswith("mean")]
        stored: dict[SourceIndex, float] = {}
        for column in columns:
            stored |= _actual_by_label(md, column)

        for label, expected in _expected_by_label(result).items():
            assert stored[label] == pytest.approx(expected), f"{label} landed on the wrong row"

    def test_inferred_placement_matches_the_source_index(self, od_dataset):
        """The default call must agree with the explicit one, value for value.

        ``add_factors(stats["stats"])`` is what the docs show and what v1.1 accepted. It
        infers placement from array lengths alone, so it is exactly the call that a change
        to the ordering convention breaks silently.
        """
        result = _stats(od_dataset)

        inferred = Metadata(od_dataset)
        with pytest.deprecated_call():
            inferred.add_factors(dict(result["stats"]))

        explicit = Metadata(od_dataset)
        explicit.add_factors(dict(result["stats"]), source_index=result["source_index"])

        assert sorted(inferred.factor_names) == sorted(explicit.factor_names)
        for column in (c for c in explicit.dataframe.columns if c.endswith("mean")):
            assert _actual_by_label(inferred, column) == _actual_by_label(explicit, column)

    def test_combined_spelling_matches_the_source_index(self, od_dataset):
        """The retired ``level="combined"`` must place the same data as its replacement."""
        result = _stats(od_dataset)

        deprecated = Metadata(od_dataset)
        with pytest.deprecated_call():
            deprecated.add_factors(dict(result["stats"]), level="combined")

        explicit = Metadata(od_dataset)
        explicit.add_factors(dict(result["stats"]), source_index=result["source_index"])

        for column in (c for c in explicit.dataframe.columns if c.endswith("mean")):
            assert _actual_by_label(deprecated, column) == _actual_by_label(explicit, column)

    def test_ic_stats_land_on_image_rows(self, ic_dataset):
        """Classification stats are one per image and must not be split."""
        result = _stats(ic_dataset)
        assert all(entry.target is None for entry in result["source_index"])

        md = Metadata(ic_dataset)
        md.add_factors(dict(result["stats"]), source_index=result["source_index"])

        assert "mean" in md.dataframe.columns
        assert md.rows_at("unit")["mean"].to_numpy() == pytest.approx(result["stats"]["mean"])

    def test_ic_stats_infer_the_image_level(self, ic_dataset):
        """One value per image matches the image level's row count, so inference suffices."""
        result = _stats(ic_dataset)

        md = Metadata(ic_dataset)
        md.add_factors(dict(result["stats"]))

        assert md.rows_at("unit")["mean"].to_numpy() == pytest.approx(result["stats"]["mean"])

    def test_item_with_no_targets_still_gets_its_image_value(self, od_dataset):
        """An empty target list contributes an image row and no instance rows."""
        result = _stats(od_dataset)
        empty = OD_LABELS.index([])

        md = Metadata(od_dataset)
        md.add_factors(dict(result["stats"]), source_index=result["source_index"])

        stored = _actual_by_label(md, "unit_mean")
        assert SourceIndex(empty, None) in stored
        assert stored[SourceIndex(empty, None)] == pytest.approx(_expected_by_label(result)[SourceIndex(empty, None)])
        assert not any(
            label.item == empty and label.target is not None for label in _actual_by_label(md, "instance_mean")
        )

    def test_vector_valued_stats_are_skipped_not_fatal(self, od_dataset):
        """ImageStats.ALL includes histogram/percentiles/center, which have no column form."""
        result = compute_stats(od_dataset, normalize_pixel_values=False)

        md = Metadata(od_dataset)
        md.add_factors(dict(result["stats"]), source_index=result["source_index"])

        assert "histogram" in md.dropped_factors
        assert "unit_mean" in md.factor_names


class TestFromFactorsFromComputeStats:
    """``Metadata.from_factors(...)`` over real stats output, with no dataset bound."""

    def test_od_two_level_stats_round_trip(self, od_dataset):
        result = _stats(od_dataset)
        md = Metadata.from_factors(dict(result["stats"]), source_index=result["source_index"])

        assert md.levels == ("unit", "instance")
        assert dict(md.level_counts) == {"unit": len(OD_LABELS), "instance": sum(len(x) for x in OD_LABELS)}

        stored = _actual_by_label(md, "unit_mean") | _actual_by_label(md, "instance_mean")
        for label, expected in _expected_by_label(result).items():
            assert stored[label] == pytest.approx(expected), f"{label} landed on the wrong row"

    def test_od_two_level_stats_accept_class_labels(self, od_dataset):
        """Labels describe the instance level, which is where they must attach."""
        result = _stats(od_dataset)
        flat = np.concatenate([np.asarray(labels, dtype=int) for labels in OD_LABELS])

        md = Metadata.from_factors(dict(result["stats"]), flat, source_index=result["source_index"])

        assert md.class_labels.tolist() == flat.tolist()

    def test_od_target_only_stats_round_trip(self, od_dataset):
        result = _stats(od_dataset, per_image=False)
        flat = np.concatenate([np.asarray(labels, dtype=int) for labels in OD_LABELS])

        md = Metadata.from_factors(dict(result["stats"]), flat, source_index=result["source_index"])

        # One kind of entry, so one level, and the factor keeps its bare name.
        assert md.levels == ("instance",)
        assert "mean" in md.factor_names

        # Rows follow the source index, which carries the (item, target) of each value.
        rows = md.dataframe.select("item_index", "target_index", "mean")
        assert [SourceIndex(item, target) for item, target, _ in rows.rows()] == list(result["source_index"])
        assert rows["mean"].to_numpy() == pytest.approx(result["stats"]["mean"])

    def test_ic_stats_round_trip(self, ic_dataset):
        result = _stats(ic_dataset)
        md = Metadata.from_factors(dict(result["stats"]), IC_LABELS, source_index=result["source_index"])

        assert "mean" in md.factor_names
        assert md.rows_at(md.levels[0])["mean"].to_numpy() == pytest.approx(result["stats"]["mean"])

    def test_vector_valued_stats_are_skipped_not_fatal(self, od_dataset):
        """Without a filter these flatten to a wrong length and abort the whole call."""
        result = compute_stats(od_dataset, normalize_pixel_values=False)

        md = Metadata.from_factors(dict(result["stats"]), source_index=result["source_index"])

        assert "histogram" in md.dropped_factors
        assert "unit_mean" in md.factor_names

    def test_item_indices_and_source_index_are_exclusive(self, od_dataset):
        result = _stats(od_dataset, per_image=False)
        with pytest.raises(ValueError, match="mutually exclusive"):
            Metadata.from_factors(
                dict(result["stats"]),
                item_indices=np.zeros(len(result["source_index"]), dtype=int),
                source_index=result["source_index"],
            )

    def test_level_and_source_index_are_exclusive(self, od_dataset):
        result = _stats(od_dataset, per_image=False)
        with pytest.raises(ValueError, match="mutually exclusive"):
            Metadata.from_factors(dict(result["stats"]), level="unit", source_index=result["source_index"])

    def test_repeated_item_indices_get_distinct_target_indices(self):
        """Several rows sharing an item are targets 0, 1, ... of it, not all target 0.

        ``item_indices`` exists so that several detections can share an image. A constant
        ``target_index`` gives every one of those rows the same ``(item_index,
        target_index)`` identity, which is what a later source index is matched against —
        so the rows could never be named, whatever index the caller passed.
        """
        md = Metadata.from_factors({"a": np.arange(4.0)}, item_indices=[0, 0, 1, 1])

        assert md.dataframe["item_index"].to_list() == [0, 0, 1, 1]
        assert md.dataframe["target_index"].to_list() == [0, 1, 0, 1]

        md.add_factors(
            {"b": np.array([10.0, 20.0, 30.0, 40.0])},
            source_index=[SourceIndex(0, 0), SourceIndex(0, 1), SourceIndex(1, 0), SourceIndex(1, 1)],
        )
        assert md.dataframe["b"].to_list() == [10.0, 20.0, 30.0, 40.0]

    def test_one_row_per_item_still_indexes_to_zero(self):
        """The default — one row per item — keeps every target_index at 0."""
        md = Metadata.from_factors({"a": np.arange(3.0)})

        assert md.dataframe["target_index"].to_list() == [0, 0, 0]

    def test_item_count_counts_items_not_rows(self):
        """Several labels can name one item, so rows and items are not the same number."""
        md = Metadata.from_factors(
            {"m": np.array([10.0, 11.0, 30.0])},
            source_index=[SourceIndex(1, 0), SourceIndex(1, 1), SourceIndex(3, 0)],
        )
        assert md.item_count == 2

    def test_item_level_rows_leave_target_index_null(self):
        """Nullness is how downstream code tells a per-item row from a per-target one.

        :class:`~dataeval.quality.Outliers` decides whether a result is per-target by
        testing ``target_index`` for nulls, so a zero there would misreport an image-level
        result as a detection-level one.
        """
        item_only = Metadata.from_factors({"m": np.arange(3.0)}, source_index=[SourceIndex(i) for i in range(3)])
        assert item_only.dataframe["target_index"].to_list() == [None, None, None]

        both = Metadata.from_factors(
            {"m": np.arange(4.0)},
            source_index=[SourceIndex(0), SourceIndex(0, 0), SourceIndex(1), SourceIndex(1, 0)],
        )
        assert both.rows_at("unit")["target_index"].to_list() == [None, None]

    def test_column_vector_factors_are_kept(self):
        """(N, 1) is one value per row; only genuinely vector-valued factors are dropped."""
        md = Metadata.from_factors({"a": np.arange(4).reshape(4, 1), "b": np.arange(4.0)})

        assert sorted(md.factor_names) == ["a", "b"]
        assert "a" not in md.dropped_factors

    def test_a_single_row_factor_is_kept(self):
        """A one-element factor is 1-D data, not a scalar to be collapsed away.

        A blanket squeeze reduces (1,) to 0-d, which then fails the ndim == 1 test and is
        reported as vector-valued — the one dataset size where a perfectly ordinary factor
        silently disappears.
        """
        md = Metadata.from_factors({"a": np.array([1.5]), "b": np.array([[2.5]])}, class_labels=[0])

        assert sorted(md.factor_names) == ["a", "b"]
        assert not md.dropped_factors
        assert md.rows_at(md.levels[0])["a"].to_list() == [1.5]

    def test_single_row_vector_stats_are_still_skipped(self):
        """(1, K) is one row of K values on a one-item dataset, not K rows of one value.

        The two readings are indistinguishable from the shape alone, and flattening picks
        the wrong one: ``center`` and ``histogram`` would be imported as K rows of data
        while a genuine one-row ``mean`` was dropped — the classification exactly inverted.
        """
        md = Metadata.from_factors(
            {"center": np.array([[1.0, 2.0]]), "histogram": np.zeros((1, 256)), "mean": np.array([0.5])},
            class_labels=[0],
        )

        assert sorted(md.factor_names) == ["mean"]
        assert sorted(md.dropped_factors) == ["center", "histogram"]

    def test_single_row_stats_result_splits_the_right_way(self):
        """The same edge against real compute_stats output, where these shapes actually arise.

        One image and one row of output is the only shape that produces ``mean`` as ``(1,)``
        alongside ``histogram`` as ``(1, 256)`` and ``center`` as ``(1, 2)`` — the three
        arrays whose classification a squeeze inverts.
        """
        dataset = _get_mock_od_dataset(_images(1), [[0]], _boxes([[0]]))
        result = compute_stats(dataset, normalize_pixel_values=False, per_target=False)
        assert np.asarray(result["stats"]["mean"]).shape == (1,)
        assert np.asarray(result["stats"]["histogram"]).shape == (1, 256)

        md = Metadata.from_factors(dict(result["stats"]), source_index=result["source_index"])

        assert "mean" in md.factor_names
        assert "histogram" in md.dropped_factors
        assert "center" in md.dropped_factors

    def test_a_duplicated_source_index_is_rejected(self):
        with pytest.raises(ValueError, match="names the same item-level row more than once"):
            Metadata.from_factors(
                {"m": np.arange(3.0)},
                source_index=[SourceIndex(0), SourceIndex(0), SourceIndex(2)],
            )

    def test_a_label_with_no_item_entry_is_rejected(self):
        """An instance row needs an image row to hang from; silently orphaning it loses it."""
        with pytest.raises(ValueError, match="no per-item entry"):
            Metadata.from_factors(
                {"m": np.arange(3.0)},
                source_index=[SourceIndex(0), SourceIndex(0, 0), SourceIndex(5, 0)],
            )


class TestStatsResultIsAccepted:
    """The whole ``StatsResult`` may stand in for its ``stats`` mapping."""

    @pytest.mark.parametrize("kwargs", [{"per_target": False}, {"per_image": False}, {}])
    def test_add_factors_unpacks_a_stats_result(self, od_dataset, kwargs):
        result = _stats(od_dataset, **kwargs)

        unpacked = Metadata(od_dataset)
        unpacked.add_factors(dict(result["stats"]), source_index=result["source_index"])

        direct = Metadata(od_dataset)
        direct.add_factors(result)

        assert sorted(direct.factor_names) == sorted(unpacked.factor_names)
        for column in (c for c in unpacked.dataframe.columns if c.endswith("mean")):
            assert _actual_by_label(direct, column) == _actual_by_label(unpacked, column)

    def test_from_factors_unpacks_a_stats_result(self, od_dataset):
        result = _stats(od_dataset)

        unpacked = Metadata.from_factors(dict(result["stats"]), source_index=result["source_index"])
        direct = Metadata.from_factors(result)

        assert sorted(direct.factor_names) == sorted(unpacked.factor_names)
        assert dict(direct.level_counts) == dict(unpacked.level_counts)

    def test_bookkeeping_keys_are_not_imported_as_factors(self, od_dataset):
        """object_count and invalid_box_count describe the run, not the images."""
        result = _stats(od_dataset)

        md = Metadata(od_dataset)
        md.add_factors(result)

        assert "object_count" not in md.factor_names
        assert "invalid_box_count" not in md.factor_names
        assert "image_count" not in md.factor_names

    def test_an_explicit_source_index_overrides_the_embedded_one(self, od_dataset):
        """The escape hatch for a corrected index must actually take effect."""
        result = _stats(od_dataset, per_image=False)
        reversed_index = list(reversed(result["source_index"]))

        md = Metadata(od_dataset)
        md.add_factors(result, source_index=reversed_index)

        # Placement follows the labels, so reversing them relabels every value.
        expected = dict(zip(reversed_index, result["stats"]["mean"], strict=True))
        assert _actual_by_label(md, "mean") == pytest.approx(expected)

    def test_an_explicit_level_is_rejected(self, od_dataset):
        result = _stats(od_dataset, per_target=False)

        md = Metadata(od_dataset)
        with pytest.raises(ValueError, match="mutually exclusive"):
            md.add_factors(result, level="unit")

    def test_a_plain_mapping_is_not_mistaken_for_a_result(self, od_dataset):
        """User factors named "stats" and "source_index" must stay factors.

        Both reserved key names are present, so only the value types tell this apart from
        a real result. A false positive here would silently discard every factor passed.
        """
        md = Metadata(od_dataset)
        md.add_factors({"stats": [0.1, 0.2, 0.3, 0.4, 0.5], "source_index": [1, 2, 3, 4, 5]}, level="unit")

        assert md.rows_at("unit")["stats"].to_list() == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert md.rows_at("unit")["source_index"].to_list() == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize(
        ("candidate", "expected"),
        [
            ({"stats": {"mean": [1.0]}, "source_index": [SourceIndex(0)]}, True),
            ({"stats": {"mean": [1.0]}}, False),
            ({"source_index": [SourceIndex(0)]}, False),
            ({"stats": [1.0], "source_index": [SourceIndex(0)]}, False),
            ({"stats": {"mean": [1.0]}, "source_index": [(0, None, None)]}, False),
            ({"stats": {"mean": [1.0]}, "source_index": 3}, False),
            ({}, False),
        ],
    )
    def test_detection_requires_both_keys_and_both_value_types(self, candidate, expected):
        assert _is_stats_result(candidate) is expected


class TestVacuousSplitsAreReported:
    """A level split that holds nothing is dropped, but never silently."""

    def test_a_pre_existing_all_null_split_is_recorded_too(self):
        """Boxes that all miss their images make every instance statistic null.

        Nothing about that is background-specific, and the column is as unusable as a
        background one, so it is dropped the same way — and reported, so that code
        expecting both halves of the split can find out where the other one went.
        """
        dataset = _od_dataset([[0], [1], [0]], [[[100, 100, 110, 110]]] * 3)
        result = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)

        md = Metadata(dataset)
        md.add_factors(result)

        assert "unit_mean" in md.factor_names
        assert "instance_mean" not in md.factor_names
        assert md.dropped_factors["instance_mean"] == ["no_values_at_level"]

    def test_combined_keeps_both_halves_it_promised(self):
        """``level="combined"`` names its columns up front, so it keeps them regardless.

        The deprecation warning tells the caller exactly which two columns it will get.
        Dropping one for holding nothing would rename the result out from under code
        following that warning, which is the thing ``qualify=True`` exists to prevent.
        """
        md = Metadata(_od_dataset([[0], [1], [0]]))

        # Three unit rows with values, three instance rows without.
        values = np.concatenate([np.arange(3.0), np.full(3, np.nan)])
        with pytest.deprecated_call():
            md.add_factors({"foo": values}, level="combined")

        assert "unit_foo" in md.factor_names
        assert "instance_foo" in md.factor_names
        assert md.dropped_factors == {}

    def test_a_factor_empty_at_every_level_is_kept_whole(self):
        """Dropping every half would make the factor vanish, which is never the answer."""
        dataset = _od_dataset([[0], [1], [0]])
        result = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)

        md = Metadata(dataset)
        md.add_factors(
            {"all_null": np.full(len(result["source_index"]), np.nan)},
            source_index=result["source_index"],
        )

        assert {"unit_all_null", "instance_all_null"} <= set(md.factor_names)
        assert md.dropped_factors == {}

"""Level-aware Outliers: restriction on re-detection, level views, aggregations, and `levels=`."""

from collections import Counter
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.core import compute_stats, track_stats
from dataeval.extractors import FlattenExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Outliers, OutliersOutput
from dataeval.quality._shared import row_kinds
from dataeval.types import SourceIndex
from dataeval.utils.thresholds import ConstantThreshold
from tests.conftest import MockMetadata, MockODDataset
from tests.metadata.test_structurers import _mot_dataset
from tests.quality.test_outliers import TestOutliersOverMetadata as _MetadataTests  # not collected here


def _od(get_od_dataset, n=20, seed=0):
    return get_od_dataset(list(np.random.default_rng(seed).random((n, 3, 16, 16))), 2, True)


def _od_stats(ds):
    return compute_stats(ds, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, normalize_pixel_values=False)


def _columns(result):
    return set(result.data().columns)


# Flags about half the pixel means of `_od`, at either level. A constant threshold is fitted to no population, so
# it judges `_od`'s twenty-odd real box readings however many there are.
_HALF = ConstantThreshold(upper=0.5)


@pytest.mark.required
class TestFromStatsRedetection:
    def test_without_targets_a_redetection_returns_items_only(self, get_od_dataset):
        stats = _od_stats(_od(get_od_dataset))
        redetected = Outliers().from_stats(stats, per_target=False).with_threshold(("zscore", 0.5))
        assert len(redetected) > 0
        assert "target_index" not in _columns(redetected)

    def test_with_targets_a_redetection_returns_both(self, get_od_dataset):
        stats = _od_stats(_od(get_od_dataset))
        redetected = Outliers().from_stats(stats, per_target=True).with_threshold(_HALF)
        assert "target_index" in _columns(redetected)

    def test_without_images_a_redetection_returns_targets_only(self, get_od_dataset):
        stats = _od_stats(_od(get_od_dataset))
        redetected = Outliers().from_stats(stats, per_image=False, per_target=True).with_threshold(_HALF)
        assert len(redetected) > 0
        assert redetected.data()["target_index"].null_count() == 0

    def test_classwise_filters_the_class_ids_with_the_stats(self, get_od_dataset):
        ds = _od(get_od_dataset)
        stats = _od_stats(ds)
        result = Outliers().from_stats(stats, per_target=False).classwise(Metadata(ds)).with_threshold(("zscore", 0.5))
        assert len(result) > 0
        assert "target_index" not in _columns(result)

    def test_a_chained_redetection_keeps_the_restriction(self, get_od_dataset):
        stats = _od_stats(_od(get_od_dataset))
        result = Outliers().from_stats(stats, per_target=False)
        twice = result.with_threshold(("zscore", 0.5)).with_threshold(("zscore", 0.4))
        assert twice.restricted_to == result.restricted_to == frozenset({None})
        assert "target_index" not in _columns(twice)

    def test_both_flags_leave_the_result_unrestricted(self, get_od_dataset):
        stats = _od_stats(_od(get_od_dataset))
        assert Outliers().from_stats(stats, per_image=True, per_target=True).restricted_to is None

    def test_an_output_without_the_new_attributes_still_loads(self, get_od_dataset):
        stats = _od_stats(_od(get_od_dataset))
        result = Outliers().from_stats(stats, per_target=True)
        vars(result).pop("restricted_to", None)
        vars(result).pop("tracking", None)
        assert result.restricted_to is None
        assert result.tracking is False
        assert len(result.with_threshold(("zscore", 0.5))) > 0


def _leveled_metadata():
    """A tracking metadata with an ordered factor at every level, each carrying one planted extreme value."""
    md = _MetadataTests._tracked()
    rng = np.random.default_rng(1)
    for level in ("sequence", "unit", "instance"):
        count = md.level_counts[level]
        md.add_factors({f"{level}_signal": np.r_[rng.normal(0, 1, count - 1), 50.0]}, level=level)
    return md


# The sequence and track levels hold twelve rows each, under the floor a statistical threshold needs, so their
# planted extremes are judged against constants. The frame and detection levels keep the default threshold.
_SMALL_LEVELS = {"sequence_signal": ConstantThreshold(upper=10.0), "n_gaps": ConstantThreshold(upper=0.5)}


def _leveled_outliers():
    return Outliers(outlier_threshold=_SMALL_LEVELS)


def _leveled_result():
    return _leveled_outliers().evaluate(_leveled_metadata(), per_target=True)


def _names(result):
    return set(result.data()["metric_name"].cast(pl.Utf8))


def _levels(result):
    frame = result.data()
    return set(frame["level"].cast(pl.Utf8)) if "level" in frame.columns else set()


def _same_rows(a, b):
    return a.select(sorted(a.columns)).rows() == b.select(sorted(b.columns)).rows()


@pytest.mark.required
class TestLevelViews:
    def test_a_tracking_result_is_recognized(self):
        assert _leveled_result().tracking is True

    def test_each_tracking_view_holds_its_own_level(self):
        result = _leveled_result()
        assert "sequence_signal" in _names(result.sequences)
        assert not {"level", "target_index"} & _columns(result.sequences)
        assert _levels(result.frames) == {"unit"}
        assert "unit_signal" in _names(result.frames)
        assert _levels(result.tracks) == {"track"}
        assert "n_gaps" in _names(result.tracks)
        assert "instance_signal" in _names(result.detections)
        assert "level" not in _columns(result.detections)
        assert "target_index" in _columns(result.detections)

    def test_items_are_frames_on_tracking_data_and_targets_are_detections(self):
        result = _leveled_result()
        assert _same_rows(result.items.data(), result.frames.data())
        assert _same_rows(result.targets.data(), result.detections.data())

    def test_on_an_image_result_items_and_targets_split_the_rows(self, get_od_dataset):
        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=("zscore", 0.5))
        result = outliers.evaluate(_od(get_od_dataset), per_target=True)
        assert result.tracking is False
        assert len(result.items) > 0
        assert "target_index" not in _columns(result.items)
        assert len(result.targets) > 0
        assert result.targets.data()["target_index"].null_count() == 0
        assert len(result.items) + len(result.targets) == len(result)
        assert all(len(getattr(result, name)) == 0 for name in ("sequences", "frames", "tracks", "detections"))

    @pytest.mark.parametrize("name", ["sequences", "frames", "tracks", "detections", "items", "targets"])
    def test_a_view_equals_its_own_redetection(self, name):
        result = _leveled_result()
        view = getattr(result, name)
        assert len(view) > 0
        assert _same_rows(view.data(), view.with_threshold(result.outlier_threshold).data())

    def test_a_view_redetects_only_its_own_level(self):
        result = _leveled_result()
        once = result.tracks.with_threshold(ConstantThreshold(upper=0.5))
        redetected = once.with_threshold(ConstantThreshold(upper=0.4))
        assert _levels(redetected) == {"track"}

    def test_narrowing_twice_to_different_levels_is_empty(self):
        view = _leveled_result().tracks.sequences
        assert len(view) == 0
        assert view.restricted_to == frozenset()

    def test_a_view_with_no_flagged_rows_is_empty_and_still_redetectable(self):
        quiet = _leveled_result().with_threshold(("zscore", 100.0))
        assert len(quiet) == 0
        for view in (quiet.tracks, quiet.items, quiet.detections):
            assert len(view) == 0
            assert len(view.with_threshold(ConstantThreshold(upper=0.5))) > 0

    def test_classwise_runs_on_a_view(self):
        md = _leveled_metadata()
        result = Outliers().evaluate(md, per_target=True)
        classwise = result.detections.classwise(md)
        assert classwise.restricted_to == frozenset({"instance"})
        redetected = classwise.with_threshold(("zscore", 0.5))
        assert len(redetected) > 0
        assert "level" not in _columns(redetected)

    def test_prune_on_a_view_names_only_that_level(self):
        plan = _leveled_result().tracks.prune()
        assert len(plan) > 0
        assert all(address.kind == "track" for address in plan)

    def test_a_multi_dataset_view_redetects_its_own_rows(self, get_od_dataset):
        first, second = _od(get_od_dataset, 12, 0), _od(get_od_dataset, 12, 1)
        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=("zscore", 0.5))
        result = outliers.evaluate(first, second, per_target=True)
        for view in (result.targets, result.items):
            assert len(view) > 0
            assert "dataset_index" in _columns(view)
            assert _same_rows(view.data(), view.with_threshold(result.outlier_threshold).data())

    def test_cluster_rows_stay_out_of_a_view_that_excludes_the_item_end(self, get_od_dataset):
        outliers = Outliers(flags=ImageStats.PIXEL, extractor=FlattenExtractor(), cluster_threshold=2.0)
        result = outliers.evaluate(_od(get_od_dataset), per_target=True)
        assert "cluster_distance" in _names(result.with_threshold(cluster_threshold=0.1))
        assert "cluster_distance" not in _names(result.targets.with_threshold(cluster_threshold=0.1))
        assert "cluster_distance" in _names(result.items.with_threshold(cluster_threshold=0.1))


@pytest.mark.required
class TestAggregateByClassJoin:
    @staticmethod
    def _metadata():
        # Item 0 holds a car and a bus; item 1 holds a van.
        return MockMetadata(
            class_labels=np.array([0, 1, 2], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_binned=[],
            index2label={0: "car", 1: "bus", 2: "van"},
            item_indices=np.array([0, 0, 1], dtype=np.int64),
        )

    @staticmethod
    def _totals(frame):
        return {row["class_name"]: row["Total"] for row in frame.iter_rows(named=True)}

    def test_a_flagged_box_counts_under_its_own_class_only(self):
        flagged = pl.DataFrame([{"item_index": 0, "target_index": 1, "metric_name": "width", "metric_value": 1.0}])
        by_class = OutliersOutput(flagged).aggregate_by_class(self._metadata())
        assert self._totals(by_class) == {"bus": 1, "Total": 1}

    def test_a_box_position_restarts_at_each_item(self):
        flagged = pl.DataFrame([{"item_index": 1, "target_index": 0, "metric_name": "width", "metric_value": 1.0}])
        by_class = OutliersOutput(flagged).aggregate_by_class(self._metadata())
        assert self._totals(by_class) == {"van": 1, "Total": 1}

    def test_an_item_row_and_a_box_row_are_both_counted(self):
        flagged = pl.DataFrame([
            {"item_index": 0, "target_index": None, "metric_name": "mean", "metric_value": 1.0},
            {"item_index": 0, "target_index": 1, "metric_name": "width", "metric_value": 1.0},
        ])
        by_class = OutliersOutput(flagged).aggregate_by_class(self._metadata())
        assert by_class.columns == ["class_name", "mean", "width", "Total"]
        assert by_class.rows() == [("bus", 1, 1, 2), ("car", 1, 0, 1), ("Total", 2, 1, 3)]

    def test_with_one_label_per_item_a_box_counts_under_its_item_label(self):
        # A class axis read at the item level, such as md.at("unit").classed_by("weather"), labels items, not boxes.
        per_item = MockMetadata(
            class_labels=np.array([0, 1], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_binned=[],
            index2label={0: "car", 1: "van"},
            item_indices=np.array([0, 1], dtype=np.int64),
        )
        flagged = pl.DataFrame([{"item_index": 0, "target_index": 1, "metric_name": "width", "metric_value": 1.0}])
        assert self._totals(OutliersOutput(flagged).aggregate_by_class(per_item)) == {"car": 1, "Total": 1}

    def test_a_class_axis_read_per_image_counts_every_box_under_its_image(self, get_od_dataset):
        weather = ["rain" if i % 3 else "sun" for i in range(20)]
        factors = [{"weather": name, "hour": float(i)} for i, name in enumerate(weather)]
        ds = get_od_dataset(list(np.random.default_rng(0).random((20, 3, 16, 16))), 2, True, metadata=factors)
        result = Outliers(flags=ImageStats.PIXEL, outlier_threshold=("zscore", 0.5)).evaluate(ds, per_target=True)
        frame = result.data()
        assert frame["target_index"].drop_nulls().n_unique() == 2
        by_class = result.aggregate_by_class(Metadata(ds).at("unit").classed_by("weather"))
        assert self._totals(by_class) == {**Counter(weather[i] for i in frame["item_index"]), "Total": frame.height}

    def test_an_item_row_still_counts_under_every_class_of_its_item(self):
        flagged = pl.DataFrame([{"item_index": 0, "target_index": None, "metric_name": "mean", "metric_value": 1.0}])
        by_class = OutliersOutput(flagged).aggregate_by_class(self._metadata())
        assert self._totals(by_class) == {"car": 1, "bus": 1, "Total": 2}

    # An image holding boxes [car, car, bus], beside an image holding a bus.
    _REPEATED = [[0, 0, 1], [1]]

    def test_an_item_row_counts_once_per_distinct_class_of_a_metadata_like(self):
        md = MockMetadata(
            class_labels=np.array([c for labels in self._REPEATED for c in labels], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_binned=[],
            index2label={0: "car", 1: "bus"},
            item_indices=np.array([i for i, labels in enumerate(self._REPEATED) for _ in labels], dtype=np.int64),
        )
        flagged = pl.DataFrame([{"item_index": 0, "target_index": None, "metric_name": "mean", "metric_value": 1.0}])
        assert self._totals(OutliersOutput(flagged).aggregate_by_class(md)) == {"car": 1, "bus": 1, "Total": 2}

    def test_an_item_row_counts_once_per_distinct_class_of_a_metadata(self):
        images = [np.random.default_rng(i).random((3, 16, 16)) for i in range(len(self._REPEATED))]
        boxes = [[[0, 0, 4, 4]] * len(labels) for labels in self._REPEATED]
        md = Metadata(MockODDataset(images, self._REPEATED, boxes, classes=["car", "bus"]))  # type: ignore
        flagged = pl.DataFrame([{"item_index": 0, "target_index": None, "metric_name": "mean", "metric_value": 1.0}])
        assert self._totals(OutliersOutput(flagged).aggregate_by_class(md)) == {"car": 1, "bus": 1, "Total": 2}


def test_a_detection_counts_under_its_frames_class_on_a_unit_view():
    md = Metadata(_mot_dataset([[1, 3, 2]]))
    md.add_factors({"weather": ["rain", "clear", "snow"]}, level="unit")
    md = md.at("unit").classed_by("weather")
    frame = pl.DataFrame({
        "item_index": [0, 0],
        "target_index": [0, 4],
        "metric_name": ["m", "m"],
        "metric_value": [1.0, 1.0],
    })
    summary = OutliersOutput(frame).aggregate_by_class(md)
    counts = dict(zip(summary["class_name"].cast(pl.Utf8), summary["m"], strict=True))
    assert counts == {"rain": 1, "snow": 1, "Total": 2}


@pytest.mark.required
class TestLeveledAggregations:
    def test_metric_totals_count_addresses_per_level(self):
        result = _leveled_result()
        totals = result.aggregate_by_metric()
        assert totals.columns == ["level", "metric_name", "Total"]
        gaps = result.tracks.data().filter(pl.col("metric_name").cast(pl.Utf8) == "n_gaps")
        flagged_tracks = gaps.select("item_index", "target_index")
        got = totals.filter((pl.col("level") == "track") & (pl.col("metric_name").cast(pl.Utf8) == "n_gaps"))
        assert got["Total"].to_list() == [flagged_tracks.unique().height]
        assert totals.filter(pl.col("level").is_null())["metric_name"].cast(pl.Utf8).to_list() == ["sequence_signal"]
        assert set(totals["level"].drop_nulls()) == {"instance", "track", "unit"}

    def test_the_summary_does_not_depend_on_which_rows_were_flagged(self):
        quiet = _leveled_result().with_threshold(("zscore", 100.0))
        assert len(quiet) == 0
        assert quiet.aggregate_by_metric().columns == ["level", "metric_name", "Total"]
        with pytest.raises(ValueError, match="no class"):
            quiet.aggregate_by_class(MagicMock())

    def test_a_class_summary_of_frames_and_tracks_is_refused(self):
        with pytest.raises(ValueError, match=r"\['track', 'unit'\] rows have no class"):
            _leveled_result().aggregate_by_class(MagicMock())

    @pytest.mark.parametrize("view", ["detections", "targets"])
    def test_the_label_end_of_a_leveled_result_can_be_summarized_by_class(self, view):
        md = _leveled_metadata()
        by_class = getattr(Outliers().evaluate(md, per_target=True), view).aggregate_by_class(md)
        assert dict(by_class.select("class_name", "Total").rows()) == {"0": 1, "Total": 1}

    def test_an_image_result_keeps_the_old_metric_summary(self, get_od_dataset):
        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=("zscore", 0.5))
        result = outliers.evaluate(_od(get_od_dataset), per_target=True)
        assert len(result) > 0
        assert result.aggregate_by_metric().columns == ["metric_name", "Total"]

    def test_a_result_without_stored_statistics_is_read_from_its_frame(self):
        # n_gaps: two tracks of item 0, one of them flagged twice, so 2 addresses in 1 item and 3 rows.
        # mean: three items. The 1-count groups tie, so their order is level (null first), then metric.
        flagged = pl.DataFrame(
            [
                {"item_index": 0, "target_index": 3, "level": "track", "metric_name": "n_gaps"},
                {"item_index": 0, "target_index": 4, "level": "track", "metric_name": "n_gaps"},
                {"item_index": 0, "target_index": 3, "level": "track", "metric_name": "n_gaps"},
                {"item_index": 0, "target_index": 3, "level": "track", "metric_name": "length"},
                {"item_index": 0, "target_index": 4, "level": "track", "metric_name": "zeros"},
                {"item_index": 0, "target_index": 0, "level": "unit", "metric_name": "brightness"},
                {"item_index": 0, "target_index": 0, "level": None, "metric_name": "width"},
                {"item_index": 1, "target_index": None, "level": None, "metric_name": "mean"},
                {"item_index": 2, "target_index": None, "level": None, "metric_name": "mean"},
                {"item_index": 5, "target_index": None, "level": None, "metric_name": "mean"},
                {"item_index": 1, "target_index": None, "level": None, "metric_name": "std"},
            ],
            schema_overrides={"target_index": pl.Int64, "level": pl.Utf8},
        ).with_columns(metric_value=pl.lit(1.0))
        result = OutliersOutput(flagged)
        assert result.aggregate_by_metric().rows() == [
            (None, "mean", 3),
            ("track", "n_gaps", 2),
            (None, "std", 1),
            ("instance", "width", 1),
            ("track", "length", 1),
            ("track", "zeros", 1),
            ("unit", "brightness", 1),
        ]
        with pytest.raises(ValueError, match="no class"):
            result.aggregate_by_class(MagicMock())
        assert result.items.aggregate_by_metric().rows() == [("mean", 3), ("std", 1)]


def _kinds_of(result):
    return set(row_kinds(result.data()).to_list())


def _two_class_tracked():
    """Twelve sequences of two tracks, of classes 0 and 1, with sequence 3's class-1 track flickering."""
    frames = [[2] * 6 for _ in range(12)]
    frames[3] = [2, 1] * 3
    ds = _mot_dataset(frames)
    md = Metadata(ds)
    md.add_factors(track_stats(ds), level="track", key="track_id")
    signal = np.r_[np.random.default_rng(1).normal(0, 1, md.level_counts["unit"] - 1), 50.0]
    md.add_factors({"unit_signal": signal}, level="unit")
    return md


@pytest.mark.required
class TestEvaluateLevels:
    def test_the_default_judges_sequences_and_frames_only(self):
        assert _kinds_of(_leveled_outliers().evaluate(_leveled_metadata())) == {None, "unit"}

    def test_a_track_level_is_asked_for_by_name(self):
        result = _leveled_outliers().evaluate(_leveled_metadata(), levels="track")
        assert _kinds_of(result) == {"track"}
        assert SourceIndex(3, 1, "track") in result.outliers

    def test_several_levels_by_tuple_or_with_a_repeat(self):
        both = _leveled_outliers().evaluate(_leveled_metadata(), levels=("unit", "track"))
        assert _kinds_of(both) == {"unit", "track"}
        repeated = _leveled_outliers().evaluate(_leveled_metadata(), levels=("track", "track"))
        assert _kinds_of(repeated) == {"track"}

    def test_per_target_adds_detections_and_tracks(self):
        result = _leveled_outliers().evaluate(_leveled_metadata(), per_target=True)
        assert _kinds_of(result) == {None, "unit", "instance", "track"}

    def test_flags_without_images_leave_the_frames_out(self):
        result = _leveled_outliers().evaluate(_leveled_metadata(), per_image=False, per_target=True)
        assert _kinds_of(result) == {"instance", "track"}

    def test_the_result_records_that_it_is_tracking_even_without_frames_or_tracks(self):
        md = _MetadataTests._tracked()
        md.add_factors({"seq_signal": np.r_[np.random.default_rng(0).normal(0, 1, 11), 50.0]}, level="sequence")
        result = Outliers(outlier_threshold=ConstantThreshold(upper=10.0)).evaluate(md, levels="sequence")
        assert len(result) > 0
        assert result.tracking is True
        assert len(result.sequences) == len(result)

    def test_levels_with_a_flag_is_refused(self):
        with pytest.raises(ValueError, match="not both"):
            Outliers().evaluate(_leveled_metadata(), levels="track", per_target=True)  # type: ignore[call-overload]

    def test_an_image_level_name_on_tracking_metadata_is_refused(self):
        with pytest.raises(ValueError, match="not a level of a tracking metadata"):
            Outliers().evaluate(_leveled_metadata(), levels="item")

    def test_a_named_factor_outside_the_plan_is_refused_with_the_fix(self):
        with pytest.raises(ValueError, match=r"levels=\['track'\] or per_target=True"):
            Outliers().evaluate(_leveled_metadata(), factors=["n_gaps"])

    def test_the_hint_for_a_named_factor_keeps_the_levels_already_judged(self):
        # unit_signal is judged by default and n_gaps is not, so the fix must name both levels.
        with pytest.raises(ValueError, match=r"levels=\['track', 'unit'\] or per_image=True and per_target=True"):
            Outliers().evaluate(_leveled_metadata(), factors=["unit_signal", "n_gaps"])
        result = _leveled_outliers().evaluate(
            _leveled_metadata(), factors=["unit_signal", "n_gaps"], levels=["track", "unit"]
        )
        assert _kinds_of(result) == {"unit", "track"}

    def test_a_named_factor_outside_an_explicit_plan_is_refused_with_the_fix(self):
        with pytest.raises(ValueError, match=r"levels=\['unit'\] or per_image=True"):
            Outliers().evaluate(_leveled_metadata(), levels="track", factors=["unit_signal"])
        with pytest.raises(ValueError, match=r"levels=\['track', 'unit'\] or per_image=True and per_target=True"):
            Outliers().evaluate(_leveled_metadata(), levels="track", factors=["n_gaps", "unit_signal"])

    def test_a_plan_with_no_ordered_factor_is_refused_and_names_where_they_live(self):
        # The fixture's ordered factors live at the frame and track levels, none at the sequence level.
        with pytest.raises(ValueError, match=r"no ordered factor lives at \['sequence'\].*levels=\['track', 'unit'\]"):
            Outliers().evaluate(_MetadataTests._tracked(), levels="sequence")

    def test_no_flag_at_all_is_refused_first(self):
        with pytest.raises(ValueError, match="At least one of per_image or per_target must be True"):
            Outliers().evaluate(_leveled_metadata(), per_image=False)

    def test_a_categorical_factor_is_refused_before_the_plan_is_checked(self):
        with pytest.raises(ValueError, match="cannot be thresholded"):
            Outliers().evaluate(_leveled_metadata(), factors=["labels"])

    def test_per_class_runs_over_levels(self):
        md = _two_class_tracked()
        assert set(md.class_labels.tolist()) == {0, 1}
        result = Outliers().evaluate(md, levels=("unit", "track"), per_class=True)
        plain = Outliers().evaluate(md, levels=("unit", "track"))
        assert result.tracking is True
        assert _kinds_of(result) == {"unit", "track"}
        for view in ("frames", "tracks"):
            assert len(getattr(plain, view)) > 0
            assert _same_rows(getattr(result, view).data(), getattr(plain, view).data())


@pytest.mark.required
class TestEvaluateLevelsOverImageMetadata:
    def test_item_names_a_single_level_metadata(self):
        md = _MetadataTests._temperature()
        result = Outliers().evaluate(md, levels="item")
        assert result.data()["item_index"].to_list() == [23]

    def test_target_alone_on_a_single_level_metadata_is_refused(self):
        md = _MetadataTests._temperature()
        with pytest.raises(ValueError, match=r"one level.*per_image=True"):
            Outliers().evaluate(md, levels="target")

    def test_per_target_beside_per_image_is_harmless_on_a_single_level_metadata(self):
        md = _MetadataTests._temperature()
        assert len(Outliers().evaluate(md, per_target=True)) == 1

    def test_item_and_target_name_the_metadata_levels(self, get_od_dataset):
        md = _MetadataTests._boxes(get_od_dataset)
        result = Outliers().evaluate(md, levels="target", factors=["iou"])
        assert SourceIndex(9, 1) in result.outliers
        with pytest.raises(ValueError, match="not a level of an image metadata"):
            Outliers().evaluate(md, levels="unit")


@pytest.mark.required
class TestEvaluateLevelsOverDatasets:
    def test_levels_on_a_dataset_equal_the_flags(self, get_od_dataset):
        ds = _od(get_od_dataset)
        outliers = Outliers(flags=ImageStats.PIXEL)
        by_levels = outliers.evaluate(ds, levels=["item", "target"])
        by_flags = outliers.evaluate(ds, per_target=True)
        assert by_levels.data().rows() == by_flags.data().rows()

    def test_item_on_a_dataset_equals_per_image_alone(self, get_od_dataset):
        ds = _od(get_od_dataset)
        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=("zscore", 0.5))
        by_levels = outliers.evaluate(ds, levels="item")
        by_flags = outliers.evaluate(ds, per_image=True, per_target=False)
        assert len(by_flags) > 0
        assert by_levels.data().rows() == by_flags.data().rows()

    def test_levels_on_several_datasets_equal_the_flags(self, get_od_dataset):
        first, second = _od(get_od_dataset, 12, 0), _od(get_od_dataset, 12, 1)
        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=("zscore", 0.5))
        by_levels = outliers.evaluate(first, second, levels=["item", "target"])
        by_flags = outliers.evaluate(first, second, per_target=True)
        assert "dataset_index" in _columns(by_levels)
        assert len(by_levels) > 0
        assert by_levels.data().rows() == by_flags.data().rows()

    def test_a_tracking_name_on_a_dataset_says_to_pass_a_metadata(self, get_od_dataset):
        with pytest.raises(ValueError, match="Pass a Metadata"):
            Outliers().evaluate(_od(get_od_dataset), levels="track")

    def test_no_flag_at_all_is_still_refused_on_a_dataset(self, get_od_dataset):
        with pytest.raises(ValueError, match="At least one of per_image or per_target must be True"):
            Outliers().evaluate(_od(get_od_dataset), per_image=False, per_target=False)


@pytest.mark.required
class TestRedetectionKeepsClasses:
    def _two_classes(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.normal(0, 1, 100), rng.normal(50, 1, 100)])
        values[3] = 6.0  # ordinary over both classes, extreme within class 0
        labels = np.repeat([0, 1], 100)
        return Metadata.from_factors({"x": values}, class_labels=labels)

    def test_with_threshold_after_classwise_stays_per_class(self):
        md = self._two_classes()
        per_class = Outliers().evaluate(md).classwise(md)
        assert 3 in per_class.data()["item_index"].to_list()
        again = per_class.with_threshold(3.5)
        assert 3 in again.data()["item_index"].to_list()

    def test_a_level_view_keeps_the_classes(self):
        md = self._two_classes()
        kept = Outliers().evaluate(md).classwise(md).items.with_threshold(3.5)
        assert 3 in kept.data()["item_index"].to_list()

    def test_itemwise_drops_the_classes(self):
        md = self._two_classes()
        assert Outliers().evaluate(md).classwise(md).itemwise().strata is None

    def test_an_output_without_strata_still_loads(self):
        md = self._two_classes()
        result = Outliers().evaluate(md).classwise(md)
        expected = result.itemwise().with_threshold(3.5).data()
        # An output pickled before these attributes existed loads with none of them set on the instance.
        for name in ("strata", "_populations", "score_input", "_rarity", "scored"):
            vars(result).pop(name, None)
        assert result.strata is None
        assert result.with_threshold(3.5).data().equals(expected)

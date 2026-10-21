"""``add_factors(overwrite=True)`` keeps the column it replaces, and every factor records its source name."""

import json
import math
import warnings
import zipfile
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.core import video_stats
from dataeval.flags import VideoStats
from tests.core.test_video_stats import _stream
from tests.embeddings.test_embeddings import MockDataset
from tests.metadata.test_add_factors_collisions import TestSourceIndexSplit, _images, _tracking
from tests.metadata.test_declared_aggregations import MEDIAN, _result, _video
from tests.metadata.test_structurers import _FrameTracks, _MOTTarget

_TRACK_SPEEDS = {"track_ids": [5, 9, 1], "item_index": [0, 0, 1], "speed": [0.5, 0.9, 0.1]}


def _unit(md: Metadata, name: str) -> list:
    return md.rows_at("unit")[name].to_list()


def _dataset(**columns: list) -> MockDataset:
    """Four classified images whose datum metadata holds the given columns."""
    rows = [{name: values[i] for name, values in columns.items()} for i in range(4)]
    return MockDataset(np.ones((4, 3, 3)), np.eye(3)[[0, 1, 2, 0]], rows)


def _encoding_warnings(read) -> list[str]:
    """Messages of every "given an encoding but is not a factor" warning ``read`` raises."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read()
    return [str(w.message) for w in caught if "given an encoding" in str(w.message)]


@pytest.mark.required
class TestOverwrite:
    def test_the_replaced_column_moves_to_original(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        before = md.rows_at("unit")["w"].dtype
        md.add_factors({"w": [5.0, 6.0, 7.0, 8.0]}, level="unit", overwrite=True)
        assert _unit(md, "w") == [5.0, 6.0, 7.0, 8.0]
        assert _unit(md, "w_original") == [1.0, 2.0, 3.0, 4.0]
        assert md.rows_at("unit")["w_original"].dtype == before
        assert {"w", "w_original"} <= set(md.factor_info)

    def test_the_moved_column_keeps_its_dtype(self):
        md = _images(n=[1, 2, 3, 4])
        md.add_factors({"f": [0.5, 1.5, 2.5, 3.5]}, level="unit")
        md.add_factors({"n": np.array([5, 6, 7, 8]), "f": [0.0] * 4}, level="unit", overwrite=True)
        assert md.rows_at("unit")["n_original"].dtype == pl.Int64
        assert md.rows_at("unit")["f_original"].dtype == pl.Float64
        assert _unit(md, "n_original") == [1, 2, 3, 4]

    def test_repeated_overwrites_count_up(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"w": [5.0, 6.0, 7.0, 8.0]}, level="unit", overwrite=True)
        md.add_factors({"w": [9.0, 9.0, 9.0, 9.0]}, level="unit", overwrite=True)
        assert _unit(md, "w") == [9.0, 9.0, 9.0, 9.0]
        assert _unit(md, "w_original") == [1.0, 2.0, 3.0, 4.0]
        assert _unit(md, "w_original_2") == [5.0, 6.0, 7.0, 8.0]

    def test_no_overwrite_is_unchanged(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"w": [5.0, 6.0, 7.0, 8.0]}, level="unit")
        assert _unit(md, "w") == [1.0, 2.0, 3.0, 4.0]
        assert _unit(md, "w_added") == [5.0, 6.0, 7.0, 8.0]
        assert "w_original" not in md.rows_at("unit").columns

    def test_a_new_name_moves_nothing_aside(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"v": [5.0, 6.0, 7.0, 8.0]}, level="unit", overwrite=True)
        assert "v_original" not in md.rows_at("unit").columns

    def test_a_column_at_an_ancestor_level_moves_aside_at_its_own(self):
        md = _tracking()
        md.add_factors({"weather": [1.0, 2.0]}, level="sequence")
        md.add_factors({"weather": [5.0, 6.0, 7.0]}, level="track", overwrite=True)
        assert md.factor_info["weather"].level == "track"
        assert md.factor_info["weather_original"].level == "sequence"
        assert md.rows_at("sequence")["weather_original"].to_list() == [1.0, 2.0]
        assert md.factor_data.shape[1] == len(md.factor_names)

    def test_a_column_at_another_level_moves_aside_at_its_own(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"x": np.arange(4.0)}, level="instance")
        md.add_factors({"x": np.zeros(4)}, level="unit", overwrite=True)
        assert md.factor_info["x"].level == "unit"
        assert md.factor_info["x_original"].level == "instance"
        assert md.rows_at("instance")["x_original"].to_list() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.required
class TestOnlyALosingWriteMovesAside:
    """A prior column is moved aside only when the write would lose a value it holds."""

    def test_rerunning_the_same_overwrite_adds_nothing(self):
        streams = [_stream(10), _stream(20)]
        targets = [_MOTTarget([_FrameTracks([0]) for _ in stream]) for stream in streams]
        dataset = MockDataset(streams, targets, [{"video_duration": 10 / 30}, {"video_duration": 1.0}])
        md = Metadata(dataset)
        stats = video_stats(dataset, stats=VideoStats.DESCRIPTIVE)
        md.add_factors(stats, overwrite=True)
        after_one = sorted(md.factor_info)
        assert md.add_factors(stats, overwrite=True, return_collisions=True) == {}
        md.add_factors(stats, overwrite=True)
        assert sorted(md.factor_info) == after_one
        assert "video_duration_original" in after_one

    def test_filling_empty_rows_moves_nothing(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"v": [1.0, None, math.nan, 4.0]}, level="unit")
        md.add_factors({"v": [1.0, 2.0, 3.0, 4.0]}, level="unit", overwrite=True)
        assert _unit(md, "v") == [1.0, 2.0, 3.0, 4.0]
        assert "v_original" not in md.rows_at("unit").columns

    def test_a_present_value_made_missing_moves(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors({"w": [1.0, None, 3.0, 4.0]}, level="unit", overwrite=True, return_collisions=True)
        assert _unit(md, "w_original") == [1.0, 2.0, 3.0, 4.0]
        # Missing on one side is not a disagreement, so the report is empty though the column moved.
        assert report == {}

    def test_a_present_value_made_nan_moves(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"w": [1.0, math.nan, 3.0, 4.0]}, level="unit", overwrite=True)
        assert _unit(md, "w_original") == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.required
class TestMovedColumnsAreKeptButNotAnalyzed:
    """A moved column is readable and described, and analyzed only when `include` names it."""

    def test_kept_and_described_but_not_analyzed(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"w": [5.0, 6.0, 7.0, 8.0]}, level="unit", overwrite=True)
        assert list(md.factor_names) == ["w"]
        assert md.factor_data.shape == (4, 1)
        assert md.factor_values.shape == (4, 1)
        assert md.factor_info["w_original"].source_name == "w"
        assert "w_original" not in md.encoding()
        assert _unit(md, "w_original") == [1, 2, 3, 4]

    def test_an_added_column_is_still_analyzed(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"w": [5.0, 6.0, 7.0, 8.0]}, level="unit")
        assert list(md.factor_names) == ["w", "w_added"]

    def test_an_excluded_factor_stays_out_when_overwritten(self):
        md = Metadata(_dataset(hour=[1.0, 2.0, 3.0, 4.0], w=[0.5, 1.5, 2.5, 3.5]), exclude=["hour"])
        md.add_factors({"hour": [4.0, 3.0, 2.0, 1.0]}, level="unit", overwrite=True)
        assert list(md.factor_names) == ["w"]
        assert "hour_original" in md.factor_info

    def test_a_class_axis_stays_out_when_overwritten(self):
        rows = [{"weather": ["sun", "rain", "fog"][i % 3], "hour": float(i) + 0.5} for i in range(12)]
        md = Metadata(MockDataset(np.ones((12, 3, 3)), np.eye(3)[[i % 3 for i in range(12)]], rows))
        axis = md.classed_by("weather")
        weather = axis.rows_at("unit")["weather"].to_list()
        weather[0] = "fog"
        axis.add_factors({"weather": weather}, level="unit", overwrite=True)
        assert "weather_original" not in axis.factor_names
        assert "weather_original" in axis.factor_info

    def test_naming_it_in_include_analyzes_it(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"w": [5.0, 6.0, 7.0, 8.0]}, level="unit", overwrite=True)
        md.include = ["w", "w_original"]
        assert list(md.factor_names) == ["w", "w_original"]
        assert md.factor_data.shape == (4, 2)
        assert "w_original" in md.encoding()

    @pytest.mark.parametrize("exclude", [[], ["other"]], ids=["cleared", "reset"])
    def test_changing_exclude_does_not_analyze_it(self, exclude):
        md = Metadata(_dataset(w=[1.0, 2.0, 3.0, 4.0]), exclude=["w"])
        md.add_factors({"w": [5.0, 6.0, 7.0, 8.0]}, level="unit", overwrite=True)
        md.exclude = exclude
        assert list(md.factor_names) == ["w"]

    def test_classed_by_refuses_it_and_names_the_way_in(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0], s=["a", "b", "a", "b"])
        md.add_factors({"s": ["b", "b", "a", "a"]}, level="unit", overwrite=True)
        with pytest.raises(ValueError, match="include"):
            md.classed_by("s_original")

    def test_survives_a_save_and_load(self, tmp_path):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"w": [5.0, 6.0, 7.0, 8.0]}, level="unit", overwrite=True)
        md.save(tmp_path / "m.dem")
        loaded = Metadata.load(tmp_path / "m.dem")
        assert list(loaded.factor_names) == ["w"]
        assert loaded.factor_info["w_original"].source_name == "w"
        assert _unit(loaded, "w_original") == [1, 2, 3, 4]


def _strict_letters() -> Metadata:
    """Eight images whose `s` holds a `c` the strict vocabulary `a`, `b` refuses."""
    rows = [{"s": ["a", "b", "c", "a"][i % 4], "w": float(i) + 0.5} for i in range(8)]
    dataset = MockDataset(np.ones((8, 3, 3)), np.eye(3)[[i % 3 for i in range(8)]], rows)
    return Metadata(dataset, factor_levels={"s": ["a", "b"]}, strict=True)


def _excluded_hours() -> Metadata:
    """Forty images with a continuous `hour`, excluded, and a categorical `weather`."""
    rng = np.random.default_rng(0)
    rows = [{"hour": float(rng.uniform(0, 24)), "weather": ["sun", "rain"][i % 2]} for i in range(40)]
    md = Metadata(MockDataset(np.ones((40, 3, 3)), np.eye(3)[[i % 3 for i in range(40)]], rows), exclude=["hour"])
    md.add_factors({"hour": md.rows_at("unit")["hour"].to_numpy() + 0.5}, level="unit", overwrite=True)
    return md


@pytest.mark.required
class TestDescribingAMovedColumnIsQuiet:
    """Describing a moved column never raises or warns; analyzing one, by name, is held to every rule."""

    def test_correcting_values_a_strict_vocabulary_refused(self):
        md = _strict_letters()
        with pytest.raises(ValueError, match="declared vocabulary"):
            md.factor_data  # noqa: B018
        md.add_factors({"s": ["a", "b", "a", "a"] * 2}, level="unit", overwrite=True)
        assert md.factor_data.shape == (8, 2)
        # Described without the strict check: the vocabulary, grown to hold what the column has.
        assert cast("Any", md.factor_info["s_original"].encoding).levels == ("a", "b", "c")
        assert cast("Any", md._encoding["s_original"]).levels == ("a", "b")

    def test_correcting_them_after_excluding_the_factor(self):
        md = _strict_letters()
        md.exclude = ["s"]
        md.add_factors({"s": ["a", "b", "a", "a"] * 2}, level="unit", overwrite=True)
        assert md.factor_data.shape == (8, 1)
        assert "s_original" in md.factor_info

    def test_copied_bins_that_cannot_cut_the_moved_values(self):
        rows = [{"x": ["lo", "hi", "lo", "mid"][i % 4], "w": float(i) + 0.5} for i in range(8)]
        dataset = MockDataset(np.ones((8, 3, 3)), np.eye(3)[[i % 3 for i in range(8)]], rows)
        md = Metadata(dataset, continuous_factor_bins={"x": 3})
        md.add_factors({"x": np.arange(8.0)}, level="unit", overwrite=True)
        assert md.factor_data.shape == (8, 2)
        assert md.factor_info["x_original"].factor_type == "categorical"

    def test_an_excluded_factor_overwritten_reads_without_a_warning(self):
        md = _excluded_hours()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert md.factor_data.shape == (40, 1)
            assert md.factor_info["hour_original"].factor_type == "continuous"

    def test_named_in_include_it_is_binned_and_announced_like_any_factor(self):
        md = _excluded_hours()
        md.include = ["weather", "hour_original"]
        with pytest.warns(UserWarning, match="hour_original"):
            assert md.factor_data.shape == (40, 2)

    def test_named_in_include_it_is_held_to_its_strict_vocabulary(self):
        md = _strict_letters()
        md.add_factors({"s": ["a", "b", "a", "a"] * 2}, level="unit", overwrite=True)
        md.include = ["s", "s_original"]
        with pytest.raises(ValueError, match="declared vocabulary"):
            md.factor_data  # noqa: B018


@pytest.mark.required
class TestANameAndItsOriginalInOneCall:
    @pytest.mark.parametrize("order", [("a", "a_original"), ("a_original", "a")])
    def test_the_prior_column_is_never_written_over(self, order):
        md = _images(a=[1.0, 2.0, 3.0, 4.0])
        values = {"a": [5.0, 6.0, 7.0, 8.0], "a_original": [9.0, 9.0, 9.0, 9.0]}
        md.add_factors({name: values[name] for name in order}, level="unit", overwrite=True)
        assert _unit(md, "a") == [5.0, 6.0, 7.0, 8.0]
        assert _unit(md, "a_original") == [9.0, 9.0, 9.0, 9.0]
        assert _unit(md, "a_original_2") == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.required
class TestMovedDeclarationsStayInThisInstance:
    """A moved column is a snapshot of this instance, so `new()` does not carry its copied declarations."""

    def test_declared_bins(self):
        md = Metadata(_dataset(hour=[1.0, 2.0, 3.0, 4.0]), continuous_factor_bins={"hour": 2})
        md.add_factors({"hour": [4.0, 3.0, 2.0, 1.0]}, level="unit", overwrite=True)
        assert "hour_original" in md.continuous_factor_bins
        fresh = md.new(_dataset(hour=[1.0, 2.0, 3.0, 4.0]))
        assert _encoding_warnings(lambda: fresh.factor_data) == []
        assert fresh.factor_data.shape == (4, 1)

    def test_declared_vocabulary(self):
        md = Metadata(_dataset(s=["a", "b", "a", "b"]), factor_levels={"s": ["b", "a"]})
        md.add_factors({"s": ["a", "a", "b", "b"]}, level="unit", overwrite=True)
        fresh = md.new(_dataset(s=["a", "b", "a", "b"]))
        assert _encoding_warnings(lambda: fresh.factor_data) == []

    def test_after_a_save_and_load(self, tmp_path):
        md = Metadata(_dataset(hour=[1.0, 2.0, 3.0, 4.0]), continuous_factor_bins={"hour": 2})
        md.add_factors({"hour": [4.0, 3.0, 2.0, 1.0]}, level="unit", overwrite=True)
        md.save(tmp_path / "m.dem")
        loaded = Metadata.load(tmp_path / "m.dem")
        assert loaded.factor_data.shape == (4, 1)
        fresh = loaded.new(_dataset(hour=[1.0, 2.0, 3.0, 4.0]))
        assert _encoding_warnings(lambda: fresh.factor_data) == []

    def test_a_dropped_moved_roll_up_takes_its_declarations_with_it(self):
        md = _video()
        md.add_factors(cast("Any", _result([10.0, 12.0, 14.0, 16.0, 18.0] * 2, [MEDIAN])), level="unit")
        md.continuous_factor_bins = {"pan_speed_median": 2}
        md.add_factors({"pan_speed_median": [1.0, 2.0]}, level="sequence", overwrite=True)
        changed = cast("Any", _result([100.0, 200.0, 300.0, 400.0, 500.0] * 2, [MEDIAN]))
        md.add_factors(changed, level="unit", overwrite=True)
        assert "pan_speed_median_original" not in md.continuous_factor_bins
        assert _encoding_warnings(lambda: md.factor_data) == []


@pytest.mark.required
class TestKeyedOverwrite:
    def test_named_rows_replaced_and_the_whole_prior_column_kept(self):
        md = _tracking()
        md.add_factors(_TRACK_SPEEDS, level="track", key="track_id")
        md.add_factors(
            {"track_ids": [9], "item_index": [0], "speed": [0.8]}, level="track", key="track_id", overwrite=True
        )
        assert md.rows_at("track")["speed"].to_list() == [0.5, 0.8, 0.1]
        assert md.rows_at("track")["speed_original"].to_list() == [0.5, 0.9, 0.1]

    def test_filling_empty_rows_moves_nothing_aside(self):
        """A fold replaces no value, so there is no prior column to keep."""
        md = _tracking()
        md.add_factors({"track_ids": [5, 9], "item_index": [0, 0], "speed": [0.5, 0.9]}, level="track", key="track_id")
        fill = {"track_ids": [1], "item_index": [1], "speed": [0.1]}
        md.add_factors(fill, level="track", key="track_id", overwrite=True)
        assert md.rows_at("track")["speed"].to_list() == [0.5, 0.9, 0.1]
        assert "speed_original" not in md.rows_at("track").columns


@pytest.mark.required
class TestBookkeepingFollowsTheMove:
    def test_a_declared_vocabulary_describes_both_columns(self):
        rows = [{"s": value} for value in ["a", "b", "a", "b"]]
        md = Metadata(MockDataset(np.ones((4, 3, 3)), np.eye(3)[[0, 1, 2, 0]], rows), factor_levels={"s": ["b", "a"]})
        md.add_factors({"s": ["a", "a", "b", "b"]}, level="unit", overwrite=True)
        assert cast("Any", md.factor_info["s_original"].encoding).levels == ("b", "a")
        assert md.encoding("s").levels == ("b", "a")

    def test_a_moved_roll_up_is_dropped_cleanly_when_its_source_changes(self):
        """The roll-up record moves with the column, so a changed source recomputes it under a new name."""
        md = _video()
        md.add_factors(cast("Any", _result([10.0, 12.0, 14.0, 16.0, 18.0] * 2, [MEDIAN])), level="unit")
        md.add_factors({"pan_speed_median": [1.0, 2.0]}, level="sequence", overwrite=True)
        changed = cast("Any", _result([100.0, 200.0, 300.0, 400.0, 500.0] * 2, [MEDIAN]))
        md.add_factors(changed, level="unit", overwrite=True)

        assert md.factor_data.shape[1] == len(md.factor_names)
        assert "pan_speed_median_original" not in md.factor_info
        sequences = md.rows_at("sequence")
        assert "pan_speed_median_original" not in sequences.columns
        assert sequences["pan_speed_median"].to_list() == [1.0, 2.0]
        (recomputed,) = (name for name in md.factor_names if name.startswith("pan_speed_median_agg"))
        assert sequences[recomputed].to_list() == [300.0, 300.0]

    def test_a_rolled_up_column_keeps_its_aggregated_from(self):
        md = _tracking().agg("instance", "unit", pl.len().alias("n"))
        md.add_factors({"n": np.zeros(md.level_counts["unit"])}, level="unit", overwrite=True)
        assert md.factor_info["n_original"].aggregated_from == "instance"
        assert md.factor_info["n"].aggregated_from is None


@pytest.mark.required
class TestSourceName:
    def test_a_dataset_factor_keeps_its_name(self):
        assert _images(w=[1.0, 2.0, 3.0, 4.0]).factor_info["w"].source_name == "w"

    def test_a_reserved_name_records_the_key_the_dataset_used(self):
        md = _images(level=[1.0, 2.0, 3.0, 4.0])
        assert md.factor_info["metadata_level"].source_name == "level"

    def test_a_nested_key_records_its_flattened_key(self):
        rows = [{"cam": {"level": float(i)}} for i in range(4)]
        md = Metadata(MockDataset(np.ones((4, 3, 3)), np.eye(3)[[0, 1, 2, 0]], rows))
        assert md.factor_info["metadata_level"].source_name == "level"

    def test_added_and_moved_columns(self):
        md = _images(level=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"level": [0.0] * 4}, level="unit")
        md.add_factors({"metadata_level": [5.0] * 4}, level="unit", overwrite=True)
        info = md.factor_info
        assert info["metadata_level_added"].source_name == "level"
        assert info["metadata_level"].source_name == "metadata_level"
        assert info["metadata_level_original"].source_name == "level"

    def test_a_source_index_split_records_the_placed_name(self):
        md = TestSourceIndexSplit()._md()
        md.add_factors(TestSourceIndexSplit._split([1.0, 2.0, 3.0, 9.0]))
        assert md.factor_info["unit_mean"].source_name == "unit_mean"
        assert md.factor_info["instance_mean_added"].source_name == "instance_mean"

    def test_a_roll_up_records_the_name_it_asked_for(self):
        md = _tracking()
        md.add_factors({"n": np.zeros(md.level_counts["unit"])}, level="unit")
        rolled = md.agg("instance", "unit", pl.len().alias("n"))
        renamed = next(name for name in rolled.factor_info if name.startswith("n_agg"))
        assert rolled.factor_info[renamed].source_name == "n"

    def test_a_structurer_derived_factor_keeps_its_name(self):
        assert _tracking().factor_info["track_length"].source_name == "track_length"


@pytest.mark.required
class TestExistingColumn:
    def test_without_overwrite_the_existing_values_keep_the_name(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors({"w": [0.0] * 4}, level="unit", return_collisions=True)
        assert report["w"].existing_column == "w"

    def test_with_overwrite_they_are_in_the_moved_column(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        first = md.add_factors({"w": [0.0] * 4}, level="unit", overwrite=True, return_collisions=True)
        second = md.add_factors({"w": [1.0] * 4}, level="unit", overwrite=True, return_collisions=True)
        assert first["w"].existing_column == "w_original"
        assert second["w"].existing_column == "w_original_2"


@pytest.mark.required
class TestPersistence:
    def test_source_names_survive_a_round_trip(self, tmp_path):
        md = _images(level=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"metadata_level": [5.0] * 4}, level="unit", overwrite=True)
        md.save(tmp_path / "m.dem")
        info = Metadata.load(tmp_path / "m.dem").factor_info
        assert info["metadata_level_original"].source_name == "level"
        assert info["metadata_level"].source_name == "metadata_level"

    def test_an_older_file_reads_each_column_as_its_own_source(self, tmp_path):
        _images(level=[1.0, 2.0, 3.0, 4.0]).save(tmp_path / "new.dem")
        old = tmp_path / "old.dem"
        with zipfile.ZipFile(tmp_path / "new.dem") as source, zipfile.ZipFile(old, "w") as target:
            for name in source.namelist():
                data = source.read(name)
                if name == "manifest.json":
                    manifest = json.loads(data)
                    del manifest["source_names"]
                    data = json.dumps(manifest)
                target.writestr(name, data)
        assert Metadata.load(old).factor_info["metadata_level"].source_name == "metadata_level"

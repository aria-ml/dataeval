from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from dataeval.config import use_seed
from dataeval.core import combination_scores
from dataeval.core._combination_scores import _numeric, _windows


def _reason(result, row):
    return result["reasons"][result["reason"][row]]


def _frame():
    return pl.DataFrame({
        "a": ["p", "q"],
        "s": ["u", "v"],
        "x": [1.0, 2.0],
        "f": [0.5, 1.5],
        "item_index": [0, 1],
        "v": [[1.0], [2.0]],
        "tags": [["n"], ["r"]],
    })


@pytest.mark.required
class TestRarity:
    def test_a_lone_snow_frame_counts_one(self):
        data = {"weather": ["clear"] * 299 + ["snow"]}
        result = combination_scores(data, groups=[("weather",)])
        assert result["methods"] == ("rarity",)
        assert result["dims"] == (1,)
        assert result["score"][-1] == 1
        assert result["n"][-1] == 300
        assert result["observed"][result["combination"][-1]] == "weather=snow"
        assert np.isnan(result["tail"]).all()

    def test_rarity_is_counted_within_each_video(self):
        data = {"item_index": [0] * 300 + [1] * 300, "weather": ["clear"] * 299 + ["snow"] * 301}
        result = combination_scores(data, groups=[("weather",)], strata="item_index")
        assert result["score"][299] == 1
        assert result["n"][299] == 300
        assert result["strata"][result["stratum"][299]] == "item_index=0"
        assert set(result["score"][300:]) == {300}

    def test_three_wrong_sensor_frames_count_three(self):
        data = {"sensor_id": ["front"] * 297 + ["rear"] * 3}
        result = combination_scores(data, groups=[("sensor_id",)])
        assert result["score"][-1] == 3

    def test_marginal_separates_a_rare_combination_from_a_rare_value(self):
        weather = ["clear"] * 50 + ["rain"] * 49 + ["rain"]
        time = ["day"] * 50 + ["night"] * 49 + ["day"]
        result = combination_scores({"weather": weather, "time": time}, groups=[("weather", "time")])
        assert result["score"][-1] == 1
        assert result["marginal"][-1] == 50
        assert result["observed"][result["combination"][-1]] == "weather=rain, time=day"

    def test_null_is_a_level(self):
        result = combination_scores({"sensor_id": ["a"] * 297 + [None] * 3}, groups=[("sensor_id",)])
        assert result["score"][-1] == 3
        assert result["observed"][result["combination"][-1]] == "sensor_id=null"

    def test_tags_are_counted_as_exact_sets(self):
        tags = [["night", "rain"]] * 50 + [["rain", "night", "night"]] + [["a|b"]] + [["a", "b"]]
        result = combination_scores({"tags": tags}, groups=[("tags",)])
        assert result["score"][50] == 51
        assert result["score"][51] == 1
        assert result["score"][52] == 1
        labels = {result["observed"][code] for code in result["combination"][51:]}
        assert labels == {"tags='a|b'", "tags=a|b"}

    def test_an_empty_tag_list_is_the_empty_set_and_a_null_list_is_missing(self):
        tags = pl.DataFrame({"tags": [[], [], None]}, schema={"tags": pl.List(pl.String)})
        result = combination_scores(tags, groups=[("tags",)])
        assert result["score"][0] == 2
        assert result["observed"][result["combination"][0]] == "tags={}"
        assert np.isnan(result["score"][2])
        assert _reason(result, 2) == "missing value"

    def test_nan_in_a_categorical_float_column_is_the_null_level(self):
        data = {"code": np.array([1.0] * 10 + [np.nan] * 5)}
        result = combination_scores(data, groups=[("code",)], kinds={"code": "categorical"})
        assert result["score"][10:].tolist() == [5.0] * 5
        assert {result["observed"][code] for code in result["combination"][10:]} == {"code=null"}

    def test_a_null_stratum_is_reported_in_every_test(self):
        data = {"w": ["a", "a", "b"], "t": ["c", "c", "c"], "s": [1, None, 1]}
        result = combination_scores(data, groups=[("w",), ("t",)], strata="s")
        assert _reason(result, 1) == "missing stratum"
        assert result["reasons"][result["reason"][4]] == "missing stratum"
        assert result["strata"][result["stratum"][1]] == "<missing>"
        assert result["n"][0] == 2

    def test_rows_are_ordered_by_group_then_row(self):
        result = combination_scores({"w": ["a", "b"], "t": ["c", "d"]}, groups=[("w",), ("t",)])
        assert result["row"].tolist() == [0, 1, 0, 1]
        assert result["group"].tolist() == [0, 0, 1, 1]


@pytest.mark.required
class TestPlanning:
    def test_pair_search_skips_reserved_and_unread_columns(self):
        data = pl.DataFrame({"a": ["p", "q"], "s": ["u", "v"], "item_index": [0, 1], "v": [[1.0], [2.0]]})
        result = combination_scores(data)
        assert result["groups"] == (("a", "s"),)
        assert result["kinds"] == {"a": "categorical", "s": "categorical"}

    def test_empty_data_returns_empty_arrays(self):
        result = combination_scores(pl.DataFrame(schema={"a": pl.String, "b": pl.String}))
        assert result["groups"] == (("a", "b"),)
        assert result["row"].size == 0
        assert result["score"].size == 0

    def test_a_dict_and_a_frame_agree(self):
        data = {"w": ["a", "b", "a"], "t": ["c", "c", "d"]}
        first = combination_scores(data, groups=[("w", "t")])
        second = combination_scores(pl.DataFrame(data), groups=[("w", "t")])
        np.testing.assert_array_equal(first["score"], second["score"])
        assert first["observed"] == second["observed"]


@pytest.mark.required
class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"groups": [("nope",)]}, "not a column"),
            ({"groups": [("a",)], "strata": "nope"}, "not a column"),
            ({"groups": [("item_index",)]}, "reserved"),
            ({"groups": [], "given": {"x": "item_index"}}, "reserved"),
            ({"groups": [("a", "s")], "strata": "s"}, "strata"),
            ({"groups": [("a", "x")]}, "mixes numeric and categorical"),
            ({"groups": [()]}, "at least one column"),
            ({"groups": [("a", "a")]}, "each once"),
            ({"groups": [], "given": {"x": "x"}}, "two different numeric"),
            ({"groups": [], "given": {"x": "a"}}, "two different numeric"),
            ({"groups": [("a",)], "kinds": {"a": "vector"}}, "cannot be read as"),
            ({"groups": [("a",)], "kinds": {"a": "shade"}}, "cannot be read as"),
            ({"groups": [("a",)], "strata": "f"}, "continuous"),
            ({"groups": [("a",)], "strata": "v"}, "list or unread"),
            ({"groups": [("a",)], "strata": "tags"}, "list or unread"),
            ({"groups": [("v",)]}, "needs kinds="),
            ({"groups": [("a",)], "kinds": {"item_index": "categorical"}}, "reserved"),
        ],
    )
    def test_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            combination_scores(_frame(), **kwargs)

    @pytest.mark.parametrize("stamp", [datetime(2026, 1, 1), timedelta(seconds=1)])
    def test_a_datetime_or_duration_stratum_is_refused(self, stamp):
        data = {"a": ["p", "q"], "when": [stamp, stamp]}
        with pytest.raises(ValueError, match="continuous"):
            combination_scores(data, groups=[("a",)], strata="when")

    def test_a_bare_string_is_not_a_group(self):
        with pytest.raises(TypeError):
            combination_scores(_frame(), groups=("a", "s"))
        with pytest.raises(TypeError):
            combination_scores(_frame(), groups="as")

    def test_nothing_to_search(self):
        with pytest.raises(ValueError, match="No pair"):
            combination_scores({"a": ["x", "y"]})

    def test_nan_in_a_categorical_float_stratum_is_a_missing_stratum(self):
        data = {"a": ["p"] * 15, "f": [1.0] * 10 + [np.nan] * 5}
        result = combination_scores(data, groups=[("a",)], strata="f", kinds={"f": "categorical"})
        assert [_reason(result, row) for row in range(10, 15)] == ["missing stratum"] * 5
        assert result["score"][:10].tolist() == [10.0] * 10

    def test_a_float_column_declared_categorical_may_be_a_stratum(self):
        result = combination_scores(_frame(), groups=[("a",)], strata="f", kinds={"f": "categorical"})
        assert result["strata"] == ("f=0.5", "f=1.5")


@pytest.mark.required
class TestMahalanobis:
    def test_a_combination_outlier_scores_highest_with_a_low_marginal(self):
        rng = np.random.default_rng(0)
        size = rng.normal(0, 1, 400)
        offset = -size + rng.normal(0, 0.2, 400)
        size[-1], offset[-1] = 1.5, 1.5
        with use_seed(0):
            result = combination_scores({"size": size, "offset": offset}, groups=[("size", "offset")])
        assert result["methods"] == ("mahalanobis",)
        assert result["dims"] == (2,)
        assert np.argmax(result["score"]) == 399
        assert result["marginal"][-1] < 2.5
        assert result["tail"][-1] < 1e-6

    def test_strata_split_the_fit(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, (100, 2))
        b = rng.normal(10, 1, (100, 2))
        b[0] = [0.0, 0.0]
        data = {"x": np.r_[a[:, 0], b[:, 0]], "y": np.r_[a[:, 1], b[:, 1]], "cls": ["a"] * 100 + ["b"] * 100}
        with use_seed(0):
            result = combination_scores(data, groups=[("x", "y")], strata="cls")
        assert np.argmax(result["score"]) == 100
        assert result["n"][0] == 100

    def test_a_vector_and_a_compositional_vector(self):
        rng = np.random.default_rng(2)
        counts = rng.poisson([20, 10, 5, 2], (200, 4)).astype(float)
        counts[-1] = [2, 5, 10, 20]
        shares = counts / counts.sum(axis=1, keepdims=True)
        with use_seed(0):
            plain = combination_scores({"hist": counts.tolist()}, groups=[("hist",)], kinds={"hist": "vector"})
            closed = combination_scores({"hist": shares.tolist()}, groups=[("hist",)], kinds={"hist": "vector"})
        assert plain["dims"] == (4,)
        assert np.argmax(plain["score"]) == 199
        assert closed["dims"] == (3,)
        assert np.argmax(closed["score"]) == 199

    def test_vectors_of_different_lengths_are_refused(self):
        with pytest.raises(ValueError, match="one length"):
            combination_scores({"v": [[1.0, 2.0], [1.0]]}, groups=[("v",)], kinds={"v": "vector"})

    def test_nan_inf_and_null_are_missing(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=60).tolist()
        y = rng.normal(size=60).tolist()
        x[0], x[1], y[2] = np.nan, np.inf, None
        with use_seed(0):
            result = combination_scores({"x": x, "y": y}, groups=[("x", "y")])
        assert [_reason(result, row) for row in range(3)] == ["missing value"] * 3
        assert result["n"][3] == 57

    def test_a_vector_with_a_missing_element_is_missing(self):
        rng = np.random.default_rng(7)
        rows = rng.normal(size=(40, 2)).tolist()
        rows[0] = [1.0, None]
        result = combination_scores({"v": rows}, groups=[("v",)], kinds={"v": "vector"})
        assert _reason(result, 0) == "missing value"

    def test_an_empty_vector_is_missing_and_the_others_are_scored(self):
        rows = np.random.default_rng(9).normal(size=(40, 2)).tolist()
        rows[0] = []
        result = combination_scores({"v": rows}, groups=[("v",)], kinds={"v": "vector"})
        assert _reason(result, 0) == "missing value"
        assert np.isfinite(result["score"][1:]).all()

    def test_nan_and_inf_inside_a_vector_are_missing(self):
        rows = np.random.default_rng(10).normal(size=(40, 2)).tolist()
        rows[0], rows[1] = [1.0, np.nan], [np.inf, 1.0]
        result = combination_scores({"v": rows}, groups=[("v",)], kinds={"v": "vector"})
        assert [_reason(result, row) for row in range(2)] == ["missing value"] * 2
        assert result["n"][2] == 38

    def test_too_few_rows(self):
        result = combination_scores({"x": [1.0, 2.0, 3.0], "y": [3.0, 1.0, 2.0]}, groups=[("x", "y")])
        assert {_reason(result, row) for row in range(3)} == {"too few rows"}

    def test_exact_fit(self):
        rng = np.random.default_rng(3)
        gaps = np.where(rng.random(300) < 0.7, 0, rng.poisson(3, 300)).astype(float)
        result = combination_scores({"gaps": gaps, "speed": rng.normal(size=300)}, groups=[("gaps", "speed")])
        assert {_reason(result, row) for row in range(300)} == {"exact fit"}

    def test_degenerate_covariance(self):
        x = np.random.default_rng(4).normal(size=100)
        result = combination_scores({"x": x, "y": 2 * x}, groups=[("x", "y")])
        assert {_reason(result, row) for row in range(100)} == {"degenerate covariance"}

    def test_the_tail_starts_at_fifty_rows_per_dimension(self):
        x = np.random.default_rng(0).standard_normal((100, 2))
        with use_seed(0):
            below = combination_scores({"a": x[:99, 0], "b": x[:99, 1]}, groups=[("a", "b")])
            at = combination_scores({"a": x[:, 0], "b": x[:, 1]}, groups=[("a", "b")])
        assert np.isnan(below["tail"]).all()
        assert np.isfinite(below["score"]).all()
        assert np.isfinite(at["tail"]).all()

    def test_exact_fit_at_the_support_size(self):
        x = np.r_[np.zeros(6), np.random.default_rng(0).normal(size=4) + 5]
        result = combination_scores({"x": x}, groups=[("x",)])
        assert {_reason(result, row) for row in range(10)} == {"exact fit"}

    def test_a_vector_column_of_nulls_is_missing(self):
        data = pl.DataFrame({"v": pl.Series([None] * 40, dtype=pl.List(pl.Float64))})
        result = combination_scores(data, groups=[("v",)], kinds={"v": "vector"})
        assert {_reason(result, row) for row in range(40)} == {"missing value"}

    def test_a_stratum_of_missing_values_is_not_scored(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=80).tolist()
        x[40:] = [None] * 40
        data = {"x": x, "y": rng.normal(size=80), "cls": ["a"] * 40 + ["b"] * 40}
        result = combination_scores(data, groups=[("x", "y")], strata="cls")
        assert {_reason(result, row) for row in range(40, 80)} == {"missing value"}
        assert np.isfinite(result["score"][:40]).all()

    def test_pair_search_keeps_families_apart_and_skips_reserved_columns(self):
        rng = np.random.default_rng(5)
        data = {
            "x": rng.normal(size=60),
            "y": rng.normal(size=60),
            "w": ["a", "b"] * 30,
            "t": ["c"] * 60,
            "track_id": np.arange(60),
        }
        with use_seed(0):
            result = combination_scores(data)
        assert result["groups"] == (("x", "y"), ("w", "t"))
        assert result["methods"] == ("mahalanobis", "rarity")

    def test_datetimes_keep_their_precision(self):
        stamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(60).astype("timedelta64[ns]")
        assert _numeric(pl.DataFrame({"t": stamps}), "t").tolist() == list(range(60))


@pytest.mark.required
class TestResidual:
    @staticmethod
    def _daylight(rng):
        hour = rng.integers(0, 24, 2000).astype(float)
        brightness = 0.5 + 0.4 * np.cos(2 * np.pi * (hour - 12) / 24) + rng.normal(0, 0.03, 2000)
        return np.r_[hour, 2.0], np.r_[brightness, 0.9]

    def test_a_bright_frame_at_two_in_the_morning_is_the_most_unusual(self):
        hour, brightness = self._daylight(np.random.default_rng(0))
        result = combination_scores({"brightness": brightness, "hour": hour}, groups=[], given={"brightness": "hour"})
        assert result["methods"] == ("residual",)
        assert result["groups"] == (("brightness", "hour"),)
        assert np.argmax(np.abs(result["score"])) == 2000
        assert result["score"][-1] > 0
        assert np.isfinite(result["marginal"]).all()
        assert np.isnan(result["tail"]).all()

    def test_a_dark_frame_at_noon_scores_negative(self):
        hour, brightness = self._daylight(np.random.default_rng(1))
        hour[-1], brightness[-1] = 12.0, 0.1
        result = combination_scores({"brightness": brightness, "hour": hour}, groups=[], given={"brightness": "hour"})
        assert np.argmax(np.abs(result["score"])) == 2000
        assert result["score"][-1] < 0

    def test_constant_residual(self):
        result = combination_scores({"y": [1.0] * 30, "x": np.arange(30.0)}, groups=[], given={"y": "x"})
        assert {_reason(result, row) for row in range(30)} == {"constant residual"}

    def test_too_few_rows(self):
        result = combination_scores({"y": np.arange(10.0), "x": np.arange(10.0)}, groups=[], given={"y": "x"})
        assert {_reason(result, row) for row in range(10)} == {"too few rows"}
        assert np.isnan(result["marginal"]).all()

    def test_a_missing_conditioner_is_a_missing_value(self):
        x = np.arange(40.0)
        x[0] = np.nan
        result = combination_scores({"y": np.sin(np.arange(40.0)), "x": x}, groups=[], given={"y": "x"})
        assert _reason(result, 0) == "missing value"
        assert result["n"][1] == 39

    def test_strata_are_scored_separately(self):
        x = np.r_[np.arange(40.0), np.arange(25.0)]
        data = {"y": np.sin(x), "x": x, "cls": ["a"] * 40 + ["b"] * 25}
        result = combination_scores(data, groups=[], given={"y": "x"}, strata="cls")
        assert set(result["n"][:40]) == {40}
        assert set(result["n"][40:]) == {25}
        assert np.isfinite(result["score"]).all()

    def test_each_window_holds_its_nearest_rows_without_the_row(self):
        windows = _windows(12, 10)
        assert windows.shape == (12, 10)
        assert all(row not in windows[row] for row in range(12))
        assert (np.diff(windows, axis=1) > 0).all()
        assert windows[0].tolist() == list(range(1, 11))
        assert windows[11].tolist() == list(range(1, 11))

    def test_an_interior_window_of_odd_width_is_centered_on_the_row(self):
        assert _windows(121, 11)[60].tolist() == [55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66]

    def test_a_block_range_matches_the_same_rows_of_the_whole(self):
        assert (_windows(40, 10, 13, 29) == _windows(40, 10)[13:29]).all()

    def test_blocking_does_not_change_the_result(self, monkeypatch):
        hour, brightness = self._daylight(np.random.default_rng(0))
        data = {"brightness": brightness, "hour": hour}
        whole = combination_scores(data, groups=[], given={"brightness": "hour"})
        monkeypatch.setattr("dataeval.core._combination_scores._BLOCK_ROWS", 7)
        blocked = combination_scores(data, groups=[], given={"brightness": "hour"})
        for field in ("score", "marginal", "reason"):
            np.testing.assert_array_equal(blocked[field], whole[field])


@pytest.mark.required
class TestDistribution:
    @staticmethod
    def _tracks(rng):
        speeds = [rng.normal(5, 1, rng.integers(3, 200)).tolist() for _ in range(300)]
        speeds.append(np.r_[rng.normal(2, 0.3, 60), rng.normal(8, 0.3, 60)].tolist())
        return speeds

    def test_a_bimodal_track_outranks_short_ordinary_ones(self):
        speeds = self._tracks(np.random.default_rng(0))
        with use_seed(0):
            result = combination_scores({"speeds": speeds}, groups=[("speeds",)], kinds={"speeds": "sample"})
        assert result["methods"] == ("distribution",)
        assert np.argmax(result["score"]) == 300
        assert np.isnan(result["tail"]).all()
        assert np.isnan(result["marginal"]).all()
        lengths = np.array([len(sample) for sample in speeds[:300]])
        score = result["score"][:300]
        assert 0.7 < np.median(score[lengths < 15]) / np.median(score[lengths >= 15]) < 1.4

    def test_a_sample_joins_a_numeric_group_as_one_dimension(self):
        rng = np.random.default_rng(1)
        data = {"speeds": self._tracks(rng), "duration": rng.normal(100, 10, 301)}
        with use_seed(0):
            result = combination_scores(data, groups=[("speeds", "duration")], kinds={"speeds": "sample"})
        assert result["methods"] == ("mahalanobis",)
        assert result["dims"] == (2,)
        assert np.argmax(result["score"]) == 300
        assert np.isnan(result["tail"]).all()

    def test_missing_elements_are_dropped_and_empty_samples_are_missing(self):
        result = combination_scores({"s": [[1.0, None], [], None, [np.inf]]}, groups=[("s",)], kinds={"s": "sample"})
        assert [_reason(result, row) for row in range(4)] == [
            "too few rows",
            "missing value",
            "missing value",
            "missing value",
        ]

    def test_identical_constant_samples_score_zero(self):
        result = combination_scores({"s": [[3.0, 3.0]] * 5}, groups=[("s",)], kinds={"s": "sample"})
        assert result["score"].tolist() == [0.0] * 5


@pytest.mark.required
class TestEpicCases:
    """The cases &393 names, each the most unusual row of its stratum by its method's direction."""

    def test_a_large_box_near_the_horizon_within_its_class(self):
        rng = np.random.default_rng(0)
        size = rng.normal(0, 1, 400)
        offset = np.where(np.arange(400) < 200, -size, size) + rng.normal(0, 0.2, 400)
        cls = ["car"] * 200 + ["sign"] * 200
        size[199], offset[199] = 1.5, 1.5  # a car: large boxes sit low, so this one is odd for cars, ordinary for signs
        with use_seed(0):
            result = combination_scores(
                {"rel_size": size, "rel_offset_y": offset, "class_name": cls},
                groups=[("rel_size", "rel_offset_y")],
                strata="class_name",
            )
        assert np.argmax(result["score"][:200]) == 199
        assert result["marginal"][199] < 2.5
        assert result["strata"][result["stratum"][199]] == "class_name=car"

    def test_a_fast_track_that_wanders(self):
        rng = np.random.default_rng(2)
        speed = rng.normal(5, 1, 300)
        straightness = np.clip(0.3 + 0.08 * speed + rng.normal(0, 0.03, 300), 0, 1)
        speed[-1], straightness[-1] = 7.0, 0.62
        with use_seed(0):
            result = combination_scores(
                {"mean_speed": speed, "straightness_index": straightness},
                groups=[("mean_speed", "straightness_index")],
            )
        assert np.argmax(result["score"]) == 299
        assert result["marginal"][-1] < 2.5


@pytest.mark.required
class TestTailCalibration:
    @pytest.mark.parametrize(("n", "d"), [(100, 2), (250, 5), (500, 10)])
    def test_clean_normal_data_rarely_has_a_small_tail(self, n, d):
        rng = np.random.default_rng(0)
        names = tuple(f"c{i}" for i in range(d))
        rates = []
        with use_seed(0):
            for _ in range(200):
                x = rng.standard_normal((n, d))
                result = combination_scores(dict(zip(names, x.T, strict=True)), groups=[names])
                rates.append(np.mean(result["tail"] < 0.01))
        assert 0.005 <= np.mean(rates) <= 0.02


@pytest.mark.required
class TestDeterminism:
    def test_two_runs_under_one_seed_agree(self):
        rng = np.random.default_rng(7)
        data = {
            "x": rng.normal(size=80),
            "y": rng.normal(size=80),
            "w": rng.choice(["a", "b"], 80).tolist(),
            "s": [rng.normal(size=int(m)).tolist() for m in rng.integers(3, 30, 80)],
        }
        runs = []
        for _ in range(2):
            with use_seed(3):
                runs.append(
                    combination_scores(
                        pl.DataFrame(data), groups=[("x", "y"), ("w",), ("s", "x")], kinds={"s": "sample"}
                    )
                )
        for key in ("score", "tail", "marginal", "n", "reason", "combination"):
            np.testing.assert_array_equal(runs[0][key], runs[1][key])

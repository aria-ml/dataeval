"""Unusual combinations of metadata factors in Outliers: combinations=, given=, kinds= (#1542)."""

from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest

import dataeval.quality._outliers_scores as scores_module
from dataeval import Metadata
from dataeval.config import use_seed
from dataeval.core._combination_scores import REASONS
from dataeval.quality import Outliers
from dataeval.types import SourceIndex
from dataeval.utils.thresholds import ConstantThreshold, ZScoreThreshold
from tests.conftest import _get_dataset
from tests.metadata.test_structurers import _mot_dataset


def _od(count, boxes):
    """An object detection dataset of `count` images with `boxes` boxes each."""
    dataset: Any = _get_dataset(count, boxes)
    return dataset


def _horizon():
    """The core test's horizon boxes, 200 cars then 200 signs, on 200 images of two boxes each."""
    rng = np.random.default_rng(0)
    size = rng.normal(0, 1, 400)
    offset = np.where(np.arange(400) < 200, -size, size) + rng.normal(0, 0.2, 400)
    size[199], offset[199] = 1.5, 1.5  # a car: large boxes sit low, so this one is odd for cars, ordinary for signs
    md = Metadata(_od(200, 2))
    md.add_factors(
        {"rel_size": size, "rel_offset_y": offset, "vehicle": ["car"] * 200 + ["sign"] * 200}, level=md.label_level
    )
    return md


def _daylight():
    """The core test's brightness by hour, with a bright frame at 02:00 last."""
    rng = np.random.default_rng(0)
    hour = rng.integers(0, 24, 2000).astype(float)
    brightness = 0.5 + 0.4 * np.cos(2 * np.pi * (hour - 12) / 24) + rng.normal(0, 0.03, 2000)
    return Metadata.from_factors({"brightness": np.r_[brightness, 0.9], "hour": np.r_[hour, 2.0]})


def _tracks():
    """The core test's 300 tracks, the last fast and wandering."""
    rng = np.random.default_rng(2)
    speed = rng.normal(5, 1, 300)
    straightness = np.clip(0.3 + 0.08 * speed + rng.normal(0, 0.03, 300), 0, 1)
    speed[-1], straightness[-1] = 7.0, 0.62
    md = Metadata(_mot_dataset([[300]]))
    md.add_factors({"mean_speed": speed, "straightness_index": straightness}, level="track")
    return md


def _correlated(n=200, seed=0, names=("a", "b")):
    """Two correlated columns with the last row off the line, plus an unrelated pair."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = -a + rng.normal(0, 0.2, n)
    a[-1], b[-1] = 1.5, 1.5
    return {names[0]: a, names[1]: b, "c": rng.normal(0, 1, n), "d": rng.normal(0, 1, n)}


def _rows(result, method):
    # Categoricals as strings: polars 1.0 compares two results' local categoricals unequal even when their values match.
    frame = result.data().with_columns(cs.categorical().cast(pl.Utf8))
    return frame.filter(pl.col("method") == method) if "method" in frame.columns else frame.clear()


def _checks(result):
    """The combination checks a result lists, as (level, metric_name, method)."""
    listed = result.populations.filter(pl.col("method").is_in(["mahalanobis", "residual", "distribution"]))
    return set(listed.select("level", "metric_name", "method").iter_rows())


class _Counting:
    """Count the calls to core's combination_scores, keeping each call's keyword arguments."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = scores_module.combination_scores

        def counted(*args, **kwargs):
            self.calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(scores_module, "combination_scores", counted)


@pytest.mark.required
class TestCombinations:
    def test_a_large_box_near_the_horizon_is_flagged_and_neither_column_alone(self):
        md = _horizon()
        with use_seed(0):
            result = Outliers().evaluate(
                md, levels="target", strata=["vehicle"], combinations=[("rel_size", "rel_offset_y")]
            )
        frame = result.data()
        planted = frame.filter((pl.col("item_index") == 99) & (pl.col("target_index") == 1))
        assert planted.select("metric_name", "method", "direction", "population").rows() == [
            ("rel_size, rel_offset_y", "mahalanobis", "upper", "vehicle=car")
        ]

    def test_a_bright_frame_at_night_is_flagged_by_the_residual(self):
        with use_seed(0):
            result = Outliers().evaluate(_daylight(), given={"brightness": "hour"})
        planted = result.data().filter(pl.col("item_index") == 2000)
        assert planted.select("metric_name", "method", "direction").rows() == [
            ("brightness given hour", "residual", "upper")
        ]
        assert "population" not in planted.columns

    def test_a_probable_id_switch_is_flagged(self):
        with use_seed(0):
            result = Outliers().evaluate(_tracks(), levels="track", combinations=[("mean_speed", "straightness_index")])
        rows = _rows(result, "mahalanobis").filter(pl.col("target_index") == 299)
        assert rows.select("level", "metric_name", "direction").rows() == [
            ("track", "mean_speed, straightness_index", "upper")
        ]

    def test_a_combination_below_ten_rows_per_dimension_is_listed(self):
        def listed(n):
            rng = np.random.default_rng(0)
            md = Metadata.from_factors({name: rng.normal(0, 1, n) for name in "xyz"})
            with use_seed(0):
                result = Outliers().evaluate(md, combinations=[("x", "y", "z")])
            check = result.populations.filter(pl.col("method") == "mahalanobis")
            return check.select("metric_name", "n", "judged", "reason").rows()

        assert listed(29) == [("x, y, z", 29, False, "below 30 rows")]
        assert listed(30) == [("x, y, z", 30, True, None)]

    def test_an_exact_fit_reports_the_floor_reason_at_fifteen_rows(self):
        md = Metadata.from_factors({"gaps": np.r_[np.zeros(10), np.arange(1.0, 6.0)], "speed": np.arange(15.0)})
        result = Outliers().evaluate(md, combinations=[("gaps", "speed")])
        check = result.populations.filter(pl.col("method") == "mahalanobis")
        assert check.select("n", "judged", "reason").rows() == [(15, False, "below 20 rows")]

    @pytest.mark.parametrize(
        ("check", "metric_name", "method"),
        [({"combinations": [("x", "y")]}, "x, y", "mahalanobis"), ({"given": {"x": "y"}}, "x given y", "residual")],
    )
    def test_a_combination_with_no_scored_row_lists_its_population(self, check, metric_name, method):
        rng = np.random.default_rng(0)
        md = Metadata.from_factors({"x": rng.normal(size=100), "y": np.full(100, np.nan)})
        result = Outliers().evaluate(md, **check)
        assert _rows(result, method).is_empty()
        listed = result.populations.filter(pl.col("method") == method).select("metric_name", "n", "judged", "reason")
        assert listed.rows() == [(metric_name, 0, False, "below 20 rows")]

    def test_an_exact_fit_lists_its_population(self):
        rng = np.random.default_rng(3)
        gaps = np.where(rng.random(300) < 0.7, 0, rng.poisson(3, 300)).astype(float)
        md = Metadata.from_factors({"gaps": gaps, "speed": rng.normal(size=300)})
        result = Outliers().evaluate(md, combinations=[("gaps", "speed")])
        assert _rows(result, "mahalanobis").is_empty()
        check = result.populations.filter(pl.col("method") == "mahalanobis")
        assert check.select("metric_name", "n", "judged", "reason").rows() == [("gaps, speed", 300, False, "exact fit")]

    def test_with_threshold_does_not_rescore(self, monkeypatch):
        counting = _Counting(monkeypatch)
        md = Metadata.from_factors(_correlated())
        with use_seed(0):
            result = Outliers().evaluate(md, combinations=[("a", "b")])
        assert len(counting.calls) == 1
        loosened = result.with_threshold(ConstantThreshold(upper=1e9))
        tightened = loosened.with_threshold({"a, b": ConstantThreshold(upper=0.0)})
        assert len(counting.calls) == 1
        assert _rows(loosened, "mahalanobis").height == _rows(result, "mahalanobis").height
        assert _rows(tightened, "mahalanobis").height == 200

    def test_stratified_rescores(self, monkeypatch):
        counting = _Counting(monkeypatch)
        factors = _correlated()
        md = Metadata.from_factors({**factors, "weather": ["rain", "clear"] * 100})
        with use_seed(0):
            result = Outliers().evaluate(md, combinations=[("a", "b")])
            stratified = result.stratified(md, "weather")
            direct = Outliers().evaluate(md, combinations=[("a", "b")], strata=["weather"])
        assert len([call for call in counting.calls if call.get("strata") is not None]) >= 2
        assert _rows(stratified, "mahalanobis").equals(_rows(direct, "mahalanobis"))
        assert stratified.populations.equals(direct.populations)
        assert set(stratified.populations.filter(pl.col("method") == "mahalanobis")["population"].cast(pl.Utf8)) == {
            "weather=rain",
            "weather=clear",
        }
        calls = len(counting.calls)
        assert _rows(stratified.itemwise(), "mahalanobis").equals(_rows(result, "mahalanobis"))
        assert len(counting.calls) > calls

    def test_a_limited_threshold_leaves_the_scores_on_the_default(self):
        md = Metadata.from_factors(_correlated())
        with use_seed(0):
            default = Outliers().evaluate(md, combinations=[("a", "b")])
            limited = Outliers(outlier_threshold=ZScoreThreshold(1.0, upper_limit=1e9)).evaluate(
                md, combinations=[("a", "b")]
            )
            unlimited = Outliers(outlier_threshold=ZScoreThreshold(1.0)).evaluate(md, combinations=[("a", "b")])
        assert _rows(limited, "mahalanobis").equals(_rows(default, "mahalanobis"))
        assert _rows(unlimited, "mahalanobis").height > _rows(default, "mahalanobis").height

    def test_a_mapping_threshold_overrides_one_check(self):
        md = Metadata.from_factors(_correlated())
        with use_seed(0):
            result = Outliers(outlier_threshold={"a, b": ConstantThreshold(upper=0.0)}).evaluate(
                md, combinations=[("a", "b"), ("c", "d")]
            )
            default = Outliers().evaluate(md, combinations=[("c", "d")])
        rows = _rows(result, "mahalanobis")
        assert rows.filter(pl.col("metric_name") == "a, b").height == 200
        assert rows.filter(pl.col("metric_name") == "c, d").equals(_rows(default, "mahalanobis"))

    def test_all_skips_a_level_with_one_factor(self):
        md = Metadata(_od(100, 2))
        md.add_factors({"hour": np.arange(100.0)}, level=md.item_level)
        md.add_factors(_correlated(), level=md.label_level)
        with use_seed(0):
            result = Outliers().evaluate(md, levels=["item", "target"], combinations="all")
        assert _checks(result) == {
            ("instance", name, "mahalanobis") for name in ("a, b", "a, c", "a, d", "b, c", "b, d", "c, d")
        }

    def test_all_with_no_pair_anywhere_raises(self):
        md = Metadata.from_factors({"x": np.arange(30.0), "weather": ["rain"] * 30})
        with pytest.raises(ValueError, match="pair"):
            Outliers().evaluate(md, combinations="all")

    def test_given_alone_calls_core_with_no_groups(self, monkeypatch):
        counting = _Counting(monkeypatch)
        with use_seed(0):
            Outliers().evaluate(_daylight(), given={"brightness": "hour"})
        assert [(call["groups"], call["given"]) for call in counting.calls] == [([], {"brightness": "hour"})]

    def test_all_stays_within_factors(self):
        md = Metadata.from_factors(_correlated())
        with use_seed(0):
            result = Outliers().evaluate(md, factors=["a", "b", "c"], combinations="all")
        assert {name for _, name, _ in _checks(result)} == {"a, b", "a, c", "b, c"}

    def test_a_categorical_group_needs_strata(self):
        rng = np.random.default_rng(0)
        factors = {
            **_correlated(),
            "weather": rng.choice(["rain", "clear"], 200).tolist(),
            "time": rng.choice(["day", "night"], 200).tolist(),
            "camera": ["front", "rear"] * 100,
        }
        md = Metadata.from_factors(factors)
        with pytest.raises(ValueError, match="strata"):
            Outliers().evaluate(md, combinations=[("weather", "time")])
        with use_seed(0):
            searched = Outliers().evaluate(md, combinations="all")
        assert {name for _, name, _ in _checks(searched)} == {"a, b", "a, c", "a, d", "b, c", "b, d", "c, d"}
        assert searched.populations.filter(pl.col("method") == "rarity").is_empty()
        stratified = Outliers().evaluate(md, combinations=[("weather", "time")], strata=["camera"])
        rarity = stratified.populations.filter(pl.col("method") == "rarity")
        assert "weather, time" in set(rarity["metric_name"].cast(pl.Utf8))

    def test_a_categorical_group_holding_a_stratum_is_refused(self):
        rng = np.random.default_rng(0)
        md = Metadata.from_factors({
            "weather": rng.choice(["rain", "clear"], 200).tolist(),
            "time": rng.choice(["day", "night"], 200).tolist(),
            "camera": ["front", "rear"] * 100,
        })
        with pytest.raises(ValueError, match=r"holds \['weather'\], also named in strata="):
            Outliers().evaluate(md, combinations=[("weather", "time")], strata=["weather", "camera"])

    @pytest.mark.parametrize("combinations", [[("cam", "x")], "all"])
    def test_a_kinds_key_outside_every_check_is_refused(self, combinations):
        rng = np.random.default_rng(0)
        md = Metadata.from_factors({"cam": rng.integers(0, 3, 200), "x": rng.normal(size=200)})
        with pytest.raises(KeyError, match=r"names \['camm'\].*columns are \['cam', 'x'\]"):
            Outliers().evaluate(md, combinations=combinations, kinds={"camm": "categorical"})

    def test_one_column_groups_are_refused(self):
        md = Metadata.from_factors({**_correlated(), "weather": ["rain", "clear"] * 100})
        with pytest.raises(ValueError, match="single column"):
            Outliers().evaluate(md, combinations=[("a",)])
        with pytest.raises(ValueError, match="single column"):
            Outliers().evaluate(md, combinations=[("a",)], kinds={"a": "numeric"})
        with pytest.raises(ValueError, match="single column"):
            Outliers().evaluate(md, combinations=[("weather",)], strata=["weather"])

    def test_a_bare_string_is_refused(self):
        md = Metadata.from_factors(_correlated())
        with pytest.raises(TypeError):
            Outliers().evaluate(md, combinations="ab")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            Outliers().evaluate(md, combinations=["ab"])

    def test_a_unit_and_track_group_is_refused(self):
        md = Metadata(_mot_dataset([[1] * 30]))
        md.add_factors({"brightness": np.arange(30.0)}, level="unit")
        md.add_factors({"speed": [1.0]}, level="track")
        with pytest.raises(ValueError, match="'unit'.*'track'|'track'.*'unit'"):
            Outliers().evaluate(md, levels=["unit", "track"], combinations=[("brightness", "speed")])

    def test_kinds_for_a_column_at_one_level_only(self):
        md = Metadata(_od(100, 2))
        md.add_factors(_correlated(100, names=("x", "y")), level=md.item_level)
        md.add_factors({key: value for key, value in _correlated(200).items() if key in "ab"}, level=md.label_level)
        with use_seed(0):
            result = Outliers().evaluate(
                md, levels=["item", "target"], combinations=[("a", "b"), ("x", "y")], kinds={"a": "numeric"}
            )
        assert {(level, name) for level, name, _ in _checks(result)} == {(None, "x, y"), ("instance", "a, b")}

    def test_prune_knows_a_combination_that_flagged_nothing(self):
        md = Metadata.from_factors(_correlated())
        with use_seed(0):
            result = Outliers(outlier_threshold={"a, b": ConstantThreshold(upper=1e9)}).evaluate(
                md, combinations=[("a", "b")]
            )
        assert _rows(result, "mahalanobis").is_empty()
        assert len(result.prune(metrics=["a, b"])) == 0

    def test_combinations_need_a_metadata(self, get_od_dataset):
        dataset = get_od_dataset(10, 2)
        with pytest.raises(ValueError, match="pass the Metadata"):
            Outliers().evaluate(dataset, combinations=[("a", "b")])
        with pytest.raises(ValueError, match="pass the Metadata"):
            Outliers().evaluate(dataset, given={"a": "b"})
        with pytest.raises(ValueError, match="pass the Metadata"):
            Outliers().evaluate(dataset, kinds={"a": "numeric"})

    def test_stratified_refuses_a_metadata_of_other_level_heights(self):
        md = _horizon()
        other = Metadata(_od(200, 3))
        other.add_factors({"vehicle": ["car"] * 600}, level=other.label_level)
        with use_seed(0):
            result = Outliers().evaluate(md, levels="target", combinations=[("rel_size", "rel_offset_y")])
        with pytest.raises(ValueError, match="rows"):
            result.stratified(other, "vehicle")
        with pytest.raises(ValueError, match="classwise: .* rows"):
            result.classwise(other)


@pytest.mark.required
class TestCombinationLevelsAndReasons:
    def test_a_score_check_with_no_spread_is_listed_unjudged(self):
        n = 40
        check = scores_module._ScoredCheck(
            level="unit",
            kind=None,
            metric_name="a, b",
            method="mahalanobis",
            dims=2,
            rows=np.arange(n),
            score=np.full(n, 1.5),
            reason=np.zeros(n, dtype=np.int8),
            reasons=REASONS,
            stratum=np.full(n, -1, dtype=np.intp),
            labels={},
        )
        found, populations = scores_module.flag_scored([check], None, lambda _, row: SourceIndex(row))
        assert found.is_empty()
        assert [(p.metric_name, p.n, p.judged, p.reason) for p in populations] == [("a, b", n, False, "no spread")]

    def test_classwise_rescores_within_each_class(self):
        md = Metadata.from_factors(_correlated(), np.repeat([0, 1], 100))
        with use_seed(0):
            result = Outliers().evaluate(md, combinations=[("a", "b")], per_class=True)
        listed = result.populations.filter(pl.col("method") == "mahalanobis")
        assert listed["n"].to_list() == [100, 100]
        assert listed["population"].null_count() == 0
        assert listed.equals(result.itemwise().classwise(md).populations.filter(pl.col("method") == "mahalanobis"))

    def test_a_coarser_column_is_carried_down_to_the_deeper_level(self):
        md = Metadata(_od(100, 2))
        md.add_factors({"hour": np.random.default_rng(1).normal(0, 1, 100)}, level=md.item_level)
        md.add_factors({key: value for key, value in _correlated(200).items() if key in "ab"}, level=md.label_level)
        with use_seed(0):
            result = Outliers().evaluate(md, levels=["item", "target"], combinations=[("hour", "a")])
        listed = result.populations.filter(pl.col("method") == "mahalanobis")
        assert listed.select("level", "metric_name", "n").rows() == [("instance", "hour, a", 200)]

    def test_a_group_scored_at_a_level_not_judged_is_refused(self):
        md = Metadata(_od(100, 2))
        md.add_factors({"hour": np.arange(100.0)}, level=md.item_level)
        md.add_factors({key: value for key, value in _correlated(200).items() if key in "ab"}, level=md.label_level)
        with pytest.raises(ValueError, match="does not judge"):
            Outliers().evaluate(md, levels="item", combinations=[("a", "b")])

    def test_an_unknown_column_is_refused(self):
        md = Metadata.from_factors(_correlated())
        with pytest.raises(KeyError, match="nope"):
            Outliers().evaluate(md, combinations=[("a", "nope")])
        with pytest.raises(KeyError, match="nope"):
            Outliers().evaluate(md, given={"a": "nope"})

    def test_a_threshold_without_an_upper_bound_judges_no_distance(self):
        md = Metadata.from_factors(_correlated())
        threshold = {"a, b": ZScoreThreshold(lower_multiplier=2.0, upper_multiplier=None)}
        with use_seed(0):
            result = Outliers(outlier_threshold=threshold).evaluate(md, combinations=[("a", "b")])
        assert _rows(result, "mahalanobis").is_empty()
        listed = result.populations.filter(pl.col("method") == "mahalanobis")
        assert listed.select("judged", "reason").rows() == [(False, "no upper bound")]

    def test_a_sample_column_alone_is_scored_by_distribution(self):
        rng = np.random.default_rng(0)
        speeds = [rng.normal(5, 1, int(rng.integers(20, 40))).tolist() for _ in range(99)]
        speeds.append(np.r_[rng.normal(2, 0.3, 15), rng.normal(8, 0.3, 15)].tolist())  # a bimodal track
        md = Metadata(_mot_dataset([[100]]))
        md.add_factors({"speeds": pl.Series(speeds)}, level="track")
        with use_seed(0):
            result = Outliers().evaluate(md, levels="track", combinations=[("speeds",)], kinds={"speeds": "sample"})
        rows = _rows(result, "distribution")
        assert 99 in rows["target_index"].to_list()
        assert set(rows["metric_name"].cast(pl.Utf8)) == {"speeds"}
        assert set(rows["direction"].cast(pl.Utf8)) == {"upper"}
        assert _checks(result) == {("track", "speeds", "distribution")}

    def test_checks_at_one_level_share_their_rows_and_strata(self):
        md = Metadata.from_factors({**_correlated(), "weather": ["rain", "clear"] * 100})
        with use_seed(0):
            result = Outliers().evaluate(
                md, combinations=[("a", "b"), ("c", "d")], given={"a": "c"}, strata=["weather"]
            )
        first, *others = result.scored
        assert len(others) == 2
        assert all(check.rows is first.rows and check.stratum is first.stratum for check in others)
        refit = result.with_threshold(2.0).scored
        assert refit[2].rows is refit[0].rows
        assert refit[2].stratum is refit[0].stratum

    def test_a_level_view_narrows_combination_rows_and_populations(self):
        md = Metadata(_od(100, 2))
        md.add_factors(_correlated(100, names=("x", "y")), level=md.item_level)
        md.add_factors({key: value for key, value in _correlated(200).items() if key in "ab"}, level=md.label_level)
        everything = {"a, b": ConstantThreshold(upper=0.0), "x, y": ConstantThreshold(upper=0.0)}
        with use_seed(0):
            result = Outliers().evaluate(md, levels=["item", "target"], combinations=[("a", "b"), ("x", "y")])
        targets = result.targets.with_threshold(everything)
        assert set(_rows(targets, "mahalanobis")["metric_name"].cast(pl.Utf8)) == {"a, b"}
        assert _rows(targets, "mahalanobis").height == 200
        assert {name for _, name, _ in _checks(targets)} == {"a, b"}

    def test_a_missing_value_is_neither_a_finding_nor_counted(self):
        factors = _correlated()
        factors["a"][0] = np.nan
        md = Metadata.from_factors(factors)
        with use_seed(0):
            result = Outliers(outlier_threshold={"a, b": ConstantThreshold(upper=0.0)}).evaluate(
                md, combinations=[("a", "b")]
            )
        rows = _rows(result, "mahalanobis")
        assert 0 not in rows["item_index"].to_list()
        assert rows.height == 199
        assert result.populations.filter(pl.col("method") == "mahalanobis")["n"].to_list() == [199]

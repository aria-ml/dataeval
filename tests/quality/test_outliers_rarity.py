"""Categorical rarity within a stratum (#1521)."""

import numpy as np
import polars as pl
import pytest
from scipy import stats

from dataeval import Metadata
from dataeval.quality import Outliers
from dataeval.quality._outliers_scores import rarity_minimum
from tests.metadata.test_structurers import _mot_dataset


def _videos(values_per_video, name="weather", **kwargs):
    """One video per list of frame values, one detection per frame, the values a frame factor called `name`."""
    md = Metadata(_mot_dataset([[1] * len(values) for values in values_per_video]), **kwargs)
    md.add_factors({name: [v for values in values_per_video for v in values]}, level="unit")
    return md


def _rarity(result):
    """The rarity rows of a result, empty when it has none."""
    frame = result.data()
    return frame.filter(pl.col("method") == "rarity") if "method" in frame.columns else frame.clear()


def _planted(value, at, common, n):
    """`n` frames of `common` with `value` at the positions `at`."""
    return [value if i in at else common for i in range(n)]


def _bh(p_values, alpha=0.05):
    """Benjamini-Hochberg by hand: which p-values are discoveries."""
    p = np.asarray(p_values)
    ranked = np.sort(p)
    passed = np.flatnonzero(ranked <= alpha * np.arange(1, len(p) + 1) / len(p))
    return p <= ranked[passed[-1]] if passed.size else np.zeros(len(p), dtype=bool)


def _p(k, n):
    return stats.binom.cdf(k, n, 0.05)


@pytest.mark.required
class TestRarity:
    def test_the_minimum_is_93(self):
        assert rarity_minimum() == 93

    def test_one_snow_frame_in_a_clear_video_is_flagged(self):
        md = _videos([_planted("snow", {120}, "clear", 300), ["snow"] * 150 + ["clear"] * 150])
        result = Outliers().evaluate(md, levels="unit", strata=["item_index"], factors=["weather"])
        rows = _rarity(result)
        assert rows.height == 1
        row = rows.row(0, named=True)
        assert (row["item_index"], row["target_index"], row["level"]) == (0, 120, "unit")
        assert (row["metric_name"], row["observed"], row["population"]) == ("weather", "weather=snow", "item_index=0")
        assert row["metric_value"] == pytest.approx(0.0157, abs=5e-4)
        assert (row["bound"], row["direction"]) == (0.05, "lower")
        assert row["percentile"] is None

    def test_three_wrong_sensor_frames_are_flagged(self):
        md = _videos([_planted("B", {10, 150, 290}, "A", 300), ["B"] * 150 + ["A"] * 150], name="sensor_id")
        rows = _rarity(Outliers().evaluate(md, levels="unit", strata=["item_index"]))
        assert rows["target_index"].to_list() == [10, 150, 290]
        assert rows["metric_value"].to_list() == pytest.approx([0.0256] * 3, abs=5e-4)
        assert set(rows["observed"].cast(pl.Utf8)) == {"sensor_id=B"}

    def test_a_stratum_below_93_rows_is_listed_not_judged(self):
        md = _videos([_planted("truck", {5, 40}, "car", 60), ["truck"] * 50 + ["car"] * 50], name="vehicle")
        result = Outliers().evaluate(md, levels="unit", strata=["item_index"])
        assert _rarity(result).is_empty()
        listed = result.populations.filter(pl.col("method") == "rarity").sort("population")
        assert listed.select("level", "metric_name", "population", "n", "judged", "reason").rows() == [
            ("unit", "vehicle", "item_index=0", 60, False, "below 93 rows"),
            ("unit", "vehicle", "item_index=1", 100, True, None),
        ]

    def test_a_value_seen_in_one_stratum_only_is_withheld(self):
        md = _videos([_planted("snow", {7}, "clear", 300), ["clear"] * 300])
        result = Outliers().evaluate(md, levels="unit", strata=["item_index"])
        assert _rarity(result).is_empty()
        assert len(result.prune(metrics=["weather"])) == 0

    def test_bh_leaves_a_cell_unflagged_that_its_raw_p_would_flag(self):
        # V0: one snow in 120 (raw p 0.0155); V1: one fog in 300 (raw p 3.5e-6); V2: every value common.
        videos = [
            _planted("snow", {60}, "clear", 120),
            _planted("fog", {100}, "clear", 300),
            ["clear"] * 40 + ["snow"] * 30 + ["fog"] * 30,
        ]
        observed = [_p(119, 120), _p(1, 120), _p(299, 300), _p(1, 300), _p(40, 100), _p(30, 100), _p(30, 100)]
        assert observed[1] <= 0.05
        assert _bh(observed).tolist() == [False, False, False, True, False, False, False]
        # Counting the absent cells (fog in V0, snow in V1) would let the V0 snow through.
        assert _bh([*observed, _p(0, 120), _p(0, 300)])[1]
        rows = _rarity(Outliers().evaluate(_videos(videos), levels="unit", strata=["item_index"]))
        assert rows.select("item_index", "target_index", "observed").rows() == [(1, 100, "weather=fog")]

    def test_bh_counts_withheld_cells(self):
        # As above, plus one hail in V1 and nowhere else: withheld, but counted, which lets the V0 snow through.
        videos = [
            _planted("snow", {60}, "clear", 120),
            [{100: "fog", 200: "hail"}.get(i, "clear") for i in range(300)],
            ["clear"] * 40 + ["snow"] * 30 + ["fog"] * 30,
        ]
        hail = _p(1, 300)
        observed = [_p(119, 120), _p(1, 120), _p(298, 300), _p(1, 300), _p(40, 100), _p(30, 100), _p(30, 100)]
        assert _bh([*observed, hail])[1]
        assert not _bh(observed)[1]
        rows = _rarity(Outliers().evaluate(_videos(videos), levels="unit", strata=["item_index"]))
        assert rows.select("item_index", "target_index", "observed").rows() == [
            (0, 60, "weather=snow"),
            (1, 100, "weather=fog"),
        ]

    def test_a_declared_vocabulary_integer_is_judged_by_rarity(self):
        values = [_planted(1, {33}, 0, 300), [1] * 150 + [0] * 150]
        md = _videos(values, name="sensor_id", factor_levels={"sensor_id": [0, 1]})
        rows = _rarity(Outliers().evaluate(md, levels="unit", strata=["item_index"]))
        assert rows.select("item_index", "target_index", "metric_name", "observed").rows() == [
            (0, 33, "sensor_id", "sensor_id=1")
        ]

    def test_rarity_needs_strata(self):
        md = _videos([["clear"] * 100])
        with pytest.raises(ValueError, match="pass strata="):
            Outliers().evaluate(md, levels="unit", factors=["weather"])

    def test_a_metadata_of_only_categorical_factors_runs_with_strata(self):
        weather = _planted("snow", {120}, "clear", 300) + ["snow"] * 150 + ["clear"] * 150
        md = Metadata.from_factors({"weather": weather, "camera": ["a"] * 300 + ["b"] * 300})
        with pytest.raises(ValueError, match="no ordered factors"):
            Outliers().evaluate(md)
        rows = _rarity(Outliers().evaluate(md, strata=["camera"]))
        assert rows.select("item_index", "observed", "population").rows() == [(120, "weather=snow", "camera=a")]

    def test_a_factor_named_in_strata_is_not_judged(self):
        md = _videos([_planted("snow", {120}, "clear", 300), ["snow"] * 150 + ["clear"] * 150])
        md.add_factors({"camera": ["a"] * 300 + ["b"] * 300}, level="unit")
        result = Outliers().evaluate(md, levels="unit", strata=["item_index", "camera"])
        rarity = result.populations.filter(pl.col("method") == "rarity")
        assert set(rarity["metric_name"].cast(pl.Utf8)) == {"weather"}

    def test_rarity_is_not_run_where_strata_do_not_reach(self):
        md = _videos([_planted("snow", {120}, "clear", 300), ["snow"] * 150 + ["clear"] * 150])
        result = Outliers().evaluate(md, levels=["unit", "track"], strata=["track_id"])
        assert _rarity(result).is_empty()
        assert result.populations.filter(pl.col("method") == "rarity").is_empty()

    def test_a_named_factor_the_strata_do_not_reach_is_refused(self):
        md = _videos([["clear"] * 100])
        with pytest.raises(ValueError, match="do not reach"):
            Outliers().evaluate(md, levels=["unit", "track"], strata=["track_id"], factors=["weather", "track_length"])


@pytest.mark.required
class TestRarityRedetection:
    @pytest.fixture
    def snowy(self):
        """The snow case, with an ordered frame factor beside it."""
        md = _videos([_planted("snow", {120}, "clear", 300), ["snow"] * 150 + ["clear"] * 150])
        rng = np.random.default_rng(0)
        md.add_factors({"brightness": rng.normal(100, 5, 600)}, level="unit")
        return md

    def test_per_class_adds_no_rarity_rows(self):
        rng = np.random.default_rng(0)
        md = Metadata.from_factors(
            {"x": rng.normal(0, 1, 200), "weather": ["clear"] * 199 + ["snow"]}, np.repeat([0, 1], 100)
        )
        result = Outliers().evaluate(md, per_class=True)
        assert "method" not in result.data().columns
        assert result.populations.filter(pl.col("method") == "rarity").is_empty()

    def test_stratified_equals_strata_with_rarity(self, snowy):
        stratified = Outliers().evaluate(snowy, levels="unit").stratified(snowy, "item_index")
        direct = Outliers().evaluate(snowy, levels="unit", strata=["item_index"])
        assert _rarity(stratified).height == 1
        assert stratified.data().equals(direct.data())
        assert stratified.populations.equals(direct.populations)

    def test_itemwise_removes_rarity_rows(self, snowy):
        result = Outliers().evaluate(snowy, levels="unit", strata=["item_index"]).itemwise()
        assert _rarity(result).is_empty()
        assert result.populations.filter(pl.col("method") == "rarity").is_empty()

    def test_classwise_removes_rarity_rows(self, snowy):
        result = Outliers().evaluate(snowy, levels="unit", strata=["item_index"]).classwise(snowy)
        assert _rarity(result).is_empty()

    def test_with_threshold_leaves_rarity_rows_unchanged(self, snowy):
        result = Outliers().evaluate(snowy, levels="unit", strata=["item_index"])
        loosened = result.with_threshold(100.0)
        assert _rarity(loosened).equals(_rarity(result))
        rarity = pl.col("method") == "rarity"
        assert loosened.populations.filter(rarity).equals(result.populations.filter(rarity))

    def test_a_level_view_narrows_rarity_rows(self, snowy):
        result = Outliers().evaluate(snowy, levels=["sequence", "unit"], strata=["item_index"])
        assert _rarity(result.frames).height == 1
        assert _rarity(result.sequences).is_empty()
        assert _rarity(result.sequences.with_threshold(2.0)).is_empty()
        assert result.sequences.populations.filter(pl.col("method") == "rarity").is_empty()
        assert _rarity(result.frames.with_threshold(2.0)).height == 1

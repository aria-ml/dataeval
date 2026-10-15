"""Metadata factors judged against their neighbors along a sequence in Outliers: neighbors= (#1523)."""

import pickle
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest

import dataeval.quality._outliers_neighbors as neighbors_module
from dataeval import Metadata
from dataeval.quality import Outliers
from dataeval.quality._outliers_scores import _ScoreInput
from dataeval.utils.thresholds import ZScoreThreshold
from tests.conftest import _get_dataset
from tests.metadata.test_structurers import _mot_dataset

GLITCH_FRAME = (2, 100)  # video, frame: brightness ordinary over the dataset, odd beside its neighbors
GLITCH_BOX = (1, 50, 2)  # video, frame, track: one frame of the track with a larger box


def _detection(md, video, frame, track):
    """The position of one detection among the instance rows, and its target_index within its video."""
    rows = md._store.frame("instance").with_row_index("position")
    hit = rows.filter((pl.col("item_index") == video) & (pl.col("unit_index") == frame) & (pl.col("track_id") == track))
    return int(hit["position"][0]), int(hit["target_index"][0])


def _videos(videos=8, frames=200, tracks=3, seed=0, glitch=False, missing=0):
    """A tracking metadata: `tracks` tracked detections per frame, a drifting frame factor and a detection factor.

    `glitch` plants the brightness frame and the box of GLITCH_FRAME and GLITCH_BOX; `missing` blanks the first
    frames' brightness. Each factor is planted before it is added, since adding a factor again keeps the first values.
    """
    md = Metadata(_mot_dataset([[tracks] * frames] * videos))
    md._structure()  # the store holds no level frames until the dataset is structured
    rng = np.random.default_rng(seed)
    video, frame = np.repeat(np.arange(videos), frames), np.tile(np.arange(frames), videos)
    brightness = 100 + 20 * video + 10 * np.sin(frame / 25) + rng.normal(0, 1, video.size)
    track = md._store.frame("instance")["track_id"].to_numpy()
    box = 50 + 5 * track + rng.normal(0, 0.5, track.size)
    if glitch:
        brightness[GLITCH_FRAME[0] * frames + GLITCH_FRAME[1]] += 15
        box[_detection(md, *GLITCH_BOX)[0]] += 6
    brightness[:missing] = np.nan
    md.add_factors({"brightness": brightness}, level="unit")
    md.add_factors({"box_size": box}, level="instance")
    return md


def _clock(videos, frames, rng, jitter=0.05):
    """Capture times: each video starts on its own hour and runs at 30 fps with jittered frame intervals."""
    stamps = []
    for video in range(videos):
        offsets = np.cumsum(1 / 30 * (1 + rng.normal(0, jitter, frames)))
        stamps += [datetime(2026, 1, 1, 8 + video) + timedelta(seconds=float(offset)) for offset in offsets]
    return stamps


def _rows(result, method):
    """The rows a check flagged, categoricals as strings so polars 1.0 compares them."""
    frame = result.data().with_columns(cs.categorical().cast(pl.Utf8))
    return frame.filter(pl.col("method") == method) if "method" in frame.columns else frame.clear()


def _at(rows, item, target):
    """The one flagged row at an address."""
    return rows.filter((pl.col("item_index") == item) & (pl.col("target_index") == target))


def _listed(result, method="neighbors"):
    """The populations a check lists, categoricals as strings."""
    return result.populations.with_columns(cs.categorical().cast(pl.Utf8)).filter(pl.col("method") == method)


class _Counting:
    """Count the calls to core, to show that re-detection does not rescore."""

    def __init__(self, monkeypatch):
        self.calls = 0
        original = neighbors_module.sequence_residuals

        def counted(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(neighbors_module, "sequence_residuals", counted)


@pytest.fixture(scope="module")
def videos():
    return _videos(glitch=True)


@pytest.mark.required
class TestNeighbors:
    def test_a_glitch_frame_is_flagged_along_its_sequence(self, videos):
        result = Outliers().evaluate(videos, factors=["brightness"], neighbors=["brightness"])
        found = _at(_rows(result, "neighbors"), *GLITCH_FRAME)
        assert found["metric_name"].to_list() == ["brightness along sequence"]
        row = found.row(0, named=True)
        planted = videos._store.column("unit", "brightness")[GLITCH_FRAME[0] * 200 + GLITCH_FRAME[1]]
        assert row["metric_value"] == pytest.approx(planted)
        assert row["direction"] == "upper"
        assert row["population_mean"] < row["bound"] < row["metric_value"]
        assert row["population_std"] > 0
        assert row["percentile"] > 99
        assert row.get("population") is None
        assert _rows(result, "neighbors").height <= 16  # the glitch, and clean rows past the bound: under 1 % of 1600

    def test_a_glitch_detection_is_flagged_along_its_track(self, videos):
        result = Outliers().evaluate(videos, levels="instance", factors=["box_size"], neighbors=["box_size"])
        _, target = _detection(videos, *GLITCH_BOX)
        found = _at(_rows(result, "neighbors"), GLITCH_BOX[0], target)
        assert found["metric_name"].to_list() == ["box_size along track"]

    def test_untracked_detections_are_not_scored(self):
        md = Metadata(_mot_dataset([[[0, -1]] * 40] * 2))
        md._structure()
        rng = np.random.default_rng(1)
        track = md._store.frame("instance")["track_id"].to_numpy()
        box = np.where(track < 0, rng.uniform(0, 100, track.size), 50 + rng.normal(0, 0.5, track.size))
        md.add_factors({"box_size": box}, level="instance")
        result = Outliers().evaluate(md, levels="instance", factors=["box_size"], neighbors=["box_size"])
        assert _listed(result)["n"].to_list() == [80]
        untracked = set(md._store.frame("instance").filter(pl.col("track_id") < 0)["target_index"].to_list())
        assert not set(_rows(result, "neighbors")["target_index"].to_list()) & untracked

    def test_missing_values_are_left_out_of_the_pool(self):
        result = Outliers().evaluate(_videos(missing=10), factors=["brightness"], neighbors=["brightness"])
        assert _listed(result)["n"].to_list() == [1600 - 10]

    def test_all_judges_every_ordered_factor_at_the_judged_sequenced_levels(self, videos):
        result = Outliers().evaluate(videos, levels=("unit", "instance"), neighbors="all")
        names = set(_listed(result)["metric_name"].to_list())
        assert {"brightness along sequence", "time_s along sequence", "box_size along track"} <= names


@pytest.mark.required
class TestOrder:
    def test_a_late_datetime_stamp_gets_a_neighbor_and_an_order_row(self):
        md = _videos(videos=4, frames=120)
        stamps = _clock(4, 120, np.random.default_rng(2))
        stamps[1 * 120 + 60] += timedelta(hours=3)
        md.add_factors({"captured": pl.Series(stamps, dtype=pl.Datetime("us"))}, level="unit")
        result = Outliers().evaluate(md, factors=["captured"], neighbors=["captured"])
        order = _rows(result, "order")
        assert order.select("item_index", "target_index", "metric_name").rows() == [(1, 60, "captured along sequence")]
        physical = pl.Series([stamps[180]], dtype=pl.Datetime("us")).to_physical()[0]
        assert order["metric_value"][0] == pytest.approx(float(physical))
        following = pl.Series([stamps[181]], dtype=pl.Datetime("us")).to_physical()[0]
        assert order["bound"][0] == pytest.approx(float(following))
        assert order["direction"][0] == "upper"
        assert order["percentile"][0] is None
        assert order["population_mean"][0] is None
        assert order["population_std"][0] is None
        assert "population" not in order.columns or order["population"][0] is None
        assert _at(_rows(result, "neighbors"), 1, 60).height == 1

    @pytest.mark.parametrize(
        ("name", "dtype", "monotonic"),
        [
            ("time_s", pl.Float64(), True),
            ("pts", pl.Int64(), True),
            ("captured", pl.Datetime("us"), True),
            ("day", pl.Date(), True),
            ("elapsed", pl.Duration("us"), True),
            ("clock", pl.Time(), False),
            ("brightness", pl.Float64(), False),
        ],
    )
    def test_a_timestamp_is_known_by_its_dtype_or_the_stream_clock_names(self, name, dtype, monotonic):
        assert neighbors_module._is_monotonic(name, dtype) is monotonic


@pytest.mark.required
class TestPopulations:
    def test_a_pool_below_20_rows_is_listed_not_judged(self):
        md = _videos(videos=3, frames=15)
        result = Outliers().evaluate(md, factors=["brightness"], neighbors=["brightness"], strata=["item_index"])
        listed = _listed(result)
        assert listed.height == 3
        assert listed["judged"].to_list() == [False] * 3
        assert set(listed["reason"].to_list()) == {"below 20 rows"}

    def test_a_level_of_short_tracks_is_listed_with_no_rows(self):
        md = Metadata(_mot_dataset([[[i // 2] for i in range(30)]] * 4))  # every track is two frames long
        md.add_factors({"box_size": np.random.default_rng(3).normal(50, 1, 120)}, level="instance")
        result = Outliers().evaluate(md, levels="instance", factors=["box_size"], neighbors=["box_size"])
        assert _listed(result).select("n", "judged", "reason").rows() == [(0, False, "below 20 rows")]

    def test_strata_split_the_pool(self, videos):
        result = Outliers().evaluate(videos, factors=["brightness"], neighbors=["brightness"], strata=["item_index"])
        assert _listed(result).height == 8
        assert _at(_rows(result, "neighbors"), *GLITCH_FRAME)["population"].to_list() == ["item_index=2"]


@pytest.mark.required
class TestRedetection:
    def test_a_named_threshold_reaches_the_check(self, videos):
        result = Outliers().evaluate(videos, factors=["brightness"], neighbors=["brightness"])
        relaxed = result.with_threshold({"brightness along sequence": ZScoreThreshold(1000.0)})
        assert _rows(relaxed, "neighbors").height == 0

    def test_restratifying_does_not_rescore(self, videos, monkeypatch):
        counter = _Counting(monkeypatch)
        result = Outliers().evaluate(videos, factors=["brightness"], neighbors=["brightness"])
        assert counter.calls == 1
        stratified = result.stratified(videos, "item_index")
        whole = stratified.itemwise()
        result.classwise(videos)
        assert counter.calls == 1
        assert _at(_rows(stratified, "neighbors"), *GLITCH_FRAME)["population"].to_list() == ["item_index=2"]
        assert _at(_rows(whole, "neighbors"), *GLITCH_FRAME).height == 1

    def test_order_rows_survive_redetection(self):
        md = _videos(videos=4, frames=120)
        stamps = [datetime(2026, 1, 1) + timedelta(seconds=frame / 30) for _ in range(4) for frame in range(120)]
        stamps[60] -= timedelta(hours=1)
        md.add_factors({"captured": pl.Series(stamps, dtype=pl.Datetime("us"))}, level="unit")
        result = Outliers().evaluate(md, factors=["captured"], neighbors=["captured"])
        redone = (
            result.with_threshold(3.0),
            result.stratified(md, "item_index"),
            result.itemwise(),
            result.classwise(md),
            result.frames,
            result.frames.with_threshold(3.0),
        )
        for each in redone:
            assert _rows(each, "order").select("item_index", "target_index").rows() == [(0, 60)]
        assert _rows(result.detections, "order").height == 0
        assert _rows(result.detections.with_threshold(3.0), "order").height == 0

    def test_views_keep_their_levels_through_redetection(self, videos):
        result = Outliers().evaluate(
            videos,
            levels=("unit", "instance"),
            factors=["brightness", "box_size"],
            neighbors=["brightness", "box_size"],
        )
        frames = result.frames.with_threshold(3.0)
        assert set(_rows(frames, "neighbors")["metric_name"].to_list()) == {"brightness along sequence"}
        detections = result.detections.with_threshold(3.0)
        assert set(_rows(detections, "neighbors")["metric_name"].to_list()) == {"box_size along track"}

    def test_a_result_without_stored_order_rows_redetects(self, videos):
        result = Outliers().evaluate(videos, factors=["brightness"], neighbors=["brightness"])
        del result.__dict__["_order"]  # as a result pickled before order rows were stored
        assert _at(_rows(result.with_threshold(3.0), "neighbors"), *GLITCH_FRAME).height == 1

    def test_a_score_input_from_before_neighbors_round_trips_through_pickle(self, videos):
        score_input = Outliers().evaluate(videos, factors=["brightness"], neighbors=["brightness"]).score_input
        assert score_input is not None
        assert score_input.scored_levels() == ["unit"]
        older = _ScoreInput(  # the six fields it had before neighbors existed
            score_input.frames,
            score_input.categorical,
            score_input.groups,
            score_input.given,
            score_input.kinds,
            score_input.level_kinds,
        )
        loaded = pickle.loads(pickle.dumps(older))
        assert loaded.scored_levels() == []
        assert loaded.frames["unit"].equals(score_input.frames["unit"])

    def test_aggregations_and_prune_read_neighbor_rows(self, videos):
        result = Outliers().evaluate(videos, factors=["brightness"], neighbors=["brightness"])
        metrics = result.aggregate_by_metric().with_columns(cs.categorical().cast(pl.Utf8))
        assert "brightness along sequence" in metrics["metric_name"].to_list()
        plan = result.prune(metrics=["brightness along sequence"])
        assert any(index.item == GLITCH_FRAME[0] for index in plan.discard)


@pytest.mark.required
class TestCleanData:
    def test_clean_data_flags_few_rows(self):
        md = _videos(videos=8, frames=200, tracks=3, seed=4)
        rng = np.random.default_rng(5)
        frame = np.tile(np.arange(200), 8)
        md.add_factors({"hour": 8 + frame / 200 * 4 + rng.normal(0, 0.01, frame.size)}, level="unit")
        md.add_factors({"captured": pl.Series(_clock(8, 200, rng), dtype=pl.Datetime("us"))}, level="unit")
        result = Outliers().evaluate(md, levels=("unit", "instance"), neighbors="all")
        flagged = _rows(result, "neighbors").height + _rows(result, "order").height
        judged = int(_listed(result).filter(pl.col("judged"))["n"].sum())  # a pool with no spread flags nothing
        assert flagged / judged < 0.005


@pytest.mark.required
class TestRefusals:
    def test_a_dataset_is_refused(self):
        with pytest.raises(ValueError, match="read metadata factors"):
            Outliers().evaluate(_get_dataset(10, 1), neighbors=["brightness"])

    def test_a_metadata_without_sequences_is_refused(self):
        md = Metadata.from_factors({"x": np.arange(30.0)})
        with pytest.raises(ValueError, match="no sequence level"):
            Outliers().evaluate(md, neighbors=["x"])

    def test_a_categorical_factor_is_refused(self):
        md = _videos(videos=2, frames=30)
        md.add_factors({"weather": ["clear"] * 60}, level="unit")
        with pytest.raises(ValueError, match="holds no numbers or times"):
            Outliers().evaluate(md, neighbors=["weather"])

    def test_a_factor_without_order_is_refused(self, videos):
        with pytest.raises(ValueError, match="only frame"):
            Outliers().evaluate(videos, levels="track", neighbors=["track_length"])

    def test_a_level_the_call_does_not_judge_is_refused(self, videos):
        with pytest.raises(ValueError, match="does not judge"):
            Outliers().evaluate(videos, neighbors=["box_size"])

    def test_all_finding_nothing_is_refused(self, videos):
        with pytest.raises(ValueError, match="finds no ordered factor"):
            Outliers().evaluate(videos, levels="track", neighbors="all")

    def test_an_unknown_factor_is_refused(self, videos):
        with pytest.raises(KeyError, match="not a factor"):
            Outliers().evaluate(videos, neighbors=["nope"])

    def test_a_string_other_than_all_is_refused(self, videos):
        with pytest.raises(TypeError, match="neighbors"):
            Outliers().evaluate(videos, neighbors="brightness")

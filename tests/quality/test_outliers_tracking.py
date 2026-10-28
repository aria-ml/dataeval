import logging
from dataclasses import replace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dataeval import Metadata
from dataeval.core import image_stats
from dataeval.data import FrameIndices, SequenceFrames, Stride
from dataeval.extractors import FlattenExtractor
from dataeval.flags import ImageStats
from dataeval.quality import Outliers
from dataeval.quality._factor_stats import factor_kind, factor_levels, factor_stats
from dataeval.quality._outliers import DEFAULT_OUTLIERS_FLAGS
from dataeval.quality._shared import is_tracking, yields_frames
from dataeval.types import FactorLevel, SourceIndex
from tests.quality.test_duplicates import make_tracking_dataset

_SHAPE = (3, 24, 32)
_ROW = ["item_index", "target_index", "level", "metric_name", "metric_value"]


def _noise(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(100, 140, _SHAPE, dtype=np.uint8)


def _planted(n_videos: int = 3, frame_offset: int = 0):
    """Videos of 30 noisy frames each; the frame at position 12 of video 1 is a flat 250."""
    rng = np.random.default_rng(0)
    sequences = [[_noise(rng) for _ in range(30)] for _ in range(n_videos)]
    sequences[1][12] = np.full(_SHAPE, 250, dtype=np.uint8)
    return make_tracking_dataset(sequences, frame_offset=frame_offset)


def _hit(rows: pl.DataFrame, item: int, target: int, metric: str) -> pl.DataFrame:
    return rows.filter(
        (pl.col("item_index") == item)
        & (pl.col("target_index") == target)
        & (pl.col("level") == "unit")
        & (pl.col("metric_name") == metric)
    )


def _names(column: pl.Series) -> set[str]:
    return set(column.cast(pl.String).drop_nulls())


class _Undecodable:
    """A video stream that fails the test if anything walks it."""

    def __iter__(self):
        raise AssertionError("a video frame was decoded")


class _Renumbered:
    """Replays a stream with its second frame numbered as its first, so a frame_index repeats."""

    def __init__(self, stream):
        self._stream = stream

    def __iter__(self):
        for position, frame in enumerate(self._stream):
            if position == 1:
                frame.frame_index -= 1
            yield frame


def _with_streams(dataset, wrap):
    """Replace every video stream of a `make_tracking_dataset` dataset with `wrap(stream)`."""
    dataset._data = [(wrap(stream), target, datum) for stream, target, datum in dataset._data]
    return dataset


def _manual_metadata(dataset):
    """`Metadata(dataset)` with the frame view's statistics added by hand, and the names of the ordered ones added."""
    frames = SequenceFrames(dataset)
    stats = image_stats(frames, stats=DEFAULT_OUTLIERS_FLAGS, per_image=True, per_target=False)
    md = Metadata(dataset)
    before = set(md.factor_names)
    scalar = {name: values for name, values in stats["stats"].items() if np.ndim(values) == 1}
    md.add_factors(cast(Any, {**stats, "stats": scalar}), source_index=frames.to_source(stats["source_index"]))
    return md, [name for name in md.factor_names if name not in before and factor_kind(md, name) == "ordered"]


_IMAGES = [np.full(_SHAPE, value, dtype=np.uint8) for value in range(100, 130)]

_REFUSALS = [
    pytest.param({"extractor": FlattenExtractor()}, (), {}, "from_clusters", id="extractor"),
    pytest.param({}, ("tracking",), {}, "several datasets", id="two-tracking-datasets"),
    pytest.param({}, ("images",), {}, "cannot combine", id="tracking-beside-images"),
    pytest.param({}, (), {"factors": ["brightness"]}, "Metadata", id="factors"),
    pytest.param({}, (), {"neighbors": ["brightness"]}, "Metadata", id="neighbors"),
]


def _refused(dataset, config, other, kwargs):
    others = [dataset if name == "tracking" else _IMAGES for name in other]
    return Outliers(**config).evaluate(dataset, *others, **kwargs)


@pytest.mark.required
class TestTrackingDataset:
    def test_a_raw_tracking_dataset_flags_the_planted_frame(self):
        rows = Outliers().evaluate(_planted()).data()
        assert len(_hit(rows, 1, 12, "brightness")) == 1

    def test_target_index_is_the_frame_index_not_the_position(self):
        rows = Outliers().evaluate(_planted(frame_offset=100)).data()
        assert len(_hit(rows, 1, 112, "brightness")) == 1

    def test_default_levels_exclude_detections(self):
        dataset = _planted()
        default = Outliers().evaluate(dataset)
        assert _names(default.populations["level"]) == {"unit"}
        assert _names(default.data()["level"]) == {"unit"}

        detections = Outliers().evaluate(dataset, per_target=True)
        assert "instance" in _names(detections.populations["level"])
        names = _names(detections.data()["metric_name"])
        assert {"unit_brightness", "instance_brightness"} <= names
        assert "brightness" not in names

    def test_a_hand_wrapped_frame_view_is_measured_as_given(self, caplog):
        caplog.set_level(logging.INFO, logger="dataeval.quality")
        dataset = _planted(n_videos=5)
        result = Outliers().evaluate(SequenceFrames(dataset, Stride(3)))
        rows = result.data()
        assert len(_hit(rows, 1, 12, "brightness")) == 1
        assert (rows["target_index"] % 3 == 0).all()
        units = result.populations.filter(pl.col("level") == "unit")
        assert set(units["n"]) == {50}
        assert units["judged"].any()
        assert "measuring 50 of 150" in caplog.text
        assert result.dataset is dataset

    def test_a_zero_dimensional_array_is_not_a_tracking_dataset(self):
        assert not is_tracking(np.zeros(()))
        assert not yields_frames(np.zeros(()))

    def test_an_image_dataset_still_refuses_tracking_levels(self):
        with pytest.raises(ValueError, match="tracking levels"):
            Outliers().evaluate(_IMAGES, levels="unit")

    @pytest.mark.parametrize(("config", "other", "kwargs", "match"), _REFUSALS)
    def test_refusals(self, config, other, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _refused(_planted(), config, other, kwargs)

    def test_tracking_after_images_cannot_combine(self):
        with pytest.raises(ValueError, match="cannot combine"):
            Outliers().evaluate(_IMAGES, _planted())

    @pytest.mark.parametrize(("config", "other", "kwargs", "match"), _REFUSALS)
    def test_refusals_happen_before_any_frame_is_decoded(self, config, other, kwargs, match):
        dataset = _with_streams(_planted(), lambda _: _Undecodable())
        with pytest.raises(ValueError, match=match):
            _refused(dataset, config, other, kwargs)

    def test_a_repeated_frame_index_raises_the_to_source_error(self):
        dataset = _with_streams(_planted(), _Renumbered)
        with pytest.raises(ValueError, match="more than one position"):
            Outliers().evaluate(dataset)

    # `rows_at` bins the unbinned factors it shows; only the measured values are read here.
    @pytest.mark.filterwarnings("ignore:.*binned automatically:UserWarning")
    def test_a_video_of_one_frame_and_a_frame_without_detections_still_evaluate(self):
        rng = np.random.default_rng(0)
        dataset = make_tracking_dataset([[_noise(rng)], [_noise(rng) for _ in range(30)]])
        tracks = dataset[1][1].frame_tracks
        tracks[5] = replace(
            tracks[5],
            track_ids=np.zeros(0, dtype=np.int64),
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            labels=np.zeros(0, dtype=np.int64),
        )
        frames = Outliers().evaluate(dataset)
        # 1 + 30 frames, each measured, the one without detections included.
        assert set(frames.populations.filter(pl.col("level") == "unit")["n"]) == {31}
        # The two videos are the sequence rows, too few to judge.
        assert set(frames.populations.filter(pl.col("level").is_null())["n"]) == {2}
        assert frames.metadata is not None
        units = frames.metadata.rows_at("unit")
        for item, unit in ((0, 0), (1, 5)):
            row = units.filter((pl.col("item_index") == item) & (pl.col("unit_index") == unit))
            assert len(row) == 1
            assert row["brightness"].is_not_null().all()

        detections = Outliers().evaluate(dataset, per_target=True)
        populations = detections.populations
        assert set(populations.filter(pl.col("level") == "unit")["n"]) == {31}
        # One detection per frame, less the frame emptied above.
        assert set(populations.filter(pl.col("level") == "instance")["n"]) == {30}

    def test_the_dataset_route_equals_the_manual_recipe(self):
        dataset = _planted()
        md, added = _manual_metadata(dataset)
        expected = Outliers().evaluate(md, factors=added, per_image=True).data().select(_ROW)
        assert_frame_equal(Outliers().evaluate(dataset).data().select(_ROW), expected)

    # `rows_at` bins the unbinned factors it shows; the count of rows is all that is read here.
    @pytest.mark.filterwarnings("ignore:.*binned automatically:UserWarning")
    def test_the_result_holds_the_source_dataset_and_metadata(self):
        dataset = _planted()
        result = Outliers().evaluate(dataset)
        assert result.dataset is dataset
        assert result.metadata is not None
        assert "sequence" in result.metadata.levels
        assert len(result.metadata.rows_at("unit")) == 90


def _videos(n_videos: int = 24, n_frames: int = 12, dark: int = 40):
    """Videos of flat frames near 120, the last one at `dark`; each frame is a little off its video's own value."""
    sequences = [[120 + (video % 3) + (frame % 2) for frame in range(n_frames)] for video in range(n_videos)]
    sequences[-1] = [dark + (frame % 2) for frame in range(n_frames)]
    return make_tracking_dataset(sequences)


def _sequence_rows(rows: pl.DataFrame, metric: str) -> pl.DataFrame:
    return rows.filter(pl.col("level").is_null() & (pl.col("metric_name") == metric))


def _population(result, metric: str) -> dict[str, Any]:
    rows = result.populations.filter(pl.col("metric_name") == metric)
    assert len(rows) == 1
    return rows.row(0, named=True)


@pytest.mark.required
class TestVideoRollUps:
    def test_one_dark_video_is_one_sequence_row(self):
        result = Outliers().evaluate(_videos())
        rows = _sequence_rows(result.data(), "sequence_brightness")
        assert rows["item_index"].to_list() == [23]
        population = _population(result, "sequence_brightness")
        assert population["n"] == 24
        assert population["judged"]

    def test_fewer_than_twenty_videos_leave_the_sequence_level_unjudged(self):
        result = Outliers().evaluate(_videos(n_videos=5))
        assert len(_sequence_rows(result.data(), "sequence_brightness")) == 0
        population = _population(result, "sequence_brightness")
        assert population["n"] == 5
        assert not population["judged"]
        assert population["reason"] == "below 20 rows"

    def test_per_target_adds_track_rows(self):
        result = Outliers().evaluate(_videos(), per_target=True)
        rows = result.data().filter(pl.col("metric_name") == "track_brightness")
        assert rows["item_index"].to_list() == [23]
        assert rows["level"].to_list() == ["track"]
        assert _population(result, "track_brightness")["n"] == 24
        assert _population(result, "sequence_brightness")["n"] == 24

    def test_levels_unit_alone_names_no_foreign_factor(self):
        dataset = _videos()
        by_level = Outliers().evaluate(dataset, levels="unit")
        assert _names(by_level.populations["level"]) == {"unit"}
        detections = Outliers().evaluate(dataset, per_image=False, per_target=True)
        assert _names(detections.populations["level"]) == {"instance", "track"}

    def test_levels_sequence_alone_returns_the_sequence_rows(self):
        result = Outliers().evaluate(_videos(), levels="sequence")
        # The sequence level is the item level, which a population spells as no level at all.
        assert set(result.populations["level"]) == {None}
        assert result.data().filter(pl.col("metric_name") == "sequence_brightness")["item_index"].to_list() == [23]

    def test_levels_track_alone_returns_the_track_rows(self):
        result = Outliers().evaluate(_videos(), levels="track")
        assert _names(result.populations["level"]) == {"track"}
        rows = result.data()
        assert all(name.startswith("track_") for name in rows["metric_name"])
        assert rows.filter(pl.col("metric_name") == "track_brightness")["item_index"].to_list() == [23]

    def test_a_plan_with_no_ordered_factor_anywhere_says_where_they_live(self):
        dataset = _videos()
        for _, target, _ in dataset._data:
            target.frame_tracks[:] = [
                replace(
                    track,
                    track_ids=np.zeros(0, dtype=np.int64),
                    boxes=np.zeros((0, 4), dtype=np.float32),
                    scores=np.zeros(0, dtype=np.float32),
                    labels=np.zeros(0, dtype=np.int64),
                )
                for track in target.frame_tracks
            ]
        for level in ("instance", "track"):
            with pytest.raises(ValueError, match=rf"no ordered factor lives at \['{level}'\].*live at \['unit'\]"):
                Outliers().evaluate(dataset, levels=level)
        with pytest.raises(ValueError, match="no detections to judge; pass per_image=True"):
            Outliers().evaluate(dataset, per_image=False, per_target=True)

    def test_flags_that_measure_no_ordered_statistic_say_so(self):
        with pytest.raises(ValueError, match="no ordered value to threshold"):
            Outliers(flags=ImageStats.HASH).evaluate(_videos(n_videos=3))

    @pytest.mark.parametrize("kwargs", [{}, {"per_image": False, "per_target": True}], ids=["frames", "detections"])
    def test_a_frame_view_that_keeps_no_frames_says_so(self, kwargs):
        with pytest.raises(ValueError, match=r"FrameIndices\(.*\) kept no frames"):
            Outliers().evaluate(SequenceFrames(_videos(n_videos=3), FrameIndices({})), **kwargs)

    def test_flags_none_is_refused_on_a_tracking_dataset(self):
        with pytest.raises(ValueError, match="flags must not be ImageStats.NONE for a tracking dataset"):
            Outliers(flags=ImageStats.NONE).evaluate(_videos(n_videos=3))

    def test_rollups_ignore_unmeasured_frames(self):
        # A ramp, so the median over the frames measured differs from the median over all 12.
        bases = [100 + video % 3 for video in range(24)]
        dataset = make_tracking_dataset([[base + frame for frame in range(12)] for base in bases])
        result = Outliers().evaluate(SequenceFrames(dataset, Stride(3)))
        assert result.metadata is not None
        got = result.metadata.rows_at("sequence").sort("item_index")["sequence_brightness"].to_list()
        # Frames 0, 3, 6, 9 are measured: base + 4.5, where all 12 frames would give base + 5.5.
        assert got[0] == 104.5
        assert got == pytest.approx([base + 4.5 for base in bases])


def _coordinates(rows: pl.DataFrame) -> set[tuple[int, int]]:
    return set(rows.select("item_index", "target_index").iter_rows())


@pytest.mark.required
class TestVideoStrataAndAccessors:
    def test_per_class_judges_boxes_within_their_class(self):
        result = Outliers().evaluate(_videos(), per_target=True, per_class=True)
        populations = result.populations
        instances = populations.filter(pl.col("level") == "instance")
        assert _names(instances["population"]) == {"class_label=thing"}
        assert set(instances["n"]) == {288}
        # A frame has no class: the other levels are judged as one population.
        others = populations.filter(pl.col("level") != "instance")
        assert others["population"].null_count() == len(others)
        assert not populations.filter(pl.col("level") == "unit")["population"].drop_nulls().len()

    def test_strata_by_video_judges_frames_within_their_own_video(self):
        short = Outliers().evaluate(_videos(n_frames=12), strata="item_index").populations
        units = short.filter(pl.col("level") == "unit")
        assert _names(units["population"]) == {f"item_index={video}" for video in range(24)}
        assert set(units["n"]) == {12}
        assert not units["judged"].any()
        assert set(units["reason"]) == {"below 20 rows"}

        # A stratum reaches every level: a video is one row of its sequence level, so nothing there is judged either.
        sequences = short.filter(pl.col("level").is_null())
        assert set(sequences["n"]) == {1}
        assert set(sequences["reason"]) == {"below 20 rows"}

        long = Outliers().evaluate(_videos(n_frames=30), strata="item_index").populations
        units = long.filter(pl.col("level") == "unit")
        assert set(units["n"]) == {30}
        judged = units.filter(pl.col("judged"))
        assert _names(judged["metric_name"]) == {"brightness", "darkness", "mean"}
        assert judged["population"].n_unique() == 24
        assert not long.filter(pl.col("level").is_null())["judged"].any()

    def test_a_metadata_you_pass_supplies_the_strata(self):
        dataset = _videos(n_frames=30)
        md = Metadata(dataset)
        md.add_factors({"weather": ["rain" if row % 2 else "sun" for row in range(24 * 30)]}, level="unit")
        before = list(md.factor_names)
        result = Outliers().evaluate(dataset, metadata=md, strata="weather")
        populations = result.populations
        units = populations.filter(pl.col("level") == "unit")
        assert _names(units["population"]) == {"weather=rain", "weather=sun"}
        assert set(units["n"]) == {360}
        assert units["judged"].any()
        # The sequence rows hold no weather, so they are judged together: one unstratified population for each of
        # the 29 scalar ordered default statistics rolled up to the sequence level.
        assert populations.filter(pl.col("level").is_null())["population"].null_count() == 29
        # The metadata you pass is read for its strata and left as it was; the statistics went on a copy.
        assert list(md.factor_names) == before
        assert "brightness" not in md.factor_names
        assert result.metadata is not None
        assert "brightness" in result.metadata.factor_names
        assert "weather" not in result.metadata.factor_names

    def test_the_metadata_of_the_dataset_supplies_classes_and_strata(self):
        dataset = _videos(n_frames=30)
        md = Metadata(dataset)
        classes = Outliers().evaluate(dataset, per_target=True, per_class=True, metadata=md).populations
        assert _names(classes.filter(pl.col("level") == "instance")["population"]) == {"class_label=thing"}
        strata = Outliers().evaluate(dataset, metadata=md, strata="item_index").populations
        assert _names(strata.filter(pl.col("level") == "unit")["population"]) == {f"item_index={v}" for v in range(24)}

    @pytest.mark.parametrize(
        ("other", "kwargs"),
        [
            pytest.param({"n_videos": 10}, {"per_target": True, "per_class": True}, id="fewer-videos-per-class"),
            pytest.param({"n_frames": 12}, {"strata": "item_index"}, id="fewer-frames-strata"),
        ],
    )
    def test_the_metadata_of_another_tracking_dataset_is_refused(self, other, kwargs):
        with pytest.raises(ValueError, match=r"holds \{.*\} rows by level.*pass the metadata of the data evaluated"):
            Outliers().evaluate(_videos(n_frames=30), metadata=Metadata(_videos(**other)), **kwargs)

    def test_a_metadata_without_tracking_levels_is_refused(self):
        md = Metadata.from_factors({"weather": ["rain", "sun"] * 12})
        with pytest.raises(ValueError, match="must be the Metadata of this tracking dataset"):
            Outliers().evaluate(_videos(), metadata=md, strata="weather")

    def test_stratified_classwise_prune_and_aggregates_run_on_a_video_result(self):
        result = Outliers().evaluate(_planted(frame_offset=100), per_target=True)
        md = result.metadata
        assert md is not None
        # A frame is addressed by its frame_index, a detection by its position in its video.
        flagged = {(0, 18), (1, 10), (1, 12), (1, 112), (2, 1)}
        assert _coordinates(result.data()) == flagged

        by_item = result.aggregate_by_item()
        assert _coordinates(by_item) == flagged
        assert by_item.filter(pl.col("target_index") == 112)["Total"].to_list() == [9]
        by_metric = result.aggregate_by_metric()
        totals = dict(zip(by_metric["metric_name"], by_metric["Total"], strict=True))
        assert totals["instance_distance_edge"] == 3
        assert totals["unit_kurtosis"] == 1

        classwise = result.classwise(md)
        boxes = classwise.populations.filter(pl.col("level") == "instance")
        assert _names(boxes["population"]) == {"class_label=thing"}
        assert _coordinates(classwise.data()) == flagged

        stratified = result.stratified(md, "item_index")
        units = stratified.populations.filter(pl.col("level") == "unit")
        assert _names(units["population"]) == {"item_index=0", "item_index=1", "item_index=2"}
        assert set(units["n"]) == {30}
        assert _coordinates(stratified.data()) == flagged - {(1, 10)}

        plan = result.prune()
        assert plan.discard == {
            SourceIndex(0, 18),
            SourceIndex(1, 10),
            SourceIndex(1, 12),
            SourceIndex(2, 1),
            SourceIndex(1, 112, "unit"),
        }

    def test_the_image_and_metadata_routes_are_unchanged(self):
        # Neither route is tracking: each equals a route that never reaches the video code.
        rng = np.random.default_rng(0)
        images = [_noise(rng) for _ in range(30)]
        images[7] = np.full(_SHAPE, 250, dtype=np.uint8)
        stats = image_stats(images, stats=DEFAULT_OUTLIERS_FLAGS, per_image=True, per_target=True)
        got = Outliers().evaluate(images).data()
        assert_frame_equal(got, Outliers().from_stats(stats).data())
        assert got.shape == (9, 8)
        assert set(got["item_index"]) == {7}

        md, added = _manual_metadata(_planted())
        levels: dict[str, FactorLevel] = {name: level for name, level in factor_levels(md).items() if name in added}
        got = Outliers().evaluate(md, factors=added, per_image=True).data().select(_ROW)
        assert_frame_equal(got, Outliers().from_stats(factor_stats(md, levels)).data().select(_ROW))
        assert got.shape == (9, 5)
        assert got.row(0) == (1, 12, "unit", "brightness", 250.0)

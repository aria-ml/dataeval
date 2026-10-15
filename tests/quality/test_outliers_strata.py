"""Stratified Outliers: strata=, .stratified(), the population column."""

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.flags import ImageStats
from dataeval.quality import Outliers
from dataeval.quality._outliers_rows import level_strata
from dataeval.utils.thresholds import ConstantThreshold
from tests.conftest import MockMetadata, MockODDataset, get_bboxes
from tests.metadata.test_structurers import _mot_dataset

_BRIGHT = ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD


def _two_weathers(n=60, planted=3):
    rng = np.random.default_rng(0)
    weather = np.array(["rain"] * (n // 2) + ["clear"] * (n // 2))
    x = np.where(weather == "rain", rng.normal(0, 1, n), rng.normal(50, 1, n))
    x[planted] = 6.0  # within rain, extreme; over the level, ordinary
    return Metadata.from_factors({"x": x, "weather": weather})


def _two_videos(**kwargs):
    """Two videos of 30 frames, two detections a frame; the first video rains, the second is clear."""
    md = Metadata(_mot_dataset([[2] * 30, [2] * 30]), **kwargs)
    md.add_factors({"weather": ["rain"] * 30 + ["clear"] * 30}, level="unit")
    return md


def _populations(result):
    return set(result.populations["population"].cast(pl.Utf8).to_list())


def _mock(class_labels, index2label=None):
    """A MetadataLike with no level schema, as the doctests build one."""
    n = len(class_labels)
    labels = np.asarray(class_labels, np.intp)
    return MockMetadata(labels, np.zeros((n, 1), np.int64), ["f"], [False], index2label or {})


def _bright_class_images(n=40, planted=5):
    """Images whose second half is bright, with one bright image planted in the first half."""
    rng = np.random.default_rng(0)
    images = rng.random((n, 3, 16, 16))
    images[n // 2 :] = images[n // 2 :] * 0.5 + 0.5
    images[planted] = images[planted] * 0.5 + 0.5
    return list(images)


@pytest.mark.required
class TestStrata:
    def test_a_value_is_judged_within_its_stratum(self):
        md = _two_weathers()
        assert 3 not in Outliers().evaluate(md).data()["item_index"].to_list()
        result = Outliers().evaluate(md, strata=["weather"])
        row = result.data().filter(pl.col("item_index") == 3).row(0, named=True)
        assert row["population"] == "weather=rain"

    def test_a_bare_string_names_one_stratum(self):
        md = _two_weathers()
        assert (
            Outliers().evaluate(md, strata="weather").data().equals(Outliers().evaluate(md, strata=["weather"]).data())
        )

    def test_stratified_equals_strata(self):
        md = _two_weathers()
        assert (
            Outliers()
            .evaluate(md)
            .stratified(md, "weather")
            .data()
            .equals(Outliers().evaluate(md, strata=["weather"]).data())
        )

    def test_two_strata_cross(self):
        rng = np.random.default_rng(1)
        md = Metadata.from_factors({
            "x": rng.normal(0, 1, 160),
            "weather": np.tile(["rain", "clear"], 80),
            "time": np.repeat(["day", "night"], 80),
        })
        pops = Outliers().evaluate(md, strata=["weather", "time"]).populations["population"].cast(pl.Utf8).to_list()
        assert sorted(pops) == [
            "weather=clear, time=day",
            "weather=clear, time=night",
            "weather=rain, time=day",
            "weather=rain, time=night",
        ]

    def test_an_all_null_stratum_is_one_population(self):
        md = Metadata.from_factors({"x": np.random.default_rng(0).normal(0, 1, 30), "weather": [None] * 30})
        pops = Outliers().evaluate(md, strata=["weather"]).populations
        assert pops["population"].cast(pl.Utf8).to_list() == ["weather=null"]

    def test_a_continuous_stratum_is_refused(self):
        md = _two_weathers()
        with pytest.raises(ValueError, match="given="):
            Outliers().evaluate(md, strata=["x"])

    def test_an_unknown_stratum_is_a_key_error(self):
        with pytest.raises(KeyError):
            Outliers().evaluate(_two_weathers(), strata=["nope"])

    def test_no_strata_and_the_class_label_are_refused(self):
        md = _two_weathers()
        with pytest.raises(ValueError, match="names no strata"):
            Outliers().evaluate(md, strata=[])
        with pytest.raises(KeyError, match="classwise"):
            Outliers().evaluate(md, strata=["class_label"])

    def test_strata_with_per_class_is_refused(self):
        md = _two_weathers()
        with pytest.raises(ValueError, match="per_class"):
            Outliers().evaluate(md, strata=["weather"], per_class=True)

    def test_a_box_takes_its_frames_stratum(self):
        md = _two_videos()
        rng = np.random.default_rng(0)
        area = np.r_[rng.normal(0, 1, 60), rng.normal(50, 1, 60)]
        area[70] = 44.0  # a clear-weather detection: extreme among clear frames, ordinary over the level
        md.add_factors({"area": area}, level="instance")
        assert Outliers().evaluate(md, levels="instance").data().is_empty()
        result = Outliers().evaluate(md, levels="instance", strata=["weather"])
        rows = result.data().filter((pl.col("item_index") == 1) & (pl.col("target_index") == 10))
        assert rows["population"].to_list() == ["weather=clear"]

    def test_a_stratum_leaves_unreached_levels_whole(self):
        md = _two_videos()
        md.add_factors({"length_m": [1.0, 5.0]}, level="sequence")
        md.add_factors({"brightness": np.r_[np.zeros(59), 3.0]}, level="unit")
        result = Outliers(outlier_threshold=ConstantThreshold(upper=2.0)).evaluate(md, strata=["weather"])
        frame = result.data()
        sequence = frame.filter(pl.col("metric_name") == "length_m")
        unit = frame.filter(pl.col("metric_name") == "brightness")
        assert sequence["item_index"].to_list() == [1]
        assert sequence["population"].to_list() == [None]
        assert unit["population"].to_list() == ["weather=clear"]

    def test_strata_reaching_no_judged_level_are_refused(self):
        md = _two_videos()
        md.add_factors({"length_m": [1.0, 5.0]}, level="sequence")
        with pytest.raises(ValueError, match="sequence"):
            Outliers().evaluate(md, levels="sequence", strata=["weather"])

    def test_an_untracked_detection_joins_the_null_population(self):
        md = Metadata(_mot_dataset([[[0, 1, -1]] * 30]))
        md.add_factors({"kind": ["car", "person"]}, level="track")
        area = np.zeros(90)
        area[2] = 3.0  # the first frame's untracked detection
        md.add_factors({"area": area}, level="instance")
        result = Outliers(outlier_threshold=ConstantThreshold(upper=2.0)).evaluate(
            md, levels="instance", strata=["kind"]
        )
        assert result.data()["population"].to_list() == ["kind=null"]
        assert _populations(result) == {"kind=car", "kind=person", "kind=null"}

    def test_a_reserved_stratum_leaves_track_whole(self):
        md = _two_videos()
        md.add_factors({"brightness": np.r_[3.0, np.zeros(59)]}, level="unit")
        md.add_factors({"speed": [0.0, 0.0, 0.0, 3.0]}, level="track")
        result = Outliers(outlier_threshold=ConstantThreshold(upper=2.0)).evaluate(
            md, levels=("unit", "track"), strata=["unit_index"]
        )
        frame = result.data()
        assert frame.filter(pl.col("metric_name") == "speed")["population"].to_list() == [None]
        assert frame.filter(pl.col("metric_name") == "brightness")["population"].to_list() == [
            "item_index=0, unit_index=0"
        ]

    def test_a_frame_stratum_is_one_frame_of_one_video(self):
        md = _two_videos()
        md.add_factors({"area": np.zeros(120)}, level="instance")
        result = Outliers().evaluate(md, levels="instance", strata=["unit_index"])
        pops = result.populations
        assert pops.height == 60
        assert pops["n"].to_list() == [2] * 60
        assert {"item_index=0, unit_index=0", "item_index=1, unit_index=0"} <= _populations(result)
        tracks = Outliers().evaluate(md, levels="instance", strata=["track_id"])
        assert _populations(tracks) == {f"item_index={i}, track_id={t}" for i in (0, 1) for t in (0, 1)}

    def test_inherited_false_is_honored(self):
        md = _two_videos(inherited=False)
        md.add_factors({"area": np.zeros(120)}, level="instance")
        with pytest.raises(ValueError, match="inherited=False"):
            Outliers().evaluate(md, levels="instance", strata=["weather"])

    def test_the_dataset_path_takes_strata_with_metadata(self, get_od_dataset):
        dataset = get_od_dataset(_bright_class_images(), 2)
        md = Metadata(dataset)
        md.add_factors({"weather": ["rain"] * 20 + ["clear"] * 20}, level=md.item_level)
        outliers = Outliers(flags=_BRIGHT)
        result = outliers.evaluate(dataset, per_target=True, strata=["weather"], metadata=md)
        assert _populations(result) == {"weather=rain", "weather=clear"}
        assert result.data().filter(pl.col("item_index") == 5)["population"].unique().to_list() == ["weather=rain"]
        with pytest.raises(ValueError, match="level schema"):
            outliers.evaluate(dataset, strata=["weather"], metadata=_mock(np.zeros(40)))
        with pytest.raises(ValueError, match="level schema"):
            outliers.evaluate(dataset, strata=["weather"])

    def test_stratified_with_no_columns_points_to_itemwise(self):
        md = _two_weathers()
        with pytest.raises(ValueError, match="itemwise"):
            Outliers().evaluate(md).stratified(md)

    def test_stratified_refuses_a_metadata_without_levels(self):
        md = _two_weathers()
        with pytest.raises(ValueError, match="level schema"):
            Outliers().evaluate(md).stratified(_mock(np.zeros(60)), "weather")  # type: ignore

    def test_stratified_refuses_a_metadata_of_another_size(self):
        with pytest.raises(ValueError, match="60"):
            Outliers().evaluate(_two_weathers()).stratified(_two_weathers(n=40), "weather")

    def test_itemwise_drops_the_strata(self):
        md = _two_weathers()
        result = Outliers().evaluate(md, strata=["weather"]).itemwise()
        assert result.strata is None
        assert result.data().equals(Outliers().evaluate(md).data())

    def test_with_threshold_keeps_the_strata(self):
        md = _two_weathers()
        result = Outliers().evaluate(md, strata=["weather"]).with_threshold(4.0)
        assert _populations(result) == {"weather=rain", "weather=clear"}
        assert result.data().filter(pl.col("item_index") == 3)["population"].to_list() == ["weather=rain"]

    def test_level_strata_codes_one_level(self):
        md = _two_videos()
        codes, labels = level_strata(md, ["weather"], "instance")
        assert codes is not None
        assert codes.dtype == pl.Utf8
        assert [labels[int(code)] for code in codes.to_list()] == ["weather=rain"] * 60 + ["weather=clear"] * 60
        assert level_strata(md, ["weather"], "sequence") == (None, {})

    def test_a_call_without_strata_has_no_population_column(self):
        frame = Outliers(outlier_threshold=ConstantThreshold(upper=40.0)).evaluate(_two_weathers()).data()
        assert not frame.is_empty()
        assert not {"population", "method", "observed"} & set(frame.columns)


@pytest.mark.required
class TestClasswiseOnMetadata:
    def test_classwise_equals_stratified_on_the_class_factor(self, get_od_dataset):
        dataset = get_od_dataset(_bright_class_images(), 2)
        md = Metadata(dataset)
        md.add_factors({"vehicle": ["car"] * 20 + ["truck"] * 20}, level=md.item_level)
        result = Outliers(flags=_BRIGHT).evaluate(dataset)
        keys = ["item_index", "metric_name", "metric_value"]
        by_class = result.classwise(md.classed_by("vehicle")).data()
        by_stratum = result.stratified(md, "vehicle").data()
        assert not by_class.is_empty()
        assert by_class.select(keys).equals(by_stratum.select(keys))
        assert by_class["population"].cast(pl.Utf8).to_list() == by_stratum["population"].cast(pl.Utf8).to_list()

    def test_mixed_class_items_are_their_own_population(self):
        images = _bright_class_images(planted=37)
        labels = [[0, 0]] * 20 + [[1, 1]] * 15 + [[0, 1]] * 5
        dataset = MockODDataset(images, labels, get_bboxes(40, 2, False))
        md = Metadata(dataset)  # type: ignore
        result = Outliers(flags=ImageStats.PIXEL_MEAN, outlier_threshold=ConstantThreshold(upper=0.7)).evaluate(dataset)
        rows = result.classwise(md).data().filter(pl.col("item_index") == 37)
        assert rows["population"].cast(pl.Utf8).to_list() == [f"{md.class_axis}=null"]

    def test_classwise_with_a_mock_metadata_still_works(self):
        images = _bright_class_images()
        result = Outliers(flags=_BRIGHT).evaluate(images).classwise(_mock(np.repeat([0, 1], 20), {0: "dim", 1: "lit"}))
        assert 5 in result.data()["item_index"].to_list()
        assert _populations(result) == {"class_label=dim", "class_label=lit"}

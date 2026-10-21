"""``add_factors(return_collisions=True)`` reports where incoming values disagree with existing ones."""

import logging
import math

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.core import video_stats
from dataeval.flags import VideoStats
from dataeval.types import FactorCollision, SourceIndex
from tests.core.test_video_stats import _stream
from tests.embeddings.test_embeddings import MockDataset
from tests.metadata.test_structurers import _FrameTracks, _mot_dataset, _MOTTarget


def _images(**columns: list) -> Metadata:
    """Four classified images whose datum metadata holds the given columns."""
    rows = [{name: values[i] for name, values in columns.items()} for i in range(4)]
    return Metadata(MockDataset(np.ones((4, 3, 3)), np.eye(3)[[0, 1, 2, 0]], rows))


def _tracking() -> Metadata:
    """Two sequences: tracks 5 and 9 in the first, track 1 in the second."""
    return Metadata(_mot_dataset([[[5, 9], [5]], [[1]]]))


@pytest.mark.required
class TestReturn:
    def test_default_returns_none(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        assert md.add_factors({"w": [1.0, 2.0, 3.0, 5.0]}, level="unit") is None

    def test_no_colliding_name_reports_nothing(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        assert md.add_factors({"v": [1.0, 2.0, 3.0, 5.0]}, level="unit", return_collisions=True) == {}

    def test_agreeing_values_report_nothing_and_still_land(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        assert md.add_factors({"w": [1.0, 2.0, 3.0, 4.0]}, level="unit", return_collisions=True) == {}
        assert md.rows_at("unit")["w_added"].to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_no_factors_with_a_tolerance_key_is_refused(self):
        with pytest.raises(ValueError, match="nope"):
            _images(w=[1.0] * 4).add_factors({}, return_collisions=True, collision_tolerance={"nope": 0.1})
        assert _images(w=[1.0] * 4).add_factors({}, return_collisions=True) == {}


@pytest.mark.required
class TestDifferences:
    def test_values_beyond_tolerance_are_reported(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors({"w": [1.0, 2.0, 3.3, 8.0]}, level="unit", return_collisions=True)
        hit = report["w"]
        assert isinstance(hit, FactorCollision)
        assert (hit.factor, hit.column, hit.level, hit.compared) == ("w", "w_added", "unit", 4)
        assert hit.differing.columns == ["item_index", "existing", "incoming", "relative_difference"]
        assert hit.differing["item_index"].to_list() == [2, 3]
        assert hit.differing["existing"].to_list() == [3.0, 4.0]
        assert hit.differing["incoming"].to_list() == [3.3, 8.0]
        assert hit.differing["relative_difference"].to_list() == pytest.approx([0.3 / 3.3, 0.5])

    def test_values_within_a_scalar_tolerance_are_not(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors(
            {"w": [1.0, 2.0, 3.3, 8.0]}, level="unit", return_collisions=True, collision_tolerance=0.1
        )
        assert report["w"].differing["item_index"].to_list() == [3]

    def test_a_mapping_gives_each_factor_its_own_tolerance(self):
        md = _images(a=[1.0, 2.0, 3.0, 4.0], b=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors(
            {"b": [1.0, 2.0, 3.0, 4.001], "a": [1.0, 2.0, 3.0, 8.0]},
            level="unit",
            return_collisions=True,
            collision_tolerance={"a": 0.5},
        )
        # `a` differs by exactly its tolerance; `b` is not in the mapping, so compared exactly.
        assert list(report) == ["b"]

    def test_report_follows_the_order_factors_were_passed(self):
        md = _images(a=[1.0, 2.0, 3.0, 4.0], b=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors({"b": [0.0] * 4, "a": [0.0] * 4}, level="unit", return_collisions=True)
        assert list(report) == ["b", "a"]

    def test_integer_columns_compare_numerically(self):
        md = _images(n=[1, 2, 3, 4])
        report = md.add_factors({"n": np.array([1, 2, 3, 5])}, level="unit", return_collisions=True)
        assert report["n"].differing["relative_difference"].to_list() == pytest.approx([0.2])

    def test_integers_past_float_precision_differ(self):
        """2**53 + 1 and 2**53 are one float apart from nothing: the ratio must not round to agreement."""
        md = _images(n=[2**53, 2**53, 1, 2])
        report = md.add_factors({"n": np.array([2**53 + 1, 2**53, 1, 2])}, level="unit", return_collisions=True)
        assert report["n"].differing["item_index"].to_list() == [0]
        assert report["n"].differing["relative_difference"][0] > 0

    def test_epoch_nanoseconds_a_hundred_apart_differ(self):
        stamp = 1_700_000_000_000_000_000
        md = _images(ts=[stamp] * 4)
        incoming = np.array([stamp + 100, stamp, stamp, stamp])
        report = md.add_factors({"ts": incoming}, level="unit", return_collisions=True)
        assert report["ts"].differing["incoming"].to_list() == [stamp + 100]
        assert report["ts"].differing["relative_difference"][0] == pytest.approx(100 / (stamp + 100))

    def test_unsigned_against_signed_integers(self):
        md = _images(n=[2**53, 2, 3, 4])
        incoming = np.array([2**53 + 1, 2, 3, 1], dtype=np.uint64)
        report = md.add_factors({"n": incoming}, level="unit", return_collisions=True)
        assert report["n"].differing["item_index"].to_list() == [0, 3]
        assert report["n"].differing["relative_difference"].to_list() == pytest.approx([1 / (2**53 + 1), 0.75])

    def test_an_infinity_against_a_value_differs_with_a_null_ratio(self):
        md = _images(w=[1.0, math.inf, -math.inf, math.inf])
        report = md.add_factors({"w": [1.0, 5.0, math.inf, math.inf]}, level="unit", return_collisions=True)
        hit = report["w"].differing
        assert hit["item_index"].to_list() == [1, 2]
        assert hit["relative_difference"].to_list() == [None, None]

    def test_both_zero_agree(self):
        md = _images(w=[0.0, 0.0, 0.0, 0.0])
        assert md.add_factors({"w": [0.0, -0.0, 0.0, 0.0]}, level="unit", return_collisions=True) == {}


@pytest.mark.required
class TestOverwrite:
    def test_overwrite_reports_the_replaced_values(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors({"w": [1.0, 2.0, 3.0, 9.0]}, level="unit", overwrite=True, return_collisions=True)
        assert report["w"].column == "w"
        assert report["w"].differing["existing"].to_list() == [4.0]
        assert report["w"].differing["incoming"].to_list() == [9.0]
        assert md.rows_at("unit")["w"].to_list() == [1.0, 2.0, 3.0, 9.0]

    def test_no_overwrite_names_the_added_column(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors({"w": [1.0, 2.0, 3.0, 9.0]}, level="unit", return_collisions=True)
        assert report["w"].column == "w_added"
        assert md.rows_at("unit")["w"].to_list() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.required
class TestKeyed:
    def test_only_the_named_rows_are_compared(self):
        md = _tracking()
        md.add_factors(
            {"track_ids": [5, 9, 1], "item_index": [0, 0, 1], "speed": [0.5, 0.9, 0.1]}, level="track", key="track_id"
        )
        report = md.add_factors(
            {"track_ids": [9], "item_index": [0], "speed": [0.8]},
            level="track",
            key="track_id",
            overwrite=True,
            return_collisions=True,
        )
        hit = report["speed"]
        assert hit.compared == 1
        assert hit.differing.columns == ["item_index", "track_id", "existing", "incoming", "relative_difference"]
        assert hit.differing.select("item_index", "track_id", "existing", "incoming").rows() == [(0, 9, 0.9, 0.8)]
        assert md.rows_at("track")["speed"].to_list() == [0.5, 0.8, 0.1]

    def test_filling_empty_rows_is_not_a_collision(self):
        """One sequence per call folds into one column; the second call fills rows the first left empty."""
        md = _tracking()
        md.add_factors({"track_ids": [5, 9], "item_index": [0, 0], "speed": [0.5, 0.9]}, level="track", key="track_id")
        report = md.add_factors(
            {"track_ids": [1], "item_index": [1], "speed": [0.1]}, level="track", key="track_id", return_collisions=True
        )
        assert report == {}
        assert md.rows_at("track")["speed"].to_list() == [0.5, 0.9, 0.1]

    def test_unit_level_keyed_by_unit_index(self):
        md = _tracking()
        md.add_factors({"glare": np.zeros(md.level_counts["unit"])}, level="unit")
        report = md.add_factors(
            {"unit_index": [1], "item_index": [0], "glare": [0.5]},
            level="unit",
            key="unit_index",
            return_collisions=True,
        )
        assert report["glare"].differing.select("item_index", "unit_index").rows() == [(0, 1)]


@pytest.mark.required
class TestMissing:
    def test_only_rows_present_on_both_sides_are_compared(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"w": [1.0, None, math.nan, 4.0]}, level="unit", overwrite=True)
        report = md.add_factors({"w": [None, 5.0, math.nan, 7.0]}, level="unit", return_collisions=True)
        # Row 0 and 1 are missing on one side, row 2 on both: only row 3 is compared.
        assert report["w"].compared == 1
        assert report["w"].differing["item_index"].to_list() == [3]

    def test_text_differs_only_where_unequal(self):
        md = _images(s=["a", "b", "c", "d"])
        report = md.add_factors({"s": ["a", "x", None, "d"]}, level="unit", return_collisions=True)
        hit = report["s"]
        assert hit.compared == 3
        assert hit.differing.select("item_index", "existing", "incoming").rows() == [(1, "b", "x")]
        assert hit.differing["relative_difference"].to_list() == [None]

    def test_booleans_are_not_numeric(self):
        md = _images(f=[True, False, True, False])
        report = md.add_factors({"f": [True, True, True, False]}, level="unit", return_collisions=True)
        assert report["f"].differing["relative_difference"].to_list() == [None]

    def test_mixed_types_compare_for_equality(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        report = md.add_factors({"w": ["1.0", "2", "x", "y"]}, level="unit", return_collisions=True)
        assert report["w"].compared == 4
        assert report["w"].differing["relative_difference"].null_count() == 4


@pytest.mark.required
class TestInvalidTolerance:
    @pytest.mark.parametrize(
        "tolerance",
        [-0.1, math.nan, math.inf, {"w": -1.0}, {"w": math.nan}, {"nope": 0.1}],
        ids=["negative", "nan", "inf", "negative-entry", "nan-entry", "unknown-key"],
    )
    def test_refused_before_anything_is_written(self, tolerance):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        before = md.rows_at("unit").columns
        with pytest.raises(ValueError, match="collision_tolerance"):
            md.add_factors({"w": [0.0] * 4}, level="unit", return_collisions=True, collision_tolerance=tolerance)
        assert md.rows_at("unit").columns == before

    def test_ignored_without_return_collisions(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        assert md.add_factors({"w": [0.0] * 4}, level="unit", collision_tolerance=-1.0) is None


@pytest.mark.required
class TestLevels:
    def test_ancestor_values_are_gathered_down(self):
        md = _tracking()
        md.add_factors({"weather": [1.0, 2.0]}, level="sequence")
        # Track rows are (0, 5), (0, 9) and (1, 1), so they read 1.0, 1.0 and 2.0 from their sequences.
        report = md.add_factors({"weather": [1.0, 3.0, 2.0]}, level="track", return_collisions=True)
        hit = report["weather"]
        assert hit.compared == 3
        assert hit.differing.select("item_index", "track_id", "existing", "incoming").rows() == [(0, 9, 1.0, 3.0)]

    def test_an_unreadable_existing_column_is_skipped_and_logged(self, caplog):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        md.add_factors({"x": np.arange(4.0)}, level="instance")
        with caplog.at_level(logging.INFO, logger="dataeval"):
            report = md.add_factors({"x": np.zeros(4)}, level="unit", return_collisions=True)
        assert report == {}
        assert any("'x'" in r.getMessage() and "not compared" in r.getMessage() for r in caplog.records)


def _boxes(n: int):
    class _Target:
        boxes = np.tile(np.array([[0.0, 0.0, 2.0, 2.0]]), (n, 1))
        labels = np.zeros(n, dtype=int)
        scores = np.ones(n)

    return _Target()


@pytest.mark.required
class TestSourceIndexSplit:
    """One factor addressed at two levels becomes two columns, each its own entry."""

    @staticmethod
    def _split(values):
        index = [SourceIndex(0), SourceIndex(1), SourceIndex(0, 0), SourceIndex(1, 0)]
        return {"stats": {"mean": np.asarray(values, dtype=float)}, "source_index": index}

    def _md(self) -> Metadata:
        md = Metadata(MockDataset(np.ones((2, 3, 3)), [_boxes(1), _boxes(1)], None))
        md.add_factors(self._split([1.0, 2.0, 3.0, 4.0]))
        return md

    def test_keyed_by_the_placed_name(self):
        report = self._md().add_factors(self._split([1.0, 2.0, 3.0, 9.0]), return_collisions=True)
        assert list(report) == ["instance_mean"]
        hit = report["instance_mean"]
        assert (hit.factor, hit.column, hit.level) == ("instance_mean", "instance_mean_added", "instance")
        assert hit.differing.columns == ["item_index", "target_index", "existing", "incoming", "relative_difference"]

    def test_a_passed_name_covers_both_columns(self):
        report = self._md().add_factors(
            self._split([1.0, 2.0, 3.0, 9.0]), return_collisions=True, collision_tolerance={"mean": 1.0}
        )
        assert report == {}

    def test_a_placed_name_takes_precedence(self):
        report = self._md().add_factors(
            self._split([1.0, 2.0, 3.0, 9.0]),
            return_collisions=True,
            collision_tolerance={"mean": 1.0, "instance_mean": 0.0},
        )
        assert list(report) == ["instance_mean"]


@pytest.mark.required
def test_video_stats_against_reported_durations():
    """The measured duration of the second video disagrees with the one its metadata reports."""
    streams = [_stream(10), _stream(20)]
    targets = [_MOTTarget([_FrameTracks([0]) for _ in stream]) for stream in streams]
    ds = MockDataset(streams, targets, [{"video_duration": 10 / 30}, {"video_duration": 1.0}])
    md = Metadata(ds)
    report = md.add_factors(
        video_stats(ds, stats=VideoStats.DESCRIPTIVE),
        return_collisions=True,
        collision_tolerance={"video_duration": 0.01},
    )
    assert list(report) == ["video_duration"]
    hit = report["video_duration"]
    assert (hit.level, hit.compared) == ("sequence", 2)
    assert hit.differing["item_index"].to_list() == [1]
    assert hit.differing["existing"].to_list() == [1.0]
    assert hit.differing["incoming"].to_list() == pytest.approx([20 / 30])

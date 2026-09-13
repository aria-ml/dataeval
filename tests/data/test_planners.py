"""Tests for segment planners."""

from typing import Any, cast

import numpy as np
import pytest

from dataeval.data import Cuts, SegmentPlanner, SequenceInfo, Window
from dataeval.data._planners import validated_plan
from dataeval.protocols import DatumMetadata


def info(n_frames: int, source_id: str = "vid0", metadata: dict[str, Any] | None = None) -> SequenceInfo:
    return SequenceInfo(
        index=0,
        source_id=source_id,
        n_frames=n_frames,
        metadata=cast(DatumMetadata, {"id": source_id, **(metadata or {})}),
    )


class _Rows(SegmentPlanner):
    """Planner stub returning fixed rows for validation testing."""

    def __init__(self, rows: Any) -> None:
        self.rows = rows

    def plan(self, info: SequenceInfo) -> Any:
        return self.rows


class TestWindow:
    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"size": 8}, [[0, 8], [8, 16], [16, 20]]),
            ({"size": 8, "drop_remainder": True}, [[0, 8], [8, 16]]),
            ({"size": 8, "stride": 4}, [[0, 8], [4, 12], [8, 16], [12, 20], [16, 20]]),
            ({"size": 8, "stride": 4, "drop_remainder": True}, [[0, 8], [4, 12], [8, 16], [12, 20]]),
            ({"size": 8, "stride": 12}, [[0, 8], [12, 20]]),
        ],
    )
    def test_spec_table(self, kwargs, expected):
        np.testing.assert_array_equal(Window(**kwargs).plan(info(20)), expected)

    def test_short_video_is_one_window_unless_remainder_is_dropped(self):
        np.testing.assert_array_equal(Window(8).plan(info(5)), [[0, 5]])
        assert Window(8, drop_remainder=True).plan(info(5)).shape == (0, 2)

    def test_empty_video_plans_nothing(self):
        assert Window(8).plan(info(0)).shape == (0, 2)

    def test_plan_dtype(self):
        assert Window(8).plan(info(20)).dtype == np.intp

    @pytest.mark.parametrize("kwargs", [{"size": 0}, {"size": -1}, {"size": 4, "stride": 0}])
    def test_rejects_bad_sizes(self, kwargs):
        with pytest.raises(ValueError, match="Window"):
            Window(**kwargs)

    def test_repr(self):
        assert repr(Window(8)) == "Window(size=8, stride=8, drop_remainder=False)"


class TestCuts:
    def test_cut_points_are_boundaries(self):
        np.testing.assert_array_equal(Cuts({"vid0": [100, 250]}).plan(info(400)), [[0, 100], [100, 250], [250, 400]])

    def test_sorted_and_deduplicated(self):
        plan = Cuts({"vid0": [250, 100, 100]}).plan(info(400))
        np.testing.assert_array_equal(plan, [[0, 100], [100, 250], [250, 400]])

    @pytest.mark.parametrize("cut", [0, 400, 401, -1])
    def test_out_of_range_raises_naming_the_cut(self, cut):
        with pytest.raises(ValueError, match=f"{cut}"):
            Cuts({"vid0": [cut]}).plan(info(400))

    def test_missing_entry_is_one_segment_and_is_logged(self, caplog):
        with caplog.at_level("INFO", logger="dataeval"):
            np.testing.assert_array_equal(Cuts({"other": [1]}).plan(info(10)), [[0, 10]])
        assert any("no cut points" in record.message for record in caplog.records)

    def test_empty_cut_sequence_is_one_segment(self):
        np.testing.assert_array_equal(Cuts({"vid0": []}).plan(info(10)), [[0, 10]])

    def test_string_names_a_metadata_key(self):
        np.testing.assert_array_equal(Cuts("shots").plan(info(10, metadata={"shots": [4]})), [[0, 4], [4, 10]])
        np.testing.assert_array_equal(Cuts("shots").plan(info(10)), [[0, 10]])

    def test_empty_video_plans_nothing(self):
        assert Cuts({}).plan(info(0)).shape == (0, 2)

    def test_repr(self):
        assert "Cuts(" in repr(Cuts({"vid0": [1]}))


class TestValidatedPlan:
    @pytest.mark.parametrize(
        "rows",
        [[[0, 0]], [[-1, 2]], [[0, 21]], [[5, 8], [0, 3]], [[0, 1, 2]], [[0.0, 1.5]], [0, 1]],
    )
    def test_rejects_bad_rows(self, rows):
        with pytest.raises(ValueError, match="_Rows"):
            validated_plan(rows, _Rows(rows), info(20))

    def test_accepts_overlap_and_gaps(self):
        rows = [[0, 8], [4, 12], [16, 20]]
        np.testing.assert_array_equal(validated_plan(rows, _Rows(rows), info(20)), rows)

    def test_empty_plan_is_allowed(self):
        assert validated_plan([], _Rows([]), info(20)).shape == (0, 2)
        assert validated_plan(np.empty((0, 2)), _Rows([]), info(20)).shape == (0, 2)

    def test_planner_must_implement_plan(self):
        with pytest.raises(TypeError):
            SegmentPlanner()  # type: ignore[abstract]

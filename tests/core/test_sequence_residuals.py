"""Each value judged against its neighbors along a sequence: dataeval.core.sequence_residuals (#1523)."""

import numpy as np
import polars as pl
import pytest

from dataeval.core import sequence_residuals
from dataeval.core._sequence_residuals import REASONS


def _one(values, monotonic=False, order=None, window=5):
    """Score one column of one sequence, ordered by position unless `order` is given."""
    n = len(values)
    data = {"x": values, "video": [0] * n, "frame": list(range(n)) if order is None else order}
    return sequence_residuals(
        data, ["x"], sequence="video", order="frame", monotonic=["x"] if monotonic else [], window=window
    )


def _flagged(scores, limit=6.0):
    """Rows whose score is past a fixed limit, for tests that need no fitted threshold."""
    return np.flatnonzero(np.abs(np.nan_to_num(scores)) > limit).tolist()


@pytest.mark.required
class TestNeighbors:
    def test_spike_scores_high_only_at_its_row(self):
        values = np.random.default_rng(0).normal(0, 1, 60)
        values[30] += 8
        assert _flagged(_one(values)["score"]) == [30]

    def test_runs_up_to_three_rows_are_caught_and_six_read_as_a_segment(self):
        rng = np.random.default_rng(1)
        three, six = rng.normal(0, 1, 80), rng.normal(0, 1, 80)
        three[40:43] += 10
        six[40:46] += 10
        assert set(_flagged(_one(three)["score"])) >= {40, 41, 42}
        assert not set(_flagged(_one(six)["score"])) & set(range(40, 46))

    @pytest.mark.parametrize(
        "shape",
        [
            np.r_[np.zeros(30), np.full(30, 10.0)],  # a scene cut
            np.arange(60) * 0.5,  # a ramp
            np.cumsum(np.random.default_rng(2).normal(0, 1, 60)),  # a random walk
        ],
        ids=["cut", "ramp", "walk"],
    )
    def test_cuts_ramps_and_walks_score_low(self, shape):
        values = shape + np.random.default_rng(3).normal(0, 0.1, 60)
        assert _flagged(_one(values)["score"]) == []

    def test_a_fade_scores_high_at_most_at_its_two_ends(self):
        values = np.random.default_rng(4).normal(0, 1, 60)
        values[25:35] += 5 * np.arange(1, 11)
        values[35:] += 50
        assert set(_flagged(_one(values)["score"])) <= {25, 26, 34, 35}

    def test_first_and_last_rows_follow_a_trend(self):
        values = np.arange(40) * 2.0 + np.random.default_rng(5).normal(0, 0.2, 40)
        result = _one(values)
        assert abs(result["score"][0]) < 6
        assert abs(result["score"][-1]) < 6

    def test_a_spike_at_the_first_row_scores_high(self):
        values = np.random.default_rng(6).normal(0, 1, 40)
        values[0] += 10
        assert _flagged(_one(values)["score"]) == [0]

    def test_a_small_spike_along_a_steep_ramp_is_missed(self):
        """The known limit: a glitch in the ramp's direction below its rise over three rows sits near the side ahead."""
        values = np.arange(60) * 3.0
        values[30] += 6.0
        assert 30 not in _flagged(_one(values)["score"], limit=3.0)

    def test_the_line_places_each_side_at_its_median_order_across_a_gap(self):
        # The row at order 10 sits between a left side whose median order is 2 (its mean is 3) and a right side across
        # a gap whose median order is 32. Read at order 10, the line through (2, 0) and (32, 10) is 8/3; through the
        # left mean it would be 70/29, so a residual of 0 shows the sides are placed at their median order.
        frames = [0, 1, 2, 3, 9, 10, 30, 31, 32, 33, 34]
        values = [0.0] * 5 + [8 / 3] + [10.0] * 5
        result = _one(values, order=frames)
        assert result["residual"][5] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("values", "order", "window"),
        [(np.arange(10.0), None, 1), (np.arange(5.0), [0] * 5, 5)],
        ids=["one-row-side", "one-order"],
    )
    def test_an_end_row_with_no_end_line_is_scored_against_its_side(self, values, order, window):
        assert np.isfinite(_one(values, order=order, window=window)["score"]).all()

    def test_sequences_are_scored_apart(self):
        values = np.r_[np.zeros(20), np.full(20, 100.0)]
        data = {"x": values, "video": [0] * 20 + [1] * 20, "frame": list(range(20)) * 2}
        result = sequence_residuals(data, ["x"], sequence="video", order="frame")
        np.testing.assert_allclose(result["residual"], 0.0)
        assert result["sequence"].tolist() == [0] * 20 + [1] * 20


@pytest.mark.required
class TestReasons:
    def test_each_reason(self):
        frame = pl.DataFrame({
            "x": [1.0, None, 2.0, 3.0, 4.0, 5.0, float("inf"), float("nan"), 6.0],
            "video": pl.Series([0, 0, 0, None, 1, 1, 0, 0, 0]),
            "frame": pl.Series([0, 1, 2, 3, None, 5, 6, 7, 8], dtype=pl.Float64),
        })
        result = sequence_residuals(frame, ["x"], sequence="video", order="frame")
        reasons = [REASONS[code] for code in result["reason"]]
        assert reasons == [
            "",
            "missing value",
            "",
            "missing sequence",
            "missing order",
            "short sequence",
            "missing value",
            "missing value",
            "",
        ]
        assert np.isnan(result["score"][1])
        assert np.isfinite(result["score"][0])

    def test_two_rows_are_a_short_sequence_and_not_checked_for_order(self):
        result = _one([5.0, 1.0], monotonic=True)
        assert [REASONS[code] for code in result["reason"]] == ["short sequence"] * 2
        assert not result["out_of_order"].any()

    def test_a_fully_null_sequence_key_marks_every_row_missing_sequence(self):
        data = {"x": [1.0, 2.0, 3.0], "video": pl.Series([None, None, None], dtype=pl.Int64), "frame": [0, 1, 2]}
        result = sequence_residuals(data, ["x"], sequence="video", order="frame")
        assert [REASONS[code] for code in result["reason"]] == ["missing sequence"] * 3
        assert result["sequence"].tolist() == [-1] * 3
        assert np.isnan(result["score"]).all()

    def test_no_scorable_row_is_not_an_error(self):
        result = _one([1.0, 2.0])
        assert np.isnan(result["score"]).all()

    def test_equal_orders_keep_row_order(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = _one(values, monotonic=True, order=[0, 1, 2, 3, 3, 4, 5])
        assert not result["out_of_order"].any()


@pytest.mark.required
class TestSpread:
    def test_shrunk_spread_lies_between_the_sequences_own_and_the_pooled(self):
        rng = np.random.default_rng(7)
        quiet, loud = rng.normal(0, 0.5, 50), rng.normal(0, 4, 50)
        data = {"x": np.r_[quiet, loud], "video": [0] * 50 + [1] * 50, "frame": list(range(50)) * 2}
        result = sequence_residuals(data, ["x"], sequence="video", order="frame")

        def mad(r):
            return 1.4826 * np.median(np.abs(r - np.median(r)))

        own, pooled = mad(result["residual"][:50]), mad(result["residual"])
        expected = np.sqrt((50 * own**2 + 30 * pooled**2) / 80)
        np.testing.assert_allclose(result["spread"][:50], expected)
        assert own < expected < pooled

    def test_a_zero_pooled_spread_leaves_scores_undivided(self):
        result = _one(list(range(20)), monotonic=True)
        assert (result["spread"] == 1.0).all()
        np.testing.assert_array_equal(result["score"], result["residual"])


@pytest.mark.required
class TestMonotonic:
    def test_jitter_and_a_dropped_frame_score_low(self):
        rng = np.random.default_rng(8)
        steps = 1 / 30 * (1 + rng.normal(0, 0.05, 200))
        steps[100] *= 2
        assert _flagged(_one(100 + np.cumsum(steps), monotonic=True)["score"], limit=8.0) == []

    def test_a_frame_rate_switch_scores_low_away_from_the_switch(self):
        rng = np.random.default_rng(10)
        steps = np.r_[np.full(120, 1 / 30), np.full(80, 1 / 15)] * (1 + rng.normal(0, 0.02, 200))
        scores = _one(100 + np.cumsum(steps), monotonic=True)["score"]
        assert all(abs(row - 120) <= 5 for row in _flagged(scores))

    @pytest.mark.parametrize(("size", "marked"), [(5, [30, 31, 32, 33, 34]), (6, [36])], ids=["glitch", "reset"])
    def test_a_set_of_window_rows_is_marked_whole_and_a_larger_one_only_where_the_clock_went_back(self, size, marked):
        stamps = np.arange(60) / 30
        stamps[30 : 30 + size] += 100
        result = _one(stamps, monotonic=True, window=5)
        assert np.flatnonzero(result["out_of_order"]).tolist() == marked

    @pytest.mark.parametrize(
        "series",
        [
            pl.Series("x", np.round(np.arange(200) * 1001 / 30).astype(np.int64)),
            pl.Series("x", (1.7e15 + np.round(np.arange(200) * 1e6 / 30)).astype(np.int64)).cast(pl.Datetime("us")),
            pl.Series("x", 1.7e9 + np.arange(200) / 30),
        ],
        ids=["pts-ms", "datetime-us", "epoch-seconds"],
    )
    def test_exact_clocks_leave_residuals_of_exactly_zero(self, series):
        frame = pl.DataFrame({"x": series, "video": [0] * 200, "frame": list(range(200))})
        result = sequence_residuals(frame, ["x"], sequence="video", order="frame", monotonic=["x"])
        assert (result["residual"] == 0).all()
        assert not result["out_of_order"].any()

    def test_a_late_stamp_is_the_only_row_out_of_order(self):
        stamps = 100 + np.arange(200) / 30
        stamps[100] += 3 * 3600
        result = _one(stamps, monotonic=True)
        assert np.flatnonzero(result["out_of_order"]).tolist() == [100]
        assert result["order_bound"][100] == pytest.approx(stamps[101])
        assert _flagged(result["score"]) == [100]

    def test_a_backward_stamp_at_row_one_marks_row_one(self):
        stamps = 100 + np.arange(20) / 30
        stamps[1] = 99.0
        assert np.flatnonzero(_one(stamps, monotonic=True)["out_of_order"]).tolist() == [1]

    def test_equal_consecutive_values_are_in_order(self):
        assert not _one([0.0, 1.0, 1.0, 2.0, 3.0], monotonic=True)["out_of_order"].any()

    def test_a_reset_marks_the_row_where_the_clock_went_back(self):
        stamps = np.r_[np.arange(50), np.arange(20, 70)] / 30
        result = _one(stamps, monotonic=True)
        assert np.flatnonzero(result["out_of_order"]).tolist() == [50]
        assert result["order_bound"][50] == pytest.approx(stamps[49])

    def test_a_clock_running_backwards_marks_every_row_but_the_first(self):
        result = _one(np.arange(20, 0, -1) / 30, monotonic=True)
        assert result["out_of_order"].tolist() == [False] + [True] * 19

    def test_a_column_not_named_monotonic_is_not_checked_for_order(self):
        assert not _one([3.0, 2.0, 1.0, 0.0])["out_of_order"].any()


@pytest.mark.required
class TestInputs:
    def test_mapping_and_frame_agree(self):
        values = np.random.default_rng(9).normal(0, 1, 30)
        data = {"x": values, "video": [0] * 30, "frame": list(range(30))}
        left = sequence_residuals(data, ["x"], sequence="video", order="frame")
        right = sequence_residuals(pl.DataFrame(data), ["x"], sequence="video", order="frame")
        np.testing.assert_array_equal(left["score"], right["score"])

    def test_two_columns_are_laid_out_column_then_row(self):
        data = {"a": np.arange(10.0), "b": np.arange(10.0), "video": [0] * 10, "frame": list(range(10))}
        result = sequence_residuals(data, ["a", "b"], sequence="video", order="frame")
        assert result["columns"] == ("a", "b")
        assert result["column"].tolist() == [0] * 10 + [1] * 10
        assert result["row"].tolist() == list(range(10)) * 2

    def test_each_sequence_has_the_residuals_it_has_alone(self):
        """Sequences are scored at once, so nothing may leak across their boundaries; only the spread is pooled."""
        rng = np.random.default_rng(11)
        lengths = rng.integers(3, 31, 20)
        sequence = np.repeat(np.arange(20), lengths)
        frame = np.concatenate([np.cumsum(rng.choice([0, 1, 2], n)) for n in lengths])
        clock = frame / 30 + rng.normal(0, 0.002, frame.size)
        clock[rng.random(frame.size) < 0.05] += 5.0
        data = {"x": rng.normal(0, 1, frame.size) + 3.0 * frame, "t": clock, "video": sequence, "frame": frame}
        together = sequence_residuals(data, ["x", "t"], sequence="video", order="frame", monotonic=["t"], window=3)
        for video in range(20):
            alone = {name: np.asarray(column)[sequence == video] for name, column in data.items()}
            single = sequence_residuals(alone, ["x", "t"], sequence="video", order="frame", monotonic=["t"], window=3)
            mask = np.tile(sequence == video, 2)
            for key in ("residual", "out_of_order", "order_bound"):
                np.testing.assert_array_equal(together[key][mask], single[key], err_msg=f"video {video}: {key}")

    def test_a_sequence_of_two_keys(self):
        data = {"x": np.zeros(12), "video": [0] * 12, "track": [0] * 6 + [1] * 6, "frame": list(range(6)) * 2}
        result = sequence_residuals(data, ["x"], sequence=["video", "track"], order="frame")
        assert result["sequence"].tolist() == [0] * 6 + [1] * 6

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"window": 0}, "window must be at least 1"),
            ({"columns": []}, "no column to score"),
            ({"columns": ["missing"]}, "not a column of data"),
            ({"columns": ["label"]}, "neither numbers nor times"),
            ({"monotonic": ["other"]}, "which columns does not score"),
            ({"sequence": "missing"}, "not a column of data"),
            ({"sequence": []}, "no column to identify a sequence"),
            ({"order": "missing"}, "not a column of data"),
            ({"order": "label"}, "neither numbers nor times"),
        ],
    )
    def test_refusals(self, kwargs, message):
        data = {"x": [1.0, 2.0, 3.0], "label": ["a", "b", "c"], "video": [0] * 3, "frame": [0, 1, 2]}
        arguments = {"columns": ["x"], "sequence": "video", "order": "frame", **kwargs}
        with pytest.raises(ValueError, match=message):
            sequence_residuals(data, **arguments)

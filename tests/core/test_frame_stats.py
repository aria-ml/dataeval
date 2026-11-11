"""`frame_stats`: per-frame elapsed time and exact repeats."""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
import weakref
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.config import use_max_processes
from dataeval.core import combine_stats_results
from dataeval.core import frame_stats as _frame_stats
from dataeval.core._frame_stats import _measure_video  # pyright: ignore[reportPrivateUsage]
from dataeval.core._video_runner import video_clock
from dataeval.data import VideoSegments, VideoStitch, Window
from dataeval.flags import FrameStats
from dataeval.types import SourceIndex
from tests.data.test_stitch import make_mot

pytestmark = [pytest.mark.required, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]


def frame_stats(source: Any, **kwargs: Any) -> Any:
    """`frame_stats` over this file's test doubles, which are not full `VideoFrame` objects."""
    return _frame_stats(cast(Any, source), **kwargs)


class _Frame:
    """A decoded frame. Built with ``pixels=None``, any read of its pixels fails the test."""

    def __init__(self, pixels: Any, time_s: float | None, frame_index: int = 0) -> None:
        self._pixels = pixels
        self.time_s = time_s
        self.frame_index = frame_index

    @property
    def pixels(self) -> Any:
        assert self._pixels is not None, "pixels were read"
        return self._pixels


class _NoTime:
    """A frame with no ``time_s`` attribute at all."""

    def __init__(self, frame_index: int) -> None:
        self.frame_index = frame_index
        self.pixels = np.zeros((1, 2, 2), np.uint8)


def _timed(times: Sequence[float | None], first_index: int = 0) -> list[_Frame]:
    """Frames that carry timestamps and no pixels, so a pixel read fails."""
    return [_Frame(None, t, first_index + k) for k, t in enumerate(times)]


def _timing(times: Sequence[float | None]) -> Any:
    return frame_stats(_timed(times), stats=FrameStats.TIMING)


def _status(result: Any) -> list[str]:
    return list(result["details"]["elapsed_time_status"])


def _mot(streams: Sequence[Sequence[_Frame]]) -> Any:
    from tests.embeddings.test_embeddings import MockDataset
    from tests.metadata.test_structurers import _FrameTracks, _MOTTarget

    targets = [_MOTTarget([_FrameTracks([0]) for _ in stream]) for stream in streams]
    return MockDataset(streams, targets, None)


def _mot_dataset(frames_per_sequence: Sequence[int]) -> Any:
    # Real pixels: building a `Metadata` reads each frame's size. `frame_stats` itself still reads none.
    return _mot([[_Frame(np.zeros((1, 2, 2), np.uint8), k / 30, k) for k in range(n)] for n in frames_per_sequence])


def _framed(arrays: Sequence[Any]) -> list[_Frame]:
    """Frames that carry these pixels, one per array."""
    return [_Frame(a, k / 30, k) for k, a in enumerate(arrays)]


def _repeats(arrays: Sequence[Any]) -> Any:
    return frame_stats(_framed(arrays), stats=FrameStats.EXACT_REPEAT)


class TestElapsedTime:
    def test_timing_never_reads_pixels(self):
        # `_timed` frames raise on any pixel read.
        result = _timing([0.0, 0.5, 1.0, 1.5, 2.0])
        assert result["stats"]["elapsed_time_s"].shape == (5,)

    def test_fixed_clock(self):
        times = [k / 30 for k in range(6)]
        result = _timing(times)
        elapsed = result["stats"]["elapsed_time_s"]
        assert elapsed.dtype == np.float64
        assert np.isnan(elapsed[0])
        np.testing.assert_array_equal(elapsed[1:], np.diff(times))
        assert _status(result) == ["first_frame", "ok", "ok", "ok", "ok", "ok"]

    def test_one_missing_time_nans_two_rows(self):
        result = _timing([0.0, 0.5, None, 1.5, 2.0])
        np.testing.assert_array_equal(result["stats"]["elapsed_time_s"], [np.nan, 0.5, np.nan, np.nan, 0.5])
        assert _status(result) == ["first_frame", "ok", "missing_time", "missing_time", "ok"]

    def test_a_nonfinite_time_is_missing(self):
        result = _timing([0.0, float("nan"), 2.0, float("inf"), 4.0])
        assert _status(result) == ["first_frame", "missing_time", "missing_time", "missing_time", "missing_time"]
        assert np.isnan(result["stats"]["elapsed_time_s"]).all()

    def test_a_frame_with_no_time_attribute_is_missing(self):
        result = frame_stats([_NoTime(k) for k in range(3)], stats=FrameStats.TIMING)
        assert _status(result) == ["first_frame", "missing_time", "missing_time"]

    def test_repeated_decreasing_and_restarting_clocks(self):
        # Steps: +1, 0, -0.5, +1.5, +1, then a clock that restarts below the last (-2.8), then a clip that
        # starts much later (+4.8), which is a long positive step and stays a number.
        result = _timing([0.0, 1.0, 1.0, 0.5, 2.0, 3.0, 0.2, 5.0])
        assert _status(result) == [
            "first_frame",
            "ok",
            "nonpositive_step",
            "nonpositive_step",
            "ok",
            "ok",
            "nonpositive_step",
            "ok",
        ]
        elapsed = result["stats"]["elapsed_time_s"]
        assert elapsed[1] == 1.0
        assert elapsed[-1] == pytest.approx(4.8)
        assert np.isnan(elapsed[[0, 2, 3, 6]]).all()

    def test_median_matches_video_clock_dt(self):
        times = [0.0, 0.03, 0.07, 0.1, 0.16, 0.2]
        elapsed = _timing(times)["stats"]["elapsed_time_s"]
        assert np.nanmedian(elapsed) == video_clock(times).dt


class TestStitchedAndRolledUp:
    def test_a_clock_that_restarts_at_a_stitch_boundary_is_a_nonpositive_step(self):
        dataset, _ = make_mot([[[0], [0], [0], [0]]])
        segments = VideoSegments(dataset, Window(2), timestamps="local")
        restarting = frame_stats(
            VideoStitch(segments, group_by="source_id", track_ids="preserve"), stats=FrameStats.TIMING
        )
        assert _status(restarting) == ["first_frame", "ok", "nonpositive_step", "ok"]
        assert [s.key for s in restarting["source_index"]] == [0, 1, 2, 3]  # stitching renumbers the frames

        repaired = VideoStitch(segments, group_by="source_id", track_ids="preserve", timestamps="offset")
        assert _status(frame_stats(repaired, stats=FrameStats.TIMING)) == ["first_frame", "ok", "ok", "ok"]

    def test_the_sequence_median_agrees_with_video_clock_to_rounding(self):
        # Four steps, an even count, where the two medians average the middle pair and can differ in the last bit.
        times = [0.0, 0.031, 0.067, 0.1, 0.163]
        ds = _mot([[_Frame(np.zeros((1, 2, 2), np.uint8), t, k) for k, t in enumerate(times)]])
        md = Metadata(ds)
        md.add_factors(frame_stats(ds, stats=FrameStats.TIMING))
        rolled = md.rows_at("sequence")["elapsed_time_s_median"][0]
        assert rolled == pytest.approx(video_clock(times).dt, rel=1e-12)

    def test_timing_rollups_need_a_quarter_of_the_frames(self):
        # The first frame is always NaN and counts as a frame: one valid step of five is 0.2, two of five is 0.4.
        thin = [0.0, 0.1, None, None, None]
        enough = [0.0, 0.1, 0.2, None, None]
        ds = _mot([
            [_Frame(np.zeros((1, 2, 2), np.uint8), t, k) for k, t in enumerate(times)] for times in (thin, enough)
        ])
        md = Metadata(ds)
        md.add_factors(frame_stats(ds, stats=FrameStats.TIMING))
        medians = md.rows_at("sequence")["elapsed_time_s_median"].to_list()
        assert medians[0] is None
        assert medians[1] == pytest.approx(0.1)


class TestAddressing:
    def test_rows_keyed_by_frame_number_attach_to_those_metadata_rows(self):
        zeros = np.zeros((1, 2, 2), np.uint8)
        ds = _mot([[_Frame(zeros, 0.0, 100), _Frame(zeros, 0.5, 102)]])
        md = Metadata(ds)
        md.add_factors(frame_stats(ds, stats=FrameStats.TIMING))
        rows = md.rows_at("unit")
        assert rows["unit_index"].to_list() == [100, 102]
        assert rows["elapsed_time_s"].to_list()[1] == 0.5

    def test_rows_are_addressed_by_frame_number(self):
        result = frame_stats(
            [_Frame(None, 0.0, 100), _Frame(None, 0.1, 102)],
            stats=FrameStats.TIMING,
        )
        assert result["source_index"] == [SourceIndex(0, 100, "unit"), SourceIndex(0, 102, "unit")]

    def test_a_sequence_of_streams_is_in_item_order(self):
        a, b = _timed([0.0, 1.0, 2.0]), _timed([0.0, 0.5], first_index=10)
        result = frame_stats([a, b], stats=FrameStats.TIMING)
        assert [(s.item, s.key) for s in result["source_index"]] == [(0, 0), (0, 1), (0, 2), (1, 10), (1, 11)]
        stats, _, _ = combine_stats_results([result, result])
        assert stats["elapsed_time_s"].shape == (10,)

    def test_reordering_items_permutes_the_rows(self):
        a, b = _timed([0.0, 1.0, 2.0]), _timed([0.0, 0.5], first_index=10)
        forward = frame_stats([a, b], stats=FrameStats.TIMING)
        backward = frame_stats([b, a], stats=FrameStats.TIMING)
        np.testing.assert_array_equal(
            backward["stats"]["elapsed_time_s"],
            np.concatenate([forward["stats"]["elapsed_time_s"][3:], forward["stats"]["elapsed_time_s"][:3]]),
        )
        assert [s.key for s in backward["source_index"]] == [10, 11, 0, 1, 2]
        assert [s.item for s in backward["source_index"]] == [0, 0, 1, 1, 1]

    def test_serial_and_threaded_runs_agree(self):
        streams = [_timed([k * (0.03 + 0.01 * i) for k in range(8)]) for i in range(4)]
        with use_max_processes(1):
            serial = frame_stats(streams, stats=FrameStats.TIMING)
        with use_max_processes(4):
            threaded = frame_stats(streams, stats=FrameStats.TIMING)
        np.testing.assert_array_equal(serial["stats"]["elapsed_time_s"], threaded["stats"]["elapsed_time_s"])
        assert serial["source_index"] == threaded["source_index"]

    def test_edges(self):
        result = frame_stats([[], _timed([0.5]), _timed([0.0, 1.0])], stats=FrameStats.TIMING)
        assert [(s.item, s.key) for s in result["source_index"]] == [(1, 0), (2, 0), (2, 1)]
        assert _status(result) == ["first_frame", "first_frame", "ok"]
        empty = frame_stats([], stats=FrameStats.TIMING)
        assert empty["stats"]["elapsed_time_s"].shape == (0,)
        assert list(empty["source_index"]) == []

    def test_progress_is_reported_once_per_video(self):
        calls: list[tuple[int, int | None]] = []

        def record(step: int, *, total: int | None = None, desc: str | None = None, extra_info: Any = None) -> None:
            calls.append((step, total))

        with use_max_processes(1):
            frame_stats([_timed([0.0, 1.0]) for _ in range(3)], stats=FrameStats.TIMING, progress_callback=record)
        assert calls == [(1, 3), (2, 3), (3, 3)]


class TestSelectionAndRollUps:
    def test_none_raises(self):
        with pytest.raises(ValueError, match="TIMING"):
            frame_stats(_timed([0.0, 1.0]), stats=FrameStats.NONE)

    def test_timing_declares_its_two_rollups(self):
        aggregations = _timing([0.0, 1.0])["aggregations"]
        assert [(a.how, a.factors, a.source, a.target, a.min_coverage) for a in aggregations] == [
            ("median", ("elapsed_time_s",), "unit", "sequence", 0.25),
            ("std", ("elapsed_time_s",), "unit", "sequence", 0.25),
        ]

    def test_attaches_and_rolls_up_to_the_sequence(self):
        ds = _mot_dataset([4, 6])
        md = Metadata(ds)
        md.add_factors(frame_stats(ds, stats=FrameStats.TIMING))
        sequences = md.rows_at("sequence")
        assert sequences["elapsed_time_s_median"].to_list() == pytest.approx([1 / 30, 1 / 30])
        assert "elapsed_time_s_std" in sequences.columns
        # Details stay on the result: `add_factors` does not attach them, and the docstring says so.
        assert "elapsed_time_status" not in md.rows_at("unit").columns
        assert "elapsed_time_s" in md.rows_at("unit").columns


class TestExperimental:
    def test_the_first_call_warns_once(self):
        code = textwrap.dedent(
            """
            import warnings
            from types import SimpleNamespace as NS
            from dataeval.core import frame_stats
            from dataeval.exceptions import ExperimentalWarning
            from dataeval.flags import FrameStats

            frames = [NS(pixels=None, time_s=float(k), frame_index=k) for k in range(2)]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                frame_stats(frames, stats=FrameStats.TIMING)
                frame_stats(frames, stats=FrameStats.TIMING)
            print(sum(issubclass(w.category, ExperimentalWarning) for w in caught))
            """,
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "1"


_BASE = np.arange(12, dtype=np.uint8).reshape(1, 3, 4)


class TestExactRepeat:
    def test_identical_frames_repeat(self):
        result = _repeats([_BASE, _BASE.copy(), _BASE.copy()])
        np.testing.assert_array_equal(result["stats"]["repeat_frame"], [np.nan, 1.0, 1.0])
        assert result["stats"]["repeat_frame"].dtype == np.float64

    def test_one_changed_pixel_is_not_a_repeat(self):
        changed = _BASE.copy()
        changed[0, 0, 0] += 1
        result = _repeats([_BASE, _BASE.copy(), changed])
        np.testing.assert_array_equal(result["stats"]["repeat_frame"], [np.nan, 1.0, 0.0])

    def test_perceptual_similarity_is_not_exact(self):
        rng = np.random.default_rng(0)
        base = rng.random((1, 8, 8), dtype=np.float32)
        noisy = base + np.float32(1e-6) * rng.standard_normal((1, 8, 8)).astype(np.float32)
        np.testing.assert_array_equal(_repeats([base, noisy])["stats"]["repeat_frame"], [np.nan, 0.0])

    def test_a_shape_or_dtype_change_is_zero_not_nan(self):
        shape_changed = np.zeros((1, 3, 5), np.uint8)
        dtype_changed = np.zeros((1, 3, 5), np.float32)
        result = _repeats([_BASE, np.zeros((1, 3, 4), np.uint8), shape_changed, dtype_changed])
        np.testing.assert_array_equal(result["stats"]["repeat_frame"], [np.nan, 0.0, 0.0, 0.0])

    def test_nan_pixels_match_only_at_matching_positions(self):
        a = np.zeros((1, 2, 3), np.float32)
        a[0, 0, 0] = np.nan
        elsewhere = np.zeros((1, 2, 3), np.float32)
        elsewhere[0, 1, 2] = np.nan
        result = _repeats([a, a.copy(), elsewhere])
        np.testing.assert_array_equal(result["stats"]["repeat_frame"], [np.nan, 1.0, 0.0])
        assert list(result["details"]["repeat_pair_has_missing"]) == [False, True, False]

    def test_two_all_nan_frames_repeat_and_say_so(self):
        everything = np.full((1, 2, 2), np.nan, np.float32)
        result = _repeats([everything, everything.copy()])
        np.testing.assert_array_equal(result["stats"]["repeat_frame"], [np.nan, 1.0])
        assert list(result["details"]["repeat_pair_has_missing"]) == [False, True]

    def test_integer_frames_never_have_missing_pixels(self):
        result = _repeats([_BASE, _BASE.copy()])
        assert result["details"]["repeat_pair_has_missing"].dtype == np.bool_
        assert not result["details"]["repeat_pair_has_missing"].any()

    def test_a_two_dimensional_frame_is_measured(self):
        flat = np.arange(12, dtype=np.uint8).reshape(3, 4)
        np.testing.assert_array_equal(_repeats([flat, flat.copy()])["stats"]["repeat_frame"], [np.nan, 1.0])

    def test_a_two_dimensional_frame_equals_its_one_channel_form(self):
        """Frames are compared as the (C, H, W) arrays every statistic reads, which the docstring says."""
        flat = np.arange(12, dtype=np.uint8).reshape(3, 4)
        np.testing.assert_array_equal(_repeats([flat, flat[None].copy()])["stats"]["repeat_frame"], [np.nan, 1.0])

    def test_the_last_frame_is_released_when_a_video_finishes(self):
        """`run_videos` keeps every video's consumer, so none may keep a copy of its last frame."""
        rows = _measure_video(FrameStats.EXACT_REPEAT, cast(Any, _framed([_BASE, _BASE.copy(), _BASE.copy()])))
        assert rows._previous is None  # pyright: ignore[reportPrivateUsage]
        assert rows.repeat == [pytest.approx(np.nan, nan_ok=True), 1.0, 1.0]

    def test_a_reused_decode_buffer_is_not_a_repeat(self):
        """A decoder that hands back one buffer, refilled each frame, must not make every frame equal itself."""

        def reused():
            buffer = np.zeros((1, 2, 2), np.uint8)
            for k in range(4):
                if k == 2:
                    buffer[0, 0, 0] = 9
                yield _Frame(buffer, k / 30, k)

        result = frame_stats(reused(), stats=FrameStats.EXACT_REPEAT)
        np.testing.assert_array_equal(result["stats"]["repeat_frame"], [np.nan, 1.0, 0.0, 1.0])

    def test_each_frame_is_decoded_once_and_only_the_previous_is_held(self):
        reads: list[int] = []
        arrays: list[weakref.ref[Any]] = []

        class _Counting(_Frame):
            @property
            def pixels(self) -> Any:
                reads[self.frame_index] += 1
                return self._pixels

        def stream():
            for k in range(6):
                if k >= 2:
                    gc.collect()
                    assert arrays[k - 2]() is None, f"frame {k - 2} is still held when frame {k} is pulled"
                pixels = np.full((1, 4, 4), k, np.uint8)
                arrays.append(weakref.ref(pixels))
                reads.append(0)
                yield _Counting(pixels, k / 30, k)
                del pixels

        frame_stats(stream(), stats=FrameStats.EXACT_REPEAT)
        assert reads == [1] * 6


class TestSelection:
    def test_each_selection_returns_only_its_own_columns_details_and_rollups(self):
        frames = _framed([_BASE, _BASE.copy()])
        timing = frame_stats(frames, stats=FrameStats.TIMING)
        repeat = frame_stats(frames, stats=FrameStats.EXACT_REPEAT)
        both = frame_stats(frames)  # the default is ALL

        assert set(timing["stats"]) == {"elapsed_time_s"}
        assert set(timing["details"]) == {"elapsed_time_status"}
        assert set(repeat["stats"]) == {"repeat_frame"}
        assert set(repeat["details"]) == {"repeat_pair_has_missing"}
        assert set(both["stats"]) == {"elapsed_time_s", "repeat_frame"}
        assert set(both["details"]) == {"elapsed_time_status", "repeat_pair_has_missing"}
        assert [(a.how, a.factors, a.min_coverage, a.suffix) for a in repeat["aggregations"]] == [
            ("mean", ("repeat_frame",), 0.0, "_fraction"),
        ]
        assert [a.how for a in both["aggregations"]] == ["median", "std", "mean"]

    def test_the_repeat_fraction_excludes_the_first_frame(self):
        x, y = np.zeros((1, 2, 2), np.uint8), np.ones((1, 2, 2), np.uint8)
        ds = _mot([_framed([x, x, y, y, y]), _framed([x])])
        md = Metadata(ds)
        md.add_factors(frame_stats(ds, stats=FrameStats.EXACT_REPEAT))
        fractions = md.rows_at("sequence")["repeat_frame_fraction"].to_list()
        assert fractions[0] == pytest.approx(3 / 4)  # repeats [nan, 1, 0, 1, 1]
        assert fractions[1] is None or np.isnan(fractions[1])  # a one-frame sequence has nothing to compare

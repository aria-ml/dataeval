"""`track_steps`: the step each detection takes from where its track was last seen."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.core import combine_stats_results
from dataeval.core import track_steps as _track_steps
from dataeval.data import VideoSegments, VideoStitch, Window
from dataeval.exceptions import MaiteShapeError
from dataeval.types import MeasurementSpec, SourceIndex
from tests.data.test_stitch import make_mot
from tests.embeddings.test_embeddings import MockDataset

pytestmark = [pytest.mark.required, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]

_SIZE = (8, 10)  # (height, width) of a frame's pixels
_AUTO: Any = object()

# Boxes are (x0, y0, x1, y1). `_A` is 10 by 10 and has diagonal sqrt(200).
_A = [10.0, 10.0, 20.0, 20.0]


def track_steps(source: Any, **kwargs: Any) -> Any:
    """`track_steps` over this file's test doubles, which are not full MAITE datasets."""
    return _track_steps(cast(Any, source), **kwargs)


class _Pixels:
    """A frame's pixels as far as `track_steps` may look: a shape, and a failure on any array conversion."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pixels were converted")


class _Frame:
    """A decoded frame that declares its own number and, optionally, its time and its size."""

    def __init__(self, key: int, time_s: float | None, size: tuple[int, int] | None, lazy: bool) -> None:
        self.frame_index = key
        if time_s is not None:
            self.time_s = time_s
        if size is None:
            self.pixels = None
        else:
            self.pixels = _Pixels((1, *size)) if lazy else np.zeros((1, *size), np.float32)


class _Detections:
    """One frame's target. Built with ``ids=None`` it has no ``track_ids`` at all."""

    def __init__(self, boxes: Any, ids: Sequence[int] | None) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(len(boxes), 4)
        self.labels = np.zeros(len(boxes), dtype=np.intp)
        self.scores = np.ones(len(boxes))
        if ids is not None:
            self.track_ids = np.asarray(ids, dtype=np.intp)


@dataclass
class _Target:
    frame_tracks: list[_Detections]


@dataclass
class F:
    """One frame of a sequence: its boxes and track ids, and what its stream frame declares."""

    boxes: Any
    ids: Sequence[int] | None = None
    key: int | None = None  # the stream frame's own number; its position when None
    time_s: Any = _AUTO  # None leaves the frame without a time; its position times a half second by default
    size: tuple[int, int] | None = _SIZE  # None leaves the frame's size unreadable


def make(*sequences: Sequence[F], lazy: bool = True) -> Any:
    """A tracking dataset with one item per sequence. ``lazy`` frames fail the test if their pixels are converted."""
    streams, targets = [], []
    for frames in sequences:
        streams.append([
            _Frame(k if f.key is None else f.key, k * 0.5 if f.time_s is _AUTO else f.time_s, f.size, lazy)
            for k, f in enumerate(frames)
        ])
        targets.append(_Target([_Detections(f.boxes, f.ids) for f in frames]))
    return MockDataset(streams, targets, None)


def one_track(*frames: F) -> Any:
    """The result for a single sequence."""
    return track_steps(make(frames))


def status(result: Any) -> list[str]:
    return result["details"]["step_status"].tolist()


class _Exploding:
    """A dataset that fails the test the moment anything is read from it."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Any:
        raise AssertionError("an item was read")


class TestKeyword:
    @pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "1", None, 2.0])
    def test_a_bad_max_frame_delta_raises_before_any_item_is_read(self, bad):
        with pytest.raises(ValueError, match="max_frame_delta"):
            track_steps(_Exploding(), max_frame_delta=bad)

    def test_the_message_names_the_value(self):
        with pytest.raises(ValueError, match=r"max_frame_delta.*-3"):
            track_steps(_Exploding(), max_frame_delta=-3)

    def test_a_numpy_integer_works(self):
        frames = [F([_A], [1]), F([], []), F([_A], [1])]
        assert status(track_steps(make(frames), max_frame_delta=np.int64(2))) == ["first_observation", "ok"]

    def test_a_dataset_that_is_not_a_tracking_dataset_is_refused(self):
        with pytest.raises(MaiteShapeError):
            track_steps([np.zeros((3, 4, 4))])


class TestExperimental:
    def test_the_first_call_warns_once(self):
        code = textwrap.dedent(
            """
            import warnings
            from dataeval.core import track_steps
            from dataeval.exceptions import ExperimentalWarning

            class Empty:
                def __len__(self): return 0
                def __getitem__(self, i): raise IndexError

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                track_steps(Empty())
                track_steps(Empty())
            print(sum(issubclass(w.category, ExperimentalWarning) for w in caught))
            """,
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "1"


class TestRows:
    def test_frames_keyed_100_and_102_at_positions_0_and_1_are_one_frame_apart(self):
        result = one_track(F([_A], [5], key=100), F([_A], [5], key=102))
        np.testing.assert_array_equal(result["stats"]["step_frame_delta"], [np.nan, 1.0])
        np.testing.assert_array_equal(result["details"]["step_frame_key"], [100, 102])
        np.testing.assert_array_equal(result["details"]["step_previous_frame_key"], [-1, 100])
        assert status(result) == ["first_observation", "ok"]

    def test_rows_follow_the_item_then_the_frame_then_the_order_the_frame_holds_them(self):
        two_apart = [F([_A, _A], [3, 4]), F([], []), F([_A], [3])]
        result = track_steps(make(two_apart, [F([_A], [3])]))
        assert list(result["source_index"]) == [
            SourceIndex(0, 0),
            SourceIndex(0, 1),
            SourceIndex(0, 2),
            SourceIndex(1, 0),
        ]
        # Track 3 returns after an empty frame: one detection back in the item's numbering, two frames back in time.
        np.testing.assert_array_equal(result["details"]["step_previous_target_index"], [-1, -1, 0, -1])
        np.testing.assert_array_equal(result["stats"]["step_frame_delta"], [np.nan, np.nan, 2.0, np.nan])

    def test_a_track_id_that_ends_one_sequence_and_starts_the_next_starts_over(self):
        result = track_steps(make([F([_A], [7]), F([_A], [7])], [F([_A], [7]), F([_A], [7])]))
        assert status(result) == ["first_observation", "ok", "first_observation", "ok"]
        np.testing.assert_array_equal(result["details"]["step_previous_target_index"], [-1, 0, -1, 0])

    def test_an_empty_dataset_gives_empty_columns(self):
        result = track_steps(make())
        assert len(result["source_index"]) == 0
        assert result["details"]["step_status"].shape == (0,)
        assert result["stats"]["step_frame_delta"].dtype == np.float64

    def test_sequences_with_no_detections_give_no_rows(self):
        result = track_steps(make([F([], []), F([], [])]))
        assert len(result["source_index"]) == 0
        assert result["details"]["step_frame_key"].dtype == np.int64

    def test_a_one_frame_sequence_gives_one_first_observation(self):
        result = one_track(F([_A], [9]))
        assert status(result) == ["first_observation"]

    def test_details_have_the_listed_dtypes(self):
        details = one_track(F([_A], [1]), F([_A], [1]))["details"]
        assert details["step_status"].dtype == object
        for name in ("step_previous_target_index", "step_frame_key", "step_previous_frame_key"):
            assert details[name].dtype == np.int64

    def test_a_stream_longer_than_its_target_raises(self):
        dataset = make([F([_A], [1]), F([_A], [1])])
        dataset.data[0].append(_Frame(2, 1.0, _SIZE, True))
        with pytest.raises(ValueError, match="more frames"):
            track_steps(dataset)

    def test_a_stream_shorter_than_its_target_raises(self):
        dataset = make([F([_A], [1]), F([_A], [1])])
        dataset.data[0].pop()
        with pytest.raises(ValueError, match="only 1"):
            track_steps(dataset)

    def test_boxes_that_are_not_four_numbers_raise_naming_the_item_and_frame(self):
        dataset = make([F([_A], [1])], [F([_A], [1]), F([_A, _A], [1, 2])])
        dataset.targets[1].frame_tracks[1].boxes = np.zeros((2, 3))
        with pytest.raises(ValueError, match=r"frame 1 of item 1"):
            track_steps(dataset)

    def test_a_track_id_on_two_detections_of_a_frame_raises_naming_the_item_and_frame(self):
        with pytest.raises(ValueError, match=r"Track id 3 .*frame 1 of item 1"):
            track_steps(make([F([_A], [3])], [F([_A], [3]), F([_A, _A], [3, 3])]))

    def test_untracked_detections_may_repeat_an_id(self):
        result = one_track(F([_A, _A], [-1, -1]), F([_A, _A], [-1, -1]))
        assert status(result) == ["untracked"] * 4


class TestUntracked:
    @pytest.mark.parametrize("ids", [[-1], [-7], None])
    def test_a_negative_id_or_none_at_all_is_untracked(self, ids):
        result = one_track(F([_A], ids), F([_A], ids))
        assert status(result) == ["untracked", "untracked"]
        np.testing.assert_array_equal(result["stats"]["step_frame_delta"], [np.nan, np.nan])
        np.testing.assert_array_equal(result["details"]["step_previous_target_index"], [-1, -1])
        np.testing.assert_array_equal(result["details"]["step_previous_frame_key"], [-1, -1])

    def test_untracked_detections_do_not_disturb_the_numbering_of_tracked_ones(self):
        result = one_track(F([_A, _A], [-1, 2]), F([_A, _A], [2, -1]))
        assert status(result) == ["untracked", "first_observation", "ok", "untracked"]
        np.testing.assert_array_equal(result["details"]["step_previous_target_index"], [-1, -1, 1, -1])


class TestStatuses:
    def test_a_gap_of_two_frames_exceeds_the_default_limit_and_is_fine_at_two(self):
        frames = [F([_A], [1]), F([], []), F([_A], [1])]
        assert status(one_track(*frames)) == ["first_observation", "gap_exceeds_limit"]
        assert status(track_steps(make(frames), max_frame_delta=2)) == ["first_observation", "ok"]

    def test_the_gap_counts_frames_the_stream_yields_not_the_difference_of_frame_numbers(self):
        result = one_track(F([_A], [1], key=100), F([_A], [1], key=140))
        assert status(result) == ["first_observation", "ok"]
        np.testing.assert_array_equal(result["stats"]["step_frame_delta"], [np.nan, 1.0])

    def test_step_frame_delta_is_defined_wherever_a_previous_observation_exists(self):
        result = one_track(F([_A], [1]), F([], []), F([], []), F([_A], [1]))
        assert status(result) == ["first_observation", "gap_exceeds_limit"]
        np.testing.assert_array_equal(result["stats"]["step_frame_delta"], [np.nan, 3.0])

    @pytest.mark.parametrize(
        "bad",
        [
            [10.0, 10.0, np.nan, 20.0],
            [10.0, 10.0, np.inf, 20.0],
            [10.0, -np.inf, 20.0, 20.0],
            [20.0, 10.0, 10.0, 20.0],  # x1 < x0
            [10.0, 20.0, 20.0, 10.0],  # y1 < y0
            [10.0, 10.0, 10.0, 20.0],  # no width
            [10.0, 10.0, 20.0, 10.0],  # no height
        ],
    )
    def test_an_invalid_box_at_either_end_is_invalid_box(self, bad):
        assert status(one_track(F([_A], [1]), F([bad], [1]))) == ["first_observation", "invalid_box"]
        assert status(one_track(F([bad], [1]), F([_A], [1]))) == ["first_observation", "invalid_box"]

    def test_the_successor_of_an_invalid_box_is_invalid_box_though_its_own_box_is_fine(self):
        bad = [10.0, 10.0, np.nan, 20.0]
        result = one_track(F([bad], [1]), F([_A], [1]), F([_A], [1]))
        assert status(result) == ["first_observation", "invalid_box", "ok"]
        # The previous observation is the invalid box. It is not skipped.
        np.testing.assert_array_equal(result["details"]["step_previous_target_index"], [-1, 0, 1])

    def test_frame_sizes_that_differ_are_a_geometry_change_and_equal_ones_are_ok(self):
        changed = one_track(F([_A], [1], size=(8, 10)), F([_A], [1], size=(8, 12)))
        assert status(changed) == ["first_observation", "geometry_change"]
        assert status(one_track(F([_A], [1]), F([_A], [1]))) == ["first_observation", "ok"]

    def test_a_change_in_height_alone_is_a_geometry_change(self):
        result = one_track(F([_A], [1], size=(8, 10)), F([_A], [1], size=(9, 10)))
        assert status(result) == ["first_observation", "geometry_change"]

    def test_a_size_that_cannot_be_read_at_either_end_is_a_geometry_change(self):
        result = one_track(F([_A], [1], size=None), F([_A], [1], size=None))
        assert status(result) == ["first_observation", "geometry_change"]

    @pytest.mark.parametrize("which", [0, 1])
    def test_a_frame_size_that_cannot_be_read_is_a_geometry_change(self, which):
        sizes: list[tuple[int, int] | None] = [_SIZE, _SIZE]
        sizes[which] = None
        result = one_track(F([_A], [1], size=sizes[0]), F([_A], [1], size=sizes[1]))
        assert status(result) == ["first_observation", "geometry_change"]

    def test_an_invalid_first_observation_is_a_first_observation(self):
        assert status(one_track(F([[10.0, 10.0, np.nan, 20.0]], [1]))) == ["first_observation"]

    def test_an_invalid_box_after_a_long_gap_is_a_gap(self):
        bad = [10.0, 10.0, np.nan, 20.0]
        result = one_track(F([_A], [1]), F([], []), F([bad], [1]))
        assert status(result) == ["first_observation", "gap_exceeds_limit"]

    def test_an_invalid_box_across_a_size_change_is_invalid_box(self):
        bad = [10.0, 10.0, np.nan, 20.0]
        result = one_track(F([_A], [1], size=(8, 10)), F([bad], [1], size=(8, 12)))
        assert status(result) == ["first_observation", "invalid_box"]

    def test_no_pixel_is_converted(self):
        # `make` builds lazy frames: a conversion of their pixels fails the test.
        result = one_track(F([_A], [1]), F([_A], [1]))
        assert status(result) == ["first_observation", "ok"]


class TestDeterminism:
    def test_reordering_the_detections_of_each_frame_permutes_the_rows_and_changes_nothing_else(self):
        second = [30.0, 10.0, 45.0, 30.0]
        original = track_steps(make([F([_A, second], [1, 2]), F([second, _A], [2, 1]), F([_A, second], [1, 2])]))
        swapped = track_steps(make([F([second, _A], [2, 1]), F([_A, second], [1, 2]), F([second, _A], [2, 1])]))
        perm = [1, 0, 3, 2, 5, 4]
        for name, column in original["stats"].items():
            np.testing.assert_array_equal(swapped["stats"][name], column[perm], err_msg=name)
        for name, column in original["details"].items():
            if name == "step_previous_target_index":
                continue  # an address, which moves with the rows
            np.testing.assert_array_equal(swapped["details"][name], column[perm], err_msg=name)

    def test_reordering_the_items_permutes_the_rows_and_changes_nothing_else(self):
        one = [F([_A], [1]), F([_A], [1])]
        two = [F([_A], [4]), F([], []), F([_A], [4])]
        forward, backward = track_steps(make(one, two)), track_steps(make(two, one))
        perm = [2, 3, 0, 1]
        for name, column in forward["stats"].items():
            np.testing.assert_array_equal(backward["stats"][name], column[perm], err_msg=name)
        assert status(backward) == [status(forward)[i] for i in perm]


def stat(result: Any, name: str) -> Any:
    return result["stats"][name]


_MEASURES = (
    "step_displacement_norm",
    "step_speed_norm",
    "step_speed_norm_per_s",
    "step_iou_with_previous",
    "step_log_scale_change",
    "step_log_scale_rate",
)


class TestMeasurements:
    def test_a_box_doubled_about_its_center_scales_by_log_2_overlaps_a_quarter_and_does_not_move(self):
        result = one_track(F([_A], [1]), F([[5.0, 5.0, 25.0, 25.0]], [1]))
        assert stat(result, "step_log_scale_change")[1] == pytest.approx(np.log(2))
        assert stat(result, "step_iou_with_previous")[1] == pytest.approx(0.25)
        assert stat(result, "step_displacement_norm")[1] == 0.0

    def test_a_translation_is_measured_in_diagonals_of_the_previous_box(self):
        result = one_track(F([_A], [1]), F([[13.0, 14.0, 23.0, 24.0]], [1]))
        assert stat(result, "step_displacement_norm")[1] == pytest.approx(5 / np.hypot(10, 10))
        assert result["details"]["step_dx_px"][1] == 3.0
        assert result["details"]["step_dy_px"][1] == 4.0

    def test_a_box_that_widens_and_shifts(self):
        result = one_track(F([_A], [1]), F([[10.0, 10.0, 30.0, 20.0]], [1]))
        assert stat(result, "step_log_scale_change")[1] == pytest.approx(0.5 * np.log(2))
        assert stat(result, "step_iou_with_previous")[1] == pytest.approx(0.5)
        assert stat(result, "step_displacement_norm")[1] == pytest.approx(5 / np.hypot(10, 10))

    @pytest.mark.parametrize("factor", [2.0, 3.0])
    def test_scaling_every_coordinate_leaves_the_normalized_columns_alone(self, factor):
        before, after = [12.0, 7.0, 40.0, 31.0], [15.0, 9.0, 50.0, 40.0]

        def scaled(by: float) -> Any:
            return one_track(F([np.multiply(before, by)], [1]), F([np.multiply(after, by)], [1]))

        base, other = scaled(1.0), scaled(factor)
        for name in _MEASURES:
            np.testing.assert_allclose(stat(other, name), stat(base, name), rtol=1e-12, err_msg=name)
        for name in ("step_dx_px", "step_dy_px"):
            np.testing.assert_allclose(other["details"][name], factor * base["details"][name], rtol=1e-12)

    def test_the_two_rates_equal_their_per_step_values_at_the_default(self):
        result = one_track(F([_A], [1]), F([[13.0, 14.0, 24.0, 26.0]], [1]))
        np.testing.assert_array_equal(stat(result, "step_speed_norm"), stat(result, "step_displacement_norm"))
        np.testing.assert_array_equal(stat(result, "step_log_scale_rate"), stat(result, "step_log_scale_change"))

    def test_a_longer_step_halves_each_rate(self):
        frames = [F([_A], [1]), F([], []), F([[13.0, 14.0, 24.0, 26.0]], [1])]
        one = one_track(F([_A], [1]), F([[13.0, 14.0, 24.0, 26.0]], [1]))
        two = track_steps(make(frames), max_frame_delta=2)
        assert stat(two, "step_frame_delta")[1] == 2.0
        assert stat(two, "step_speed_norm")[1] == pytest.approx(stat(one, "step_displacement_norm")[1] / 2)
        assert stat(two, "step_log_scale_rate")[1] == pytest.approx(stat(one, "step_log_scale_change")[1] / 2)
        assert stat(two, "step_displacement_norm")[1] == pytest.approx(stat(one, "step_displacement_norm")[1])

    def test_the_per_second_rate_divides_by_the_elapsed_time(self):
        result = one_track(F([_A], [1], time_s=1.0), F([[13.0, 14.0, 23.0, 24.0]], [1], time_s=3.0))
        assert stat(result, "step_elapsed_time_s")[1] == 2.0
        assert stat(result, "step_speed_norm_per_s")[1] == pytest.approx(stat(result, "step_displacement_norm")[1] / 2)

    @pytest.mark.parametrize("times", [(1.0, None), (None, 1.0), (1.0, 1.0), (2.0, 1.0), (1.0, np.nan), (np.inf, 5.0)])
    def test_a_time_that_gives_no_step_leaves_the_frame_based_columns_valid(self, times):
        result = one_track(F([_A], [1], time_s=times[0]), F([[13.0, 14.0, 23.0, 24.0]], [1], time_s=times[1]))
        assert status(result) == ["first_observation", "ok"]
        assert np.isnan(stat(result, "step_elapsed_time_s")[1])
        assert np.isnan(stat(result, "step_speed_norm_per_s")[1])
        for name in ("step_frame_delta", "step_displacement_norm", "step_speed_norm", "step_log_scale_rate"):
            assert np.isfinite(stat(result, name)[1]), name

    @pytest.mark.parametrize(
        "after",
        [[30.0, 10.0, 40.0, 20.0], [20.0, 10.0, 30.0, 20.0]],  # apart, and touching along an edge
    )
    def test_boxes_that_do_not_overlap_or_only_touch_have_an_iou_of_zero(self, after):
        assert stat(one_track(F([_A], [1]), F([after], [1])), "step_iou_with_previous")[1] == 0.0

    def test_identical_boxes_overlap_completely_and_have_not_moved_or_scaled(self):
        result = one_track(F([_A], [1]), F([_A], [1]))
        assert stat(result, "step_iou_with_previous")[1] == 1.0
        assert stat(result, "step_displacement_norm")[1] == 0.0
        assert stat(result, "step_log_scale_change")[1] == 0.0

    def test_a_box_far_from_the_origin_keeps_its_overlap(self):
        before, after = [1e6, 1e6, 1e6 + 10, 1e6 + 10], [1e6 + 5, 1e6, 1e6 + 15, 1e6 + 10]
        result = one_track(F([before], [1]), F([after], [1]))
        assert stat(result, "step_iou_with_previous")[1] == pytest.approx(1 / 3)

    @pytest.mark.parametrize("frames", [[F([_A], [1]), F([], []), F([_A], [1])], [F([_A], [-1]), F([_A], [-1])]])
    def test_a_row_that_is_not_ok_has_nan_in_every_measure(self, frames):
        result = one_track(*frames)
        assert "ok" not in status(result)
        for name in _MEASURES:
            assert np.isnan(stat(result, name)).all(), name

    def test_the_pixel_difference_is_there_wherever_a_previous_observation_exists_and_the_centers_are_finite(self):
        far = [13.0, 14.0, 23.0, 24.0]
        result = one_track(F([_A], [1]), F([], []), F([far], [1]), F([[10.0, 10.0, np.nan, 20.0]], [1]))
        assert status(result) == ["first_observation", "gap_exceeds_limit", "invalid_box"]
        np.testing.assert_array_equal(result["details"]["step_dx_px"], [np.nan, 3.0, np.nan])
        np.testing.assert_array_equal(result["details"]["step_dy_px"], [np.nan, 4.0, np.nan])

    def test_every_measure_is_float64(self):
        result = one_track(F([_A], [1]), F([_A], [1]))
        for name in (*_MEASURES, "step_frame_delta", "step_elapsed_time_s"):
            assert stat(result, name).dtype == np.float64, name
        assert result["details"]["step_dx_px"].dtype == np.float64


_FACTOR_NAMES = (
    "step_frame_delta",
    "step_elapsed_time_s",
    "step_displacement_norm",
    "step_speed_norm",
    "step_speed_norm_per_s",
    "step_iou_with_previous",
    "step_log_scale_change",
    "step_log_scale_rate",
)


def _two_items(**kwargs: Any) -> Any:
    """Two sequences with a tracked and an untracked detection in most frames, on real pixels for `Metadata`."""
    moved = [13.0, 14.0, 23.0, 24.0]
    first = [F([_A, _A], [1, -1]), F([moved, _A], [1, -1]), F([_A, moved], [-1, 1])]
    return make(first, [F([_A], [1]), F([moved], [1])], lazy=False, **kwargs)


class TestResult:
    def test_it_holds_the_eight_factors_the_details_and_no_roll_ups(self):
        result = track_steps(_two_items())
        assert tuple(result["stats"]) == _FACTOR_NAMES
        assert set(result["details"]) == {
            "step_status",
            "step_dx_px",
            "step_dy_px",
            "step_previous_target_index",
            "step_frame_key",
            "step_previous_frame_key",
        }
        assert "aggregations" not in result
        assert "categorical" not in result

    def test_every_column_has_a_value_for_each_detection(self):
        result = track_steps(_two_items())
        assert len(result["source_index"]) == 8
        for column in (*result["stats"].values(), *result["details"].values()):
            assert len(column) == 8

    def test_the_sentinels_fall_exactly_where_there_is_no_previous_observation(self):
        result = track_steps(_two_items())
        none = np.isin(result["details"]["step_status"], ["untracked", "first_observation"])
        np.testing.assert_array_equal(result["details"]["step_previous_target_index"] == -1, none)
        np.testing.assert_array_equal(result["details"]["step_previous_frame_key"] == -1, none)
        np.testing.assert_array_equal(np.isnan(result["stats"]["step_frame_delta"]), none)

    def test_add_factors_attaches_every_factor_to_the_instance_rows_and_no_detail(self):
        dataset = _two_items()
        result = track_steps(dataset)
        metadata = Metadata(dataset)
        metadata.add_factors(result)
        rows = metadata.rows_at("instance")
        assert rows.height == 8
        for name in _FACTOR_NAMES:
            np.testing.assert_array_equal(rows[name].to_numpy(), result["stats"][name], err_msg=name)
        for name in result["details"]:
            assert name not in rows.columns

    def test_measurement_specs_cover_every_factor_and_record_the_limit(self):
        specs = track_steps(_two_items(), max_frame_delta=3)["measurement_specs"]
        assert tuple(specs) == _FACTOR_NAMES
        assert all(isinstance(spec, MeasurementSpec) for spec in specs.values())
        assert {spec.config["max_frame_delta"] for spec in specs.values()} == {3}
        assert all(
            "shot" in spec.config["conditioning"] and "camera" in spec.config["conditioning"] for spec in specs.values()
        )
        assert len({(spec.units, spec.algorithm) for spec in specs.values()}) == len(specs)

    def test_results_with_different_limits_are_not_combined(self):
        one, three = track_steps(_two_items()), track_steps(_two_items(), max_frame_delta=3)
        with pytest.raises(ValueError, match=r"max_frame_delta"):
            combine_stats_results([one, three])

    def test_results_with_the_same_limit_are_combined(self):
        stats, source_index, _ = combine_stats_results([track_steps(_two_items()), track_steps(_two_items())])
        assert len(source_index) == 16
        assert len(stats["step_frame_delta"]) == 16


class TestStitching:
    def test_a_step_across_a_segment_boundary_is_measured_because_shots_are_not_conditioned_on(self):
        dataset, _ = make_mot([[[0], [0], [0], [0]]])
        segments = VideoSegments(dataset, Window(2), timestamps="local")
        result = track_steps(VideoStitch(segments, group_by="source_id", track_ids="preserve"))
        assert status(result) == ["first_observation", "ok", "ok", "ok"]
        # The clock restarts at the boundary, so that step has frames and no seconds.
        np.testing.assert_array_equal(np.isnan(result["stats"]["step_elapsed_time_s"]), [True, False, True, False])
        assert np.isfinite(result["stats"]["step_displacement_norm"][2])


class TestOddTimes:
    """A time that gives no usable step is NaN quietly: no warning reaches the user for data that is merely odd."""

    @pytest.mark.parametrize("times", [(np.inf, np.inf), (-1e308, 1e308), (np.nan, 1.0), (0.0, 5e-324)])
    def test_no_warning_escapes(self, times):
        moved = [13.0, 14.0, 23.0, 24.0]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = one_track(F([_A], [1], time_s=times[0]), F([moved], [1], time_s=times[1]))
        assert status(result) == ["first_observation", "ok"]

    def test_a_rate_that_overflows_is_nan_rather_than_infinite(self):
        moved = [13.0, 14.0, 23.0, 24.0]
        result = one_track(F([_A], [1], time_s=0.0), F([moved], [1], time_s=5e-324))
        assert np.isnan(stat(result, "step_speed_norm_per_s")[1])
        assert np.isfinite(stat(result, "step_displacement_norm")[1])


def _reference(sequences: Sequence[Sequence[F]], limit: int) -> list[tuple[str, int, int]]:
    """Each detection's (status, previous target index, frames between), by remembering every track's last sighting."""
    rows = []
    for frames in sequences:
        last: dict[int, tuple[int, int, bool, Any]] = {}  # id -> (position, target index, box valid, size)
        index = 0
        for position, frame in enumerate(frames):
            for box, track in zip(frame.boxes, frame.ids or [-1] * len(frame.boxes), strict=True):
                valid = bool(np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1])
                if track < 0:
                    rows.append(("untracked", -1, -1))
                elif track not in last:
                    rows.append(("first_observation", -1, -1))
                else:
                    before, before_index, before_valid, before_size = last[track]
                    delta = position - before
                    if delta > limit:
                        name = "gap_exceeds_limit"
                    elif not (valid and before_valid):
                        name = "invalid_box"
                    elif frame.size is None or before_size is None or frame.size != before_size:
                        name = "geometry_change"
                    else:
                        name = "ok"
                    rows.append((name, before_index, delta))
                if track >= 0:
                    last[track] = (position, index, valid, frame.size)
                index += 1
    return rows


class TestAgainstAReference:
    @pytest.mark.parametrize("limit", [1, 2, 5])
    def test_long_tracks_with_gaps_bad_boxes_and_size_changes_match_a_brute_force_walk(self, limit):
        rng = np.random.default_rng(limit)
        sequences = []
        for _ in range(3):
            frames = []
            for _ in range(60):
                ids = rng.permutation(8)[: rng.integers(0, 6)].tolist()
                ids = [-1 if rng.random() < 0.15 else i for i in ids]
                boxes = rng.uniform(0, 50, (len(ids), 4))
                boxes[:, 2:] = boxes[:, :2] + rng.uniform(-2, 20, (len(ids), 2))  # some reversed or empty
                boxes[rng.random(len(ids)) < 0.05, 0] = np.nan
                size = None if rng.random() < 0.05 else (8, 10 + int(rng.integers(0, 2)))
                frames.append(F(boxes, ids, size=size))
            sequences.append(frames)
        result = track_steps(make(*sequences), max_frame_delta=limit)
        expected = _reference(sequences, limit)
        assert status(result) == [name for name, _, _ in expected]
        np.testing.assert_array_equal(result["details"]["step_previous_target_index"], [i for _, i, _ in expected])
        np.testing.assert_array_equal(
            result["stats"]["step_frame_delta"], [np.nan if d < 0 else float(d) for _, _, d in expected]
        )
        assert {"ok", "invalid_box", "geometry_change", "gap_exceeds_limit", "untracked"} <= set(status(result))

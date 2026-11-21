"""Tests for track_switches: encounters of two tracks scored by whether their appearance fits better crossed."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import track_switches
from dataeval.exceptions import ExperimentalWarning

pytestmark = [pytest.mark.required, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]

D = 8  # embedding width of every scene


def _rows(axis: int | list[int], n: int, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """``n`` rows near the unit vector ``e[axis]`` (or near ``e[axis[i]]`` row by row), plus Gaussian noise."""
    axes = np.broadcast_to(np.asarray(axis), (n,))
    rows = np.eye(D)[axes]
    return rows + np.random.default_rng(seed).normal(0.0, noise, (n, D)) if noise else rows


def _track(
    video: int, track: int, frames: Any, xs: Any, rows: np.ndarray, labels: Any = 0, y: float = 0.0
) -> tuple[np.ndarray, ...]:
    """One track: 10 px square boxes centered at ``(xs, y)``, one per frame."""
    frames = np.asarray(frames)
    x = np.broadcast_to(np.asarray(xs, dtype=float), frames.shape)
    boxes = np.stack([x - 5, np.full(x.shape, y) - 5, x + 5, np.full(x.shape, y) + 5], axis=1)
    return (
        np.full(len(frames), video),
        np.full(len(frames), track),
        np.broadcast_to(np.asarray(labels), frames.shape),
        frames,
        boxes,
        np.asarray(rows, dtype=float),
    )


def _scene(*tracks: tuple[np.ndarray, ...]) -> dict[str, Any]:
    """The six inputs of ``track_switches`` for the given tracks, stacked in the order given."""
    video, track, labels, frames, boxes, rows = (np.concatenate(column) for column in zip(*tracks, strict=True))
    return {
        "embeddings": rows,
        "track_ids": track,
        "frame_indices": frames,
        "sequence_indices": video,
        "boxes": boxes,
        "labels": labels,
    }


def _lists(result: Any) -> dict[str, list[Any]]:
    """Every column of a result as a list, for comparing whole results."""
    return {name: np.asarray(column).tolist() for name, column in dict(result).items()}


def _apart_except(frames: np.ndarray, meets: list[int], x: float) -> np.ndarray:
    """x positions of a track that sits at ``x`` on the frames in ``meets`` and 100 px away otherwise."""
    return np.where(np.isin(frames, meets), x, x + 100.0)


class TestEncounters:
    def test_tracks_that_never_meet_give_no_row(self):
        """Boxes 1,000 px apart never meet, so the table is empty, with every column typed."""
        f = np.arange(30)
        result = track_switches(**_scene(_track(0, 1, f, 0.0, _rows(0, 30)), _track(0, 2, f, 1000.0, _rows(1, 30))))
        assert all(len(column) == 0 for column in _lists(result).values())
        assert result["switch_score"].dtype == np.float64
        assert result["flagged"].dtype == np.bool_
        assert result["row_a"].dtype == np.intp

    def test_a_meeting_too_short_to_score_keeps_a_row(self):
        """Tracks that meet without `window` rows on a side keep a row with no score, and no warning is raised."""
        f = np.arange(6)
        scene = _scene(_track(0, 1, f, _apart_except(f, [3], 0.0), _rows(0, 6)), _track(0, 2, f, 0.0, _rows(1, 6)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=ExperimentalWarning)
            result = track_switches(**scene, window=10)
        assert result["frame"].tolist() == [3]
        assert np.isnan(result["switch_score"][0])
        assert not result["flagged"][0]
        assert (result["row_a"][0], result["row_b"][0]) == (3, 9)

    def test_different_classes_never_pair(self):
        f = np.arange(30)
        scene = _scene(_track(0, 1, f, 0.0, _rows(0, 30), labels=0), _track(0, 2, f, 0.0, _rows(1, 30), labels=1))
        assert len(track_switches(**scene)["frame"]) == 0

    def test_a_track_pairs_by_its_most_common_label(self):
        """Track 1's minority label 2 falls on the meeting frame; it pairs as class 1 all the same."""
        f = np.arange(3)
        scene = _scene(
            _track(0, 1, f, _apart_except(f, [2], 0.0), _rows(0, 3), labels=[1, 1, 2]),
            _track(0, 2, f, 0.0, _rows(1, 3), labels=1),
        )
        result = track_switches(**scene, window=2)
        assert result["label"].tolist() == [1]
        assert result["frame"].tolist() == [2]

    def test_a_tie_between_labels_takes_the_smaller(self):
        f = np.arange(2)
        scene = _scene(_track(0, 1, f, 0.0, _rows(0, 2), labels=[2, 1]), _track(0, 2, f, 0.0, _rows(1, 2), labels=1))
        assert track_switches(**scene, window=2)["label"].tolist() == [1]

    def test_the_same_id_in_two_videos_never_pairs_across_them(self):
        f = np.arange(30)
        scene = _scene(_track(0, 1, f, 0.0, _rows(0, 30)), _track(1, 1, f, 0.0, _rows(1, 30)))
        assert len(track_switches(**scene)["frame"]) == 0

    @pytest.mark.parametrize(("second", "expected"), [(8, 1), (9, 2)])
    def test_meetings_more_than_five_positions_apart_are_two_encounters(self, second: int, expected: int):
        """Meetings at shared positions 3 and 8 (5 apart) are one encounter; at 3 and 9 (6 apart), two."""
        f = np.arange(21)
        scene = _scene(
            _track(0, 1, f, _apart_except(f, [3, second], 0.0), _rows(0, 21)),
            _track(0, 2, f, 0.0, _rows(1, 21)),
        )
        assert len(track_switches(**scene, window=2)["frame"]) == expected

    def test_a_thinned_video_splits_encounters_as_full_frames_do(self):
        """The same scene on every 2nd frame index gives the same rows at twice the frame."""
        f = np.arange(21)

        def scene(frames: np.ndarray) -> dict[str, Any]:
            x1 = np.where(np.isin(f, [3, 9]), 0.0, 100.0)
            return _scene(_track(0, 1, frames, x1, _rows(0, 21)), _track(0, 2, frames, 0.0, _rows(1, 21)))

        full, thin = track_switches(**scene(f), window=2), track_switches(**scene(2 * f), window=2)
        assert thin["frame"].tolist() == (2 * full["frame"]).tolist()
        assert sorted(thin["frame"].tolist()) == [6, 18]
        np.testing.assert_array_equal(thin["switch_score"], full["switch_score"])
        others = {k: v for k, v in _lists(thin).items() if k not in ("frame", "switch_score")}
        assert others == {k: v for k, v in _lists(full).items() if k not in ("frame", "switch_score")}

    def test_thinning_keeps_meetings_five_positions_apart_in_one_encounter(self):
        """Meetings at 3 and 8 are 5 positions apart at any frame rate, though 10 frame indices apart when thinned."""
        f = np.arange(21)

        def scene(frames: np.ndarray) -> dict[str, Any]:
            x1 = np.where(np.isin(f, [3, 8]), 0.0, 100.0)
            return _scene(_track(0, 1, frames, x1, _rows(0, 21)), _track(0, 2, frames, 0.0, _rows(1, 21)))

        assert len(track_switches(**scene(f), window=2)["frame"]) == 1
        assert len(track_switches(**scene(2 * f), window=2)["frame"]) == 1

    def test_positions_count_the_frames_both_tracks_share(self):
        """Track 2 skips frames 4 to 7, so meetings at 2 and 10 are 4 shared frames apart: one encounter."""
        f1 = np.arange(21)
        f2 = np.r_[0:4, 8:21]
        x1 = np.where(np.isin(f1, [2, 10]), 0.0, 100.0)
        scene = _scene(_track(0, 1, f1, x1, _rows(0, 21)), _track(0, 2, f2, 0.0, _rows(1, len(f2))))
        assert len(track_switches(**scene, window=2)["frame"]) == 1

    def test_untracked_rows_are_ignored_and_rows_point_into_the_input(self):
        """Untracked rows first in the input, on top of the pair: rows still address the caller's arrays."""
        f = np.arange(30)
        scene = _scene(
            _track(0, -1, f, 0.0, _rows(2, 30)),
            _track(0, 1, f, _apart_except(f, [12], 0.0), _rows(0, 30)),
            _track(0, 2, f, 0.0, _rows(1, 30)),
        )
        result = track_switches(**scene, window=5)
        assert result["frame"].tolist() == [12]
        assert scene["track_ids"][result["row_a"]].tolist() == [1]
        assert scene["track_ids"][result["row_b"]].tolist() == [2]
        assert scene["frame_indices"][result["row_a"]].tolist() == [12]
        assert scene["frame_indices"][result["row_b"]].tolist() == [12]


def _pair(n: int = 30) -> dict[str, Any]:
    f = np.arange(n)
    return _scene(_track(0, 1, f, 0.0, _rows(0, n)), _track(0, 2, f, 0.0, _rows(1, n)))


class TestFailClosed:
    def test_inputs_of_different_lengths_raise(self):
        scene = _pair()
        scene["track_ids"] = scene["track_ids"][:-1]
        with pytest.raises(ValueError, match="has .* entries"):
            track_switches(**scene)

    def test_one_dimensional_embeddings_raise_and_name_them(self):
        scene = _pair()
        scene["embeddings"] = scene["embeddings"][:, 0]
        with pytest.raises(ValueError, match="embeddings must be two-dimensional"):
            track_switches(**scene)

    def test_a_non_finite_embedding_raises(self):
        scene = _pair()
        scene["embeddings"][4, 2] = np.nan
        with pytest.raises(ValueError, match="embeddings must be finite"):
            track_switches(**scene)

    def test_a_track_id_that_is_not_whole_raises(self):
        scene = _pair()
        scene["track_ids"] = scene["track_ids"].astype(float)
        scene["track_ids"][3] = 1.5
        with pytest.raises(ValueError, match="track_ids"):
            track_switches(**scene)

    @pytest.mark.parametrize("shape", [(60, 3), (59, 4), (60, 4, 1)])
    def test_boxes_not_one_box_per_row_raise(self, shape: tuple[int, ...]):
        scene = _pair()
        scene["boxes"] = np.ones(shape)
        with pytest.raises(ValueError, match="boxes"):
            track_switches(**scene)

    @pytest.mark.parametrize(
        "box", [[0.0, 0.0, np.inf, 10.0], [10.0, 0.0, 10.0, 10.0], [0.0, 10.0, 10.0, 5.0]], ids=["inf", "x", "y"]
    )
    def test_a_box_that_is_not_finite_or_has_no_area_raises(self, box: list[float]):
        scene = _pair()
        scene["boxes"][7] = box
        with pytest.raises(ValueError, match="boxes"):
            track_switches(**scene)

    def test_a_label_that_is_not_whole_raises(self):
        scene = _pair()
        scene["labels"] = scene["labels"].astype(float)
        scene["labels"][3] = 0.5
        with pytest.raises(ValueError, match="labels"):
            track_switches(**scene)

    def test_input_with_no_tracked_row_raises(self):
        scene = _pair()
        scene["track_ids"] = np.full(60, -1)
        with pytest.raises(ValueError, match="No row has a track id"):
            track_switches(**scene)

    def test_input_with_no_track_of_two_rows_raises(self):
        scene = _pair()
        scene["sequence_indices"] = np.arange(60)
        with pytest.raises(ValueError, match="No track has two rows"):
            track_switches(**scene)

    def test_a_repeated_frame_raises(self):
        scene = _pair()
        scene["frame_indices"][5] = 4
        with pytest.raises(ValueError, match=r"track \d+ of sequence \d+ has two rows at frame"):
            track_switches(**scene)

    @pytest.mark.parametrize("bad", [1, 2.5, True, "10"])
    def test_a_window_that_is_not_an_integer_of_at_least_two_raises(self, bad: Any):
        with pytest.raises(ValueError, match="window"):
            track_switches(**_pair(), window=bad)


class TestExperimental:
    def test_it_is_exported(self):
        from dataeval.core import TrackSwitchesResult, track_switches  # noqa: F401

    def test_the_first_call_warns_once(self):
        code = textwrap.dedent(
            """
            import warnings
            import numpy as np
            from dataeval.core import track_switches
            from dataeval.exceptions import ExperimentalWarning

            rows = np.random.default_rng(0).normal(size=(6, 4))
            boxes = np.tile([0.0, 0.0, 10.0, 10.0], (6, 1))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(2):
                    track_switches(rows, np.r_[np.zeros(3, int), np.ones(3, int)], np.r_[0:3, 0:3], np.zeros(6, int),
                                   boxes=boxes, labels=np.zeros(6, int), window=2)
            print(sum(issubclass(w.category, ExperimentalWarning) for w in caught))
            """,
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "1"


def _swap_scene(n: int = 60, meet: tuple[int, int] = (20, 40), at: int = 30, swapped: bool = True) -> dict[str, Any]:
    """Two tracks within reach over frames ``meet[0]..meet[1]``, their objects exchanged from frame ``at``."""
    f = np.arange(n)
    x1 = np.where((f >= meet[0]) & (f <= meet[1]), 0.0, 100.0)
    before, after = (f < at), (f >= at)
    first = np.where(after & swapped, 1, 0)
    second = np.where(after & swapped, 0, 1)
    assert before.any()
    return _scene(
        _track(0, 1, f, x1, _rows(first.tolist(), n, seed=1)), _track(0, 2, f, 0.0, _rows(second.tolist(), n, seed=2))
    )


class TestPlantedSwitch:
    def test_a_swap_is_flagged_at_its_frame(self):
        result = track_switches(**_swap_scene())
        assert result["flagged"].tolist() == [True]
        assert result["switch_score"][0] > 0
        assert result["frame"].tolist() == [30]
        assert (result["row_a"][0], result["row_b"][0]) == (30, 90)

    def test_the_same_tracks_unswapped_are_not_flagged(self):
        result = track_switches(**_swap_scene(swapped=False))
        assert np.isfinite(result["switch_score"][0])
        assert result["switch_score"][0] < 0
        assert not result["flagged"][0]

    def test_a_half_swap_is_flagged(self):
        """Track 1 takes track 2's object at frame 30, and track 2 goes on to a third object."""
        f = np.arange(60)
        x1 = np.where((f >= 20) & (f <= 40), 0.0, 100.0)
        first = np.where(f >= 30, 1, 0).tolist()
        second = np.where(f >= 30, 2, 1).tolist()
        scene = _scene(_track(0, 1, f, x1, _rows(first, 60, seed=1)), _track(0, 2, f, 0.0, _rows(second, 60, seed=2)))
        result = track_switches(**scene)
        assert result["flagged"].tolist() == [True]
        assert result["frame"].tolist() == [30]

    def test_a_pair_meeting_twice_flags_only_the_swap(self):
        f = np.arange(70)
        x1 = np.where(((f >= 12) & (f <= 16)) | ((f >= 40) & (f <= 50)), 0.0, 100.0)
        first = np.where(f >= 45, 1, 0).tolist()
        second = np.where(f >= 45, 0, 1).tolist()
        scene = _scene(_track(0, 1, f, x1, _rows(first, 70, seed=1)), _track(0, 2, f, 0.0, _rows(second, 70, seed=2)))
        result = track_switches(**scene, window=5)
        assert result["flagged"].tolist() == [True, False]
        assert result["frame"][0] == 45
        assert 12 <= result["frame"][1] <= 16

    def test_an_equal_score_takes_the_earlier_frame(self):
        """Noise-free straight tracks score the same at every frame, so the first scored frame is reported."""
        f = np.arange(10)
        scene = _scene(_track(0, 1, f, 0.0, _rows(0, 10, noise=0.0)), _track(0, 2, f, 0.0, _rows(1, 10, noise=0.0)))
        result = track_switches(**scene, window=2)
        assert result["frame"].tolist() == [2]
        assert result["switch_score"][0] == pytest.approx(-2 * np.sqrt(2))

    def test_a_window_with_a_zero_row_is_not_scored(self):
        f = np.arange(10)
        scene = _scene(_track(0, 1, f, _apart_except(f, [5], 0.0), _rows(0, 10)), _track(0, 2, f, 0.0, _rows(1, 10)))
        scene["embeddings"][4] = 0.0
        result = track_switches(**scene, window=2)
        assert result["frame"].tolist() == [5]
        assert np.isnan(result["switch_score"][0])

    def test_a_later_window_without_the_zero_row_is_scored(self):
        f = np.arange(14)
        scene = _scene(_track(0, 1, f, _apart_except(f, [5, 9], 0.0), _rows(0, 14)), _track(0, 2, f, 0.0, _rows(1, 14)))
        scene["embeddings"][4] = 0.0
        result = track_switches(**scene, window=2)
        assert result["frame"].tolist() == [9]
        assert np.isfinite(result["switch_score"][0])

    def test_rows_sort_by_score_then_video_tracks_and_frame(self):
        """Finite scores first, highest first; then the rows with no score, by video, tracks and frame."""
        f60, f6 = np.arange(60), np.arange(6)
        swap = _swap_scene()
        short = _scene(
            _track(0, 3, f6, 1000.0, _rows(2, 6)),
            _track(0, 4, f6, 1000.0, _rows(3, 6)),
            _track(1, 1, f6, 0.0, _rows(2, 6)),
            _track(1, 2, f6, 0.0, _rows(3, 6)),
            _track(1, 5, f60, 2000.0, _rows(4, 60)),
            _track(1, 6, f60, 2000.0, _rows(5, 60)),
        )
        scene: dict[str, Any] = {name: np.concatenate([swap[name], short[name]]) for name in swap}
        result = track_switches(**scene)
        rows = list(
            zip(result["item_index"].tolist(), result["track_a"].tolist(), result["track_b"].tolist(), strict=True)
        )
        assert rows == [(0, 1, 2), (1, 5, 6), (0, 3, 4), (1, 1, 2)]
        assert np.isfinite(result["switch_score"][:2]).all()
        assert np.isnan(result["switch_score"][2:]).all()


class TestDeterminism:
    def test_reordering_the_input_permutes_rows_and_changes_nothing_else(self):
        scene = _swap_scene()
        perm = np.random.default_rng(7).permutation(len(scene["embeddings"]))
        inverse = np.argsort(perm)
        original = track_switches(**scene)
        shuffled = track_switches(**{name: column[perm] for name, column in scene.items()})
        assert original["flagged"].tolist() == [True]
        assert shuffled["row_a"].tolist() == inverse[original["row_a"]].tolist()
        assert shuffled["row_b"].tolist() == inverse[original["row_b"]].tolist()
        for name in ("item_index", "label", "track_a", "track_b", "frame", "switch_score", "flagged"):
            assert shuffled[name].tolist() == original[name].tolist()  # pyright: ignore[reportGeneralTypeIssues]

    def test_two_runs_are_identical(self):
        first, second = track_switches(**_swap_scene()), track_switches(**_swap_scene())
        assert first["flagged"].tolist() == [True]
        assert _lists(first) == _lists(second)

    def test_scaling_a_row_by_two_changes_nothing(self):
        scene = _swap_scene()
        original = track_switches(**scene)
        scene["embeddings"][[25, 35, 85]] *= 2.0
        scaled = track_switches(**scene)
        assert original["flagged"].tolist() == [True]
        assert scaled["switch_score"].tolist() == original["switch_score"].tolist()

    def test_scaling_a_row_by_3_7_changes_nothing_beyond_rounding(self):
        scene = _swap_scene()
        original = track_switches(**scene)
        scene["embeddings"][[25, 35, 85]] *= 3.7
        scaled = track_switches(**scene)
        assert np.isfinite(original["switch_score"]).all()
        np.testing.assert_allclose(scaled["switch_score"], original["switch_score"], rtol=1e-12)
        assert scaled["frame"].tolist() == original["frame"].tolist()

    def test_boxes_in_another_unit_give_the_same_rows(self):
        """One scale on both axes changes no meeting, since centers are compared with diagonals."""
        scene = _swap_scene()
        original = track_switches(**scene)
        scene["boxes"] = scene["boxes"] / 1920.0
        scaled = track_switches(**scene)
        assert original["flagged"].tolist() == [True]
        assert _lists(scaled) == _lists(original)

    def test_another_video_does_not_change_a_score(self):
        """A video sorted ahead, with large non-negative rows, leaves the planted swap's score bit for bit the same."""
        swap = _swap_scene()
        alone = track_switches(**swap)
        f = np.arange(20_000)
        ahead = _scene(_track(-1, 7, f, 5000.0, np.abs(_rows(0, 20_000, noise=1.0, seed=3)) + 1.0))
        combined: dict[str, Any] = {name: np.concatenate([ahead[name], swap[name]]) for name in swap}
        both = track_switches(**combined)
        assert alone["flagged"].tolist() == [True]
        assert both["switch_score"].tolist() == alone["switch_score"].tolist()

    def test_an_embeddings_object_gives_the_same_result(self):
        from dataeval import Embeddings
        from dataeval.extractors import FlattenExtractor

        scene = _swap_scene()
        embeddings = Embeddings([row.reshape(1, 1, D) for row in scene["embeddings"]], extractor=FlattenExtractor())
        as_object = track_switches(**{**scene, "embeddings": embeddings})
        as_array = track_switches(**{**scene, "embeddings": np.asarray(embeddings)})
        assert as_array["flagged"].tolist() == [True]
        assert _lists(as_object) == _lists(as_array)

"""`track_path_stats`: how far, and in which direction, a track's embeddings travel."""

from __future__ import annotations

import itertools
import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.core import track_path_stats
from tests.metadata.test_structurers import _mot_dataset

pytestmark = [pytest.mark.required, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]

DIM = 32


def _path(n: int, amount: float = 0.0, noise: float = 0.02, seed: int = 0, axis: int = 1) -> np.ndarray:
    """``n`` rows of one path: a unit vector along axis 0 plus a ramp of ``amount`` along ``axis``, plus noise.

    The ramp runs from ``-amount / 2`` to ``amount / 2``, so over the first and last thirds its means sit at about
    ``-amount / 3`` and ``amount / 3``, which is ``2 * amount / 3`` apart.
    """
    rows = np.zeros((n, DIM))
    rows[:, 0] = 1.0
    rows[:, axis] = amount * (np.arange(n) / (n - 1) - 0.5)
    return rows + np.random.default_rng(seed).normal(0.0, noise, rows.shape)


def _flat(tracks: list[tuple[int, int, np.ndarray]]):
    """The four per-row arrays of ``[(sequence, track id, rows), ...]``, frames counting 0, 1, 2 ... in each track."""
    embeddings = np.concatenate([rows for _, _, rows in tracks])
    tracks_ = np.concatenate([np.full(len(rows), track) for _, track, rows in tracks])
    frames = np.concatenate([np.arange(len(rows)) for _, _, rows in tracks])
    sequences = np.concatenate([np.full(len(rows), sequence) for sequence, _, rows in tracks])
    return embeddings, tracks_, frames, sequences


def _stack(tracks: list[tuple[int, int, np.ndarray]], **kwargs: Any) -> Any:
    embeddings, track_ids, frames, sequences = _flat(tracks)
    return track_path_stats(embeddings, track_ids, frames, sequences, **kwargs)


def _same(a: Any, b: Any) -> None:
    """Every column of two results is bit-identical, NaN matching NaN."""
    assert a["stats"].keys() == b["stats"].keys()
    for name in a["stats"]:
        np.testing.assert_array_equal(a["stats"][name], b["stats"][name])
    np.testing.assert_array_equal(a["details"]["path_direction"], b["details"]["path_direction"])


class TestPlantedDrift:
    def test_the_displacement_ratio_and_direction_follow_the_planted_path(self):
        result = _stack([(0, 1, _path(90, amount=0.6))])
        # The ramp runs from -0.3 to 0.3. Its first and last thirds average about -0.2 and +0.2, so 0.4 apart.
        displacement = result["stats"]["path_displacement"][0]
        assert displacement == pytest.approx(0.4, abs=0.03)
        assert displacement / result["stats"]["path_displacement_null"][0] > 5
        assert result["details"]["path_direction"][0] @ np.eye(DIM)[1] > 0.95

    def test_the_length_of_a_row_does_not_matter_only_its_direction(self):
        rows = _path(90, amount=0.6)
        scales = np.random.default_rng(7).uniform(0.2, 5.0, size=(90, 1))
        plain, scaled = _stack([(0, 1, rows)]), _stack([(0, 1, rows * scales)])
        for name in ("path_displacement", "path_displacement_null"):
            np.testing.assert_allclose(scaled["stats"][name], plain["stats"][name], rtol=1e-12)
        np.testing.assert_allclose(scaled["details"]["path_direction"], plain["details"]["path_direction"], atol=1e-12)

    def test_a_larger_planted_amount_gives_a_larger_displacement(self):
        result = _stack([(0, 1, _path(90, amount=0.3, seed=1)), (0, 2, _path(90, amount=0.9, seed=2))])
        small, large = result["stats"]["path_displacement"]
        assert small == pytest.approx(0.2, abs=0.03)
        assert large == pytest.approx(0.6, abs=0.05)
        assert small < large

    def test_a_track_that_does_not_move_has_a_ratio_near_one(self):
        result = _stack([(0, 1, _path(90, amount=0.0, seed=3))])
        # Over random orders the squared ratio averages 1, so a ratio of 3 is far in the tail.
        assert result["stats"]["path_displacement"][0] / result["stats"]["path_displacement_null"][0] < 3


class TestNullIsExact:
    @pytest.mark.parametrize("n", [3, 6, 7])
    def test_the_null_is_the_root_mean_square_displacement_over_every_order(self, n: int):
        rows = np.random.default_rng(n).normal(size=(n, 5))
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        k = max(n // 3, 1)
        squares = [
            np.sum((rows[list(order)][-k:].mean(axis=0) - rows[list(order)][:k].mean(axis=0)) ** 2)
            for order in itertools.permutations(range(n))
        ]
        result = track_path_stats(rows, np.zeros(n, int), np.arange(n), np.zeros(n, int), min_points=3)
        assert result["stats"]["path_displacement_null"][0] ** 2 == pytest.approx(np.mean(squares), rel=0, abs=1e-12)


class TestDeterminism:
    @staticmethod
    def tracks():
        return [(0, 1, _path(70, 0.5, seed=1)), (0, 2, _path(80, 0.2, seed=2)), (1, 1, _path(75, 0.8, seed=3))]

    def test_shuffling_the_input_rows_changes_nothing(self):
        embeddings, track_ids, frames, sequences = _flat(self.tracks())
        order = np.random.default_rng(0).permutation(len(embeddings))
        shuffled = track_path_stats(embeddings[order], track_ids[order], frames[order], sequences[order])
        _same(track_path_stats(embeddings, track_ids, frames, sequences), shuffled)

    def test_rows_are_sorted_by_video_then_track(self):
        scrambled = [self.tracks()[2], self.tracks()[1], self.tracks()[0]]
        result = _stack(scrambled)
        assert result["stats"]["item_index"].tolist() == [0, 0, 1]
        assert result["stats"]["track_ids"].tolist() == [1, 2, 1]
        assert result["stats"]["n_points"].tolist() == [70, 80, 75]

    def test_a_track_does_not_depend_on_the_other_tracks(self):
        everything = _stack(self.tracks())
        alone = _stack([self.tracks()[1]])
        for name in alone["stats"]:
            np.testing.assert_array_equal(alone["stats"][name][0], everything["stats"][name][1])
        np.testing.assert_array_equal(alone["details"]["path_direction"][0], everything["details"]["path_direction"][1])

    def test_directions_have_unit_length(self):
        directions = _stack(self.tracks())["details"]["path_direction"]
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)


class TestFailClosed:
    def test_a_short_track_gets_nan_and_a_warning_with_the_counts(self):
        with pytest.warns(UserWarning, match=r"1 of 2 tracks have fewer than 60 points.*min_points"):
            result = _stack([(0, 1, _path(90, 0.5)), (0, 2, _path(40, 0.5, seed=1))])
        assert result["stats"]["n_points"].tolist() == [90, 40]
        assert np.isfinite(result["stats"]["path_displacement"][0])
        assert np.isnan(result["stats"]["path_displacement"][1])
        assert np.isnan(result["stats"]["path_displacement_null"][1])
        assert np.isnan(result["details"]["path_direction"][1]).all()

    def test_untracked_rows_are_not_part_of_any_track(self):
        track = _path(90, 0.5)
        untracked = _path(10, 0.9, seed=5)
        embeddings = np.concatenate([track, untracked])
        track_ids = np.concatenate([np.full(90, 1), np.full(10, -1)])
        frames = np.concatenate([np.arange(90), np.arange(10)])
        result = track_path_stats(embeddings, track_ids, frames, np.zeros(100, int))
        assert result["stats"]["track_ids"].tolist() == [1]
        assert result["stats"]["n_points"].tolist() == [90]
        _same(result, _stack([(0, 1, track)]))

    def test_the_same_id_in_two_videos_is_two_tracks(self):
        result = _stack([(0, 7, _path(70, 0.5, seed=1)), (1, 7, _path(70, 0.5, seed=2))])
        assert result["stats"]["track_ids"].tolist() == [7, 7]
        assert result["stats"]["item_index"].tolist() == [0, 1]

    def test_a_repeated_frame_raises_and_says_how_to_fix_it(self):
        frames = np.arange(70)
        frames[10] = 9
        with pytest.raises(ValueError, match=r"track 1 of sequence 0 has two rows at frame 9.*own frame index"):
            track_path_stats(_path(70, 0.5), np.ones(70, int), frames, np.zeros(70, int))

    def test_input_with_no_tracked_row_raises_and_names_sequenceframes(self):
        with pytest.raises(ValueError, match="SequenceFrames"):
            track_path_stats(_path(70), np.full(70, -1), np.arange(70), np.zeros(70, int))

    def test_one_row_per_track_raises_and_names_sequence_indices(self):
        # Each row in a video of its own: what passing `crops.item_indices` for `sequence_indices` gives.
        with pytest.raises(ValueError, match="sequence_indices"):
            track_path_stats(_path(70), np.zeros(70, int), np.arange(70), np.arange(70))

    @pytest.mark.parametrize("bad", [0, 2, -1, 3.5, True, "60"])
    def test_min_points_below_three_or_not_an_integer_raises(self, bad: Any):
        with pytest.raises(ValueError, match="min_points"):
            _stack([(0, 1, _path(70, 0.5))], min_points=bad)

    def test_a_non_finite_embedding_raises(self):
        rows = _path(70, 0.5)
        rows[5, 3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            _stack([(0, 1, rows)])

    def test_a_zero_length_row_gives_its_track_nan_and_no_other(self):
        flawed = _path(70, 0.5, seed=1)
        flawed[7] = 0.0
        result = _stack([(0, 1, flawed), (0, 2, _path(70, 0.5, seed=2))])
        assert np.isnan(result["stats"]["path_displacement"][0])
        assert np.isfinite(result["stats"]["path_displacement"][1])

    def test_inputs_of_different_lengths_raise_and_name_the_input(self):
        embeddings, track_ids, frames, sequences = _flat([(0, 1, _path(70, 0.5))])
        with pytest.raises(ValueError, match="track_ids"):
            track_path_stats(embeddings, track_ids[:-1], frames, sequences)

    def test_track_ids_that_are_not_whole_numbers_raise(self):
        embeddings, _, frames, sequences = _flat([(0, 1, _path(70, 0.5))])
        with pytest.raises(ValueError, match="track_ids"):
            track_path_stats(embeddings, np.full(70, 1.5), frames, sequences)


class TestAttach:
    def test_the_result_attaches_to_the_track_rows(self):
        # Video 0 holds tracks 0 and 1 over 6 frames, and video 1 holds track 0 over 4.
        metadata = Metadata(_mot_dataset([[2] * 6, [1] * 4]))
        result = _stack(
            [(0, 0, _path(6, 0.5, seed=1)), (0, 1, _path(6, 0.9, seed=2)), (1, 0, _path(4, 0.5, seed=3))],
            min_points=3,
        )
        metadata.add_factors(result, level="track", key="track_id")
        rows = metadata.rows_at("track")
        assert rows.height == 3
        assert rows["path_displacement"].to_list() == result["stats"]["path_displacement"].tolist()


class TestExperimental:
    def test_the_first_call_warns_once(self):
        code = textwrap.dedent(
            """
            import warnings
            import numpy as np
            from dataeval.core import track_path_stats
            from dataeval.exceptions import ExperimentalWarning

            rows = np.random.default_rng(0).normal(size=(6, 4))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(2):
                    track_path_stats(rows, np.zeros(6, int), np.arange(6), np.zeros(6, int), min_points=3)
            print(sum(issubclass(w.category, ExperimentalWarning) for w in caught))
            """,
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "1"

"""Tests for the core track-statistics module (``_track_stats.py``)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from dataeval.core import TrackStatsResult, track_stats
from dataeval.core._track_stats import (
    _at_edge,
    _centers,
    _compute_appearances_and_duration,
    _compute_edge_flags,
    _compute_gaps,
    _compute_jitter_rate,
    _compute_speed_stats,
    _compute_step_speeds,
    _compute_straightness,
    _jitter_sparc,
)
from dataeval.types import Track


def make_track(boxes, frames, track_id: int = 0, labels=None, scores=None) -> Track:
    """Build a real ``Track`` from a list of boxes and frame indices.

    ``Track`` is a concrete (pydantic) dataclass, not a protocol, so we
    construct the real type rather than a stand-in: it satisfies the
    ``Mapping[int, Track]`` parameter of ``track_stats`` directly and is
    validated by the typing pipeline for free. ``frame_indices`` must be an
    ndarray (``Track`` validates each field with an ``isinstance`` check).
    ``labels`` and ``scores`` default to arrays aligned with ``frames`` — all
    class 0 and all confidence 1.0 (ground-truth-like) respectively. Pass
    explicit sequences to exercise the majority-label, ``label_confidence``
    and ``mean_score`` reductions.
    """
    frames_arr = np.asarray(frames, dtype=np.int64)
    n = len(frames_arr)
    labels_arr = np.zeros(n, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
    scores_arr = np.ones(n, dtype=np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
    return Track(
        track_id=track_id,
        boxes=np.array(boxes, dtype=np.float64).reshape(-1, 4),
        frame_indices=frames_arr,
        scores=scores_arr,
        labels=labels_arr,
    )


ALL_FIELDS = (
    "track_ids",
    "labels",
    "label_confidence",
    "mean_score",
    "n_appearances",
    "track_duration",
    "n_gaps",
    "total_gap_length",
    "mean_speed",
    "speed_variance",
    "net_displacement",
    "straightness_index",
    "jitter_rate",
    "entry_at_edge",
    "exit_at_edge",
)


class TestCenters:
    """Tests for ``_centers``."""

    def test_centers_basic(self):
        boxes = np.array([[0, 0, 10, 20], [4, 6, 8, 10]], dtype=np.float64)
        ctrs = _centers(boxes)
        np.testing.assert_array_equal(ctrs, np.array([[5.0, 10.0], [6.0, 8.0]]))

    def test_centers_shape_and_dtype(self):
        boxes = np.array([[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 4, 4]], dtype=np.float32)
        ctrs = _centers(boxes)
        assert ctrs.shape == (3, 2)
        assert ctrs.dtype == np.float64

    def test_centers_single_box(self):
        ctrs = _centers(np.array([[10, 20, 30, 40]], dtype=np.float64))
        np.testing.assert_array_equal(ctrs, np.array([[20.0, 30.0]]))


class TestAtEdge:
    """Tests for ``_at_edge``."""

    def test_interior_box_is_not_at_edge(self):
        assert _at_edge(np.array([50, 50, 60, 60]), 100, 100, 5) is False

    def test_left_edge(self):
        assert _at_edge(np.array([2, 50, 40, 60]), 100, 100, 5) is True

    def test_top_edge(self):
        assert _at_edge(np.array([50, 3, 60, 40]), 100, 100, 5) is True

    def test_right_edge(self):
        assert _at_edge(np.array([50, 50, 97, 60]), 100, 100, 5) is True

    def test_bottom_edge(self):
        assert _at_edge(np.array([50, 50, 60, 96]), 100, 100, 5) is True

    def test_exactly_on_threshold_left_is_inclusive(self):
        # x1 == threshold -> within threshold (inclusive).
        assert _at_edge(np.array([5, 50, 40, 60]), 100, 100, 5) is True

    def test_exactly_on_threshold_right_is_inclusive(self):
        # x2 == frame_w - threshold -> within threshold (inclusive).
        assert _at_edge(np.array([50, 50, 95, 60]), 100, 100, 5) is True

    def test_returns_python_bool(self):
        # Guards against returning numpy.bool_ from the comparisons.
        assert type(_at_edge(np.array([50, 50, 60, 60]), 100, 100, 5)) is bool


class TestAppearancesAndDuration:
    """Tests for ``_compute_appearances_and_duration``."""

    def test_contiguous(self):
        assert _compute_appearances_and_duration(np.array([0, 1, 2])) == (3, 3)

    def test_with_gaps(self):
        # First appearance 0, last 5 -> duration 6; 3 appearances.
        assert _compute_appearances_and_duration(np.array([0, 2, 5])) == (3, 6)

    def test_single_frame(self):
        assert _compute_appearances_and_duration(np.array([7])) == (1, 1)

    def test_empty(self):
        assert _compute_appearances_and_duration(np.array([])) == (0, 0)


class TestComputeGaps:
    """Tests for ``_compute_gaps``."""

    def test_contiguous_has_no_gaps(self):
        assert _compute_gaps(np.array([0, 1, 2]), 3, 3) == (0, 0)

    def test_single_multi_frame_gap_run(self):
        # frames [0, 1, 5]: one run of missing frames (2,3,4) -> 1 gap, length 3.
        assert _compute_gaps(np.array([0, 1, 5]), 3, 6) == (1, 3)

    def test_multiple_gap_runs(self):
        # frames [0, 2, 5]: two separate gap runs -> 2 gaps, total length 3.
        assert _compute_gaps(np.array([0, 2, 5]), 3, 6) == (2, 3)

    def test_single_observation_no_gaps(self):
        assert _compute_gaps(np.array([4]), 1, 1) == (0, 0)

    def test_empty_no_gaps(self):
        assert _compute_gaps(np.array([]), 0, 0) == (0, 0)


class TestComputeStepSpeeds:
    """Tests for ``_compute_step_speeds``."""

    def test_basic_contiguous(self):
        boxes = np.array([[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]], dtype=np.float64)
        delta_pos, step_speeds = _compute_step_speeds(_centers(boxes), np.array([0, 1, 2]), 3)
        np.testing.assert_allclose(step_speeds, [10.0, 10.0])
        np.testing.assert_allclose(delta_pos, [[10.0, 0.0], [10.0, 0.0]])

    def test_normalised_by_inter_frame_delta(self):
        # Gaps must not inflate the speed estimate: displacement / frame-delta.
        boxes = np.array([[0, 0, 2, 2], [10, 0, 12, 2], [40, 0, 42, 2]], dtype=np.float64)
        _, step_speeds = _compute_step_speeds(_centers(boxes), np.array([0, 2, 5]), 3)
        # step 1: 10px over 2 frames = 5; step 2: 30px over 3 frames = 10.
        np.testing.assert_allclose(step_speeds, [5.0, 10.0])

    def test_single_observation_returns_empty(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float64)
        delta_pos, step_speeds = _compute_step_speeds(_centers(boxes), np.array([0]), 1)
        assert step_speeds.shape == (0,)
        assert delta_pos.shape == (0, 2)


class TestComputeSpeedStats:
    """Tests for ``_compute_speed_stats``."""

    def test_basic(self):
        mean, var = _compute_speed_stats(np.array([5.0, 10.0]))
        assert mean == pytest.approx(7.5)
        assert var == pytest.approx(6.25)

    def test_constant_speed_zero_variance(self):
        mean, var = _compute_speed_stats(np.array([10.0, 10.0, 10.0]))
        assert mean == pytest.approx(10.0)
        assert var == pytest.approx(0.0)

    def test_empty_returns_zeros(self):
        assert _compute_speed_stats(np.array([])) == (0.0, 0.0)


class TestComputeStraightness:
    """Tests for ``_compute_straightness``."""

    def test_perfectly_straight(self):
        delta_pos = np.array([[10.0, 0.0], [10.0, 0.0]])
        assert _compute_straightness(20.0, delta_pos, 3) == pytest.approx(1.0)

    def test_tortuous_path_below_one(self):
        # Out-and-back: net displacement small, path length large.
        delta_pos = np.array([[10.0, 0.0], [-10.0, 0.0]])
        assert _compute_straightness(0.0, delta_pos, 3) == pytest.approx(0.0)

    def test_single_observation_is_nan(self):
        assert np.isnan(_compute_straightness(0.0, np.empty((0, 2)), 1))

    def test_zero_path_length_is_nan(self):
        # Stationary track: total path length 0 -> undefined straightness.
        assert np.isnan(_compute_straightness(0.0, np.zeros((2, 2)), 3))


class TestComputeEdgeFlags:
    """Tests for ``_compute_edge_flags``."""

    def test_no_frame_dims_returns_false_false(self):
        boxes = np.array([[0, 0, 10, 10], [90, 90, 100, 100]], dtype=np.float64)
        assert _compute_edge_flags(boxes, False, 0, 0, 5) == (False, False)

    def test_first_and_last_box_at_edge(self):
        boxes = np.array([[0, 0, 10, 10], [90, 90, 100, 100]], dtype=np.float64)
        assert _compute_edge_flags(boxes, True, 100, 100, 5) == (True, True)

    def test_interior_track_not_at_edge(self):
        boxes = np.array([[40, 40, 50, 50], [45, 45, 55, 55]], dtype=np.float64)
        assert _compute_edge_flags(boxes, True, 100, 100, 5) == (False, False)

    def test_entry_only(self):
        # First box at edge, last box interior.
        boxes = np.array([[0, 0, 10, 10], [40, 40, 50, 50]], dtype=np.float64)
        assert _compute_edge_flags(boxes, True, 100, 100, 5) == (True, False)


class TestJitterSparc:
    """Tests for ``_jitter_sparc``."""

    def test_returns_non_negative_finite(self):
        val = _jitter_sparc(np.ones(20), fc_norm=0.5)
        assert np.isfinite(val)
        assert val >= 0.0

    def test_deterministic(self):
        a = _jitter_sparc(np.ones(20), fc_norm=0.5)
        b = _jitter_sparc(np.ones(20), fc_norm=0.5)
        assert a == b

    def test_amplitude_invariance(self):
        # SPARC normalizes the magnitude spectrum by its DC component, so
        # scaling the speed profile by a positive constant must not change the
        # result. This is the paper's first requirement for a valid smoothness
        # measure (dimensionless / amplitude-independent) and holds exactly up
        # to floating-point rounding, so it is asserted directly rather than
        # pinned to a captured number.
        rng = np.random.RandomState(1)
        profile = 3.0 + rng.standard_normal(64)
        base = _jitter_sparc(profile, fc_norm=0.5)
        scaled = _jitter_sparc(5.0 * profile, fc_norm=0.5)
        assert scaled == pytest.approx(base, rel=1e-9)

    def test_jittery_profile_scores_higher_than_smooth(self):
        # Higher return value == less smooth. A broadband, noisy speed profile
        # must score strictly above a flat (constant-speed) one.
        rng = np.random.RandomState(0)
        smooth = np.full(64, 5.0)
        jittery = 5.0 + 2.0 * rng.standard_normal(64)
        assert _jitter_sparc(jittery, fc_norm=0.5) > _jitter_sparc(smooth, fc_norm=0.5)

    def test_lower_cutoff_changes_result(self):
        full = _jitter_sparc(np.ones(20), fc_norm=0.5)
        narrow = _jitter_sparc(np.ones(20), fc_norm=0.4)
        assert narrow != full

    @pytest.mark.parametrize("bad_fc", [0.0, -0.1, 0.6, 1.0])
    def test_fc_norm_out_of_bounds_raises(self, bad_fc):
        with pytest.raises(ValueError, match="fc_norm must be in"):
            _jitter_sparc(np.ones(10), fc_norm=bad_fc)

    def test_zero_dc_component_returns_zero(self):
        # A zero-mean signal has mag[0] == 0 -> early return 0.0.
        signal = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        assert _jitter_sparc(signal, fc_norm=0.5) == 0.0


class TestComputeJitterRate:
    """Tests for ``_compute_jitter_rate``."""

    def test_just_below_min_frames_is_nan(self):
        # tl one short of the threshold is gated out -> NaN.
        assert np.isnan(_compute_jitter_rate(np.ones(8), tl=9, jitter_min_frames=10, jitter_fc=0.5))

    def test_exactly_at_min_frames_is_computed(self):
        # tl exactly equal to the threshold clears the gate (inclusive boundary).
        val = _compute_jitter_rate(np.ones(9), tl=10, jitter_min_frames=10, jitter_fc=0.5)
        assert np.isfinite(val)
        assert val >= 0.0


@pytest.mark.required
class TestTrackStats:
    """Functional tests for the public ``track_stats`` entry point."""

    def test_basic_contiguous_track(self):
        track = make_track(
            [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]],
            [0, 1, 2],
        )
        stats = track_stats({1: track})

        assert stats["track_ids"] == [1]
        assert stats["n_appearances"] == [3]
        assert stats["track_duration"] == [3]
        assert stats["n_gaps"] == [0]
        assert stats["total_gap_length"] == [0]
        assert stats["mean_speed"][0] == pytest.approx(10.0)
        assert stats["speed_variance"][0] == pytest.approx(0.0)
        assert stats["net_displacement"][0] == pytest.approx(20.0)
        assert stats["straightness_index"][0] == pytest.approx(1.0)
        # Short track (< jitter_min_frames) -> jitter_rate is NaN, no _jitter_sparc call.
        assert np.isnan(stats["jitter_rate"][0])
        assert stats["entry_at_edge"] == [False]
        assert stats["exit_at_edge"] == [False]

    def test_track_with_gaps(self):
        # frames [0, 2, 5]: two gap runs, total 3 missing frames.
        track = make_track(
            [[0, 0, 2, 2], [10, 0, 12, 2], [40, 0, 42, 2]],
            [0, 2, 5],
        )
        stats = track_stats({1: track})

        assert stats["n_appearances"] == [3]
        assert stats["track_duration"] == [6]
        assert stats["n_gaps"] == [2]
        assert stats["total_gap_length"] == [3]
        # total_gap_length == track_duration - n_appearances
        assert stats["total_gap_length"][0] == stats["track_duration"][0] - stats["n_appearances"][0]
        # Speeds normalised by inter-frame delta: [10/2, 30/3] = [5, 10].
        assert stats["mean_speed"][0] == pytest.approx(7.5)
        assert stats["speed_variance"][0] == pytest.approx(6.25)
        assert stats["net_displacement"][0] == pytest.approx(40.0)
        assert stats["straightness_index"][0] == pytest.approx(1.0)

    def test_single_appearance_track(self):
        # Edge case: a track of length 1.
        track = make_track([[0, 0, 10, 10]], [3])
        stats = track_stats({1: track})

        assert stats["n_appearances"] == [1]
        assert stats["track_duration"] == [1]
        assert stats["n_gaps"] == [0]
        assert stats["total_gap_length"] == [0]
        assert stats["mean_speed"][0] == pytest.approx(0.0)
        assert stats["speed_variance"][0] == pytest.approx(0.0)
        assert stats["net_displacement"][0] == pytest.approx(0.0)
        assert np.isnan(stats["straightness_index"][0])
        assert np.isnan(stats["jitter_rate"][0])

    def test_stationary_track_straightness_is_nan(self):
        # Multi-frame but never moves: zero path length -> straightness NaN.
        track = make_track([[0, 0, 10, 10]] * 3, [0, 1, 2])
        stats = track_stats({1: track})

        assert stats["mean_speed"][0] == pytest.approx(0.0)
        assert stats["net_displacement"][0] == pytest.approx(0.0)
        assert np.isnan(stats["straightness_index"][0])

    def test_diagonal_net_displacement(self):
        # Centers (0,0) -> (3,4): Euclidean distance 5.
        track = make_track([[0, 0, 0, 0], [3, 4, 3, 4]], [0, 1])
        stats = track_stats({1: track})
        assert stats["net_displacement"][0] == pytest.approx(5.0)

    def test_multiple_tracks_sorted_by_id(self):
        a = make_track([[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]], [0, 1, 2])
        b = make_track([[0, 0, 2, 2], [10, 0, 12, 2], [40, 0, 42, 2]], [0, 2, 5])
        stats = track_stats({5: a, 2: b})

        # Output is indexed by position in *sorted* track-ID order.
        assert stats["track_ids"] == [2, 5]
        assert stats["n_appearances"] == [3, 3]
        assert stats["n_gaps"] == [2, 0]  # track 2 has gaps, track 5 does not

    def test_label_is_majority_per_frame_label(self):
        # Track is mostly class 3 with one dissenting class-1 frame.
        track = make_track([[0, 0, 10, 10]] * 4, [0, 1, 2, 3], labels=[3, 3, 1, 3])
        stats = track_stats({1: track})
        assert stats["labels"] == [3]

    def test_label_confidence_is_one_for_single_class_track(self):
        track = make_track([[0, 0, 10, 10]] * 3, [0, 1, 2], labels=[2, 2, 2])
        stats = track_stats({1: track})
        assert stats["label_confidence"][0] == pytest.approx(1.0)

    def test_label_confidence_is_frame_purity_for_uniform_scores(self):
        # 3 of 4 frames agree with the majority label; scores are uniform (1.0).
        track = make_track([[0, 0, 10, 10]] * 4, [0, 1, 2, 3], labels=[3, 3, 1, 3])
        stats = track_stats({1: track})
        assert stats["label_confidence"][0] == pytest.approx(0.75)

    def test_label_confidence_is_score_weighted(self):
        # Majority label 0 (2 frames) carries little score; label 1 dominates score.
        track = make_track([[0, 0, 10, 10]] * 3, [0, 1, 2], labels=[0, 0, 1], scores=[0.1, 0.1, 0.8])
        stats = track_stats({1: track})
        assert stats["labels"] == [0]  # majority by frame count
        # confidence in label 0 = (0.1 + 0.1) / 1.0
        assert stats["label_confidence"][0] == pytest.approx(0.2)

    def test_mean_score(self):
        track = make_track([[0, 0, 10, 10]] * 3, [0, 1, 2], scores=[0.2, 0.4, 0.9])
        stats = track_stats({1: track})
        assert stats["mean_score"][0] == pytest.approx(0.5)

    def test_labels_aligned_with_sorted_track_ids(self):
        a = make_track([[0, 0, 10, 10], [10, 0, 20, 10]], [0, 1], labels=[7, 7])
        b = make_track([[0, 0, 2, 2], [10, 0, 12, 2]], [0, 1], labels=[4, 4])
        stats = track_stats({5: a, 2: b})
        # labels follow the same sorted-ID order as every other field.
        assert stats["track_ids"] == [2, 5]
        assert stats["labels"] == [4, 7]

    def test_all_fields_present_and_aligned(self):
        track = make_track([[0, 0, 10, 10], [10, 0, 20, 10]], [0, 1])
        stats = track_stats({1: track})
        for fieldname in ALL_FIELDS:
            assert fieldname in stats, f"missing field {fieldname}"
            assert len(stats[fieldname]) == 1

    def test_empty_tracks_mapping(self):
        stats = track_stats({})
        for fieldname in ALL_FIELDS:
            assert stats[fieldname] == []

    # --- edge flags -------------------------------------------------------

    def test_edge_flags_with_frame_dims(self):
        # First box top-left corner, last box bottom-right corner.
        track = make_track([[0, 0, 10, 10], [90, 90, 100, 100]], [0, 1])
        stats = track_stats({1: track}, frame_width=100, frame_height=100, edge_threshold=5)
        assert stats["entry_at_edge"] == [True]
        assert stats["exit_at_edge"] == [True]

    def test_interior_track_no_edge_flags(self):
        track = make_track([[40, 40, 50, 50], [45, 45, 55, 55]], [0, 1])
        stats = track_stats({1: track}, frame_width=100, frame_height=100, edge_threshold=5)
        assert stats["entry_at_edge"] == [False]
        assert stats["exit_at_edge"] == [False]

    def test_missing_frame_dims_warns_and_flags_false(self, caplog):
        track = make_track([[0, 0, 10, 10], [90, 90, 100, 100]], [0, 1])
        with caplog.at_level(logging.WARNING):
            stats = track_stats({1: track})  # no frame_width/frame_height
        assert any("entry_at_edge and exit_at_edge will always be False" in r.message for r in caplog.records)
        assert stats["entry_at_edge"] == [False]
        assert stats["exit_at_edge"] == [False]

    def test_large_edge_threshold_warns(self, caplog):
        track = make_track([[40, 40, 50, 50], [45, 45, 55, 55]], [0, 1])
        with caplog.at_level(logging.WARNING):
            track_stats({1: track}, frame_width=100, frame_height=100, edge_threshold=60)
        assert any("edge flags will be unreliable" in r.message for r in caplog.records)

    # --- frame-dimension validation --------------------------------------

    def test_non_positive_frame_width_raises(self):
        track = make_track([[0, 0, 10, 10], [10, 0, 20, 10]], [0, 1])
        with pytest.raises(ValueError, match="frame_width must be positive"):
            track_stats({1: track}, frame_width=0, frame_height=100)

    def test_non_positive_frame_height_raises(self):
        track = make_track([[0, 0, 10, 10], [10, 0, 20, 10]], [0, 1])
        with pytest.raises(ValueError, match="frame_height must be positive"):
            track_stats({1: track}, frame_width=100, frame_height=-1)

    # --- jitter -----------------------------------------------------------

    def test_jitter_computed_for_long_track(self):
        # A track long enough to clear the default jitter_min_frames is computed
        # using the default jitter_fc (0.5).
        frames = list(range(12))
        boxes = [[i * 1.0, 0, i * 1.0 + 5, 5] for i in range(12)]
        track = make_track(boxes, frames)
        stats = track_stats({1: track})
        assert np.isfinite(stats["jitter_rate"][0])
        assert stats["jitter_rate"][0] >= 0.0

    def test_jitter_min_frames_override(self):
        # A 5-frame track is gated out under the default jitter_min_frames (10);
        # lowering the threshold lets a finite jitter be computed.
        frames = [0, 1, 2, 3, 4]
        boxes = [[i * 2.0, 0, i * 2.0 + 4, 4] for i in range(5)]
        track = make_track(boxes, frames)

        gated = track_stats({1: track})
        assert np.isnan(gated["jitter_rate"][0])

        computed = track_stats({1: track}, jitter_min_frames=3)
        assert np.isfinite(computed["jitter_rate"][0])


class TestTrackStatsResult:
    """Tests for the ``TrackStatsResult`` typed dict."""

    def test_declares_all_fields(self):
        assert set(TrackStatsResult.__annotations__) == set(ALL_FIELDS)

    def test_track_stats_returns_mapping_with_those_keys(self):
        track = make_track([[0, 0, 10, 10], [10, 0, 20, 10]], [0, 1])
        stats = track_stats({1: track})
        assert set(stats.keys()) == set(ALL_FIELDS)

"""Tests for matching one ordered run of hashes against another."""

import numpy as np
import pytest

from dataeval.core import match_segments, sequence_containment, sequence_fingerprint


def diagonal(query_start: int, offset: int, length: int, gaps: tuple[int, ...] = ()):
    """Matched pairs along one diagonal, optionally skipping some query positions."""
    return np.array(
        [[query_start + i, query_start + i + offset] for i in range(length) if i not in gaps],
        dtype=np.intp,
    )


def zeros(pairs):
    return np.zeros(len(pairs), dtype=np.intp)


@pytest.mark.required
class TestSequenceFingerprint:
    def test_identical_sequences_share_a_digest(self):
        hashes = ["ff00ff00ff00ff00", "0123456789abcdef", "deadbeefcafef00d"]
        assert sequence_fingerprint(hashes)["exact"] == sequence_fingerprint(list(hashes))["exact"]

    def test_order_changes_the_digest(self):
        """The same frames in a different order are not the same video."""
        hashes = ["ff00ff00ff00ff00", "0123456789abcdef"]
        assert sequence_fingerprint(hashes)["exact"] != sequence_fingerprint(hashes[::-1])["exact"]

    def test_one_changed_frame_changes_the_digest(self):
        base = ["ff00ff00ff00ff00", "0123456789abcdef"]
        changed = ["ff00ff00ff00ff01", "0123456789abcdef"]
        assert sequence_fingerprint(base)["exact"] != sequence_fingerprint(changed)["exact"]

    def test_a_prefix_is_not_the_whole(self):
        hashes = ["ff00ff00ff00ff00", "0123456789abcdef"]
        assert sequence_fingerprint(hashes[:1])["exact"] != sequence_fingerprint(hashes)["exact"]

    def test_codes_and_validity_come_from_the_hashes(self):
        result = sequence_fingerprint(["ff00ff00ff00ff00", "", "0123456789abcdef"])
        assert result["codes"].shape == (3, 1)
        assert result["valid"].tolist() == [True, False, True]

    def test_frame_indices_default_to_position(self):
        assert sequence_fingerprint(["ab" * 8] * 3)["frame_indices"].tolist() == [0, 1, 2]

    def test_frame_indices_carry_a_sampled_selection(self):
        result = sequence_fingerprint(["ab" * 8] * 3, frame_indices=[0, 5, 10])
        assert result["frame_indices"].tolist() == [0, 5, 10]

    def test_times_are_all_or_nothing(self):
        """A partly populated timing makes every derived duration wrong for the frames without it."""
        assert sequence_fingerprint(["ab" * 8] * 3, times=[0.0, 0.1, 0.2])["times"] is not None
        assert sequence_fingerprint(["ab" * 8] * 3, times=[0.0, None, 0.2])["times"] is None

    def test_no_times_given(self):
        assert sequence_fingerprint(["ab" * 8] * 2)["times"] is None

    def test_empty_sequence(self):
        result = sequence_fingerprint([])
        assert result["codes"].shape[0] == 0
        assert isinstance(result["exact"], str)

    @pytest.mark.parametrize("field", ["frame_indices", "times"])
    def test_mismatched_lengths_rejected(self, field):
        with pytest.raises(ValueError, match="must agree"):
            sequence_fingerprint(["ab" * 8] * 3, **{field: [0, 1]})


@pytest.mark.required
class TestMatchSegments:
    def test_an_excerpt_is_one_segment_on_one_offset(self):
        pairs = diagonal(0, 20, 10)
        segments = match_segments(pairs, zeros(pairs), min_length=5)
        assert segments["query_start"].tolist() == [0]
        assert segments["query_end"].tolist() == [9]
        assert segments["candidate_start"].tolist() == [20]
        assert segments["candidate_end"].tolist() == [29]
        assert segments["offset"].tolist() == [20]
        assert segments["density"].tolist() == [1.0]

    def test_unrelated_content_scatters_and_yields_nothing(self):
        rng = np.random.default_rng(0)
        pairs = np.stack([rng.integers(0, 400, 200), rng.integers(0, 400, 200)], axis=1).astype(np.intp)
        assert match_segments(pairs, zeros(pairs), min_length=10)["offset"].size == 0

    def test_two_overlapping_clips_at_different_offsets(self):
        pairs = np.concatenate([diagonal(0, 5, 12), diagonal(40, 100, 12)])
        segments = match_segments(pairs, zeros(pairs), min_length=6)
        assert sorted(segments["offset"].tolist()) == [5, 100]
        assert sorted(segments["query_start"].tolist()) == [0, 40]

    @pytest.mark.parametrize(("gaps", "max_gap", "expected"), [((), 0, 1), ((5,), 0, 2), ((5,), 1, 1), ((5, 6), 1, 2)])
    def test_max_gap_bridges_dropped_frames_but_not_cuts(self, gaps, max_gap, expected):
        pairs = diagonal(0, 3, 14, gaps)
        segments = match_segments(pairs, zeros(pairs), min_length=2, max_gap=max_gap)
        assert len(segments["offset"]) == expected

    def test_min_length_filters_short_coincidences(self):
        """Shared intros and title cards are what a low min_length finds."""
        pairs = np.concatenate([diagonal(0, 7, 3), diagonal(50, 90, 20)])
        assert len(match_segments(pairs, zeros(pairs), min_length=2)["offset"]) == 2
        assert match_segments(pairs, zeros(pairs), min_length=10)["query_start"].tolist() == [50]

    def test_density_reports_how_much_of_the_span_matched(self):
        pairs = diagonal(0, 0, 10, gaps=(2, 5))
        segments = match_segments(pairs, zeros(pairs), min_length=5, max_gap=2)
        assert segments["n_matched"].tolist() == [8]
        assert segments["density"].tolist() == pytest.approx([0.8])

    def test_mean_distance_is_over_the_segments_own_pairs(self):
        pairs = diagonal(0, 4, 4)
        segments = match_segments(pairs, np.array([0, 2, 4, 6], dtype=np.intp), min_length=2)
        assert segments["mean_distance"].tolist() == pytest.approx([3.0])

    def test_offset_tolerance_joins_a_drifting_diagonal(self):
        """A slight frame-rate difference spreads one shared stretch across neighbouring offsets."""
        pairs = np.concatenate([diagonal(0, 10, 8), diagonal(8, 11, 8), diagonal(16, 12, 8)])
        strict = match_segments(pairs, zeros(pairs), min_length=6)
        joined = match_segments(pairs, zeros(pairs), min_length=6, offset_tolerance=1)
        assert len(strict["offset"]) == 3
        assert len(joined["offset"]) == 1
        assert joined["query_start"].tolist() == [0]
        assert joined["query_end"].tolist() == [23]
        assert joined["n_matched"].tolist() == [24]

    def test_offset_tolerance_does_not_join_distant_diagonals(self):
        pairs = np.concatenate([diagonal(0, 10, 8), diagonal(8, 40, 8)])
        assert len(match_segments(pairs, zeros(pairs), min_length=6, offset_tolerance=1)["offset"]) == 2

    def test_pair_order_does_not_matter(self):
        pairs = diagonal(0, 20, 10)
        shuffled = pairs[np.random.default_rng(1).permutation(len(pairs))]
        a = match_segments(pairs, zeros(pairs), min_length=5)
        b = match_segments(shuffled, zeros(shuffled), min_length=5)
        assert a["query_start"].tolist() == b["query_start"].tolist()
        assert a["offset"].tolist() == b["offset"].tolist()

    def test_no_pairs(self):
        segments = match_segments(np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.intp), min_length=5)
        assert segments["offset"].size == 0
        assert segments["mean_distance"].size == 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"min_length": 0}, "at least 1"),
            ({"min_length": 2, "max_gap": -1}, "non-negative"),
            ({"min_length": 2, "offset_tolerance": -1}, "non-negative"),
        ],
    )
    def test_bad_arguments_rejected(self, kwargs, match):
        pairs = diagonal(0, 0, 4)
        with pytest.raises(ValueError, match=match):
            match_segments(pairs, zeros(pairs), **kwargs)

    def test_shape_mismatches_rejected(self):
        with pytest.raises(ValueError, match=r"\(M, 2\)"):
            match_segments(np.zeros((4, 3), dtype=np.intp), np.zeros(4, dtype=np.intp), min_length=2)
        with pytest.raises(ValueError, match="must agree"):
            match_segments(diagonal(0, 0, 4), np.zeros(3, dtype=np.intp), min_length=2)


@pytest.mark.required
class TestSequenceContainment:
    def test_a_clip_inside_a_source_is_asymmetric(self):
        """The asymmetry is the leakage signal a symmetric score cannot express."""
        pairs = diagonal(0, 100, 10)
        query, candidate = sequence_containment(pairs, n_query=10, n_candidate=1000)
        assert query == 1.0
        assert candidate == pytest.approx(0.01)

    def test_a_re_encode_is_symmetric_and_high(self):
        pairs = diagonal(0, 0, 50)
        assert sequence_containment(pairs, 50, 50) == (1.0, 1.0)

    def test_partial_overlap(self):
        pairs = diagonal(0, 0, 25)
        assert sequence_containment(pairs, 50, 100) == (0.5, 0.25)

    def test_frames_are_counted_once_however_often_they_matched(self):
        pairs = np.array([[0, 5], [0, 6], [0, 7], [1, 8]], dtype=np.intp)
        assert sequence_containment(pairs, 10, 20) == (pytest.approx(0.2), pytest.approx(0.2))

    def test_no_pairs(self):
        assert sequence_containment(np.empty((0, 2), dtype=np.intp), 10, 10) == (0.0, 0.0)

    @pytest.mark.parametrize(("n_query", "n_candidate"), [(0, 5), (5, 0), (-1, 5)])
    def test_non_positive_counts_rejected(self, n_query, n_candidate):
        with pytest.raises(ValueError, match="must be positive"):
            sequence_containment(diagonal(0, 0, 2), n_query, n_candidate)

    def test_shape_rejected(self):
        with pytest.raises(ValueError, match=r"\(M, 2\)"):
            sequence_containment(np.zeros((3, 4), dtype=np.intp), 5, 5)

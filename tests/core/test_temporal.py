"""Tests for temporal redundancy in an ordered run of hashes."""

from collections.abc import Sequence

import numpy as np
import pytest

from dataeval.core import pack_hashes, redundant_runs


def codes_from(values: Sequence[int | None], bits: int = 64):
    """Pack a list of integer digests, where None marks a hash that was never computed."""
    width = bits // 4
    return pack_hashes(["" if value is None else f"{value:0{width}x}" for value in values])


def brute_runs(values: Sequence[int | None], radius: int, min_length: int):
    """Independent reference: walk the sequence and grow runs by hand."""
    runs, start = [], None
    for index in range(len(values) - 1):
        left, right = values[index], values[index + 1]
        linked = left is not None and right is not None and bin(left ^ right).count("1") <= radius
        if linked and start is None:
            start = index
        elif not linked and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return [(a, b) for a, b in runs if b - a + 1 >= min_length]


@pytest.mark.required
class TestRedundantRuns:
    def test_a_static_stretch_is_one_run(self):
        codes, valid = codes_from([1, 1, 1, 1, 999, 12345])
        runs = redundant_runs(codes, radius=0, valid=valid)
        assert runs["start"].tolist() == [0]
        assert runs["end"].tolist() == [3]
        assert runs["length"].tolist() == [4]
        assert runs["redundant_fraction"] == pytest.approx(3 / 6)

    def test_nothing_redundant(self):
        codes, valid = codes_from([1, 1 << 20, 1 << 40, 1 << 60])
        runs = redundant_runs(codes, radius=0, valid=valid)
        assert runs["start"].size == 0
        assert runs["redundant_fraction"] == 0.0

    def test_everything_redundant(self):
        codes, valid = codes_from([7] * 10)
        runs = redundant_runs(codes, radius=0, valid=valid)
        assert runs["length"].tolist() == [10]
        assert runs["redundant_fraction"] == pytest.approx(9 / 10)

    def test_two_runs_separated_by_a_change(self):
        codes, valid = codes_from([1, 1, 1, 999, 2, 2, 2])
        runs = redundant_runs(codes, radius=0, valid=valid)
        assert list(zip(runs["start"].tolist(), runs["end"].tolist(), strict=True)) == [(0, 2), (4, 6)]
        assert runs["redundant_fraction"] == pytest.approx(4 / 7)

    def test_radius_widens_what_counts_as_unchanged(self):
        codes, valid = codes_from([0b0, 0b1, 0b11, 0b111])
        assert redundant_runs(codes, radius=0, valid=valid)["start"].size == 0
        assert redundant_runs(codes, radius=1, valid=valid)["length"].tolist() == [4]

    def test_a_run_is_transitive_and_its_ends_can_be_far_apart(self):
        """Nothing along the way carried new information, which is the intended reading."""
        codes, valid = codes_from([0b0, 0b1, 0b11, 0b111, 0b1111])
        runs = redundant_runs(codes, radius=1, valid=valid)
        assert runs["length"].tolist() == [5]

    def test_mean_distance_is_over_the_runs_own_links(self):
        codes, valid = codes_from([0b0, 0b1, 0b1, 0b11])
        runs = redundant_runs(codes, radius=1, valid=valid)
        assert runs["mean_distance"].tolist() == pytest.approx([(1 + 0 + 1) / 3])

    def test_invalid_positions_break_a_run(self):
        """A hash that was never computed is not evidence that nothing changed."""
        codes, valid = codes_from([1, 1, None, 1, 1])
        runs = redundant_runs(codes, radius=0, valid=valid)
        assert list(zip(runs["start"].tolist(), runs["end"].tolist(), strict=True)) == [(0, 1), (3, 4)]

    def test_without_valid_an_absent_hash_is_just_a_code(self):
        codes, _ = codes_from([1, 1, None, 1, 1])
        assert redundant_runs(codes, radius=0)["start"].size == 2

    @pytest.mark.parametrize("min_length", [2, 3, 5])
    def test_min_length_filters_and_lowers_the_fraction(self, min_length):
        values = [1, 1, 999, 2, 2, 2, 2, 3]
        codes, valid = codes_from(values)
        runs = redundant_runs(codes, radius=0, valid=valid, min_length=min_length)
        assert list(zip(runs["start"].tolist(), runs["end"].tolist(), strict=True)) == brute_runs(values, 0, min_length)
        assert runs["redundant_fraction"] == pytest.approx(sum(runs["length"] - 1) / len(values))

    @pytest.mark.parametrize("radius", [0, 1, 4, 12])
    @pytest.mark.parametrize("seed", range(6))
    def test_agrees_with_a_hand_walked_reference(self, radius, seed):
        rng = np.random.default_rng(seed * 10 + radius)
        values: list[int | None] = []
        for _ in range(60):
            if values and rng.random() < 0.5 and values[-1] is not None:
                drifted = values[-1]
                for bit in rng.choice(64, size=int(rng.integers(0, 6)), replace=False):
                    drifted ^= 1 << int(bit)
                values.append(drifted)
            elif rng.random() < 0.1:
                values.append(None)
            else:
                values.append(int(rng.integers(0, 1 << 62)))
        codes, valid = codes_from(values)
        runs = redundant_runs(codes, radius=radius, valid=valid)
        assert list(zip(runs["start"].tolist(), runs["end"].tolist(), strict=True)) == brute_runs(values, radius, 2)

    @pytest.mark.parametrize("count", [0, 1])
    def test_degenerate_inputs(self, count):
        codes, valid = codes_from([1] * count)
        runs = redundant_runs(codes, radius=0, valid=valid)
        assert runs["start"].size == 0
        assert runs["redundant_fraction"] == 0.0

    @pytest.mark.parametrize("bits", [64, 128, 256])
    def test_wider_digests(self, bits):
        codes, valid = codes_from([5, 5, 5, 1 << (bits - 2)], bits=bits)
        assert redundant_runs(codes, radius=0, valid=valid)["length"].tolist() == [3]

    def test_negative_radius_rejected(self):
        codes, _ = codes_from([1, 1])
        with pytest.raises(ValueError, match="non-negative"):
            redundant_runs(codes, radius=-1)

    def test_min_length_below_two_rejected(self):
        codes, _ = codes_from([1, 1])
        with pytest.raises(ValueError, match="at least 2"):
            redundant_runs(codes, radius=0, min_length=1)

    def test_mismatched_valid_rejected(self):
        codes, _ = codes_from([1, 1, 1])
        with pytest.raises(ValueError, match="must agree"):
            redundant_runs(codes, radius=0, valid=np.array([True, True]))

    def test_order_is_the_whole_meaning(self):
        """Shuffling the input changes the answer; it is a time series, not a bag."""
        codes, valid = codes_from([1, 1, 1, 999, 2, 2, 2])
        shuffled = np.array([0, 3, 1, 4, 2, 5, 6])
        ordered = redundant_runs(codes, radius=0, valid=valid)
        scrambled = redundant_runs(codes[shuffled], radius=0, valid=valid[shuffled])
        assert ordered["length"].tolist() == [3, 3]
        assert ordered["redundant_fraction"] > scrambled["redundant_fraction"]

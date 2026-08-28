"""Tests for aligning one run of frames against another under uneven timing."""

import numpy as np
import pytest

from dataeval.core import align_subsequence, pack_hashes

INF = float("inf")


def reference(query, candidate, band=None, offset=0):
    """A plain-Python subsequence DTW, written for clarity rather than speed."""
    n, m = len(query), len(candidate)
    cost = [[abs(float(query[i]) - float(candidate[j])) for j in range(m)] for i in range(n)]

    def allowed(row: int, column: int) -> bool:
        return band is None or abs((column - offset) - row) <= band

    table = [[INF] * m for _ in range(n)]
    for column in range(m):
        if allowed(0, column):
            table[0][column] = cost[0][column]
    for row in range(1, n):
        for column in range(m):
            if not allowed(row, column):
                continue
            best = table[row - 1][column] if allowed(row - 1, column) else INF
            if column:
                if allowed(row - 1, column - 1):
                    best = min(best, table[row - 1][column - 1])
                if allowed(row, column - 1):
                    best = min(best, table[row][column - 1])
            if best < INF:
                table[row][column] = cost[row][column] + best
    return min(table[n - 1])


def codes(*hashes):
    """Pack hex digests the way the hamming metric consumes them."""
    return pack_hashes(list(hashes))[0]


@pytest.mark.required
class TestExactMatch:
    def test_a_clip_is_found_where_it_sits(self):
        clip = np.arange(10, dtype=np.float64)
        source = np.concatenate([np.full(20, -1.0), clip, np.full(20, -1.0)])
        found = align_subsequence(clip, source, metric="euclidean")
        assert (found["start"], found["end"], found["cost"]) == (20, 29, 0.0)

    def test_a_run_aligned_against_itself_costs_nothing(self):
        run = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        found = align_subsequence(run, run, metric="euclidean")
        assert found["cost"] == 0.0
        assert (found["start"], found["end"], found["path_length"]) == (0, 5, 6)

    def test_a_single_frame_query_finds_its_best_frame(self):
        found = align_subsequence(np.array([7.0]), np.array([1.0, 9.0, 7.0, 2.0]), metric="euclidean")
        assert (found["start"], found["end"], found["cost"]) == (2, 2, 0.0)

    def test_a_single_frame_candidate_collapses_the_query_onto_it(self):
        found = align_subsequence(np.array([1.0, 2.0, 3.0]), np.array([2.0]), metric="euclidean")
        assert (found["start"], found["end"]) == (0, 0)
        assert found["path_length"] == 3


@pytest.mark.required
class TestWarping:
    def test_half_speed_playback_aligns_at_no_cost(self):
        """The relation a fixed offset cannot express: the diagonal is sloped, not shifted."""
        clip = np.arange(10, dtype=np.float64)
        slowed = np.concatenate([np.full(20, -1.0), np.repeat(clip, 2), np.full(20, -1.0)])
        found = align_subsequence(clip, slowed, metric="euclidean")
        assert found["cost"] == 0.0
        assert found["path_length"] == 18

    def test_double_speed_playback_aligns_at_no_cost(self):
        clip = np.repeat(np.arange(10, dtype=np.float64), 2)
        faster = np.concatenate([np.full(5, -1.0), np.arange(10, dtype=np.float64), np.full(5, -1.0)])
        found = align_subsequence(clip, faster, metric="euclidean")
        assert found["cost"] == 0.0
        assert (found["start"], found["end"]) == (5, 14)

    def test_a_pause_partway_through_is_absorbed(self):
        clip = np.arange(8, dtype=np.float64)
        held = np.concatenate([np.arange(4, dtype=np.float64), np.full(6, 3.0), np.arange(4, 8, dtype=np.float64)])
        found = align_subsequence(clip, held, metric="euclidean")
        assert found["cost"] == 0.0
        assert found["path_length"] == 14

    def test_path_length_is_never_shorter_than_the_query(self):
        rng = np.random.default_rng(3)
        query, candidate = rng.normal(size=12), rng.normal(size=30)
        found = align_subsequence(query, candidate, metric="euclidean")
        assert found["path_length"] >= len(query)


@pytest.mark.required
class TestUnrelatedContent:
    def test_an_unrelated_run_costs_a_lot_per_step(self):
        clip = np.arange(10, dtype=np.float64)
        found = align_subsequence(clip, np.full(40, 99.0), metric="euclidean")
        assert found["normalized_cost"] > 90

    def test_normalized_cost_is_the_comparable_figure(self):
        """A longer alignment accumulates more cost simply by being longer."""
        clip = np.zeros(20)
        short = align_subsequence(clip[:5], np.ones(5), metric="euclidean")
        long = align_subsequence(clip, np.ones(20), metric="euclidean")
        assert long["cost"] > short["cost"]
        assert long["normalized_cost"] == short["normalized_cost"] == 1.0


@pytest.mark.required
class TestBand:
    def test_a_band_confines_the_alignment_to_its_diagonal(self):
        clip = np.arange(10, dtype=np.float64)
        source = np.concatenate([np.full(20, -1.0), clip, np.full(20, -1.0)])
        found = align_subsequence(clip, source, metric="euclidean", band=3, offset=20)
        assert (found["start"], found["end"], found["cost"]) == (20, 29, 0.0)

    def test_the_wrong_offset_finds_nothing_worth_reporting(self):
        clip = np.arange(10, dtype=np.float64)
        source = np.concatenate([np.full(20, -1.0), clip, np.full(20, -1.0)])
        found = align_subsequence(clip, source, metric="euclidean", band=3, offset=0)
        assert found["normalized_cost"] > 0

    def test_a_zero_band_is_a_straight_diagonal(self):
        query = np.arange(5, dtype=np.float64)
        candidate = np.arange(20, dtype=np.float64)
        found = align_subsequence(query, candidate, metric="euclidean", band=0, offset=0)
        assert (found["start"], found["end"], found["cost"], found["path_length"]) == (0, 4, 0.0, 5)

    def test_a_band_rejects_a_warp_it_cannot_reach(self):
        """Half-speed needs the path to drift by five frames; a band of one forbids it."""
        clip = np.arange(10, dtype=np.float64)
        slowed = np.repeat(clip, 2)
        tight = align_subsequence(clip, slowed, metric="euclidean", band=1, offset=0)
        loose = align_subsequence(clip, slowed, metric="euclidean", band=9, offset=0)
        assert loose["cost"] == 0.0
        assert tight["cost"] > 0.0

    def test_a_band_whose_diagonal_leaves_the_candidate_is_refused(self):
        with pytest.raises(ValueError, match="no path reaches the end"):
            align_subsequence(np.arange(5.0), np.arange(5.0), metric="euclidean", band=0, offset=50)

    def test_a_band_wider_than_the_candidate_matches_the_unbanded_answer(self):
        rng = np.random.default_rng(11)
        query, candidate = rng.normal(size=8), rng.normal(size=15)
        wide = align_subsequence(query, candidate, metric="euclidean", band=100, offset=0)
        free = align_subsequence(query, candidate, metric="euclidean")
        assert wide["cost"] == pytest.approx(free["cost"])


@pytest.mark.required
class TestAgainstAReference:
    @pytest.mark.parametrize(("band", "offset"), [(None, 0), (0, 0), (1, 0), (2, 3), (3, -2)])
    def test_cost_matches_a_plain_python_implementation(self, band, offset):
        rng = np.random.default_rng(0)
        for _ in range(25):
            query = rng.integers(0, 5, int(rng.integers(1, 12))).astype(np.float64)
            candidate = rng.integers(0, 5, int(rng.integers(2, 20))).astype(np.float64)
            expected = reference(query, candidate, band, offset)
            if expected == INF:
                with pytest.raises(ValueError, match="no path reaches the end"):
                    align_subsequence(query, candidate, metric="euclidean", band=band, offset=offset)
                continue
            found = align_subsequence(query, candidate, metric="euclidean", band=band, offset=offset)
            assert found["cost"] == pytest.approx(expected)

    def test_the_reported_window_is_one_the_alignment_really_uses(self):
        """Aligning the query against just the reported window must cost the same."""
        rng = np.random.default_rng(5)
        for _ in range(25):
            query = rng.integers(0, 5, int(rng.integers(2, 10))).astype(np.float64)
            candidate = rng.integers(0, 5, int(rng.integers(4, 20))).astype(np.float64)
            found = align_subsequence(query, candidate, metric="euclidean")
            window = candidate[found["start"] : found["end"] + 1]
            assert reference(query, window) == pytest.approx(found["cost"])


@pytest.mark.required
class TestMetrics:
    def test_hamming_consumes_packed_hashes(self):
        query = codes("0000000000000001", "0000000000000002", "0000000000000004")
        candidate = codes("ffffffffffffffff", *("0000000000000001", "0000000000000002", "0000000000000004"))
        found = align_subsequence(query, candidate)
        assert (found["start"], found["end"], found["cost"]) == (1, 3, 0.0)

    def test_hamming_counts_differing_bits(self):
        found = align_subsequence(codes("0000000000000000"), codes("000000000000000f"))
        assert found["cost"] == 4.0

    def test_cosine_ignores_magnitude(self):
        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        candidate = np.array([[5.0, 0.0], [0.0, 9.0]])
        assert align_subsequence(query, candidate, metric="cosine")["cost"] == pytest.approx(0.0)

    def test_cosine_calls_a_zero_vector_maximally_distant(self):
        """A zero vector has no direction, so no angle to anything -- not a perfect match."""
        found = align_subsequence(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]]), metric="cosine")
        assert found["cost"] == 1.0

    def test_hamming_refuses_float_descriptors(self):
        with pytest.raises(ValueError, match="packed uint64 codes"):
            align_subsequence(np.zeros((3, 2)), np.zeros((4, 2)))


@pytest.mark.required
class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"metric": "manhattan"}, "metric must be one of"),
            ({"metric": "euclidean", "band": -1}, "band must be non-negative"),
        ],
    )
    def test_bad_policy_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            align_subsequence(np.arange(3.0), np.arange(5.0), **kwargs)

    @pytest.mark.parametrize("name", ["query", "candidate"])
    def test_an_empty_run_is_rejected(self, name):
        runs = {"query": np.arange(3.0), "candidate": np.arange(5.0)}
        runs[name] = np.empty(0)
        with pytest.raises(ValueError, match=f"{name} must hold at least one frame"):
            align_subsequence(runs["query"], runs["candidate"], metric="euclidean")

    def test_runs_described_differently_are_rejected(self):
        with pytest.raises(ValueError, match="describe frames the same way"):
            align_subsequence(np.zeros((3, 2)), np.zeros((4, 5)), metric="euclidean")

    def test_a_three_dimensional_run_is_rejected(self):
        with pytest.raises(ValueError, match="must be 1- or 2-D"):
            align_subsequence(np.zeros((3, 2, 2)), np.zeros((4, 2)), metric="euclidean")

    def test_an_oversized_matrix_is_refused_and_names_the_way_through(self):
        query, candidate = np.zeros(5000), np.zeros(5000)
        with pytest.raises(ValueError, match="Pass a band"):
            align_subsequence(query, candidate, metric="euclidean", max_cells=1000)

    def test_a_band_bounds_the_work_the_refusal_was_counting(self):
        query, candidate = np.zeros(5000), np.zeros(5000)
        found = align_subsequence(query, candidate, metric="euclidean", band=2, max_cells=100_000)
        assert found["cost"] == 0.0

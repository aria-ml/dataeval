import logging
from unittest.mock import patch

import numpy as np
import pytest

from dataeval.core._bin import (
    CONTINUOUS_MIN_SAMPLE_SIZE,
    _bin_by_clusters,
    _gcd_ratio,
    bin_data,
    digitize_data,
    is_continuous,
)
from dataeval.exceptions import ShapeMismatchError


@pytest.mark.required
class TestDigitizeDataUnit:
    def test_nbins_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = digitize_data(factors, bincounts)
        assert type(hist) is np.ndarray

    def test_bin_edges_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [-np.inf, 1, np.inf]
        hist = digitize_data(factors, bin_edges)
        assert type(hist) is np.ndarray

    def test_crashes_with_negative_nbins(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = -10
        with pytest.raises(ValueError, match="bins"):
            digitize_data(factors, bincounts)

    def test_crashes_with_wrong_order(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [np.inf, 1, 2]
        with pytest.raises(ValueError, match="monotonically"):
            digitize_data(factors, bin_edges)

    def test_mixed_type(self):
        factors = [1, "a", 4.0]
        bins = 3
        err_msg = "Encountered a data value with non-numeric type when digitizing a factor."
        with pytest.raises(TypeError) as e:
            digitize_data(factors, bins)
        assert err_msg in str(e.value)


@pytest.mark.required
class TestMissingValueBinning:
    """NaN is missing data, so it gets a bin of its own rather than joining an observed one."""

    @pytest.mark.parametrize("method", ["uniform_width", "uniform_count", "clusters"])
    def test_bin_data_gives_nan_its_own_bin(self, method):
        rng = np.random.default_rng(0)
        data = rng.normal(size=200)
        data[5] = data[7] = np.nan

        binned = bin_data(data, method)
        missing = np.isnan(data)
        assert np.unique(binned[missing]).size == 1
        assert not set(binned[missing]) & set(binned[~missing])

    def test_bin_data_edges_ignore_missing(self):
        """Edges are placed on the observed values, so a NaN does not shift them."""
        rng = np.random.default_rng(0)
        clean = rng.normal(size=200)
        with_missing = clean.copy()
        with_missing[5] = np.nan

        observed = ~np.isnan(with_missing)
        assert np.array_equal(
            bin_data(with_missing, "uniform_width")[observed], bin_data(clean, "uniform_width")[observed]
        )

    def test_bin_data_separates_nan_from_infinity(self):
        """An infinity is an observed extreme and belongs in an end bin, not the missing bin."""
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(size=100), [np.inf, -np.inf, np.nan]])

        binned = bin_data(data, "uniform_width")
        pos_inf, neg_inf, nan = binned[-3], binned[-2], binned[-1]
        assert neg_inf == binned[:100].min()  # absorbed by the -inf outer edge
        assert nan not in (pos_inf, neg_inf)

    def test_bin_data_all_missing(self):
        """With nothing observed there are no edges to place, so every entry is missing."""
        assert np.array_equal(bin_data(np.full(30, np.nan), "uniform_width"), np.zeros(30))

    def test_digitize_data_gives_nan_its_own_bin(self):
        data = np.array([0.1, 1.1, np.nan, 1.2, 0.5])
        for bins in (2, [-np.inf, 1.0, np.inf]):
            binned = digitize_data(data, bins)
            assert binned[2] not in np.delete(binned, 2)

    def test_is_continuous_ignores_missing(self):
        """A continuous sample stays continuous when a few values go missing."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=200)
        assert is_continuous(data) is True

        data[5] = data[7] = np.nan
        assert is_continuous(data) is True

    def test_is_continuous_counts_only_observed(self):
        """The 20-observation floor counts values, not placeholders for absent ones."""
        rng = np.random.default_rng(0)
        data = np.full(100, np.nan)
        data[:18] = rng.normal(size=18)
        assert is_continuous(data) is False


@pytest.mark.optional
class TestDigitizeDataFunctional:
    def test_udb_regression_nbins(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = digitize_data(factors, bincounts)
        assert np.all(hist == [1, 2, 2])

    def test_udb_regression_bin_edges(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [-np.inf, 1, np.inf]
        hist = digitize_data(factors, bin_edges)
        assert np.all(hist == [1, 2, 2])

    def test_udb_regression_flipped_bin_edges(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [np.inf, 1, -np.inf]
        hist = digitize_data(factors, bin_edges)
        assert np.all(hist == [2, 1, 1])

    def test_narrow_bin_edges(self):
        factors = [0.1, 1.1, 1.5]
        bin_edges = [-10, 1, 1.2]
        hist = digitize_data(factors, bin_edges)
        assert np.all(hist == [1, 2, 3])


@pytest.mark.optional
class TestBinDataFunctional:
    @pytest.mark.parametrize(
        ("method", "data", "expected_result"),
        [
            ("uniform_width", np.array([0, 4, 8, 5, 6, 15] * 300), 6),
            ("uniform_width", np.concatenate([np.arange(2), np.arange(140, 1500)]), 10),
            # ("uniform_count", np.array([0, 4, 3, 5, 6, 8] * 10 + [5] * 30), 6), # BROKEN IN NUMPY 2.1+
            ("uniform_count", np.array([0, 4, 8, 5, 6, 15] * 10 + [5] * 30), 6),
            ("clusters", np.array([0, 4, 8, 5, 6, 15] * 300), 5),
        ],
    )
    def test_binning_method(self, method, data, expected_result):
        output = bin_data(data, method)
        unq, vals = np.unique(output, return_inverse=True)
        print(unq)
        print(data[:20])
        print(vals[:20])
        assert np.unique(output).size == expected_result


@pytest.mark.required
class TestIsContinuousFunctional:
    @pytest.mark.parametrize(
        ("data", "groups"),
        [
            # Every row its own group: nothing to collapse.
            (np.array([0, 4, 3, 5, 6, 8] * 15), np.arange(15 * 6)),
            # Factor varies within each group, so the rows are kept at full length.
            (np.array([0, 1, 9, 4, 3, 5, 2, 7, 8] * 10), np.array([0, 4, 3, 5, 6, 8] * 15)),
            # Factor constant within group: collapses to 20 integers, still a lattice.
            (
                np.concatenate([np.repeat(val, 3) for val in range(20)]),
                np.repeat(np.arange(20), 3),
            ),
            (
                np.concatenate(
                    [
                        np.repeat(val, 3)
                        for val in [0, 5, 13, 18, 2, 14, 1, 19, 16, 7, 15, 17, 4, 9, 10, 8, 12, 6, 11, 3]
                    ],
                ),
                np.repeat(np.arange(20), 3),
            ),
        ],
    )
    def test_is_continuous_repeats(self, data, groups):
        output = is_continuous(data, groups)
        assert output is not True

    def test_is_continuous_no_groups(self):
        data = np.array([0, 4, 3, 5, 6, 8] * 15)
        output = is_continuous(data)
        assert output is not True

    def test_is_coninuous_warning(self, caplog):
        # Six values, each replicated across a group of 15 rows, collapse under the floor.
        data = np.repeat([0, 4, 3, 5, 6, 8], 15)
        groups = np.repeat(np.arange(6), 15)
        warn_msg = f"All samples look discrete with so few data points (6 < {CONTINUOUS_MIN_SAMPLE_SIZE})"
        with caplog.at_level(logging.WARNING):
            output = is_continuous(data, groups)
        assert warn_msg in caplog.text
        # The count that tripped the floor is the post-grouping one, so say so.
        assert "after grouping" in caplog.text
        assert output is not True


@pytest.mark.required
class TestIsContinuousUnit:
    def test_small_sample_size_returns_false(self):
        """Test that small samples (< CONTINUOUS_MIN_SAMPLE_SIZE) return False."""
        small_data = np.array([1, 2, 3, 4, 5])  # < 20 points
        result = is_continuous(small_data)
        assert result is False

    def test_fewer_than_three_unique_values_returns_false(self):
        """Test that data with < 3 unique values returns False."""
        # Two unique values, enough samples
        data = np.array([1, 1, 1, 2, 2, 2] * 5)  # 30 points, 2 unique
        result = is_continuous(data)
        assert result is False

        # Single unique value
        data_single = np.array([5] * 25)  # 25 points, 1 unique
        result_single = is_continuous(data_single)
        assert result_single is False

    @patch("dataeval.core._bin.wasserstein_distance")
    def test_continuous_data_returns_true(self, mock_wd):
        """Test that continuous data (low Wasserstein distance) returns True."""
        mock_wd.return_value = 0.03  # < DISCRETE_MIN_WD
        continuous_data = np.random.normal(0, 1, 50)  # 50 continuous points
        result = is_continuous(continuous_data)
        assert result is True
        assert mock_wd.called

    @patch("dataeval.core._bin.wasserstein_distance")
    def test_discrete_data_returns_false(self, mock_wd):
        """Test that discrete data (high Wasserstein distance) returns False."""
        mock_wd.return_value = 0.08  # > DISCRETE_MIN_WD
        discrete_data = np.array([1, 2, 3, 4, 5] * 10)  # 50 discrete points
        result = is_continuous(discrete_data)
        assert result is False
        assert mock_wd.called

    def test_groups_scores_one_entry_per_group(self):
        """The Wasserstein test sees one entry per group, not one per row."""
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(30), 4)
        data = rng.normal(size=30)[groups]  # 120 rows, 30 distinct values

        with patch("dataeval.core._bin.wasserstein_distance", return_value=0.03) as mock_wd:
            assert is_continuous(data, groups) is True
        # dx holds n - 2 near-neighbor samples, so 28 for the collapsed sample and
        # 118 had the repeats been counted.
        assert mock_wd.call_args.args[0].size == 28

    def test_groups_collapses_replicated_factor(self):
        """A factor constant within each group is scored once per group, not once per row."""
        rng = np.random.default_rng(0)
        per_group = rng.normal(size=40)
        groups = np.repeat(np.arange(40), 3)
        replicated = per_group[groups]

        # The replicated rows are two-thirds exact duplicates, which reads as discrete.
        assert is_continuous(replicated) is False
        # Collapsed to one value per group, the same factor is continuous.
        assert is_continuous(replicated, groups) is True

    def test_groups_preserves_within_group_variation(self):
        """A factor that varies within a group is left at full length."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=120)
        groups = np.repeat(np.arange(40), 3)

        assert is_continuous(data, groups) == is_continuous(data)

    def test_groups_tolerates_missing_values(self):
        """A missing value is still constant within its group, so grouping still applies."""
        rng = np.random.default_rng(0)
        per_group = rng.normal(size=40)
        per_group[5] = np.nan
        groups = np.repeat(np.arange(40), 3)
        replicated = per_group[groups]

        # NaN != NaN, so an exact equality check would fall back to all 120 rows here.
        assert is_continuous(replicated, groups) is True

    def test_groups_logs_when_replication_check_fails(self, caplog):
        """Falling back to the ungrouped sample is recorded, since the verdicts differ."""
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(40), 3)
        data = rng.normal(size=40)[groups]
        data[1] += 1e-9  # one entry out of place

        with caplog.at_level(logging.DEBUG, logger="dataeval.core"):
            assert is_continuous(data, groups) is False
        assert "not constant within every group" in caplog.text

    def test_groups_length_mismatch_raises(self):
        """A groups array that does not align with the data is an error, not a no-op."""
        data = np.arange(30, dtype=np.float64)
        with pytest.raises(ShapeMismatchError, match="groups length 5 does not match data length 30"):
            is_continuous(data, np.arange(5))

    def test_duplicate_values_handling(self):
        """Test that duplicate values are handled correctly in NNN calculation."""
        # Data with some duplicates
        data = np.array([1, 1, 2, 3, 3, 4, 5, 6, 7, 8] * 3)  # 30 points
        with patch("dataeval.core._bin.wasserstein_distance", return_value=0.03):
            result = is_continuous(data)
            assert isinstance(result, bool)

    def test_sorted_data_processing(self):
        """Test that unsorted data is processed correctly."""
        unsorted_data = np.array([5, 1, 9, 3, 7, 2, 8, 4, 6] * 3)  # 27 points
        with patch("dataeval.core._bin.wasserstein_distance", return_value=0.03):
            result = is_continuous(unsorted_data)
            assert isinstance(result, bool)


@pytest.mark.required
class TestBinByClustersUnit:
    @patch("dataeval.core._clusterer.cluster")
    def test_basic_clustering_and_binning(self, mock_cluster):
        """Test basic clustering and bin edge creation."""
        # Mock cluster result
        mock_cluster_result = {
            "clusters": np.array([0, 0, 1, 1, 2, 2, -1, -1]),  # 3 clusters + outliers
            "k_neighbors": np.array(
                [
                    [1, 2, 3],
                    [0, 2, 3],
                    [1, 3, 4],
                    [2, 4, 5],  # non-outliers
                    [5, 6, 7],
                    [4, 6, 7],
                    [0, 1, 4],
                    [1, 5, 6],  # outliers (indices 6, 7)
                ],
            ),
        }
        mock_cluster.return_value = mock_cluster_result

        data = np.array([1.0, 1.1, 5.0, 5.2, 10.0, 10.1, 15.0, 15.5])
        result = _bin_by_clusters(data)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) >= 3  # At least 3 clusters + 1
        assert np.all(result[:-1] <= result[1:])  # Should be sorted
        mock_cluster.assert_called_once_with(data)

    @patch("dataeval.core._clusterer.cluster")
    def test_outlier_handling_with_sufficient_count(self, mock_cluster):
        """Test outlier handling when there are >= 4 outliers with same neighbor."""
        mock_cluster_result = {
            "clusters": np.array([0, 0, 1, 1, -1, -1, -1, -1]),
            "k_neighbors": np.array(  # 4 outliers
                [
                    [1, 2, 3],
                    [0, 2, 3],
                    [1, 3, 4],
                    [2, 4, 5],  # non-outliers
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],  # all outliers point to same neighbor
                ],
            ),
        }
        mock_cluster.return_value = mock_cluster_result

        data = np.array([1.0, 1.1, 5.0, 5.2, 0.1, 0.2, 0.3, 0.4])  # outliers are smaller
        result = _bin_by_clusters(data)

        assert isinstance(result, np.ndarray)
        assert len(result) > 2  # Should have extended bins
        mock_cluster.assert_called_once_with(data)

    @patch("dataeval.core._clusterer.cluster")
    def test_outlier_handling_with_insufficient_count(self, mock_cluster):
        """Test outlier handling when there are < 4 outliers with same neighbor."""
        mock_cluster_result = {
            "clusters": np.array([0, 0, 1, 1, -1, -1]),  # 2 outliers
            "k_neighbors": np.array(
                [
                    [1, 2, 3],
                    [0, 2, 3],
                    [1, 3, 4],
                    [2, 4, 5],  # non-outliers
                    [0, 1, 2],
                    [0, 1, 2],  # outliers point to same neighbor
                ],
            ),
        }
        mock_cluster.return_value = mock_cluster_result

        data = np.array([1.0, 1.1, 5.0, 5.2, 0.5, 0.6])  # outliers smaller than neighbor
        result = _bin_by_clusters(data)

        assert isinstance(result, np.ndarray)
        assert np.all(result[:-1] <= result[1:])  # Should be sorted
        mock_cluster.assert_called_once_with(data)

    @patch("dataeval.core._clusterer.cluster")
    def test_no_outliers(self, mock_cluster):
        """Test behavior when there are no outliers."""
        mock_cluster_result = {
            "clusters": np.array([0, 0, 1, 1, 2, 2]),  # No -1 values
            "k_neighbors": np.array([[1, 2, 3], [0, 2, 3], [1, 3, 4], [2, 4, 5], [3, 4, 5], [4, 5, 0]]),
        }
        mock_cluster.return_value = mock_cluster_result

        data = np.array([1.0, 1.1, 5.0, 5.2, 10.0, 10.1])
        result = _bin_by_clusters(data)

        assert isinstance(result, np.ndarray)
        assert len(result) == 4  # 3 clusters + 1 end bin
        mock_cluster.assert_called_once_with(data)

    @patch("dataeval.core._clusterer.cluster")
    def test_outliers_with_no_valid_neighbors(self, mock_cluster):
        """Test outliers that have no non-outlier neighbors."""
        mock_cluster_result = {
            "clusters": np.array([0, 0, -1, -1]),  # 2 outliers
            "k_neighbors": np.array(
                [
                    [1, 2, 3],
                    [0, 2, 3],  # non-outliers
                    [2, 3, 0],
                    [2, 3, 1],  # outliers pointing to other outliers and non-outliers
                ],
            ),
        }
        mock_cluster.return_value = mock_cluster_result

        data = np.array([1.0, 1.1, 15.0, 15.5])
        result = _bin_by_clusters(data)

        assert isinstance(result, np.ndarray)
        assert np.all(result[:-1] <= result[1:])  # Should be sorted
        mock_cluster.assert_called_once_with(data)


"""Unit tests for _gcd_ratio."""


class TestGcdRatioEdgeCases:
    """Edge cases and degenerate inputs."""

    def test_fewer_than_three_unique_values_returns_zero(self) -> None:
        assert _gcd_ratio(np.array([1.0, 1.0, 1.0])) == 0.0
        assert _gcd_ratio(np.array([1.0, 2.0])) == 0.0
        assert _gcd_ratio(np.array([5.0])) == 0.0

    def test_empty_array_returns_zero(self) -> None:
        assert _gcd_ratio(np.array([], dtype=np.float64)) == 0.0

    def test_all_identical_values_returns_zero(self) -> None:
        assert _gcd_ratio(np.full(100, 42.0)) == 0.0

    def test_two_unique_among_many_returns_zero(self) -> None:
        data = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        assert _gcd_ratio(data) == 0.0

    def test_all_gaps_below_tolerance_returns_zero(self) -> None:
        # Three "unique" values within 1e-12 of each other
        data = np.array([0.0, 1e-12, 2e-12])
        assert _gcd_ratio(data, tol=1e-9) == 0.0


class TestGcdRatioPerfectLattice:
    """Data on a perfect integer or regular grid should score 1.0."""

    def test_consecutive_integers(self) -> None:
        data = np.arange(10, dtype=np.float64)
        assert _gcd_ratio(data) == 1.0

    def test_even_integers(self) -> None:
        data = np.arange(0, 20, 2, dtype=np.float64)
        assert _gcd_ratio(data) == 1.0

    def test_half_integer_grid(self) -> None:
        data = np.arange(0, 5, 0.5)
        assert _gcd_ratio(data) == 1.0

    def test_sparse_subset_of_integer_grid(self) -> None:
        # {0, 3, 7, 10} — gaps are 3, 4, 3; min gap is 3; 4/3 ≈ 1.33 is not near-integer
        # so only 2 of 3 gaps qualify → ratio should be 2/3
        data = np.array([0.0, 3.0, 7.0, 10.0])
        result = _gcd_ratio(data)
        assert 0.6 < result < 0.7

    def test_sparse_multiples_of_base(self) -> None:
        # {0, 3, 6, 12} — gaps are 3, 3, 6; min gap is 3; all ratios (1, 1, 2) are integer
        data = np.array([0.0, 3.0, 6.0, 12.0])
        assert _gcd_ratio(data) == 1.0

    def test_large_integer_lattice(self) -> None:
        data = np.arange(0, 10000, 7, dtype=np.float64)
        assert _gcd_ratio(data) == 1.0

    def test_negative_integers(self) -> None:
        data = np.array([-10.0, -7.0, -4.0, -1.0, 2.0, 5.0])
        assert _gcd_ratio(data) == 1.0

    def test_duplicates_on_lattice_still_score_one(self) -> None:
        data = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
        assert _gcd_ratio(data) == 1.0


class TestGcdRatioContinuousData:
    """Continuous data should score well below 1.0."""

    def test_uniform_random_scores_low(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 200)
        assert _gcd_ratio(data) < 0.25

    def test_normal_random_scores_low(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        assert _gcd_ratio(data) < 0.25

    def test_exponential_random_scores_low(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.exponential(1, 200)
        assert _gcd_ratio(data) < 0.25

    def test_continuous_scores_consistently_low_across_seeds(self) -> None:
        for seed in range(10):
            rng = np.random.default_rng(seed)
            data = rng.uniform(0, 100, 200)
            assert _gcd_ratio(data) < 0.30, f"Failed on seed {seed}"


class TestGcdRatioDiscreteNonInteger:
    """Discrete data on non-integer but regular grids."""

    def test_multiples_of_pi(self) -> None:
        data = np.array([np.pi * k for k in range(10)])
        assert _gcd_ratio(data) == 1.0

    def test_multiples_of_third(self) -> None:
        # Gaps are all 1/3, but floating-point repr means we test tolerance
        data = np.arange(0, 5, 1 / 3)
        assert _gcd_ratio(data) > 0.90

    def test_irregular_discrete_support(self) -> None:
        # {0, 1, 3} — gaps are 1 and 2, ratio 2.0 is near-integer → scores 1.0
        data = np.array([0.0, 1.0, 3.0])
        assert _gcd_ratio(data) == 1.0

    def test_irregular_support_non_lattice(self) -> None:
        # {0, 1, sqrt(2)} — gap ratio ≈ 1.414, not near any integer
        data = np.array([0.0, 1.0, np.sqrt(2)])
        assert _gcd_ratio(data) < 0.85


class TestGcdRatioReturnType:
    """Return value properties."""

    def test_returns_float(self) -> None:
        data = np.arange(5, dtype=np.float64)
        result = _gcd_ratio(data)
        assert isinstance(result, float)

    def test_return_bounded_zero_one(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            data = rng.uniform(0, 100, 50)
            result = _gcd_ratio(data)
            assert 0.0 <= result <= 1.0


class TestGcdRatioInputOrdering:
    """Function should be invariant to input ordering."""

    def test_shuffled_lattice_same_as_sorted(self) -> None:
        rng = np.random.default_rng(42)
        data_sorted = np.arange(20, dtype=np.float64)
        data_shuffled = data_sorted.copy()
        rng.shuffle(data_shuffled)
        assert _gcd_ratio(data_sorted) == _gcd_ratio(data_shuffled)

    def test_shuffled_continuous_same_as_sorted(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data_shuffled = data.copy()
        rng.shuffle(data_shuffled)
        assert _gcd_ratio(data) == _gcd_ratio(data_shuffled)


class TestGcdRatioTolerance:
    """The tol parameter controls what counts as a zero gap."""

    def test_custom_tolerance_excludes_small_gaps(self) -> None:
        # Values: 0, 3e-7, 0.5, 1.0
        # Default tol (1e-9): gaps are [3e-7, ~0.5, 0.5]; min_gap = 3e-7;
        #   ratios ≈ [1, 1666666.7, 1666666.7] — the large ratios are NOT near-integer → low score
        # Large tol (1e-3): 3e-7 gap excluded; remaining gaps ≈ [0.5, 0.5]; min=0.5;
        #   ratio = [1, 1] — both near-integer → score = 1.0
        data = np.array([0.0, 3e-7, 0.5, 1.0])
        result_default = _gcd_ratio(data, tol=1e-9)
        result_large_tol = _gcd_ratio(data, tol=1e-3)
        assert result_default < 0.5
        assert result_large_tol == 1.0

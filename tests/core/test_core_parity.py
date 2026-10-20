"""Unit tests for dataeval.core._parity module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.stats import hypergeom

from dataeval.core._association import benjamini_hochberg
from dataeval.core._parity import parity
from dataeval.core._rao_scott import grouped_table
from dataeval.utils.scipy.stats import chi2_contingency, crosstab

pytestmark = pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")


def expand(table):
    """Raw rows reproducing a contingency table (level i, class j), so `parity` builds it back."""
    table = np.asarray(table)
    levels, classes = np.indices(table.shape)
    counts = table.ravel()
    rows = np.repeat(levels.ravel(), counts).astype(np.intp)
    return rows[:, None], np.repeat(classes.ravel(), counts).astype(np.intp)


def cell(cells, level, cls, factor=0):
    """Position of one (factor, level, class) cell in the flat `cells` arrays."""
    match = (cells["factor"] == factor) & (cells["level"] == level) & (cells["class_label"] == cls)
    (index,) = np.flatnonzero(match)
    return index


def _sequences(seed=0, count=40, length=25):
    """A sequence-level factor over `count` sequences, each showing one class drawn independently of it."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(count), length)
    return (groups % 2)[:, None].astype(np.intp), rng.integers(0, 3, count)[groups].astype(np.intp), groups


@pytest.mark.required
class TestParity:
    """Test cases for the parity function on independent rows."""

    def test_basic_functionality(self):
        """Test basic parity calculation with valid inputs."""
        binned_data = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)
        chi_scores = result["scores"]
        p_values = result["p_values"]

        assert isinstance(chi_scores, np.ndarray)
        assert isinstance(p_values, np.ndarray)
        assert chi_scores.shape == (2,)
        assert p_values.shape == (2,)
        assert chi_scores.dtype == np.float64
        assert p_values.dtype == np.float64

    def test_single_factor(self):
        """Test with single factor column."""
        binned_data = np.array([[0], [1], [0], [1]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert result["scores"].shape == (1,)
        assert result["p_values"].shape == (1,)

    def test_high_correlation(self):
        """Test with highly correlated factor and labels.

        Perfect association is the top of Cramér's V, so it reads exactly 1.0 whatever the
        sample size: Bergsma's correction subtracts the same term from phi-squared and from
        the dimensions it is divided by, and at phi-squared = 1 they cancel. Read off the
        G statistic instead this was 0.84, and off G with no continuity correction, 1.20 --
        above the maximum this statistic is documented to have.
        """
        data = [0] * 5 + [1] * 5
        binned_data = np.array([data], dtype=np.intp).T
        class_labels = np.array(data, dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert result["scores"][0] == pytest.approx(1.0)
        assert result["p_values"][0] < 0.05

    def test_no_correlation(self):
        """A factor whose levels hold the same class mix is independent of the labels."""
        binned_data = np.array([[0], [1], [0], [1]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert_array_almost_equal(result["scores"], [0.0])
        assert_array_almost_equal(result["p_values"], [1.0])

    def test_return_types(self):
        """Every key of the result, with the per-factor arrays one entry per factor."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        per_factor = (
            "scores",
            "chi_squared",
            "df_num",
            "df_den",
            "p_values",
            "p_values_adjusted",
            "design_effects",
            "effective_groups",
            "insufficient",
            "testable",
        )
        assert set(result) == {*per_factor, "cells", "test", "n_rows", "n_groups"}
        for key in per_factor:
            assert isinstance(result[key], np.ndarray)
            assert result[key].shape == (1,)
        assert (result["test"], result["n_rows"], result["n_groups"]) == ("pearson", 2, 2)
        assert set(result["cells"]) == {
            "factor",
            "level",
            "class_label",
            "observed",
            "expected",
            "deficit",
            "lift",
            "p_values",
            "p_values_adjusted",
            "sparse",
            "testable",
        }

    def test_insufficient_data_detection(self):
        """Test detection of tables with expected counts below Cochran's minimum."""
        binned_data = np.array([[0], [1], [0]], dtype=np.intp)
        class_labels = np.array([0, 1, 0], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert result["insufficient"][0]
        assert result["cells"]["sparse"].all()

    def test_multiple_factors(self):
        """Test with multiple factors."""
        binned_data = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.intp)
        class_labels = np.array([0, 1, 0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert result["scores"].shape == (3,)
        assert result["p_values"].shape == (3,)

    def test_zero_row_removal(self):
        """A factor with a single observed level does not raise."""
        binned_data = np.array([[0], [0], [0], [0]], dtype=np.intp)
        class_labels = np.array([0, 0, 1, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert len(result["scores"]) == 1
        assert len(result["p_values"]) == 1

    def test_single_sample_is_untestable(self):
        """A single sample holds one level and one class, so nothing can be tested."""
        result = parity(np.array([[0]], dtype=np.intp), np.array([0], dtype=np.intp), assume_independent=True)

        assert np.isnan(result["scores"][0])
        assert not result["testable"][0]

    def test_single_class(self):
        """With one class, every level has the same class mix by construction: untestable, not independent."""
        binned_data = np.array([[0], [1], [0], [1]], dtype=np.intp)
        class_labels = np.array([0, 0, 0, 0], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert np.isnan(result["scores"][0])
        assert np.isnan(result["p_values"][0])
        assert not result["testable"][0]

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        binned_data = np.array([], dtype=np.intp).reshape(0, 1)
        class_labels = np.array([], dtype=np.intp)

        with pytest.raises((ValueError, IndexError)):
            parity(binned_data, class_labels, assume_independent=True)

    def test_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1, 2], dtype=np.intp)  # Different length

        with pytest.raises((ValueError, IndexError)):
            parity(binned_data, class_labels, assume_independent=True)

    def test_mismatched_groups(self):
        with pytest.raises(ValueError, match="groups"):
            parity(np.zeros((4, 1), np.intp), np.arange(4) % 2, np.arange(3))

    def test_large_factor_values(self):
        """Test with large factor values.

        Note: Uses larger sample size (n=20) to avoid bias correction
        reducing the score to 0 in very small samples.
        """
        binned_data = np.array([[100] * 5 + [200] * 5], dtype=np.intp).T
        class_labels = np.array([0] * 5 + [1] * 5, dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert result["scores"][0] > 0
        assert 0 <= result["p_values"][0] <= 1
        assert set(result["cells"]["level"].tolist()) == {100, 200}

    def test_dtype_consistency(self):
        """Test that output arrays have correct dtypes."""
        binned_data = np.array([[0], [1]], dtype=np.intp)
        class_labels = np.array([0, 1], dtype=np.intp)

        result = parity(binned_data, class_labels, assume_independent=True)

        assert result["scores"].dtype == np.float64
        assert result["p_values"].dtype == np.float64
        assert result["insufficient"].dtype == np.bool_
        assert result["cells"]["observed"].dtype == np.intp

    def test_class_labels_as_list(self):
        """Test with class_labels as different sequence types."""
        binned_data = np.array([[0], [1]], dtype=np.intp)

        result1 = parity(binned_data, [0, 1], assume_independent=True)  # type: ignore
        result2 = parity(binned_data, (0, 1), assume_independent=True)  # type: ignore
        result3 = parity(binned_data, np.array([0, 1]), assume_independent=True)

        assert_array_almost_equal(result1["scores"], result2["scores"])
        assert_array_almost_equal(result2["scores"], result3["scores"])
        assert_array_almost_equal(result1["p_values"], result2["p_values"])
        assert_array_almost_equal(result2["p_values"], result3["p_values"])


@pytest.mark.required
class TestSamplingUnit:
    """D2: the caller must say what the independent unit is, and say it once."""

    def test_none_groups_without_declaration_raises(self):
        with pytest.raises(ValueError, match="Cannot establish independent sampling units"):
            parity(np.zeros((10, 1), int), np.arange(10) % 2)

    def test_groups_and_declaration_together_raise(self):
        with pytest.raises(ValueError, match="both"):
            parity(np.zeros((10, 1), int), np.arange(10) % 2, np.arange(10), assume_independent=True)

    def test_one_row_per_group_is_the_ungrouped_test(self):
        """Singleton groups are the rows being independent, so the exact Pearson path runs."""
        rng = np.random.default_rng(0)
        f, y = rng.integers(0, 3, (200, 1)), rng.integers(0, 2, 200)
        grouped = parity(f, y, rng.permutation(200))
        ungrouped = parity(f, y, assume_independent=True)
        assert grouped["test"] == "pearson"
        assert grouped["n_groups"] == 200
        assert grouped["p_values"][0] == ungrouped["p_values"][0]
        np.testing.assert_array_equal(grouped["cells"]["p_values"], ungrouped["cells"]["p_values"])


@pytest.mark.required
class TestUngrouped:
    def test_ungrouped_matches_scipy(self):
        rng = np.random.default_rng(1)
        f, y = rng.integers(0, 3, (300, 1)), rng.integers(0, 4, 300)
        table = crosstab(f[:, 0], y).count

        r = parity(f, y, assume_independent=True)
        ref = chi2_contingency(table, correction=False)
        assert r["chi_squared"][0] == pytest.approx(ref.statistic)
        assert r["p_values"][0] == pytest.approx(ref.pvalue)
        assert r["test"] == "pearson"
        assert r["design_effects"][0] == 1.0
        assert np.isinf(r["df_den"][0])
        assert r["df_num"][0] == 6
        assert r["effective_groups"][0] == 300

        # Each cell is the exact hypergeometric test of its 2x2 collapse.
        c = r["cells"]
        i = cell(c, level=2, cls=1)
        o, row, col = table[2, 1], table[2].sum(), table[:, 1].sum()
        dist = hypergeom(300, row, col)
        assert c["p_values"][i] == pytest.approx(min(1.0, 2 * min(dist.cdf(o), dist.sf(o - 1))))
        assert c["expected"][i] == pytest.approx(ref.expected_freq[2, 1])

    def test_cells_include_observed_zeros_and_deficit(self):
        """The reviewer's table: 0 of 30 expected is a deficit, whatever the factor's V."""
        r = parity(*expand([[0, 60], [1500, 1440]]), assume_independent=True)
        c = r["cells"]
        i = cell(c, level=0, cls=0)
        assert c["observed"][i] == 0
        assert c["expected"][i] == pytest.approx(30)
        assert c["deficit"][i] == pytest.approx(30)
        assert c["lift"][i] == 0.0
        assert c["p_values_adjusted"][i] < 1e-15
        assert r["scores"][0] < 0.3
        assert c["observed"].size == 4

    def test_untestable_factor_is_nan_not_independent(self):
        r = parity(np.zeros((50, 1), int), np.arange(50) % 2, assume_independent=True)
        assert np.isnan(r["scores"][0])
        assert np.isnan(r["p_values"][0])
        assert not r["testable"][0]

        single_class = parity(np.arange(50)[:, None] % 2, np.zeros(50, int), assume_independent=True)
        assert np.isnan(single_class["scores"][0])
        assert not single_class["testable"][0]
        assert not single_class["cells"]["testable"].any()

    def test_all_factors_untestable_returns_nan(self):
        r = parity(np.zeros((50, 3), int), np.arange(50) % 2, assume_independent=True)
        assert np.isnan(r["p_values_adjusted"]).all()
        assert np.isnan(r["cells"]["p_values_adjusted"]).all()

    def test_bh_runs_over_testable_factors_and_cells_only(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 120)
        f = np.column_stack([y ^ (rng.random(120) < 0.3), np.zeros(120, int), rng.integers(0, 3, 120)])
        r = parity(f, y, assume_independent=True)

        assert r["testable"].tolist() == [True, False, True]
        assert np.isnan(r["p_values_adjusted"][1])
        np.testing.assert_allclose(r["p_values_adjusted"][[0, 2]], benjamini_hochberg(r["p_values"][[0, 2]]))

        c = r["cells"]
        assert not c["testable"][c["factor"] == 1].any()
        assert np.isnan(c["p_values_adjusted"][~c["testable"]]).all()
        np.testing.assert_allclose(
            c["p_values_adjusted"][c["testable"]], benjamini_hochberg(c["p_values"][c["testable"]])
        )


@pytest.mark.required
class TestGrouped:
    def test_grouped_path_reports_rao_scott(self):
        f, y, groups = _sequences()
        r = parity(f, y, groups)
        table = grouped_table(f[:, 0], y, groups)

        assert r["test"] == "rao_scott_f"
        assert np.isfinite(r["df_den"][0])
        assert r["n_groups"] == 40
        assert r["n_rows"] == 1000
        assert r["chi_squared"][0] == pytest.approx(table.chi_squared / table.design_effect)
        assert r["p_values"][0] == pytest.approx(table.p_value)
        assert r["design_effects"][0] == pytest.approx(table.design_effect)
        assert r["effective_groups"][0] == pytest.approx(table.effective_groups)
        np.testing.assert_allclose(r["cells"]["p_values"], table.cell_p_values.ravel())
        np.testing.assert_array_equal(r["cells"]["sparse"], table.cell_sparse.ravel())

    def test_a_class_seen_in_one_group_has_untestable_cells(self):
        """The Korn-Graubard cell has no design df there, so the cell drops out of BH."""
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(30), 20)
        f = (groups % 2)[:, None]
        y = rng.integers(0, 2, groups.size)
        y[groups == 0] = 2  # class 2 only ever appears in group 0
        r = parity(f, y, groups)
        c = r["cells"]

        assert r["testable"][0]
        rare = c["class_label"] == 2
        assert not c["testable"][rare].any()
        assert c["testable"][~rare].all()
        assert np.isnan(c["p_values_adjusted"][rare]).all()
        np.testing.assert_allclose(c["p_values_adjusted"][~rare], benjamini_hochberg(c["p_values"][~rare]))

    def test_one_group_is_untestable(self):
        r = parity(np.arange(20)[:, None] % 2, np.arange(20) % 3, np.zeros(20, int))
        assert not r["testable"][0]
        assert np.isnan(r["p_values"][0])
        assert r["insufficient"][0]


@pytest.mark.required
class TestPearsonRatherThanG:
    """Verify Pearson chi-square behavior across table densities."""

    @staticmethod
    def _null_rejection_rate(r, k, n, trials=1500, seed=0):
        rng = np.random.default_rng(seed)
        rejected = 0
        for _ in range(trials):
            factor = rng.integers(0, r, n).astype(np.intp)[:, None]
            labels = rng.integers(0, k, n).astype(np.intp)
            rejected += float(parity(factor, labels, assume_independent=True)["p_values"][0]) <= 0.05
        return rejected / trials

    def test_the_test_holds_its_level_on_a_thin_table(self):
        """Verify test maintains nominal significance level on sparse tables."""
        assert self._null_rejection_rate(32, 3, 200) < 0.08

    def test_it_holds_its_level_on_a_dense_table_too(self):
        """Verify test maintains nominal significance level on dense tables."""
        assert 0.02 < self._null_rejection_rate(3, 3, 600) < 0.08

    def test_perfect_association_is_exactly_one(self):
        """Verify Cramér's V reaches 1.0 for perfect association."""
        for categories, n in [(2, 10), (3, 30), (5, 20)]:
            values = (np.arange(n) % categories).astype(np.intp)
            score = float(parity(values[:, None], values, assume_independent=True)["scores"][0])
            assert score == pytest.approx(1.0), f"{categories} categories at n={n}"

    def test_the_score_never_exceeds_one(self):
        """Verify Cramér's V stays bounded in [0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(8, 60))
            factor = rng.integers(0, 5, n).astype(np.intp)[:, None]
            labels = rng.integers(0, 5, n).astype(np.intp)
            score = float(parity(factor, labels, assume_independent=True)["scores"][0])
            assert np.isnan(score) or 0.0 <= score <= 1.0


@pytest.mark.required
class TestCochranCriterion:
    """Verify Cochran sufficiency criterion based on expected counts."""

    def test_a_table_the_observed_rule_could_not_see_is_flagged(self):
        """Verify tables with low expected counts are flagged even if observed counts are zero."""
        r = parity(*expand([[100, 0, 0], [0, 100, 0], [0, 0, 6]]), assume_independent=True)
        assert r["insufficient"][0], "Cochran is breached here and the old observed-count rule was silent"
        c = r["cells"]
        assert c["expected"][c["sparse"]].min() == pytest.approx(0.175, abs=1e-3)

    def test_a_table_with_small_observed_counts_but_ample_expected_is_not_flagged(self):
        """Verify tables with small observed counts but sufficient expected counts are not flagged."""
        r = parity(*expand([[47, 3], [3, 47]]), assume_independent=True)
        assert not r["insufficient"][0]
        assert not r["cells"]["sparse"].any()

    def test_a_single_expected_count_below_one_is_enough(self):
        """Verify table is flagged if any expected count is below 1.0."""
        assert parity(*expand([[235, 235, 24], [235, 235, 24], [5, 5, 2]]), assume_independent=True)["insufficient"][0]

    def test_a_fifth_of_cells_below_five_is_tolerated(self):
        """Verify table is not flagged when at most 20% of expected counts are below 5.0."""
        assert not parity(*expand([[40, 40, 40], [40, 40, 40]]), assume_independent=True)["insufficient"][0]

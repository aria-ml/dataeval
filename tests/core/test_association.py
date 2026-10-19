import numpy as np
import pytest
from scipy.stats import chi2_contingency

from dataeval.core._association import (
    benjamini_hochberg,
    bias_corrected_cramers_v,
    cochran_breach,
    cochran_breach_2x2,
)


def _reference_bergsma(chi_stat: float, n: float, r: int, k: int) -> float:
    """Scalar, branch-by-branch transcription of Bergsma (2013), used as an independent oracle."""
    if n <= 1:
        return 0.0
    phi2_corr = max(0.0, chi_stat / n - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    min_dim_corr = min(k_corr - 1, r_corr - 1)
    if min_dim_corr <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / min_dim_corr))


def _expected_2x2(n_a: int, n_b: int, n: int) -> np.ndarray:
    return np.array(
        [
            [n_a * n_b / n, n_a * (n - n_b) / n],
            [(n - n_a) * n_b / n, (n - n_a) * (n - n_b) / n],
        ],
    )


class TestCochranBreach:
    def test_empty_table_is_not_a_breach(self):
        assert cochran_breach(np.empty((0, 3))) is False

    def test_sufficient_table(self):
        assert cochran_breach(np.full((3, 4), 6.0)) is False

    def test_any_expected_count_below_one_is_a_breach(self):
        expected = np.full((3, 4), 50.0)
        expected[1, 2] = 0.9
        assert cochran_breach(expected) is True

    def test_twenty_percent_share_below_five_is_allowed_but_more_is_not(self):
        expected = np.full((2, 5), 20.0)  # 10 cells
        expected.flat[:2] = 3.0  # 20%: not more than 20%
        assert cochran_breach(expected) is False
        expected.flat[2] = 3.0  # 30%
        assert cochran_breach(expected) is True


class TestCochranBreach2x2:
    def test_matches_generic_criterion_on_every_small_table(self):
        for n in (1, 2, 5, 10, 20, 40):
            for n_a in range(n + 1):
                for n_b in range(n + 1):
                    expected = _expected_2x2(n_a, n_b, n)
                    assert bool(cochran_breach_2x2(n_a, n_b, n)) == cochran_breach(expected), (n_a, n_b, n)

    def test_reduces_to_any_expected_count_below_five(self):
        for n, n_a, n_b in [(100, 50, 40), (100, 4, 30), (100, 10, 10), (30, 10, 10), (30, 3, 29)]:
            assert bool(cochran_breach_2x2(n_a, n_b, n)) == bool((_expected_2x2(n_a, n_b, n) < 5).any())

    def test_vectorized_over_pairs(self):
        n_a = np.array([50, 4, 10])
        n_b = np.array([40, 30, 10])
        out = cochran_breach_2x2(n_a, n_b, 100)
        assert out.shape == (3,)
        assert out.tolist() == [bool(cochran_breach_2x2(a, b, 100)) for a, b in zip(n_a, n_b, strict=True)]

    def test_zero_margin_and_empty_samples_are_breaches(self):
        assert bool(cochran_breach_2x2(0, 10, 100))  # class never present
        assert bool(cochran_breach_2x2(100, 10, 100))  # class present in every sample
        assert bool(cochran_breach_2x2(0, 0, 0))  # no samples: undefined expected counts


class TestBiasCorrectedCramersV:
    def test_scalar_matches_reference_on_random_inputs(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            chi = float(rng.uniform(0, 200))
            n = float(rng.choice([0, 1, 2, 3, 10, 57, 1000]))
            r, k = (int(x) for x in rng.integers(1, 8, 2))
            assert float(bias_corrected_cramers_v(chi, n, r, k)) == pytest.approx(_reference_bergsma(chi, n, r, k))

    def test_array_input_equals_elementwise_scalars(self):
        rng = np.random.default_rng(1)
        chi = rng.uniform(0, 100, 50)
        n = rng.integers(0, 400, 50)
        out = bias_corrected_cramers_v(chi, n, 2, 2)
        assert out.shape == (50,)
        np.testing.assert_allclose(out, [_reference_bergsma(c, m, 2, 2) for c, m in zip(chi, n, strict=True)])

    def test_scalar_input_returns_a_scalar(self):
        assert np.ndim(bias_corrected_cramers_v(10.0, 100, 2, 2)) == 0

    def test_independence_and_perfect_association_on_2x2(self):
        assert float(bias_corrected_cramers_v(0.0, 100, 2, 2)) == 0.0
        assert float(bias_corrected_cramers_v(100.0, 100, 2, 2)) == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_too_few_samples_scores_zero(self, n):
        assert float(bias_corrected_cramers_v(5.0, n, 2, 2)) == 0.0

    def test_never_negative_on_real_tables(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            table = rng.integers(1, 30, (int(rng.integers(2, 5)), int(rng.integers(2, 5))))
            res = chi2_contingency(table, correction=False)
            assert float(bias_corrected_cramers_v(res.statistic, table.sum(), *table.shape)) >= 0.0  # type: ignore


def _reference_bh(p):
    """Textbook step-up adjustment: ``scipy.stats.false_discovery_control`` needs scipy 1.11, we support 1.10."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


class TestBenjaminiHochberg:
    @pytest.mark.parametrize("seed", range(5))
    def test_matches_reference(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0, 1, int(rng.integers(1, 200)))
        p[: len(p) // 4] = np.round(p[: len(p) // 4], 2)  # introduce ties
        np.testing.assert_allclose(benjamini_hochberg(p), _reference_bh(p))

    def test_n_tests_equals_padding_with_ones(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 0.2, 30)
        padded = np.concatenate([p, np.ones(70)])
        expected = _reference_bh(padded)[:30]
        np.testing.assert_allclose(benjamini_hochberg(p, n_tests=100), expected)

    def test_adjusted_values_are_bounded_and_order_preserving(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(0, 1, 100)
        adj = benjamini_hochberg(p, n_tests=250)
        assert np.all(adj >= p)
        assert np.all(adj <= 1.0)
        assert np.all(np.diff(adj[np.argsort(p, kind="stable")]) >= 0)

    def test_input_is_not_modified(self):
        p = np.array([0.04, 0.01, 0.03])
        before = p.copy()
        benjamini_hochberg(p)
        np.testing.assert_array_equal(p, before)

    def test_empty_input(self):
        assert benjamini_hochberg([]).shape == (0,)

    @pytest.mark.parametrize("bad", [[0.1, np.nan], [-0.1, 0.5], [0.5, 1.2]])
    def test_invalid_p_values_raise(self, bad):
        with pytest.raises(ValueError, match="within"):
            benjamini_hochberg(bad)

    def test_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            benjamini_hochberg(np.full((2, 2), 0.5))

    def test_n_tests_smaller_than_p_values_raises(self):
        with pytest.raises(ValueError, match="n_tests"):
            benjamini_hochberg([0.1, 0.2, 0.3], n_tests=2)

from unittest.mock import patch

import numpy as np
import pytest

from dataeval.utils.scipy.stats import (
    AndersonKsampResult,
    BwsTestResult,
    Chi2ContingencyResult,
    CramerVonMisesResult,
    CrosstabResult,
    KstestResult,
    MannwhitneyuResult,
    anderson_ksamp,
    bws_test,
    chi2_contingency,
    cramervonmises_2samp,
    crosstab,
    ks_2samp,
    mannwhitneyu,
)


class TestChi2Contingency:
    def test_returns_typed_result(self):
        table = np.array([[10, 10], [10, 10]])
        result = chi2_contingency(table, correction=False)

        assert isinstance(result, Chi2ContingencyResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.dof, int)
        assert isinstance(result.expected_freq, np.ndarray)
        assert result.expected_freq.dtype == np.float64

    def test_tuple_indexing_and_unpacking(self):
        table = np.array([[12, 5], [4, 15]])
        result = chi2_contingency(table, correction=False)

        assert result[0] == result.statistic
        assert result[1] == result.pvalue
        assert result[2] == result.dof
        assert np.array_equal(result[3], result.expected_freq)

        stat, pval, dof, expected = result
        assert stat == result.statistic
        assert pval == result.pvalue
        assert dof == result.dof
        assert np.array_equal(expected, result.expected_freq)

    def test_validation_type_checks(self):
        with (
            patch(
                "dataeval.utils.scipy.stats._scipy_chi2_contingency",
                return_value=("invalid", 0.05, 1, np.ones((2, 2))),
            ),
            pytest.raises(TypeError, match="Expected statistic to be numeric"),
        ):
            chi2_contingency([[1, 2], [3, 4]])

        with (
            patch(
                "dataeval.utils.scipy.stats._scipy_chi2_contingency",
                return_value=(1.0, "invalid", 1, np.ones((2, 2))),
            ),
            pytest.raises(TypeError, match="Expected pvalue to be numeric"),
        ):
            chi2_contingency([[1, 2], [3, 4]])

        with (
            patch(
                "dataeval.utils.scipy.stats._scipy_chi2_contingency",
                return_value=(1.0, 0.05, "invalid", np.ones((2, 2))),
            ),
            pytest.raises(TypeError, match="Expected dof to be an integer"),
        ):
            chi2_contingency([[1, 2], [3, 4]])


class TestCrosstab:
    def test_returns_typed_result(self):
        a = np.array([1, 1, 2, 2])
        b = np.array([0, 1, 0, 1])
        result = crosstab(a, b)

        assert isinstance(result, CrosstabResult)
        assert isinstance(result.elements, tuple)
        assert len(result.elements) == 2
        assert isinstance(result.count, np.ndarray)
        assert np.issubdtype(result.count.dtype, np.integer)
        assert result.count.shape == (2, 2)

    def test_tuple_indexing_and_unpacking(self):
        a = np.array([1, 2])
        b = np.array([0, 1])
        result = crosstab(a, b)

        assert result[0] == result.elements
        assert np.array_equal(result[1], result.count)

        elements, count = result
        assert elements == result.elements
        assert np.array_equal(count, result.count)

    def test_validation_type_checks(self):
        with (
            patch(
                "dataeval.utils.scipy.stats._scipy_crosstab",
                return_value=("not_a_tuple", np.ones((2, 2))),
            ),
            pytest.raises(TypeError, match="Expected elements to be sequence of arrays"),
        ):
            crosstab([1, 2], [3, 4])


class TestKs2Samp:
    def test_returns_typed_result(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.5, 2.5, 3.5, 4.5])
        result = ks_2samp(a, b)

        assert isinstance(result, KstestResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.statistic_location, float)
        assert isinstance(result.statistic_sign, int)

    def test_tuple_indexing_and_unpacking(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        result = ks_2samp(a, b)

        assert result[0] == result.statistic
        assert result[1] == result.pvalue
        assert result[2] == result.statistic_location
        assert result[3] == result.statistic_sign

        stat, pval, loc, sign = result
        assert stat == result.statistic
        assert pval == result.pvalue
        assert loc == result.statistic_location
        assert sign == result.statistic_sign

    def test_validation_type_checks(self):
        with (
            patch("dataeval.utils.scipy.stats._scipy_ks_2samp", return_value=("bad", 0.1, 1.0, 1)),
            pytest.raises(TypeError, match="Expected statistic to be numeric"),
        ):
            ks_2samp([1], [2])

        with (
            patch("dataeval.utils.scipy.stats._scipy_ks_2samp", return_value=(0.5, "bad", 1.0, 1)),
            pytest.raises(TypeError, match="Expected pvalue to be numeric"),
        ):
            ks_2samp([1], [2])

        with (
            patch("dataeval.utils.scipy.stats._scipy_ks_2samp", return_value=(0.5, 0.1, "bad", 1)),
            pytest.raises(TypeError, match="Expected statistic_location to be numeric"),
        ):
            ks_2samp([1], [2])

        with (
            patch("dataeval.utils.scipy.stats._scipy_ks_2samp", return_value=(0.5, 0.1, 1.0, "bad")),
            pytest.raises(TypeError, match="Expected statistic_sign to be an integer"),
        ):
            ks_2samp([1], [2])


class TestMannWhitneyU:
    def test_returns_typed_result(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])
        result = mannwhitneyu(a, b)

        assert isinstance(result, MannwhitneyuResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)

    def test_tuple_indexing_and_unpacking(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = mannwhitneyu(a, b)

        assert result[0] == result.statistic
        assert result[1] == result.pvalue

        stat, pval = result
        assert stat == result.statistic
        assert pval == result.pvalue

    def test_validation_type_checks(self):
        with (
            patch("dataeval.utils.scipy.stats._scipy_mannwhitneyu", return_value=("bad", 0.1)),
            pytest.raises(TypeError, match="Expected statistic to be numeric"),
        ):
            mannwhitneyu([1], [2])

        with (
            patch("dataeval.utils.scipy.stats._scipy_mannwhitneyu", return_value=(1.0, "bad")),
            pytest.raises(TypeError, match="Expected pvalue to be numeric"),
        ):
            mannwhitneyu([1], [2])


class TestOtherTwoSampleTests:
    def test_cramervonmises_2samp(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        result = cramervonmises_2samp(a, b)
        assert isinstance(result, CramerVonMisesResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)

    def test_bws_test(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        result = bws_test(a, b)
        assert isinstance(result, BwsTestResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)

    def test_anderson_ksamp(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        result = anderson_ksamp([a, b])
        assert isinstance(result, AndersonKsampResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)

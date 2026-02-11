"""Verify that bias analysis evaluators produce correct output types.

Maps to meta repo test cases:
  - TC-2.1: Bias analysis suite (Balance, Diversity, Parity)
"""

import polars as pl
import pytest

from verification.helpers import make_metadata


@pytest.mark.test_case("2-1")
class TestBiasEvaluation:
    """Verify Balance, Diversity, and Parity evaluators."""

    def test_balance_returns_output(self):
        from dataeval.bias import Balance, BalanceOutput

        metadata = make_metadata()
        result = Balance().evaluate(metadata)
        assert isinstance(result, BalanceOutput)

    def test_balance_output_has_dataframes(self):
        from dataeval.bias import Balance

        result = Balance().evaluate(make_metadata())
        assert isinstance(result.balance, pl.DataFrame)
        assert isinstance(result.factors, pl.DataFrame)
        assert isinstance(result.classwise, pl.DataFrame)

    def test_balance_classwise_has_imbalance_flag(self):
        from dataeval.bias import Balance

        result = Balance().evaluate(make_metadata())
        assert "is_imbalanced" in result.classwise.columns

    def test_diversity_returns_output(self):
        from dataeval.bias import Diversity, DiversityOutput

        result = Diversity().evaluate(make_metadata())
        assert isinstance(result, DiversityOutput)

    def test_diversity_output_has_dataframes(self):
        from dataeval.bias import Diversity

        result = Diversity().evaluate(make_metadata())
        assert isinstance(result.factors, pl.DataFrame)
        assert isinstance(result.classwise, pl.DataFrame)

    def test_diversity_factors_has_low_diversity_flag(self):
        from dataeval.bias import Diversity

        result = Diversity().evaluate(make_metadata())
        assert "is_low_diversity" in result.factors.columns

    def test_diversity_supports_shannon_method(self):
        from dataeval.bias import Diversity

        result = Diversity(method="shannon").evaluate(make_metadata())
        assert isinstance(result.factors, pl.DataFrame)

    def test_parity_returns_output(self):
        from dataeval.bias import Parity, ParityOutput

        result = Parity().evaluate(make_metadata())
        assert isinstance(result, ParityOutput)

    def test_parity_output_has_factors_and_insufficient_data(self):
        from dataeval.bias import Parity

        result = Parity().evaluate(make_metadata())
        assert isinstance(result.factors, pl.DataFrame)
        assert isinstance(result.insufficient_data, dict)

    def test_parity_factors_has_correlation_flag(self):
        from dataeval.bias import Parity

        result = Parity().evaluate(make_metadata())
        assert "is_correlated" in result.factors.columns

    def test_all_bias_outputs_support_meta(self):
        from dataeval.bias import Balance, Diversity, Parity

        metadata = make_metadata()
        for cls in (Balance, Diversity, Parity):
            result = cls().evaluate(metadata)
            meta = result.meta()
            assert meta is not None

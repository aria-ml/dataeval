from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
from numpy.random import Generator

from dataeval.metadata._ood import find_ood_predictors
from tests.metadata._shared import mock_metadata


def mock_ood_output(is_ood) -> MagicMock:
    """
    Creates a mock object with only is_ood as scores are not relevant to the calculations
    """

    m = MagicMock()
    m.is_ood = is_ood
    return m


@pytest.mark.required
class TestOODPredictors:
    def test_empty_ood(self):
        """Test that no outliers returns early with 0.0 mutual information for all factors"""

        fnames = ["a", "b", "c", "d"]
        m = mock_metadata(fnames, np.zeros((len(fnames), 1)))
        ood = mock_ood_output(np.array([False]))

        predictors = find_ood_predictors(m, ood)

        assert fnames == list(predictors.keys())
        assert all(v == 0.0 for v in predictors.values())

    def test_mismatched_data_and_mask(self, RNG: Generator):
        m = mock_metadata(["a"], discrete_data=RNG.random((5, 1)), continuous_names=[])
        ood = mock_ood_output(np.array([True]))

        error_msg = "ood and metadata must have the same length, got 1 and 5 respectively."

        with pytest.raises(ValueError, match=error_msg):
            find_ood_predictors(m, ood)

    @pytest.mark.parametrize("factors", [("a"), ("a", "b")])
    @pytest.mark.parametrize("discretes", [0, 5])
    @pytest.mark.parametrize("conts", (0, 5))
    def test_valid_inputs(self, RNG: Generator, factors, discretes, conts):
        factor_count = len(factors)
        m = mock_metadata(
            discrete_names=[f"d_{f}" for f in factors],
            discrete_data=RNG.integers(2, size=(discretes, factor_count)),
            continuous_names=[f"c_{f}" for f in factors],
            continuous_data=RNG.random((conts, factor_count)),
        )

        find_ood_predictors(m, MagicMock())

    @patch("dataeval.metadata._ood.mutual_info_classif")
    def test_mutual_info_inputs(self, mock: MagicMock, RNG: Generator):
        """
        Tests `scaled_data` and `discrete_features` are correctly calculated before calling :func:`mutual_info_classif`
        """

        ddata = RNG.random((10, 3))
        cdata = RNG.random((10, 2))

        m = mock_metadata(
            discrete_names=["d1", "d2", "d3"],
            discrete_data=ddata,
            continuous_names=["c1", "c2"],
            continuous_data=cdata,
        )

        flagged = np.array([True] * 5 + [False] * 5)
        ood = mock_ood_output(np.array(flagged))

        find_ood_predictors(m, ood)

        # Manually calculate expected values
        stacked_data = np.hstack([ddata, cdata])
        expected_scaled_data = (stacked_data - np.mean(stacked_data, axis=0)) / np.std(stacked_data, axis=0, ddof=1)
        is_discrete = np.array([True] * 3 + [False] * 2)

        expected_kwargs = {
            "X": expected_scaled_data,
            "y": ood.is_ood,
            "discrete_features": is_discrete,
            "random_state": 0,
        }

        # Manually check arguments as `assert_called_once_with` cannot handle multidimensional array equality
        args, kwargs = mock.call_args[0], mock.call_args[1]

        assert args == ()
        assert isinstance(kwargs, dict)

        for k, expected in expected_kwargs.items():
            npt.assert_array_equal(kwargs[k], expected)

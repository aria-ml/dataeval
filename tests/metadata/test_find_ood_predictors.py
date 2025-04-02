from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
from numpy.random import Generator

from dataeval.metadata._ood import _combine_discrete_continuous, find_ood_predictors


def mock_ood_output(is_ood) -> MagicMock:
    """
    Creates a mock object with only is_ood as scores are not relevant to the calculations
    """

    m = MagicMock()
    m.is_ood = is_ood
    return m


def mock_metadata(factor_names, discrete_data=None, cont_factor_names=None, continuous_data=None) -> MagicMock:
    """
    Creates a magic mock method that contains discrete and continuous data and factors
    but has no hard dependency on Metadata.
    """

    m = MagicMock()

    m.discrete_factor_names = factor_names
    m.continuous_factor_names = cont_factor_names if cont_factor_names is not None else []
    m.discrete_data = discrete_data if discrete_data is not None else np.array([])
    m.continuous_data = continuous_data if continuous_data is not None else np.array([])

    m.total_num_factors = len(m.discrete_factor_names) + len(m.continuous_factor_names)

    return m


@pytest.mark.required
class TestOODPredictors:
    def test_empty_ood(self):
        """Test that no outliers returns early with 0.0 mutual information for all factors"""

        fnames = ["a", "b", "c", "d"]
        m = mock_metadata(fnames, np.array([0.0]))
        ood = mock_ood_output(np.array([False]))

        predictors = find_ood_predictors(m, ood)

        assert fnames == list(predictors.keys())
        assert all(v == 0.0 for v in predictors.values())

    def test_mismatched_data_and_mask(self, RNG: Generator):
        m = mock_metadata(["a"], discrete_data=RNG.random((5, 1)), cont_factor_names=[])
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
            factor_names=factors,
            discrete_data=RNG.integers(2, size=(discretes, factor_count)),
            cont_factor_names=factors,
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
            factor_names=["d1", "d2", "d3"],
            discrete_data=ddata,
            cont_factor_names=["c1", "c2"],
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


class TestCombineData:
    @pytest.mark.parametrize("cnames, cshape", [(["continuous"], (3, 1)), ([], (0, 0))])
    @pytest.mark.parametrize("dnames, dshape", [(["discrete"], (3, 1)), ([], (0, 0))])
    def test_combine_data(self, dnames: list[str], dshape: tuple[int, int], cnames: list[str], cshape: tuple[int, int]):
        """Tests combinations of discrete, continuous, both, and neither"""

        m = mock_metadata(dnames, np.ones(dshape, dtype=np.int8), cnames, np.ones(cshape, dtype=np.int8))
        names, data = _combine_discrete_continuous(m)

        expected_names = []
        expected_names.extend(dnames)
        expected_names.extend(cnames)

        # Empty numpy object has shape of (0,) rather than (0,0)
        expected_shape = (0,) if max(dshape + cshape) == 0 else (max(dshape[0], cshape[0]), dshape[1] + cshape[1])

        assert names == expected_names
        assert data.shape == expected_shape

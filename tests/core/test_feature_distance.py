from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
from numpy.random import Generator
from numpy.typing import NDArray

from dataeval.core._feature_distance import _calculate_drift, feature_distance
from dataeval.data._metadata import FactorInfo, Metadata


def mock_metadata(
    discrete_names: list[str] | None = None,
    discrete_data: NDArray[Any] | None = None,
    continuous_names: list[str] | None = None,
    continuous_data: NDArray[Any] | None = None,
) -> Metadata:
    """
    Creates a magic mock method that contains discrete and continuous data and factors
    but has no hard dependency on Metadata.
    """

    m = MagicMock(spec=Metadata)

    _factors = {}
    _factor_data = []

    if discrete_names and discrete_data is not None and discrete_data.size > 0:
        _factor_data.append(discrete_data)
        _factors |= dict.fromkeys(discrete_names, FactorInfo("discrete"))

    if continuous_names and continuous_data is not None and continuous_data.size > 0:
        _factor_data.append(continuous_data)
        _factors |= dict.fromkeys(continuous_names, FactorInfo("continuous"))

    m.factor_names = list(_factors)
    m.factor_data = np.hstack(_factor_data) if _factor_data else np.array([])
    m.factor_info = _factors
    m.dataframe = pl.DataFrame(m.factor_data, schema=m.factor_names)

    m.filter_by_factor = lambda x: Metadata.filter_by_factor(m, x)
    m.calculate_distance = lambda x: Metadata.calculate_distance(m, x)

    return m


@pytest.mark.required
class TestFeatureDistance:
    """Tests valid combinations of input metadata run without errors"""

    @pytest.mark.parametrize(
        "factors, shape1, shape2",
        [
            pytest.param(["a"], (1, 1), (1, 1), id="1 sample, 1 factor"),
            pytest.param(["a"], (3, 1), (3, 1), id="multi samples, 1 factor"),
            pytest.param(["a", "b", "c"], (1, 3), (1, 3), id="1 sample, multi factors"),
            pytest.param(["a", "b", "c"], (3, 3), (3, 3), id="multi samples, multi factors"),
            pytest.param(["a", "b", "c"], (1, 3), (4, 3), id="mismatched samples, multi factors"),
        ],
    )
    def test_input_shapes(self, RNG: Generator, factors: list[str], shape1, shape2):
        m1 = mock_metadata(continuous_names=factors, continuous_data=RNG.random(shape1))
        m2 = mock_metadata(continuous_names=factors, continuous_data=RNG.random(shape2))

        with pytest.warns(UserWarning):
            result = m1.calculate_distance(m2)

        assert len(result) == len(factors)
        assert all(factor in result for factor in factors)
        assert isinstance(result[factors[0]]["statistic"], float)

    def test_no_warn_on_many_samples(self, RNG: Generator):
        """Solving the equation where N==M brings the sample count to 32 to make a valid solution"""

        c1 = RNG.random((32, 1))
        c2 = RNG.random((32, 1))

        import warnings

        with warnings.catch_warnings():
            feature_distance(c1, c2)

    def test_empty_inputs(self):
        """
        Test an empty array produces no value, but does not error

        While there seems to be a DivisionByZero, the np.atleast_2d forces the array to have an empty
        length of 1, rather than 0
        """

        c1 = np.array([])
        c2 = np.array([])

        assert feature_distance(c1, c2) == []

    def test_min_equals_max(self):
        """Test that any factors that have no deviation return an empty MetadataDistanceValues"""

        c1 = np.ones((32, 1))
        c2 = np.ones((32, 1))

        result = feature_distance(c1, c2)

        assert list(result[0].values()) == [0.0, 0.0, 0.0, 1.0]

    def test_inconsistent_features(self):
        """Test that value error is raised with inconsistent number of features"""
        c1 = np.ones((32, 1))
        c2 = np.ones((32, 2))

        with pytest.raises(ValueError):
            feature_distance(c1, c2)


@pytest.mark.required
class TestCalculateDrift:
    def test_valid_X(self):
        """When IQR is not zero, scale distance by IQR"""

        # IQR = 1.5, distance = 1.5
        res = _calculate_drift(np.arange(4), np.arange(4) + 1.5)

        assert isinstance(res, float)
        assert res == 1.0

    def test_min_eq_max(self):
        """When IQR is zero and min equals max, return distance"""

        # Array with identical values has an iqr of 0.0, return emd
        res = _calculate_drift([1.0, 1.0, 1.0], np.zeros(3))

        assert isinstance(res, float)
        assert np.isclose(res, 1.0)

    def test_min_neq_max(self):
        """When IQR is 0 and min does not equal max, return scaled distance"""
        res = _calculate_drift([0.0, 1.0, 1.0, 1.0, 2.0], np.ones(5))

        assert isinstance(res, float)
        assert np.isclose(res, 0.2)

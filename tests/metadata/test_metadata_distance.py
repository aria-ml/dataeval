import numpy as np
import pytest
from numpy.random import Generator

from dataeval.metadata import metadata_distance
from dataeval.metadata._distance import _calculate_drift
from dataeval.outputs import MetadataDistanceValues
from tests.metadata._shared import mock_metadata


@pytest.mark.required
class TestMetadataDistance:
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
            result = metadata_distance(m1, m2)

        assert len(result) == len(factors)
        assert all(factor in result for factor in factors)
        assert isinstance(result[factors[0]].statistic, float)

    def test_no_warn_on_many_samples(self, RNG: Generator):
        """Solving the equation where N==M brings the sample count to 32 to make a valid solution"""

        m1 = mock_metadata(continuous_names=["a"], continuous_data=RNG.random((32, 1)))
        m2 = mock_metadata(continuous_names=["a"], continuous_data=RNG.random((32, 1)))

        import warnings

        with warnings.catch_warnings():
            metadata_distance(m1, m2)

    def test_empty_inputs(self):
        """
        Test an empty array produces no value, but does not error

        While there seems to be a DivisionByZero, the np.atleast_2d forces the array to have an empty
        length of 1, rather than 0
        """

        m1 = mock_metadata(continuous_names=[], continuous_data=np.array([]))
        m2 = mock_metadata(continuous_names=[], continuous_data=np.array([]))

        assert metadata_distance(m1, m2) == {}

    def test_min_equals_max(self):
        """Test that any factors that have no deviation return an empty MetadataDistanceValues"""

        m1 = mock_metadata(continuous_names=["a"], continuous_data=np.ones((32, 1)))
        m2 = mock_metadata(continuous_names=["a"], continuous_data=np.ones((32, 1)))

        result = metadata_distance(m1, m2)

        assert result == {"a": MetadataDistanceValues(0.0, 0.0, 0.0, 1.0)}


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

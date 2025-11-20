import numpy as np
import pytest

from dataeval.core._completeness import completeness


@pytest.mark.required
class TestCompletenessUnit:
    def test_fails_with_too_many_quantiles(self):
        embs = np.zeros((100, 2))
        quantiles = 5
        with pytest.raises(ValueError):
            completeness(embs, quantiles)

    def test_fails_with_negative_quantiles(self):
        embs = np.zeros((100, 2))
        quantiles = -1
        with pytest.raises(ValueError):
            completeness(embs, quantiles)

    def test_p_too_large(self):
        embs = np.zeros((3, 11))
        with pytest.raises(ValueError):
            completeness(embs, 1)

    def test_high_dim_data_valueerror(self):
        """High dimensional data should raise valueerror"""
        embs = np.random.random(size=(100, 16, 16))
        with pytest.raises(ValueError):
            completeness(embs, 1)

    def test_zero_dim_data_valueerror(self):
        """Low dimensional data should raise valueerror"""
        embs = np.array([[]])
        with pytest.raises(ValueError):
            completeness(embs, 1)

    def test_completeness(self):
        embs = np.array([[1, 0], [0, 1], [1, 1]])
        result = completeness(embs, 1)
        np.testing.assert_array_equal(result[0], 0.75)
        np.testing.assert_array_equal(result[1], np.array([[0.5, 0.5]]))

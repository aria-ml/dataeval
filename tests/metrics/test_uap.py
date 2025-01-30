import numpy as np
import pytest

from dataeval.metrics.estimators._uap import uap


@pytest.mark.required
class TestUAP:
    def test_uap(self):
        labels = np.arange(10).repeat(100)
        scores = np.zeros((1000, 10), dtype=float)
        result = uap(labels, scores)
        assert result.uap > 0

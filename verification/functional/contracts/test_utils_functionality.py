"""Verify that utility components are available and functional.

Maps to meta repo test cases:
  - TC-12.1: Utility components (Data, Thresholds, Preprocessing, ONNX)
"""

import numpy as np
import pytest


@pytest.mark.test_case("12-1")
class TestUtilsFunctionality:
    """Verify utility components."""

    def test_utils_thresholds_zscore(self):
        from dataeval.utils.thresholds import ZScoreThreshold

        # ZScoreThreshold uses 'multiplier' as first positional arg
        threshold = ZScoreThreshold(3.0)
        assert threshold.upper_multiplier == 3.0

    def test_utils_preprocessing_conversion(self):
        # to_numpy is in dataeval.utils._internal
        from dataeval.utils._internal import to_numpy

        data = [np.array([1, 2]), np.array([3, 4])]
        result = to_numpy(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_utils_data_loaders_exist(self):
        from dataeval.utils import data

        assert data is not None

    def test_utils_onnx_available(self):
        from dataeval.utils import onnx

        assert onnx is not None

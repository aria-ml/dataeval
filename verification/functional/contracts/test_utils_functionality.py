"""Verify that utility components are available and functional.

Maps to meta repo test cases:
  - TC-12.1: Utility components (Data, Thresholds, Preprocessing, ONNX)
"""

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
        from dataeval.utils.preprocessing import to_int_box

        box = (1.2, 2.7, 3.4, 4.9)
        result = to_int_box(box)
        assert result == (1, 2, 4, 5)

    def test_data_operations_module_exists(self):
        from dataeval.utils import data

        assert data is not None

    def test_utils_onnx_available(self):
        from dataeval.utils import onnx

        assert onnx is not None

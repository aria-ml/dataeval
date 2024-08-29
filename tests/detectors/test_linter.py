import numpy as np
import pytest

from dataeval._internal.detectors.linter import Linter, _get_outlier_mask
from dataeval.flags import ImageStat


class TestLinter:
    def test_linter(self):
        linter = Linter()
        results = linter.evaluate(np.random.random((1000, 3, 16, 16)))
        assert results is not None

    def test_linter_custom(self):
        linter = Linter(ImageStat.ENTROPY)
        results = linter.evaluate(np.random.random((1000, 3, 16, 16)))
        assert results is not None

    def test_linter_value_error(self):
        with pytest.raises(ValueError):
            Linter(ImageStat.XXHASH)

    @pytest.mark.parametrize("method", ["zscore", "modzscore", "iqr"])
    def test_get_outlier_mask(self, method):
        mask_value = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, 2.5)
        mask_none = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, None)
        np.testing.assert_array_equal(mask_value, mask_none)

    def test_get_outlier_mask_valueerror(self):
        with pytest.raises(ValueError):
            _get_outlier_mask(np.zeros([0]), "error", None)  # type: ignore

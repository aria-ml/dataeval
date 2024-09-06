import numpy as np
import pytest

from dataeval._internal.detectors.linter import Linter, _get_outlier_mask
from dataeval._internal.metrics.stats import imagestats
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

    def test_get_outliers_with_extra_stats(self):
        linter = Linter()
        dataset = np.random.random((100, 3, 16, 16)) / 5
        dataset[0] = 1
        linter.stats = imagestats(dataset, ImageStat.ALL_HASHES | ImageStat.MEAN)
        assert len(linter.stats.dict()) == 3
        results = linter._get_outliers()
        assert len(results) == 1
        assert len(results[0]) == 1
        assert "mean" in results[0]

    def test_linter_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        stats = imagestats(data, ImageStat.MEAN)
        linter = Linter(ImageStat.MEAN)
        results = linter.evaluate(stats)
        assert results is not None

    def test_linter_with_stats_no_mean(self):
        data = np.random.random((20, 3, 16, 16))
        stats = imagestats(data, ImageStat.VAR)
        linter = Linter(ImageStat.MEAN)
        with pytest.raises(ValueError):
            linter.evaluate(stats)

    def test_linter_with_stats_mean_plus(self):
        data = np.random.random((20, 3, 16, 16))
        stats = imagestats(data, ImageStat.MEAN | ImageStat.VAR)
        linter = Linter(ImageStat.MEAN)
        results = linter.evaluate(stats)
        assert results is not None

    def test_linter_with_stats_no_mean_plus(self):
        data = np.random.random((20, 3, 16, 16))
        stats = imagestats(data, ImageStat.MEAN)
        linter = Linter(ImageStat.MEAN | ImageStat.VAR)
        with pytest.raises(ValueError):
            linter.evaluate(stats)

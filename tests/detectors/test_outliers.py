import numpy as np
import pytest

from dataeval.detectors.linters.outliers import Outliers, _get_outlier_mask
from dataeval.metrics.stats import DatasetStatsOutput, dimensionstats, pixelstats, visualstats


class TestOutliers:
    def test_outliers(self):
        outliers = Outliers()
        results = outliers.evaluate(np.random.random((100, 3, 16, 16)))
        assert outliers.stats.dimensionstats is not None
        assert outliers.stats.pixelstats is not None
        assert outliers.stats.visualstats is not None
        assert results is not None

    @pytest.mark.parametrize("method", ["zscore", "modzscore", "iqr"])
    def test_get_outlier_mask(self, method):
        mask_value = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, 2.5)
        mask_none = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, None)
        np.testing.assert_array_equal(mask_value, mask_none)

    def test_get_outlier_mask_valueerror(self):
        with pytest.raises(ValueError):
            _get_outlier_mask(np.zeros([0]), "error", None)  # type: ignore

    def test_outliers_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        stats = pixelstats(data)
        outliers = Outliers()
        results = outliers.from_stats(stats)
        assert results is not None

    def test_outliers_with_multiple_stats(self):
        dataset1 = np.zeros((50, 3, 16, 16))
        dataset2 = np.zeros((50, 3, 16, 16))
        dataset2[0] = 1
        stats1 = pixelstats(dataset1)
        stats2 = pixelstats(dataset2)
        stats3 = dimensionstats(dataset1)
        outliers = Outliers()
        results = outliers.from_stats((stats1, stats2, stats3))
        assert results is not None

    def test_outliers_with_merged_stats(self):
        dataset1 = np.zeros((50, 3, 16, 16))
        dataset2 = np.zeros((50, 3, 16, 16))
        dataset2[0] = 1
        stats3 = visualstats(dataset1)
        stats2 = pixelstats(dataset2)
        stats1 = dimensionstats(dataset1)
        outliers = Outliers()
        stats = DatasetStatsOutput(stats1, stats2, stats3)
        results = outliers.from_stats(stats)
        assert results is not None

    def test_outliers_with_invalid_stats_type(self):
        outliers = Outliers()
        with pytest.raises(TypeError):
            outliers.from_stats(1234)  # type: ignore
        with pytest.raises(TypeError):
            outliers.from_stats([1234])  # type: ignore

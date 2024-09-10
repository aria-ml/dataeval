import numpy as np
import pytest

from dataeval._internal.detectors.outliers import Outliers, _get_outlier_mask
from dataeval._internal.metrics.stats import imagestats
from dataeval.flags import ImageStat


class TestOutliers:
    def test_outliers(self):
        outliers = Outliers()
        results = outliers.evaluate(np.random.random((1000, 3, 16, 16)))
        assert results is not None

    def test_outliers_custom(self):
        outliers = Outliers(ImageStat.ENTROPY)
        results = outliers.evaluate(np.random.random((1000, 3, 16, 16)))
        assert results is not None

    def test_outliers_value_error(self):
        with pytest.raises(ValueError):
            Outliers(ImageStat.XXHASH)

    @pytest.mark.parametrize("method", ["zscore", "modzscore", "iqr"])
    def test_get_outlier_mask(self, method):
        mask_value = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, 2.5)
        mask_none = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, None)
        np.testing.assert_array_equal(mask_value, mask_none)

    def test_get_outlier_mask_valueerror(self):
        with pytest.raises(ValueError):
            _get_outlier_mask(np.zeros([0]), "error", None)  # type: ignore

    def test_get_outliers_with_extra_stats(self):
        outliers = Outliers()
        dataset = np.zeros((100, 3, 16, 16))
        dataset[0] = 1
        outliers.stats = imagestats(dataset, ImageStat.ALL_HASHES | ImageStat.MEAN)
        assert len(outliers.stats.dict()) == 3
        results = outliers._get_outliers()
        assert len(results) == 1
        assert len(results[0]) == 1
        assert "mean" in results[0]

    def test_outliers_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        stats = imagestats(data, ImageStat.MEAN)
        outliers = Outliers(ImageStat.MEAN)
        results = outliers.evaluate(stats)
        assert results is not None

    def test_outliers_with_stats_no_mean(self):
        data = np.random.random((20, 3, 16, 16))
        stats = imagestats(data, ImageStat.VAR)
        outliers = Outliers(ImageStat.MEAN)
        with pytest.raises(ValueError):
            outliers.evaluate(stats)

    def test_outliers_with_stats_mean_plus(self):
        data = np.random.random((20, 3, 16, 16))
        stats = imagestats(data, ImageStat.MEAN | ImageStat.VAR)
        outliers = Outliers(ImageStat.MEAN)
        results = outliers.evaluate(stats)
        assert results is not None

    def test_outliers_with_stats_no_mean_plus(self):
        data = np.random.random((20, 3, 16, 16))
        stats = imagestats(data, ImageStat.MEAN)
        outliers = Outliers(ImageStat.MEAN | ImageStat.VAR)
        with pytest.raises(ValueError):
            outliers.evaluate(stats)

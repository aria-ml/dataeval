from unittest.mock import patch

import numpy as np
import pytest

from dataeval.config import use_max_processes
from dataeval.detectors.linters.outliers import Outliers, OutliersOutput, _get_outlier_mask
from dataeval.metrics.stats import dimensionstats, pixelstats, visualstats
from dataeval.metrics.stats._imagestats import imagestats
from dataeval.outputs._stats import ImageStatsOutput, LabelStatsOutput


@pytest.mark.required
class TestOutliers:
    def test_outliers(self):
        outliers = Outliers()
        results = outliers.evaluate(np.random.random((100, 3, 16, 16)))
        assert len(outliers.stats) == 100
        assert results is not None

    def test_get_outlier_mask_empty(self):
        mask = _get_outlier_mask(np.zeros([0]), "zscore", None)
        assert mask is not None
        assert len(mask) == 0

    @pytest.mark.parametrize("method", ["zscore", "modzscore", "iqr"])
    def test_get_outlier_mask(self, method):
        mask_value = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, 2.5)
        mask_none = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, None)
        np.testing.assert_array_equal(mask_value, mask_none)

    @pytest.mark.parametrize("method", ["zscore", "modzscore", "iqr"])
    @patch("dataeval.detectors.linters.outliers.EPSILON", 100.0)
    def test_get_outlier_mask_with_large_epsilon(self, method):
        mask_value = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, 2.5)
        mask_none = _get_outlier_mask(np.array([0.1, 0.2, 0.1, 1.0]), method, None)
        np.testing.assert_array_equal(mask_value, mask_none)

    def test_get_outlier_mask_valueerror(self):
        with pytest.raises(ValueError):
            _get_outlier_mask(np.random.random((10, 1, 16, 16)), "error", None)  # type: ignore

    def test_get_outlier_mask_all_nan(self):
        mask_none = _get_outlier_mask(np.array([np.nan, np.nan, np.nan]), "zscore", None)
        np.testing.assert_array_equal(mask_none, np.array([False, False, False]))

    def test_outliers_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        stats = pixelstats(data)
        outliers = Outliers()
        results = outliers.from_stats(stats)
        assert results is not None

    def test_outliers_with_nan_stats(self, get_od_dataset):
        images = np.random.random((20, 3, 64, 64))
        images = images / 2.0
        images[10] = 1.0
        dataset = get_od_dataset(images, 2, True, {10: [(-5, -5, -1, -1), (1, 1, 5, 5)]})
        with use_max_processes(1):
            stats = imagestats(dataset, per_box=True)
        outliers = Outliers()
        results = outliers.from_stats(stats)
        assert all(vv != np.nan for k, v in results.issues.items() for vv in v.values())

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

    def test_outliers_with_combined_stats(self):
        dataset1 = np.zeros((50, 3, 16, 16))
        dataset2 = np.zeros((50, 3, 16, 16))
        dataset2[0] = 1
        stats3 = visualstats(dataset1)
        stats2 = pixelstats(dataset2)
        stats1 = dimensionstats(dataset1)
        outliers = Outliers()
        stats = ImageStatsOutput(**{k: v for d in (stats1, stats2, stats3) for k, v in d.data().items()})
        results = outliers.from_stats(stats)
        assert results is not None

    def test_outliers_with_invalid_stats_type(self):
        outliers = Outliers()
        with pytest.raises(TypeError):
            outliers.from_stats(1234)  # type: ignore
        with pytest.raises(TypeError):
            outliers.from_stats([1234])  # type: ignore


@pytest.mark.required
class TestOutliersOutput:
    outlier = {1: {"a": 1.0, "b": 1.0}, 3: {"a": 1.0, "b": 1.0}, 5: {"a": 1.0, "b": 1.0}}
    outlier2 = {2: {"a": 2.0, "d": 2.0}, 6: {"a": 1.0, "d": 1.0}, 7: {"a": 0.5, "c": 0.5}}
    lstat = LabelStatsOutput(
        {0: 3, 1: 4, 2: 3},
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        {0: 3, 1: 4, 2: 3},
        {0: [0, 3, 7], 1: [1, 4, 6, 9], 2: [2, 5, 8]},
        10,
        3,
        10,
        ["horse", "dog", "mule"],
    )

    def test_dict_len(self):
        output = OutliersOutput(self.outlier)
        assert len(output) == 3

    def test_list_len(self):
        output = OutliersOutput([self.outlier, self.outlier2])
        assert len(output) == 6

    def test_to_table(self):
        output = OutliersOutput(self.outlier)
        assert len(output) == 3
        table_result = output.to_table(self.lstat)
        assert isinstance(table_result, str)
        assert table_result[:35] == "  Class |    a    |    b    | Total"

    def test_to_table_list(self):
        output = OutliersOutput([self.outlier2, self.outlier])
        assert len(output) == 6
        table_result = output.to_table(self.lstat)
        assert isinstance(table_result, str)
        print(table_result)
        assert table_result[:45] == "  Class |    a    |    c    |    d    | Total"

    def test_to_dataframe_list(self):
        output = OutliersOutput([self.outlier2, self.outlier])
        assert len(output) == 6
        output_df = output.to_dataframe(self.lstat)
        assert output_df.shape == (6, 7)

    def test_to_dataframe_dict(self):
        output = OutliersOutput(self.outlier)
        assert len(output) == 3
        output_df = output.to_dataframe(self.lstat)
        assert output_df.shape == (3, 4)

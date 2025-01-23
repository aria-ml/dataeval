import numpy as np
import pytest

from dataeval.detectors.linters.outliers import Outliers, OutliersOutput, _get_outlier_mask
from dataeval.metrics.stats import DatasetStatsOutput, LabelStatsOutput, dimensionstats, pixelstats, visualstats


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


class TestOutliersOutput:
    outlier = {1: {"a": 1.0, "b": 1.0}, 3: {"a": 1.0, "b": 1.0}, 5: {"a": 1.0, "b": 1.0}}
    outlier2 = {2: {"a": 2.0, "d": 2.0}, 6: {"a": 1.0, "d": 1.0}, 7: {"a": 0.5, "c": 0.5}}

    def test_dict_len(self):
        output = OutliersOutput(self.outlier)
        assert len(output) == 3

    def test_list_len(self):
        output = OutliersOutput([self.outlier, self.outlier2])
        assert len(output) == 6

    def test_to_table(self):
        output = OutliersOutput(self.outlier)
        assert len(output) == 3
        lstat = LabelStatsOutput(
            {"horse": 3, "dog": 4, "mule": 3},
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            {"horse": 3, "dog": 4, "mule": 3},
            {"horse": [0, 3, 7], "dog": [1, 4, 6, 9], "mule": [2, 5, 8]},
            10,
            3,
            10,
        )
        table_result = output.to_table(lstat)
        assert isinstance(table_result, str)
        assert table_result[:35] == "  Class |    a    |    b    | Total"

    def test_to_table_list(self):
        output = OutliersOutput([self.outlier2, self.outlier])
        assert len(output) == 6
        lstat = LabelStatsOutput(
            {"horse": 3, "dog": 4, "mule": 3},
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            {"horse": 3, "dog": 4, "mule": 3},
            {"horse": [0, 3, 7], "dog": [1, 4, 6, 9], "mule": [2, 5, 8]},
            10,
            3,
            10,
        )
        table_result = output.to_table(lstat)
        assert isinstance(table_result, str)
        print(table_result)
        assert table_result[:45] == "  Class |    a    |    c    |    d    | Total"

    def test_to_dataframe_list(self):
        output = OutliersOutput([self.outlier2, self.outlier])
        assert len(output) == 6
        lstat = LabelStatsOutput(
            {"horse": 3, "dog": 4, "mule": 3},
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            {"horse": 3, "dog": 4, "mule": 3},
            {"horse": [0, 3, 7], "dog": [1, 4, 6, 9], "mule": [2, 5, 8]},
            10,
            3,
            10,
        )
        output_df = output.to_dataframe(lstat)
        assert output_df.shape == (6, 7)

    def test_to_dataframe_dict(self):
        output = OutliersOutput(self.outlier)
        assert len(output) == 3
        lstat = LabelStatsOutput(
            {"horse": 3, "dog": 4, "mule": 3},
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            {"horse": 3, "dog": 4, "mule": 3},
            {"horse": [0, 3, 7], "dog": [1, 4, 6, 9], "mule": [2, 5, 8]},
            10,
            3,
            10,
        )
        output_df = output.to_dataframe(lstat)
        assert output_df.shape == (3, 4)

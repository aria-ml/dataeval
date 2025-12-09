from unittest.mock import patch

import numpy as np
import pytest

from dataeval.config import use_max_processes
from dataeval.core import calculate
from dataeval.core._clusterer import ClusterResult
from dataeval.core._label_stats import LabelStatsResult
from dataeval.core.flags import ImageStats
from dataeval.evaluators.linters.outliers import Outliers, OutliersOutput, _get_outlier_mask


@pytest.mark.required
class TestOutliers:
    def test_outliers(self):
        outliers = Outliers()
        results = outliers.evaluate(np.random.random((100, 3, 16, 16)))
        assert len(outliers.stats["stats"]) > 0
        assert len(outliers.stats["source_index"]) == 100
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
    @patch("dataeval.evaluators.linters.outliers.EPSILON", 100.0)
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
        stats = calculate(data, None, ImageStats.PIXEL)
        outliers = Outliers()
        results = outliers.from_stats(stats)
        assert results is not None

    def test_outliers_with_nan_stats(self, get_od_dataset):
        images = np.random.random((20, 3, 64, 64))
        images = images / 2.0
        images[10] = 1.0
        dataset = get_od_dataset(images, 2, True, {10: [(-5, -5, -1, -1), (1, 1, 5, 5)]})
        from dataeval.utils import unzip_dataset

        with use_max_processes(1):
            stats = calculate(
                *unzip_dataset(dataset, True), stats=ImageStats.PIXEL | ImageStats.VISUAL, per_target=True
            )
        outliers = Outliers()
        results = outliers.from_stats(stats)
        assert all(vv != np.nan for k, v in results.issues.items() for vv in v.values())

    def test_outliers_with_multiple_stats(self):
        dataset1 = np.zeros((50, 3, 16, 16))
        dataset2 = np.zeros((50, 3, 16, 16))
        dataset2[0] = 1
        stats1 = calculate(dataset1, None, ImageStats.PIXEL)
        stats2 = calculate(dataset2, None, ImageStats.PIXEL)
        stats3 = calculate(dataset1, None, ImageStats.DIMENSION)
        outliers = Outliers()
        results = outliers.from_stats((stats1, stats2, stats3))
        assert results is not None

    def test_outliers_with_invalid_stats_type(self):
        outliers = Outliers()
        with pytest.raises(TypeError):
            outliers.from_stats(1234)  # type: ignore
        with pytest.raises(TypeError):
            outliers.from_stats([1234])  # type: ignore

    def test_outliers_all_false(self):
        outliers = Outliers(False, False, False)
        with pytest.raises(ValueError):
            outliers.evaluate(np.zeros([]))

    @pytest.mark.parametrize(
        "params, expected, not_expected",
        (
            ((True, False, False), "width", {"mean", "brightness"}),
            ((False, True, False), "mean", {"width", "brightness"}),
            ((False, False, True), "brightness", {"width", "mean"}),
        ),
    )
    def test_outliers_use_flags(self, params, expected, not_expected):
        outliers = Outliers(*params)
        outliers.evaluate(np.zeros((50, 1, 16, 16)))
        assert expected in outliers.stats["stats"]
        assert not not_expected & set(outliers.stats["stats"])

    def test_outliers_from_clusters_basic(self):
        """Test basic cluster-based outlier detection"""

        # Create simple embeddings
        embeddings = np.random.randn(10, 5)

        # Create ClusterResult as a TypedDict-compatible dictionary
        # Assign most points to cluster 0, one point to cluster -1 (outlier)
        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "condensed_tree": {
                "parent": np.array([]),
                "child": np.array([]),
                "lambda_val": np.array([]),
                "child_size": np.array([]),
            },
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int32),
            "k_distances": np.array([], dtype=np.float32),
        }

        # Find outliers using new method
        detector = Outliers()
        result = detector.from_clusters(embeddings, mock_cluster_result, threshold=2.0)

        # Should be a dict of indices
        assert isinstance(result.issues, dict)
        # Should not raise an error
        assert len(result.issues) >= 0

    def test_outliers_from_clusters_threshold_variations(self):
        """Test that different thresholds produce different numbers of outliers"""

        # Create embeddings where some points are far from cluster center
        main_cluster = np.random.randn(8, 5) * 0.5
        outlier_points = np.random.randn(2, 5) * 2.0 + 5.0
        embeddings = np.vstack([main_cluster, outlier_points])

        # Create ClusterResult - all in same cluster for adaptive detection
        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "condensed_tree": {
                "parent": np.array([]),
                "child": np.array([]),
                "lambda_val": np.array([]),
                "child_size": np.array([]),
            },
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int32),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Outliers()

        # Strict threshold should find more outliers
        result_strict = detector.from_clusters(embeddings, mock_cluster_result, threshold=1.5)
        # Permissive threshold should find fewer outliers
        result_permissive = detector.from_clusters(embeddings, mock_cluster_result, threshold=3.5)

        # Strict should have more (or equal) outliers than permissive
        assert len(result_strict.issues) >= len(result_permissive.issues)

    def test_outliers_from_clusters_no_outliers(self):
        """Test with well-clustered data that has no clear outliers"""

        # Create tight cluster
        embeddings = np.random.randn(10, 5) * 0.1

        # Create ClusterResult
        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "condensed_tree": {
                "parent": np.array([]),
                "child": np.array([]),
                "lambda_val": np.array([]),
                "child_size": np.array([]),
            },
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int32),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Outliers()
        result = detector.from_clusters(embeddings, mock_cluster_result, threshold=3.0)

        # With permissive threshold and tight cluster, should find few or no outliers
        assert isinstance(result.issues, dict)
        assert len(result.issues) == 0


@pytest.mark.required
class TestOutliersOutput:
    outlier = {1: {"a": 1.0, "b": 1.0}, 3: {"a": 1.0, "b": 1.0}, 5: {"a": 1.0, "b": 1.0}}
    outlier2 = {2: {"a": 2.0, "d": 2.0}, 6: {"a": 1.0, "d": 1.0}, 7: {"a": 0.5, "c": 0.5}}
    lstat: LabelStatsResult = {
        "label_counts_per_class": {0: 3, 1: 4, 2: 3},
        "label_counts_per_image": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "image_counts_per_class": {0: 3, 1: 4, 2: 3},
        "image_indices_per_class": {0: [0, 3, 7], 1: [1, 4, 6, 9], 2: [2, 5, 8]},
        "classes_per_image": [[0], [1], [2], [0], [1], [2], [1], [0], [2], [1]],
        "image_count": 10,
        "class_count": 3,
        "label_count": 10,
        "index2label": {0: "horse", 1: "dog", 2: "mule"},
        "empty_image_indices": [],
        "empty_image_count": 0,
    }

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
        assert table_result.splitlines()[0] == "Class |   a   |   b   | Total"

    def test_to_table_list(self):
        output = OutliersOutput([self.outlier2, self.outlier])
        assert len(output) == 6
        table_result = output.to_table(self.lstat)
        assert isinstance(table_result, str)
        assert table_result.splitlines()[0] == "Class |   a   |   c   |   d   | Total"

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

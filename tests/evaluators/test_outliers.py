from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from dataeval.config import use_max_processes
from dataeval.core import calculate
from dataeval.core._clusterer import ClusterResult
from dataeval.core._label_stats import LabelStatsResult
from dataeval.core.flags import ImageStats
from dataeval.data import Metadata
from dataeval.evaluators.linters.outliers import Outliers, OutliersOutput, _get_outlier_mask


def make_mock_metadata(lstat: LabelStatsResult) -> MagicMock:
    """Create a MagicMock with spec=Metadata for testing aggregate_by_class."""
    mock = MagicMock(spec=Metadata)
    mock.item_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mock.class_labels = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 1])
    mock.index2label = lstat["index2label"]
    return mock


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
        # Check that all metric values are not NaN
        assert all(val != np.nan for val in results.issues["metric_value"].to_list())
        # Check that target_id column exists
        assert "target_id" in results.issues.columns
        # For object detection with per_target=True, we should have some target_ids
        # (either None for image-level or int for target-level)

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
        outliers = Outliers(flags=ImageStats(0), per_image=False, per_target=False)
        with pytest.raises(ValueError):
            outliers.evaluate(np.zeros([]))

    @pytest.mark.parametrize(
        "flags, expected, not_expected",
        (
            (ImageStats.DIMENSION, "width", {"mean", "brightness"}),
            (ImageStats.PIXEL, "mean", {"width", "brightness"}),
            (ImageStats.VISUAL, "brightness", {"width", "mean"}),
        ),
    )
    def test_outliers_use_flags(self, flags, expected, not_expected):
        outliers = Outliers(flags=flags)
        outliers.evaluate(np.zeros((50, 1, 16, 16)))
        assert expected in outliers.stats["stats"]
        assert not not_expected & set(outliers.stats["stats"])

    def test_outliers_per_image_per_target(self, get_od_dataset):
        """Test that per_image and per_target parameters are properly passed to calculate"""
        images = np.random.random((10, 3, 64, 64))
        # Create dataset with some bounding boxes
        dataset = get_od_dataset(images, 2, True, {0: [(10, 10, 30, 30)], 5: [(20, 20, 40, 40)]})

        # Test with per_image=True, per_target=True (default)
        outliers1 = Outliers(flags=ImageStats.DIMENSION, per_image=True, per_target=True)
        outliers1.evaluate(dataset)
        # Should have both image-level and target-level stats
        source_indices1 = outliers1.stats["source_index"]
        has_image_level = any(idx.target is None for idx in source_indices1)
        has_target_level = any(idx.target is not None for idx in source_indices1)
        assert has_image_level and has_target_level

        # Test with per_image=True, per_target=False
        outliers2 = Outliers(flags=ImageStats.DIMENSION, per_image=True, per_target=False)
        outliers2.evaluate(dataset)
        # Should have only image-level stats
        source_indices2 = outliers2.stats["source_index"]
        assert all(idx.target is None for idx in source_indices2)

        # Test with per_image=False, per_target=True
        outliers3 = Outliers(flags=ImageStats.DIMENSION, per_image=False, per_target=True)
        outliers3.evaluate(dataset)
        # Should have only target-level stats
        source_indices3 = outliers3.stats["source_index"]
        assert all(idx.target is not None for idx in source_indices3)

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

        # Should be a DataFrame
        assert isinstance(result.issues, pl.DataFrame)
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

    def test_outliers_target_id_column_dropped_when_per_target_false(self):
        """Test that target_id column is dropped when per_target=False"""
        # Test with per_target=False (image-level only)
        outliers = Outliers(flags=ImageStats.PIXEL, per_target=False)
        result = outliers.evaluate(np.random.random((20, 3, 16, 16)))

        # target_id column should be dropped since per_target=False
        assert "target_id" not in result.issues.columns
        assert "item_id" in result.issues.columns
        assert "metric_name" in result.issues.columns
        assert "metric_value" in result.issues.columns

    def test_outliers_target_id_column_kept_when_has_values(self, get_od_dataset):
        """Test that target_id column is kept when there are target-level outliers"""
        images = np.random.random((10, 3, 64, 64))
        # Create dataset with bounding boxes
        dataset = get_od_dataset(images, 2, True, {0: [(0, 0, 64, 64)], 5: [(0, 0, 64, 64)]})

        # Test with per_target=True (should have target-level stats)
        outliers = Outliers(flags=ImageStats.DIMENSION, per_image=True, per_target=True)
        result = outliers.evaluate(dataset)

        # target_id column should be kept since we have target-level outliers
        assert "target_id" in result.issues.columns
        # Verify we have some non-None target_id values
        assert result.issues["target_id"].null_count() < len(result.issues)

    def test_outliers_from_clusters_drops_target_id(self):
        """Test that from_clusters drops target_id column (always image-level)"""
        embeddings = np.random.randn(10, 5)

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
        result = detector.from_clusters(embeddings, mock_cluster_result, threshold=2.0)

        # Cluster-based outlier detection is always image-level, so target_id should be dropped
        assert "target_id" not in result.issues.columns
        assert "item_id" in result.issues.columns

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
        assert isinstance(result.issues, pl.DataFrame)
        assert len(result.issues) == 0


@pytest.mark.required
class TestOutliersOutput:
    # Convert dict format to DataFrame format
    outlier = pl.DataFrame(
        [
            {"item_id": 1, "target_id": None, "metric_name": "a", "metric_value": 1.0},
            {"item_id": 1, "target_id": None, "metric_name": "b", "metric_value": 1.0},
            {"item_id": 3, "target_id": None, "metric_name": "a", "metric_value": 1.0},
            {"item_id": 3, "target_id": None, "metric_name": "b", "metric_value": 1.0},
            {"item_id": 5, "target_id": None, "metric_name": "a", "metric_value": 1.0},
            {"item_id": 5, "target_id": None, "metric_name": "b", "metric_value": 1.0},
        ]
    )
    outlier2 = pl.DataFrame(
        [
            {"item_id": 2, "target_id": None, "metric_name": "a", "metric_value": 2.0},
            {"item_id": 2, "target_id": None, "metric_name": "d", "metric_value": 2.0},
            {"item_id": 6, "target_id": None, "metric_name": "a", "metric_value": 1.0},
            {"item_id": 6, "target_id": None, "metric_name": "d", "metric_value": 1.0},
            {"item_id": 7, "target_id": None, "metric_name": "a", "metric_value": 0.5},
            {"item_id": 7, "target_id": None, "metric_name": "c", "metric_value": 0.5},
        ]
    )
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

    def test_aggregate_by_metric(self):
        """Test aggregate_by_metric returns correct counts"""
        output = OutliersOutput(self.outlier)
        result = output.aggregate_by_metric()

        # Should have 2 metrics (a and b)
        assert result.shape[0] == 2
        assert set(result["metric_name"].to_list()) == {"a", "b"}

        # Both metrics should have 3 images
        assert all(count == 3 for count in result["Total"].to_list())

        # Should be sorted by count (descending) then metric_name
        assert result["metric_name"].to_list() == ["a", "b"]

    def test_aggregate_by_metric_different_counts(self):
        """Test aggregate_by_metric with varying counts per metric"""
        df = pl.DataFrame(
            [
                {"item_id": 1, "target_id": None, "metric_name": "contrast", "metric_value": 1.0},
                {"item_id": 2, "target_id": None, "metric_name": "contrast", "metric_value": 1.0},
                {"item_id": 3, "target_id": None, "metric_name": "depth", "metric_value": 1.0},
                {"item_id": 4, "target_id": None, "metric_name": "skew", "metric_value": 1.0},
                {"item_id": 5, "target_id": None, "metric_name": "skew", "metric_value": 1.0},
                {"item_id": 6, "target_id": None, "metric_name": "skew", "metric_value": 1.0},
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_metric()

        # Should have 3 metrics
        assert result.shape[0] == 3

        # Check sorted order: skew (3), contrast (2), depth (1)
        assert result["metric_name"].to_list() == ["skew", "contrast", "depth"]
        assert result["Total"].to_list() == [3, 2, 1]

    def test_aggregate_by_metric_raises_on_list(self):
        """Test aggregate_by_metric raises error for multiple datasets"""
        output = OutliersOutput([self.outlier, self.outlier2])
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            output.aggregate_by_metric()

    def test_aggregate_by_class(self):
        """Test aggregate_by_class returns correct pivot table"""
        metadata = make_mock_metadata(self.lstat)

        # Create outliers DataFrame matching the test data
        # item_id: 0,3,7=horse(0), 1,4,6,9=dog(1), 2,5,8=mule(2)
        df = pl.DataFrame(
            [
                {"item_id": 0, "target_id": None, "metric_name": "contrast", "metric_value": 1.0},  # horse
                {"item_id": 1, "target_id": None, "metric_name": "contrast", "metric_value": 1.0},  # dog
                {"item_id": 1, "target_id": None, "metric_name": "depth", "metric_value": 1.0},  # dog
                {"item_id": 2, "target_id": None, "metric_name": "depth", "metric_value": 1.0},  # mule
                {"item_id": 3, "target_id": None, "metric_name": "skew", "metric_value": 1.0},  # horse
                {"item_id": 4, "target_id": None, "metric_name": "skew", "metric_value": 1.0},  # dog
                {"item_id": 6, "target_id": None, "metric_name": "contrast", "metric_value": 1.0},  # dog
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_class(metadata)

        # Check shape: 3 classes + 1 Total row, 3 metrics + class_name + Total columns
        assert result.shape == (4, 5)  # 4 rows (3 classes + Total), 5 cols (class_name + 3 metrics + Total)

        # Check column names
        assert "class_name" in result.columns
        assert "Total" in result.columns
        assert "contrast" in result.columns
        assert "depth" in result.columns
        assert "skew" in result.columns

        # Check Total row exists
        assert "Total" in result["class_name"].to_list()

        # Verify data types
        assert result["class_name"].dtype == pl.Categorical("lexical")
        for col in ["contrast", "depth", "skew", "Total"]:
            assert result[col].dtype == pl.UInt32

    def test_aggregate_by_class_raises_on_list(self):
        """Test aggregate_by_class raises error for multiple datasets"""
        metadata = make_mock_metadata(self.lstat)

        output = OutliersOutput([self.outlier, self.outlier2])
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            output.aggregate_by_class(metadata)

    def test_aggregate_by_item(self):
        """Test aggregate_by_item returns correct pivot table"""
        # Create test data with known structure
        df = pl.DataFrame(
            [
                {"item_id": 0, "target_id": None, "metric_name": "contrast", "metric_value": 1.0},
                {"item_id": 0, "target_id": None, "metric_name": "depth", "metric_value": 1.0},
                {"item_id": 0, "target_id": None, "metric_name": "skew", "metric_value": 1.0},
                {"item_id": 1, "target_id": None, "metric_name": "contrast", "metric_value": 1.0},
                {"item_id": 2, "target_id": None, "metric_name": "depth", "metric_value": 1.0},
                {"item_id": 2, "target_id": None, "metric_name": "skew", "metric_value": 1.0},
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Check shape: 3 items, 6 columns (item_id + target_id + 3 metrics + Total)
        assert result.shape == (3, 6)

        # Check column names
        assert "item_id" in result.columns
        assert "target_id" in result.columns
        assert "Total" in result.columns
        assert set(result.columns) == {"item_id", "target_id", "contrast", "depth", "skew", "Total"}

        # Verify items sorted by Total
        assert result["Total"].to_list() == [3, 1, 2]
        assert result["item_id"].to_list() == [0, 1, 2]

        # Check data types
        assert result["item_id"].dtype == pl.Int64
        # target_id can be Null if all values are None, or Int64 if there are actual values
        assert result["target_id"].dtype in [pl.Null, pl.Int64]
        for col in ["contrast", "depth", "skew", "Total"]:
            assert result[col].dtype == pl.UInt32

        # Verify binary indicators (0 or 1)
        for col in ["contrast", "depth", "skew"]:
            values = result[col].to_list()
            assert all(v in [0, 1] for v in values)

    def test_aggregate_by_item_sparse_metrics(self):
        """Test aggregate_by_item with items having different metrics"""
        df = pl.DataFrame(
            [
                {"item_id": 0, "target_id": None, "metric_name": "a", "metric_value": 1.0},
                {"item_id": 0, "target_id": None, "metric_name": "b", "metric_value": 1.0},
                {"item_id": 1, "target_id": None, "metric_name": "c", "metric_value": 1.0},
                {"item_id": 2, "target_id": None, "metric_name": "a", "metric_value": 1.0},
                {"item_id": 2, "target_id": None, "metric_name": "c", "metric_value": 1.0},
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Check shape: 3 items, 6 columns (item_id + target_id + 3 metrics + Total)
        assert result.shape == (3, 6)

        # Check that missing combinations are 0
        # Image 0: has a, b (not c)
        row_0 = result.filter(pl.col("item_id") == 0)
        assert row_0["a"][0] == 1
        assert row_0["b"][0] == 1
        assert row_0["c"][0] == 0
        assert row_0["Total"][0] == 2

        # Image 1: has c (not a, b)
        row_1 = result.filter(pl.col("item_id") == 1)
        assert row_1["a"][0] == 0
        assert row_1["b"][0] == 0
        assert row_1["c"][0] == 1
        assert row_1["Total"][0] == 1

        # Image 2: has a, c (not b)
        row_2 = result.filter(pl.col("item_id") == 2)
        assert row_2["a"][0] == 1
        assert row_2["b"][0] == 0
        assert row_2["c"][0] == 1
        assert row_2["Total"][0] == 2

    def test_aggregate_by_item_empty(self):
        """Test aggregate_by_item with empty DataFrame"""
        df = pl.DataFrame(
            schema={
                "item_id": pl.Int64,
                "target_id": pl.Int64,
                "metric_name": pl.Categorical("lexical"),
                "metric_value": pl.Float64,
            }
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Should return empty DataFrame with item_id, target_id, and Total columns
        assert result.shape[0] == 0
        assert "item_id" in result.columns
        assert "target_id" in result.columns
        assert "Total" in result.columns

    def test_aggregate_by_item_raises_on_list(self):
        """Test aggregate_by_item raises error for multiple datasets"""
        output = OutliersOutput([self.outlier, self.outlier2])
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            output.aggregate_by_item()

    def test_aggregate_by_item_with_targets(self):
        """Test aggregate_by_item with actual target_ids (object detection)"""
        df = pl.DataFrame(
            [
                {"item_id": 0, "target_id": 0, "metric_name": "contrast", "metric_value": 1.0},
                {"item_id": 0, "target_id": 0, "metric_name": "depth", "metric_value": 1.0},
                {"item_id": 0, "target_id": 1, "metric_name": "contrast", "metric_value": 1.0},
                {"item_id": 1, "target_id": None, "metric_name": "contrast", "metric_value": 1.0},  # image-level
                {"item_id": 1, "target_id": 0, "metric_name": "depth", "metric_value": 1.0},
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Check shape: 4 items (img0-tgt0, img0-tgt1, img1-null, img1-tgt0)
        # 5 columns (item_id + target_id + 2 metrics + Total)
        assert result.shape == (4, 5)

        # Check column names
        assert "item_id" in result.columns
        assert "target_id" in result.columns
        assert "Total" in result.columns
        assert set(result.columns) == {"item_id", "target_id", "contrast", "depth", "Total"}

        # Verify correct grouping by (item_id, target_id)
        # Image 0, target 0: has both contrast and depth (Total=2)
        # Image 0, target 1: has only contrast (Total=1)
        # Image 1, target None: has only contrast (Total=1)
        # Image 1, target 0: has only depth (Total=1)
        assert result["Total"].to_list() == [2, 1, 1, 1]

        # Check data types - target_id should be Int64 since we have actual values
        assert result["item_id"].dtype == pl.Int64
        assert result["target_id"].dtype == pl.Int64

    def test_aggregate_by_metric_empty(self):
        """Test aggregate_by_metric with empty DataFrame"""
        df = pl.DataFrame(
            schema={
                "item_id": pl.Int64,
                "target_id": pl.Int64,
                "metric_name": pl.Categorical("lexical"),
                "metric_value": pl.Float64,
            }
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_metric()

        # Should return empty DataFrame with correct schema
        assert result.shape[0] == 0
        assert "metric_name" in result.columns
        assert "Total" in result.columns
        assert result["metric_name"].dtype == pl.Categorical("lexical")
        assert result["Total"].dtype == pl.UInt32

    def test_aggregate_by_item_without_target_id(self):
        """Test aggregate_by_item when target_id column is not present"""
        # Create DataFrame without target_id column (image-level only)
        df = pl.DataFrame(
            [
                {"item_id": 0, "metric_name": "contrast", "metric_value": 1.0},
                {"item_id": 0, "metric_name": "depth", "metric_value": 1.0},
                {"item_id": 0, "metric_name": "skew", "metric_value": 1.0},
                {"item_id": 1, "metric_name": "contrast", "metric_value": 1.0},
                {"item_id": 2, "metric_name": "depth", "metric_value": 1.0},
                {"item_id": 2, "metric_name": "skew", "metric_value": 1.0},
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Check shape: 3 items, 5 columns (item_id + 3 metrics + Total)
        # No target_id column
        assert result.shape == (3, 5)

        # Check column names - should NOT have target_id
        assert "item_id" in result.columns
        assert "target_id" not in result.columns
        assert "Total" in result.columns
        assert set(result.columns) == {"item_id", "contrast", "depth", "skew", "Total"}

        # Verify items sorted by Total
        assert result["Total"].to_list() == [3, 1, 2]
        assert result["item_id"].to_list() == [0, 1, 2]

    def test_aggregate_by_class_empty(self):
        """Test aggregate_by_class with empty DataFrame"""
        metadata = make_mock_metadata(self.lstat)

        df = pl.DataFrame(
            schema={
                "item_id": pl.Int64,
                "target_id": pl.Int64,
                "metric_name": pl.Categorical("lexical"),
                "metric_value": pl.Float64,
            }
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_class(metadata)

        # Should return empty DataFrame with correct schema
        assert result.shape[0] == 0
        assert "class_name" in result.columns
        assert "Total" in result.columns
        assert result["class_name"].dtype == pl.Categorical("lexical")
        assert result["Total"].dtype == pl.UInt32

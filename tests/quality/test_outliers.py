from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from dataeval.config import use_max_processes
from dataeval.core import compute_stats
from dataeval.core._clusterer import ClusterResult
from dataeval.core._label_stats import LabelStatsResult
from dataeval.extractors import FlattenExtractor
from dataeval.flags import ImageStats
from dataeval.quality._outliers import Outliers, OutliersOutput, _build_class_ids, _get_outlier_mask
from dataeval.types import SourceIndex
from dataeval.utils.data import unzip_dataset
from dataeval.utils.thresholds import (
    IQRThreshold,
    ModifiedZScoreThreshold,
    ZScoreThreshold,
)
from tests.conftest import MockMetadata


def make_mock_metadata(lstat: LabelStatsResult) -> MockMetadata:
    """Create a MockMetadata (Metadata) for testing aggregate_by_class."""
    # class_labels maps item_id -> class: [0,1,2,0,1,2,1,0,2,1] (from lstat)
    # Nested structure: each item has one class label

    return MockMetadata(
        class_labels=np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 1], dtype=np.intp),
        factor_data=np.array([], dtype=np.int64),  # Not used by aggregate_by_class
        factor_names=[],  # Not used by aggregate_by_class
        is_discrete=[],  # Not used by aggregate_by_class
        index2label=lstat["index2label"],
    )


@pytest.mark.required
class TestOutliers:
    def test_outliers(self):
        outliers = Outliers()
        results = outliers.evaluate(np.random.random((100, 3, 16, 16)))
        assert len(outliers.stats["stats"]) > 0
        assert len(outliers.stats["source_index"]) == 100
        assert results is not None

    def test_get_outlier_mask_empty(self):
        mask = _get_outlier_mask(np.zeros([0]), ZScoreThreshold())
        assert mask is not None
        assert len(mask) == 0

    @pytest.mark.parametrize(
        "threshold",
        [ZScoreThreshold(2.5), ModifiedZScoreThreshold(2.5), IQRThreshold(2.5)],
    )
    def test_get_outlier_mask(self, threshold):
        data = np.array([0.1, 0.2, 0.1, 1.0])
        mask = _get_outlier_mask(data, threshold)
        # With only 4 values, 2.5x multiplier should not flag anything
        assert mask is not None
        assert len(mask) == len(data)

    def test_get_outlier_mask_all_nan(self):
        mask = _get_outlier_mask(np.array([np.nan, np.nan, np.nan]), ZScoreThreshold())
        np.testing.assert_array_equal(mask, np.array([False, False, False]))

    def test_outliers_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        stats = compute_stats(data, stats=ImageStats.PIXEL)
        outliers = Outliers()
        results = outliers.from_stats(stats)
        assert results is not None

    def test_outliers_with_nan_stats(self, get_od_dataset):
        images = np.random.random((20, 3, 64, 64))
        images = images / 2.0
        images[10] = 1.0
        dataset = get_od_dataset(images, 2, True, {10: [(-5, -5, -1, -1), (1, 1, 5, 5)]})

        with use_max_processes(1):
            _images, _boxes = unzip_dataset(dataset, True)
            stats = compute_stats(
                _images,
                boxes=_boxes,
                stats=ImageStats.PIXEL | ImageStats.VISUAL,
                per_target=True,
            )
        outliers = Outliers()
        results = outliers.from_stats(stats, per_target=True)
        # Check that all metric values are not NaN
        assert all(val != np.nan for val in results.data()["metric_value"].to_list())
        # Check that target_id column exists
        assert "target_index" in results.data().columns
        # For object detection with per_target=True, we should have some target_ids
        # (either None for image-level or int for target-level)

    def test_outliers_with_multiple_stats(self):
        dataset1 = np.zeros((50, 3, 16, 16))
        dataset2 = np.zeros((50, 3, 16, 16))
        dataset2[0] = 1
        stats1 = compute_stats(dataset1, stats=ImageStats.PIXEL)
        stats2 = compute_stats(dataset2, stats=ImageStats.PIXEL)
        stats3 = compute_stats(dataset1, stats=ImageStats.DIMENSION)
        outliers = Outliers()
        results = outliers.from_stats((stats1, stats2, stats3))
        assert results is not None
        assert isinstance(results.data(), pl.DataFrame)
        assert "dataset_index" in results.data().columns

    def test_outliers_with_invalid_stats_type(self):
        outliers = Outliers()
        with pytest.raises(TypeError):
            outliers.from_stats(1234)  # type: ignore
        with pytest.raises(TypeError):
            outliers.from_stats([1234])  # type: ignore

    def test_outliers_with_metric_thresholds(self):
        data = np.random.random((20, 3, 16, 16))
        stats = compute_stats(data, stats=ImageStats.VISUAL)
        outliers = Outliers(outlier_threshold={"contrast": 2.0, "brightness": ("zscore", 2.0), "sharpness": "iqr"})
        results = outliers.from_stats(stats)
        assert results is not None

    def test_outliers_all_false(self):
        outliers = Outliers(flags=ImageStats(0))
        with pytest.raises(ValueError, match="Either flags must not be ImageStats.NONE or extractor must be provided."):
            outliers.evaluate(np.zeros([]), per_image=False, per_target=False)

    @pytest.mark.parametrize(
        ("flags", "expected", "not_expected"),
        [
            (ImageStats.DIMENSION, "width", {"mean", "brightness"}),
            (ImageStats.PIXEL, "mean", {"width", "brightness"}),
            (ImageStats.VISUAL, "brightness", {"width", "mean"}),
        ],
    )
    def test_outliers_use_flags(self, flags, expected, not_expected):
        outliers = Outliers(flags=flags)
        outliers.evaluate(np.zeros((50, 1, 16, 16)))
        assert expected in outliers.stats["stats"]
        assert not not_expected & set(outliers.stats["stats"])

    def test_outliers_per_image_per_target(self, get_od_dataset):
        """Test that per_image and per_target parameters are properly passed to calculate."""
        images = np.random.random((10, 3, 64, 64))
        # Create dataset with some bounding boxes
        dataset = get_od_dataset(images, 2, True, {0: [(10, 10, 30, 30)], 5: [(20, 20, 40, 40)]})

        # Test with per_image=True, per_target=True (default)
        outliers1 = Outliers(flags=ImageStats.DIMENSION)
        outliers1.evaluate(dataset, per_image=True, per_target=True)
        # Should have both image-level and target-level stats
        source_indices1 = outliers1.stats["source_index"]
        has_image_level = any(idx.target is None for idx in source_indices1)
        has_target_level = any(idx.target is not None for idx in source_indices1)
        assert has_image_level
        assert has_target_level

        # Test with per_image=True, per_target=False
        outliers2 = Outliers(flags=ImageStats.DIMENSION)
        outliers2.evaluate(dataset, per_image=True, per_target=False)
        # Should have only image-level stats
        source_indices2 = outliers2.stats["source_index"]
        assert all(idx.target is None for idx in source_indices2)

        # Test with per_image=False, per_target=True
        outliers3 = Outliers(flags=ImageStats.DIMENSION)
        outliers3.evaluate(dataset, per_image=False, per_target=True)
        # Should have only target-level stats
        source_indices3 = outliers3.stats["source_index"]
        assert all(idx.target is not None for idx in source_indices3)

    def test_outliers_from_clusters_basic(self):
        """Test basic cluster-based outlier detection."""
        # Create simple embeddings
        embeddings = np.random.randn(10, 5)

        # Create ClusterResult as a TypedDict-compatible dictionary
        # Assign most points to cluster 0, one point to cluster -1 (outlier)
        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        # Find outliers using new method
        detector = Outliers()
        result = detector.from_clusters(embeddings, mock_cluster_result, cluster_threshold=2.0)

        # Should be a DataFrame
        assert isinstance(result.data(), pl.DataFrame)
        # Should not raise an error
        assert len(result.data()) >= 0

    def test_outliers_from_clusters_threshold_variations(self):
        """Test that different thresholds produce different numbers of outliers."""
        # Create embeddings where some points are far from cluster center
        main_cluster = np.random.randn(8, 5) * 0.5
        outlier_points = np.random.randn(2, 5) * 2.0 + 5.0
        embeddings = np.vstack([main_cluster, outlier_points])

        # Create ClusterResult - all in same cluster for adaptive detection
        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Outliers()

        # Strict threshold should find more outliers
        result_strict = detector.from_clusters(embeddings, mock_cluster_result, cluster_threshold=1.5)
        # Permissive threshold should find fewer outliers
        result_permissive = detector.from_clusters(embeddings, mock_cluster_result, cluster_threshold=3.5)

        # Strict should have more (or equal) outliers than permissive
        assert len(result_strict.data()) >= len(result_permissive.data())

    def test_outliers_target_id_column_dropped_when_per_target_false(self):
        """Test that target_id column is dropped when per_target=False."""
        # Test with per_target=False (image-level only)
        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(np.random.random((20, 3, 16, 16)), per_target=False)

        # target_id column should be dropped since per_target=False
        assert "target_index" not in result.data().columns
        assert "item_index" in result.data().columns
        assert "metric_name" in result.data().columns
        assert "metric_value" in result.data().columns

    def test_outliers_target_id_column_kept_when_has_values(self, get_od_dataset):
        """Test that target_id column is kept when there are target-level outliers."""
        images = np.random.random((10, 3, 64, 64))
        # Create dataset with bounding boxes
        dataset = get_od_dataset(images, 2, True, {0: [(0, 0, 64, 64)], 5: [(0, 0, 64, 64)]})

        # Use a tight threshold to guarantee outliers are detected in this small dataset
        outliers = Outliers(flags=ImageStats.DIMENSION, outlier_threshold=1.0)
        result = outliers.evaluate(dataset, per_image=True, per_target=True)

        # target_id column should be kept since we have target-level outliers
        assert "target_index" in result.data().columns
        # Verify we have some non-None target_id values
        assert result.data()["target_index"].null_count() < len(result.data())

    def test_outliers_from_clusters_drops_target_id(self):
        """Test that from_clusters drops target_id column (always image-level)."""
        embeddings = np.random.randn(10, 5)

        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Outliers()
        result = detector.from_clusters(embeddings, mock_cluster_result, cluster_threshold=2.0)

        # Cluster-based outlier detection is always image-level, so target_id should be dropped
        assert "target_index" not in result.data().columns
        assert "item_index" in result.data().columns

    def test_outliers_from_clusters_no_outliers(self):
        """Test with well-clustered data that has no clear outliers."""
        # Create tight cluster
        embeddings = np.random.randn(10, 5) * 0.1

        # Create ClusterResult
        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Outliers()
        result = detector.from_clusters(embeddings, mock_cluster_result, cluster_threshold=3.0)

        # With permissive threshold and tight cluster, should find few or no outliers
        assert isinstance(result.data(), pl.DataFrame)
        assert len(result.data()) == 0


@pytest.mark.required
class TestOutliersOutput:
    # Convert dict format to DataFrame format
    outlier = pl.DataFrame(
        [
            {"item_index": 1, "target_index": None, "metric_name": "a", "metric_value": 1.0},
            {"item_index": 1, "target_index": None, "metric_name": "b", "metric_value": 1.0},
            {"item_index": 3, "target_index": None, "metric_name": "a", "metric_value": 1.0},
            {"item_index": 3, "target_index": None, "metric_name": "b", "metric_value": 1.0},
            {"item_index": 5, "target_index": None, "metric_name": "a", "metric_value": 1.0},
            {"item_index": 5, "target_index": None, "metric_name": "b", "metric_value": 1.0},
        ],
    )
    outlier2 = pl.DataFrame(
        [
            {"item_index": 2, "target_index": None, "metric_name": "a", "metric_value": 2.0},
            {"item_index": 2, "target_index": None, "metric_name": "d", "metric_value": 2.0},
            {"item_index": 6, "target_index": None, "metric_name": "a", "metric_value": 1.0},
            {"item_index": 6, "target_index": None, "metric_name": "d", "metric_value": 1.0},
            {"item_index": 7, "target_index": None, "metric_name": "a", "metric_value": 0.5},
            {"item_index": 7, "target_index": None, "metric_name": "c", "metric_value": 0.5},
        ],
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

    def test_multi_dataset_len(self):
        combined = pl.concat(
            [
                self.outlier.with_columns(pl.lit(0).alias("dataset_index")),
                self.outlier2.with_columns(pl.lit(1).alias("dataset_index")),
            ]
        ).select(["dataset_index"] + self.outlier.columns)
        output = OutliersOutput(combined)
        assert len(output) == 6

    def test_aggregate_by_metric(self):
        """Test aggregate_by_metric returns correct counts."""
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
        """Test aggregate_by_metric with varying counts per metric."""
        df = pl.DataFrame(
            [
                {"item_index": 1, "target_index": None, "metric_name": "contrast", "metric_value": 1.0},
                {"item_index": 2, "target_index": None, "metric_name": "contrast", "metric_value": 1.0},
                {"item_index": 3, "target_index": None, "metric_name": "depth", "metric_value": 1.0},
                {"item_index": 4, "target_index": None, "metric_name": "skew", "metric_value": 1.0},
                {"item_index": 5, "target_index": None, "metric_name": "skew", "metric_value": 1.0},
                {"item_index": 6, "target_index": None, "metric_name": "skew", "metric_value": 1.0},
            ],
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_metric()

        # Should have 3 metrics
        assert result.shape[0] == 3

        # Check sorted order: skew (3), contrast (2), depth (1)
        assert result["metric_name"].to_list() == ["skew", "contrast", "depth"]
        assert result["Total"].to_list() == [3, 2, 1]

    def test_aggregate_by_metric_raises_on_multi_dataset(self):
        """Test aggregate_by_metric raises error for multiple datasets."""
        combined = pl.concat(
            [
                self.outlier.with_columns(pl.lit(0).alias("dataset_index")),
                self.outlier2.with_columns(pl.lit(1).alias("dataset_index")),
            ]
        )
        output = OutliersOutput(combined)
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            output.aggregate_by_metric()

    def test_aggregate_by_class(self):
        """Test aggregate_by_class returns correct pivot table."""
        metadata = make_mock_metadata(self.lstat)

        # Create outliers DataFrame matching the test data
        # item_id: 0,3,7=horse(0), 1,4,6,9=dog(1), 2,5,8=mule(2)
        df = pl.DataFrame(
            [
                {"item_index": 0, "target_index": None, "metric_name": "contrast", "metric_value": 1.0},  # horse
                {"item_index": 1, "target_index": None, "metric_name": "contrast", "metric_value": 1.0},  # dog
                {"item_index": 1, "target_index": None, "metric_name": "depth", "metric_value": 1.0},  # dog
                {"item_index": 2, "target_index": None, "metric_name": "depth", "metric_value": 1.0},  # mule
                {"item_index": 3, "target_index": None, "metric_name": "skew", "metric_value": 1.0},  # horse
                {"item_index": 4, "target_index": None, "metric_name": "skew", "metric_value": 1.0},  # dog
                {"item_index": 6, "target_index": None, "metric_name": "contrast", "metric_value": 1.0},  # dog
            ],
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

    def test_aggregate_by_class_raises_on_multi_dataset(self):
        """Test aggregate_by_class raises error for multiple datasets."""
        metadata = make_mock_metadata(self.lstat)

        combined = pl.concat(
            [
                self.outlier.with_columns(pl.lit(0).alias("dataset_index")),
                self.outlier2.with_columns(pl.lit(1).alias("dataset_index")),
            ]
        )
        output = OutliersOutput(combined)
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            output.aggregate_by_class(metadata)

    def test_aggregate_by_item(self):
        """Test aggregate_by_item returns correct pivot table."""
        # Create test data with known structure
        df = pl.DataFrame(
            [
                {"item_index": 0, "target_index": None, "metric_name": "contrast", "metric_value": 1.0},
                {"item_index": 0, "target_index": None, "metric_name": "depth", "metric_value": 1.0},
                {"item_index": 0, "target_index": None, "metric_name": "skew", "metric_value": 1.0},
                {"item_index": 1, "target_index": None, "metric_name": "contrast", "metric_value": 1.0},
                {"item_index": 2, "target_index": None, "metric_name": "depth", "metric_value": 1.0},
                {"item_index": 2, "target_index": None, "metric_name": "skew", "metric_value": 1.0},
            ],
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Check shape: 3 items, 6 columns (item_id + target_id + 3 metrics + Total)
        assert result.shape == (3, 6)

        # Check column names
        assert "item_index" in result.columns
        assert "target_index" in result.columns
        assert "Total" in result.columns
        assert set(result.columns) == {"item_index", "target_index", "contrast", "depth", "skew", "Total"}

        # Verify items sorted by Total
        assert result["Total"].to_list() == [3, 1, 2]
        assert result["item_index"].to_list() == [0, 1, 2]

        # Check data types
        assert result["item_index"].dtype == pl.Int64
        # target_id can be Null if all values are None, or Int64 if there are actual values
        assert result["target_index"].dtype in [pl.Null, pl.Int64]
        for col in ["contrast", "depth", "skew", "Total"]:
            assert result[col].dtype == pl.UInt32

        # Verify binary indicators (0 or 1)
        for col in ["contrast", "depth", "skew"]:
            values = result[col].to_list()
            assert all(v in [0, 1] for v in values)

    def test_aggregate_by_item_sparse_metrics(self):
        """Test aggregate_by_item with items having different metrics."""
        df = pl.DataFrame(
            [
                {"item_index": 0, "target_index": None, "metric_name": "a", "metric_value": 1.0},
                {"item_index": 0, "target_index": None, "metric_name": "b", "metric_value": 1.0},
                {"item_index": 1, "target_index": None, "metric_name": "c", "metric_value": 1.0},
                {"item_index": 2, "target_index": None, "metric_name": "a", "metric_value": 1.0},
                {"item_index": 2, "target_index": None, "metric_name": "c", "metric_value": 1.0},
            ],
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Check shape: 3 items, 6 columns (item_id + target_id + 3 metrics + Total)
        assert result.shape == (3, 6)

        # Check that missing combinations are 0
        # Image 0: has a, b (not c)
        row_0 = result.filter(pl.col("item_index") == 0)
        assert row_0["a"][0] == 1
        assert row_0["b"][0] == 1
        assert row_0["c"][0] == 0
        assert row_0["Total"][0] == 2

        # Image 1: has c (not a, b)
        row_1 = result.filter(pl.col("item_index") == 1)
        assert row_1["a"][0] == 0
        assert row_1["b"][0] == 0
        assert row_1["c"][0] == 1
        assert row_1["Total"][0] == 1

        # Image 2: has a, c (not b)
        row_2 = result.filter(pl.col("item_index") == 2)
        assert row_2["a"][0] == 1
        assert row_2["b"][0] == 0
        assert row_2["c"][0] == 1
        assert row_2["Total"][0] == 2

    def test_aggregate_by_item_empty(self):
        """Test aggregate_by_item with empty DataFrame."""
        df = pl.DataFrame(
            schema={
                "item_index": pl.Int64,
                "target_index": pl.Int64,
                "metric_name": pl.Categorical("lexical"),
                "metric_value": pl.Float64,
            },
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Should return empty DataFrame with item_id, target_id, and Total columns
        assert result.shape[0] == 0
        assert "item_index" in result.columns
        assert "target_index" in result.columns
        assert "Total" in result.columns

    def test_aggregate_by_item_raises_on_multi_dataset(self):
        """Test aggregate_by_item raises error for multiple datasets."""
        combined = pl.concat(
            [
                self.outlier.with_columns(pl.lit(0).alias("dataset_index")),
                self.outlier2.with_columns(pl.lit(1).alias("dataset_index")),
            ]
        )
        output = OutliersOutput(combined)
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            output.aggregate_by_item()

    def test_aggregate_by_item_with_targets(self):
        """Test aggregate_by_item with actual target_ids (object detection)."""
        df = pl.DataFrame(
            [
                {"item_index": 0, "target_index": 0, "metric_name": "contrast", "metric_value": 1.0},
                {"item_index": 0, "target_index": 0, "metric_name": "depth", "metric_value": 1.0},
                {"item_index": 0, "target_index": 1, "metric_name": "contrast", "metric_value": 1.0},
                {"item_index": 1, "target_index": None, "metric_name": "contrast", "metric_value": 1.0},  # image-level
                {"item_index": 1, "target_index": 0, "metric_name": "depth", "metric_value": 1.0},
            ],
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Check shape: 4 items (img0-tgt0, img0-tgt1, img1-null, img1-tgt0)
        # 5 columns (item_id + target_id + 2 metrics + Total)
        assert result.shape == (4, 5)

        # Check column names
        assert "item_index" in result.columns
        assert "target_index" in result.columns
        assert "Total" in result.columns
        assert set(result.columns) == {"item_index", "target_index", "contrast", "depth", "Total"}

        # Verify correct grouping by (item_id, target_id)
        # Image 0, target 0: has both contrast and depth (Total=2)
        # Image 0, target 1: has only contrast (Total=1)
        # Image 1, target None: has only contrast (Total=1)
        # Image 1, target 0: has only depth (Total=1)
        assert result["Total"].to_list() == [2, 1, 1, 1]

        # Check data types - target_id should be Int64 since we have actual values
        assert result["item_index"].dtype == pl.Int64
        assert result["target_index"].dtype == pl.Int64

    def test_aggregate_by_metric_empty(self):
        """Test aggregate_by_metric with empty DataFrame."""
        df = pl.DataFrame(
            schema={
                "item_index": pl.Int64,
                "target_index": pl.Int64,
                "metric_name": pl.Categorical("lexical"),
                "metric_value": pl.Float64,
            },
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
        """Test aggregate_by_item when target_id column is not present."""
        # Create DataFrame without target_id column (image-level only)
        df = pl.DataFrame(
            [
                {"item_index": 0, "metric_name": "contrast", "metric_value": 1.0},
                {"item_index": 0, "metric_name": "depth", "metric_value": 1.0},
                {"item_index": 0, "metric_name": "skew", "metric_value": 1.0},
                {"item_index": 1, "metric_name": "contrast", "metric_value": 1.0},
                {"item_index": 2, "metric_name": "depth", "metric_value": 1.0},
                {"item_index": 2, "metric_name": "skew", "metric_value": 1.0},
            ],
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_item()

        # Check shape: 3 items, 5 columns (item_id + 3 metrics + Total)
        # No target_id column
        assert result.shape == (3, 5)

        # Check column names - should NOT have target_id
        assert "item_index" in result.columns
        assert "target_index" not in result.columns
        assert "Total" in result.columns
        assert set(result.columns) == {"item_index", "contrast", "depth", "skew", "Total"}

        # Verify items sorted by Total
        assert result["Total"].to_list() == [3, 1, 2]
        assert result["item_index"].to_list() == [0, 1, 2]

    def test_aggregate_by_class_empty(self):
        """Test aggregate_by_class with empty DataFrame."""
        metadata = make_mock_metadata(self.lstat)

        df = pl.DataFrame(
            schema={
                "item_index": pl.Int64,
                "target_index": pl.Int64,
                "metric_name": pl.Categorical("lexical"),
                "metric_value": pl.Float64,
            },
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_class(metadata)

        # Should return empty DataFrame with correct schema
        assert result.shape[0] == 0
        assert "class_name" in result.columns
        assert "Total" in result.columns
        assert result["class_name"].dtype == pl.Categorical("lexical")
        assert result["Total"].dtype == pl.UInt32


@pytest.mark.required
class TestOutliersCoverageImprovements:
    """Additional tests to improve coverage in _outliers.py."""

    def test_outliers_get_outliers_with_no_outliers_found(self):
        """Test _get_outliers when no outliers are detected (line 509)."""
        # Create data with very small variation (no outliers expected)
        images = np.ones((20, 3, 16, 16)) * 0.5
        # Add tiny variation to avoid completely constant data
        images += np.random.random(images.shape) * 0.001

        outliers = Outliers(outlier_threshold=ZScoreThreshold(3.0))
        result = outliers.evaluate(images)

        # Should return empty DataFrame with correct schema
        assert isinstance(result.data(), pl.DataFrame)
        assert "item_index" in result.data().columns
        assert "metric_name" in result.data().columns
        assert "metric_value" in result.data().columns

    def test_outliers_evaluate_with_per_target_false_no_boxes(self):
        """Test evaluate with no boxes (line 461-509, 851)."""
        images = np.random.random((10, 3, 16, 16))
        images[5] = 1.0  # Make one image an outlier

        outliers = Outliers(
            flags=ImageStats.PIXEL,
            outlier_threshold=ZScoreThreshold(2.0),
        )

        result = outliers.evaluate(images)

        assert result is not None

        # Should have image-level stats only
        assert len(outliers.stats["source_index"]) > 0
        # All source indices should have target=None (no boxes in input)
        assert all(idx.target is None for idx in outliers.stats["source_index"])

    def test_outliers_from_stats_with_empty_result(self):
        """Test from_stats when no outliers are found."""
        # Create data with no outliers
        images1 = np.ones((20, 3, 16, 16)) * 0.5
        images2 = np.ones((20, 3, 16, 16)) * 0.5
        images1 += np.random.random(images1.shape) * 0.001
        images2 += np.random.random(images2.shape) * 0.001

        stats1 = compute_stats(images1, stats=ImageStats.PIXEL)
        stats2 = compute_stats(images2, stats=ImageStats.PIXEL)

        outliers = Outliers(outlier_threshold=ZScoreThreshold(5.0))  # Very high threshold
        result = outliers.from_stats([stats1, stats2])

        # Should return a single DataFrame with dataset_index column
        assert isinstance(result.data(), pl.DataFrame)
        assert "item_index" in result.data().columns
        assert "metric_name" in result.data().columns
        assert "metric_value" in result.data().columns


@pytest.mark.required
class TestOutliersEdgeCases:
    def test_output_len_empty(self):
        """Covers __len__ for empty and multi-dataset DataFrames."""
        # Single DF - must have the required columns for __len__ to work
        empty_schema = {
            "item_index": pl.Int64,
            "target_index": pl.Int64,
            "metric_name": pl.Categorical("lexical"),
            "metric_value": pl.Float64,
        }
        out = OutliersOutput(pl.DataFrame(schema=empty_schema))
        assert len(out) == 0

        # Multi-dataset empty DF
        multi_schema = {"dataset_index": pl.Int64, **empty_schema}
        out_multi = OutliersOutput(pl.DataFrame(schema=multi_schema))
        assert len(out_multi) == 0

    def test_aggregate_empty_dfs(self):
        """Covers aggregations returning empty structures."""
        out = OutliersOutput(pl.DataFrame())  # Empty

        # Test aggregate_by_metric on empty
        res_metric = out.aggregate_by_metric()
        assert res_metric.shape[0] == 0
        assert "metric_name" in res_metric.columns

        # Test aggregate_by_item on empty
        res_item = out.aggregate_by_item()
        assert res_item.shape[0] == 0
        assert "Total" in res_item.columns

        # Test aggregate_by_class failure on multi-dataset
        multi_df = pl.DataFrame({"dataset_index": [], "item_index": []})
        out_multi = OutliersOutput(multi_df)
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            out_multi.aggregate_by_class(MagicMock())

        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            out_multi.aggregate_by_item()

    def test_from_stats_skips_non_numeric_stats(self):
        """Regression test: hash stats (string dtype) should be skipped in outlier detection.

        Previously, stats like 'xxhash' containing hex strings (e.g. '9cca8a3736741ab7')
        would cause a ValueError when _get_outliers tried to cast them to float64.
        """
        detector = Outliers(outlier_threshold=ZScoreThreshold(1.0))

        # Simulate stats containing both numeric and non-numeric (hash) values
        mock_stats = {
            "stats": {
                "mean": np.array([100.0, 1.0, 100.0]),
                "xxhash": np.array(["9cca8a3736741ab7", "a1b2c3d4e5f60718", "ff00ff00ff00ff00"], dtype=object),
                "phash": np.array(["abcd1234", "efgh5678", "ijkl9012"], dtype=object),
            },
            "source_index": [
                SourceIndex(0, None, None),
                SourceIndex(1, None, None),
                SourceIndex(2, None, None),
            ],
        }

        # Should not raise ValueError on string-to-float conversion
        output = detector.from_stats(mock_stats)  # type: ignore
        assert isinstance(output.data(), pl.DataFrame)
        # Only numeric stats should produce outliers — hash stats should be silently skipped
        assert all(name != "xxhash" for name in output.data()["metric_name"].to_list())
        assert all(name != "phash" for name in output.data()["metric_name"].to_list())

    def test_evaluate_drops_null_target_id(self):
        """Covers evaluate logic dropping 'target_id' column if all null."""
        # Use zscore with threshold 1.0
        # With values [100.0, 1.0, 100.0]:
        # - Mean = 67.0, Std ≈ 46.67
        # - Z-score for 100.0: |100-67|/46.67 ≈ 0.707
        # - Z-score for 1.0: |1-67|/46.67 ≈ 1.41
        # Threshold 1.0 means only item 1 (z-score 1.41 > 1.0) is flagged
        detector = Outliers(outlier_threshold=ZScoreThreshold(1.0))

        # We manually construct a result that has all null target_ids
        mock_stats = {
            "stats": {"brightness": np.array([100.0, 1.0, 100.0])},
            "source_index": [
                SourceIndex(0, None, None),
                SourceIndex(1, None, None),  # Outlier (z-score ≈ 1.41)
                SourceIndex(2, None, None),
            ],
        }

        # Test from_stats which uses the same logic
        output = detector.from_stats(mock_stats)  # type: ignore
        assert "target_index" not in output.data().columns
        assert output.data()["item_index"].to_list() == [1]

    def test_from_stats_multiple_datasets_filtering(self):
        """
        Covers from_stats logic when adding dataset_index for multiple datasets.

        The from_stats method combines stats, runs outlier detection, then adds
        dataset_index with local item indices via get_dataset_step_from_idx.
        """
        # Use threshold 1.5 so only the extreme value (1000.0) is flagged as outlier
        # With combined values [10.0, 10.0, 1000.0, 10.0]:
        # - mean ≈ 257.5, std ≈ 428.5
        # - z-score for 1000.0 ≈ 1.73 > 1.5 (outlier)
        # - z-score for 10.0 ≈ 0.58 < 1.5 (not outlier)
        detector = Outliers(outlier_threshold=1.5)

        # Create stats that mimic 2 datasets
        # Dataset 1: 1 item (value 10.0)
        # Dataset 2: 3 items (values 10.0, 1000.0, 10.0)
        stats1 = {
            "stats": {"mean": np.array([10.0])},
            "source_index": [SourceIndex(0, None, None)],
        }
        stats2 = {
            "stats": {"mean": np.array([10.0, 1000.0, 10.0])},
            "source_index": [SourceIndex(0, None, None), SourceIndex(1, None, None), SourceIndex(2, None, None)],
        }

        output = detector.from_stats([stats1, stats2])  # type: ignore

        assert isinstance(output.data(), pl.DataFrame)
        assert "dataset_index" in output.data().columns

        # Dataset 0 should have no outliers
        ds0 = output.data().filter(pl.col("dataset_index") == 0)
        assert ds0.shape[0] == 0

        # Dataset 1 should have 1 outlier
        # The outlier is at SourceIndex(1) in stats2 (value 1000.0).
        # With offset correction, combined item=2, remapped to dataset 1, local item 1.
        ds1 = output.data().filter(pl.col("dataset_index") == 1)
        assert ds1.shape[0] == 1
        assert ds1["item_index"][0] == 1

    def test_get_outlier_mask_branches(self):
        """Covers _get_outlier_mask specific branches (all nan, empty)."""
        t = ZScoreThreshold(3.0)
        # Empty
        assert len(_get_outlier_mask(np.array([]), t)) == 0

        # All NaNs
        res = _get_outlier_mask(np.array([np.nan, np.nan]), t)
        assert not np.any(res)

    def test_evaluate_with_tuple_dataset(self, get_mock_ic_dataset):
        """Regression test: evaluate with cluster-based detection should handle tuple datasets.

        When a dataset returns (image, label, metadata) tuples, the extractor should
        receive only the images, not the full tuples.
        """
        data = np.random.random((20, 3, 16, 16))
        labels = list(range(len(data)))
        dataset = get_mock_ic_dataset(list(data), labels)

        outliers = Outliers(flags=ImageStats.NONE, extractor=FlattenExtractor(), cluster_threshold=2.0)
        result = outliers.evaluate(dataset)
        assert result is not None
        assert isinstance(result.data(), pl.DataFrame)

    def test_evaluate_with_tuple_dataset_combined(self, get_mock_ic_dataset):
        """Regression test: combined hash + cluster detection on tuple datasets should not crash."""
        data = np.random.random((20, 3, 16, 16))
        labels = list(range(len(data)))
        dataset = get_mock_ic_dataset(list(data), labels)

        outliers = Outliers(flags=ImageStats.PIXEL, extractor=FlattenExtractor(), cluster_threshold=2.0)
        result = outliers.evaluate(dataset)
        assert result is not None
        assert isinstance(result.data(), pl.DataFrame)


@pytest.mark.required
class TestBuildClassIds:
    def test_ic_dataset_1to1(self):
        """IC dataset: each image has exactly one class."""
        source_index = [
            SourceIndex(0, None, None),
            SourceIndex(1, None, None),
            SourceIndex(2, None, None),
        ]
        metadata = MockMetadata(
            class_labels=np.array([0, 1, 0], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "cat", 1: "dog"},
        )
        class_ids = _build_class_ids(source_index, metadata)
        np.testing.assert_array_equal(class_ids, [0, 1, 0])

    def test_od_target_level(self):
        """OD dataset: target-level entries get correct class."""
        # Image 0 has 2 targets (classes 0, 1), Image 1 has 1 target (class 0)
        source_index = [
            SourceIndex(0, None, None),  # image-level for img 0
            SourceIndex(0, 0, None),  # target 0 of img 0
            SourceIndex(0, 1, None),  # target 1 of img 0
            SourceIndex(1, None, None),  # image-level for img 1
            SourceIndex(1, 0, None),  # target 0 of img 1
        ]
        metadata = MockMetadata(
            class_labels=np.array([0, 1, 0], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "cat", 1: "dog"},
            item_indices=np.array([0, 0, 1], dtype=np.int64),
        )
        class_ids = _build_class_ids(source_index, metadata)
        # img 0 image-level: has classes {0, 1} -> -1
        # img 0 target 0: class 0
        # img 0 target 1: class 1
        # img 1 image-level: has class {0} -> 0
        # img 1 target 0: class 0
        np.testing.assert_array_equal(class_ids, [-1, 0, 1, 0, 0])

    def test_od_single_class_image(self):
        """OD image with all targets same class gets that class for image-level."""
        source_index = [
            SourceIndex(0, None, None),
            SourceIndex(0, 0, None),
            SourceIndex(0, 1, None),
        ]
        metadata = MockMetadata(
            class_labels=np.array([1, 1], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={1: "dog"},
            item_indices=np.array([0, 0], dtype=np.int64),
        )
        class_ids = _build_class_ids(source_index, metadata)
        np.testing.assert_array_equal(class_ids, [1, 1, 1])

    def test_empty_source_index(self):
        """Empty source_index returns empty array."""
        metadata = MockMetadata(
            class_labels=np.array([], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={},
        )
        class_ids = _build_class_ids([], metadata)
        assert len(class_ids) == 0


@pytest.mark.required
class TestOutliersPerClass:
    def test_per_class_requires_metadata(self):
        """per_class=True without metadata raises ValueError."""
        outliers = Outliers()
        with pytest.raises(ValueError, match="metadata must be provided"):
            outliers.evaluate(np.random.random((10, 3, 16, 16)), per_class=True)

    def test_per_class_basic(self):
        """per_class=True with IC dataset runs and returns valid output."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        # Class 0 (images 0-9): bright
        images[:10] = images[:10] * 0.3 + 0.7
        # Class 1 (images 10-19): dark
        images[10:] = images[10:] * 0.3
        # Make image 5 dark (outlier within class 0, but similar to class 1 globally)
        images[5] = 0.01

        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "bright", 1: "dark"},
        )

        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=1.5)
        result = outliers.evaluate(images, per_class=True, metadata=metadata)
        assert isinstance(result.data(), pl.DataFrame)
        # Image 5 should be detected as an outlier within class 0
        assert 5 in result.data()["item_index"].to_list()

    def test_per_class_false_unchanged(self):
        """Default per_class=False produces same result as before."""
        rng = np.random.default_rng(123)
        images = rng.random((20, 3, 16, 16))
        outliers = Outliers(flags=ImageStats.PIXEL)

        result_default = outliers.evaluate(images)
        result_explicit = outliers.evaluate(images, per_class=False)

        assert result_default.data().shape == result_explicit.data().shape
        assert result_default.data().equals(result_explicit.data())

    def test_per_class_small_class_no_crash(self):
        """Classes with very few samples don't crash."""
        rng = np.random.default_rng(99)
        images = rng.random((12, 3, 16, 16))
        # Class 0: 10 images, Class 1: 1 image, Class 2: 1 image
        labels = [0] * 10 + [1, 2]
        metadata = MockMetadata(
            class_labels=np.array(labels, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "a", 1: "b", 2: "c"},
        )
        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images, per_class=True, metadata=metadata)
        assert isinstance(result.data(), pl.DataFrame)

    def test_per_class_with_metadata_no_per_class(self):
        """metadata provided but per_class=False uses global detection."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "a", 1: "b"},
        )
        outliers = Outliers(flags=ImageStats.PIXEL)

        result_global = outliers.evaluate(images, per_class=False)
        result_with_metadata = outliers.evaluate(images, per_class=False, metadata=metadata)

        assert result_global.data().equals(result_with_metadata.data())

    def test_classwise_output_method(self):
        """classwise() on output produces per-class outliers."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        images[:10] = images[:10] * 0.3 + 0.7
        images[10:] = images[10:] * 0.3
        images[5] = 0.01

        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "bright", 1: "dark"},
        )

        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=1.5)
        result = outliers.evaluate(images)
        classwise_result = result.classwise(metadata)
        assert isinstance(classwise_result.data(), pl.DataFrame)
        assert 5 in classwise_result.data()["item_index"].to_list()

    def test_classwise_matches_evaluate_per_class(self):
        """classwise() produces same results as evaluate(per_class=True)."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        images[:10] = images[:10] * 0.3 + 0.7
        images[10:] = images[10:] * 0.3
        images[5] = 0.01

        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "bright", 1: "dark"},
        )

        outliers = Outliers(flags=ImageStats.PIXEL)
        via_evaluate = outliers.evaluate(images, per_class=True, metadata=metadata)
        via_output = outliers.evaluate(images).classwise(metadata)
        assert via_evaluate.data().equals(via_output.data())

    def test_itemwise_output_method(self):
        """itemwise() re-runs global detection from stored stats."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "a", 1: "b"},
        )

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images, per_class=True, metadata=metadata)
        global_result = result.itemwise()
        assert isinstance(global_result.data(), pl.DataFrame)

        # itemwise should match plain evaluate
        plain_result = outliers.evaluate(images)
        assert plain_result.data().equals(global_result.data())

    def test_classwise_requires_stored_stats(self):
        """classwise() raises when output has no stored stats."""
        output = OutliersOutput(
            pl.DataFrame(
                schema={
                    "item_index": pl.Int64,
                    "metric_name": pl.Categorical("lexical"),
                    "metric_value": pl.Float64,
                }
            )
        )
        metadata = MockMetadata(
            class_labels=np.array([0], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "a"},
        )
        with pytest.raises(ValueError, match="requires statistics"):
            output.classwise(metadata)

    def test_itemwise_requires_stored_stats(self):
        """itemwise() raises when output has no stored stats."""
        output = OutliersOutput(
            pl.DataFrame(
                schema={
                    "item_index": pl.Int64,
                    "metric_name": pl.Categorical("lexical"),
                    "metric_value": pl.Float64,
                }
            )
        )
        with pytest.raises(ValueError, match="requires statistics"):
            output.itemwise()

    def test_classwise_chaining_with_aggregation(self):
        """classwise() result supports aggregate methods."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        images[:10] = images[:10] * 0.3 + 0.7
        images[10:] = images[10:] * 0.3
        images[5] = 0.01

        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "bright", 1: "dark"},
        )

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images)
        classwise_result = result.classwise(metadata)

        # Can chain with aggregation methods
        by_class = classwise_result.aggregate_by_class(metadata)
        assert isinstance(by_class, pl.DataFrame)
        assert "class_name" in by_class.columns

        by_metric = classwise_result.aggregate_by_metric()
        assert isinstance(by_metric, pl.DataFrame)
        assert "metric_name" in by_metric.columns

    def test_from_stats_classwise(self):
        """classwise() works on output from from_stats()."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        images[:10] = images[:10] * 0.3 + 0.7
        images[10:] = images[10:] * 0.3
        images[5] = 0.01

        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "bright", 1: "dark"},
        )

        stats = compute_stats(images, stats=ImageStats.PIXEL)
        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.from_stats(stats)
        classwise_result = result.classwise(metadata)
        assert isinstance(classwise_result.data(), pl.DataFrame)

    def test_with_threshold_changes_results(self):
        """with_threshold() produces different outliers with different sensitivity."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        images[0] = 0.99  # extreme value

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images)

        strict = result.with_threshold(1.5)
        lenient = result.with_threshold(10.0)

        # Stricter threshold should flag at least as many outliers as lenient
        assert len(strict) >= len(lenient)

    def test_with_threshold_named(self):
        """with_threshold() accepts named threshold strings."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images)

        iqr_result = result.with_threshold("iqr")
        assert isinstance(iqr_result.data(), pl.DataFrame)

        zscore_result = result.with_threshold(("zscore", 2.5))
        assert isinstance(zscore_result.data(), pl.DataFrame)

    def test_with_threshold_per_metric(self):
        """with_threshold() accepts per-metric dict."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images)

        custom = result.with_threshold({"mean": 1.5, "brightness": ("zscore", 2.0)})
        assert isinstance(custom.data(), pl.DataFrame)

    def test_with_threshold_chained_with_classwise(self):
        """with_threshold() can be chained after classwise()."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        images[:10] = images[:10] * 0.3 + 0.7
        images[10:] = images[10:] * 0.3
        images[5] = 0.01

        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "bright", 1: "dark"},
        )

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images)
        chained = result.classwise(metadata).with_threshold(2.0)
        assert isinstance(chained.data(), pl.DataFrame)

    def test_with_threshold_preserves_stored_state(self):
        """with_threshold() result can be further re-detected."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images)

        # Chain: change threshold, then switch back to original via itemwise
        modified = result.with_threshold(2.0)
        back_to_global = modified.itemwise()
        assert isinstance(back_to_global.data(), pl.DataFrame)

    def test_with_threshold_requires_stored_stats(self):
        """with_threshold() raises when output has no stored stats."""
        output = OutliersOutput(
            pl.DataFrame(
                schema={
                    "item_index": pl.Int64,
                    "metric_name": pl.Categorical("lexical"),
                    "metric_value": pl.Float64,
                }
            )
        )
        with pytest.raises(ValueError, match="requires statistics"):
            output.with_threshold(2.0)

    def test_with_threshold_no_args_raises(self):
        """with_threshold() raises when no arguments provided."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images)
        with pytest.raises(ValueError, match="At least one"):
            result.with_threshold()

    def test_with_cluster_threshold_from_clusters(self):
        """with_threshold(cluster_threshold=X) works on from_clusters() output."""
        main_cluster = np.random.RandomState(42).randn(8, 5) * 0.5
        outlier_points = np.random.RandomState(42).randn(2, 5) * 2.0 + 5.0
        embeddings = np.vstack([main_cluster, outlier_points])

        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        detector = Outliers()
        result = detector.from_clusters(embeddings, mock_cluster_result, cluster_threshold=3.5)

        # Strict threshold should find more outliers
        strict = result.with_threshold(cluster_threshold=1.5)
        assert len(strict.data()) >= len(result.data())

        # Permissive threshold should find fewer or equal
        permissive = result.with_threshold(cluster_threshold=5.0)
        assert len(strict.data()) >= len(permissive.data())

    def test_with_cluster_threshold_evaluate(self):
        """with_threshold(cluster_threshold=X) works on evaluate() output with extractor."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))

        outliers = Outliers(flags=ImageStats.NONE, extractor=FlattenExtractor(), cluster_threshold=2.5)
        result = outliers.evaluate(images)

        # Should be able to adjust cluster threshold
        strict = result.with_threshold(cluster_threshold=1.0)
        permissive = result.with_threshold(cluster_threshold=5.0)
        assert len(strict.data()) >= len(permissive.data())

    def test_with_threshold_preserves_cluster_outliers(self):
        """with_threshold(outlier_threshold=X) preserves cluster outlier rows."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))

        outliers = Outliers(flags=ImageStats.PIXEL, extractor=FlattenExtractor(), cluster_threshold=2.0)
        result = outliers.evaluate(images)

        # Get original cluster rows
        original_cluster = result.data().filter(pl.col("metric_name") == "cluster_distance")

        # Change only outlier threshold — cluster rows should be preserved
        modified = result.with_threshold(outlier_threshold=10.0)
        modified_cluster = modified.data().filter(pl.col("metric_name") == "cluster_distance")

        assert original_cluster.shape == modified_cluster.shape
        assert original_cluster["item_index"].to_list() == modified_cluster["item_index"].to_list()

    def test_with_threshold_both_params(self):
        """with_threshold() accepts both outlier_threshold and cluster_threshold."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))

        outliers = Outliers(flags=ImageStats.PIXEL, extractor=FlattenExtractor(), cluster_threshold=2.5)
        result = outliers.evaluate(images)

        # Both params simultaneously
        modified = result.with_threshold(outlier_threshold=10.0, cluster_threshold=5.0)
        assert isinstance(modified.data(), pl.DataFrame)

        # Stats outliers should be fewer with lenient threshold
        original_stats = result.data().filter(pl.col("metric_name") != "cluster_distance")
        modified_stats = modified.data().filter(pl.col("metric_name") != "cluster_distance")
        assert len(original_stats) >= len(modified_stats)

    def test_with_cluster_threshold_chained_classwise(self):
        """result.classwise(metadata).with_threshold(cluster_threshold=X) works."""
        rng = np.random.default_rng(42)
        images = rng.random((20, 3, 16, 16))
        images[:10] = images[:10] * 0.3 + 0.7
        images[10:] = images[10:] * 0.3

        metadata = MockMetadata(
            class_labels=np.array([0] * 10 + [1] * 10, dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={0: "bright", 1: "dark"},
        )

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(images)

        # Should be chainable — cluster_threshold with no cluster stats is a no-op
        chained = result.classwise(metadata).with_threshold(cluster_threshold=2.0)
        assert isinstance(chained.data(), pl.DataFrame)

    def test_with_cluster_threshold_object(self):
        """cluster_threshold accepts Threshold objects (ThresholdLike)."""
        main_cluster = np.random.RandomState(42).randn(8, 5) * 0.5
        outlier_points = np.random.RandomState(42).randn(2, 5) * 2.0 + 5.0
        embeddings = np.vstack([main_cluster, outlier_points])

        mock_cluster_result: ClusterResult = {
            "clusters": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp),
            "mst": np.array([], dtype=np.float32),
            "linkage_tree": np.array([], dtype=np.float32),
            "membership_strengths": np.array([], dtype=np.float32),
            "k_neighbors": np.array([], dtype=np.int64),
            "k_distances": np.array([], dtype=np.float32),
        }

        # Use a Threshold object directly
        detector = Outliers(cluster_threshold=ZScoreThreshold(upper_multiplier=2.0, lower_multiplier=None))
        result = detector.from_clusters(embeddings, mock_cluster_result)
        assert isinstance(result.data(), pl.DataFrame)

        # Use IQR threshold via with_threshold
        detector2 = Outliers()
        result2 = detector2.from_clusters(embeddings, mock_cluster_result, cluster_threshold=3.5)
        iqr_result = result2.with_threshold(cluster_threshold=IQRThreshold(1.5))
        assert isinstance(iqr_result.data(), pl.DataFrame)

        # Use string shorthand
        str_result = result2.with_threshold(cluster_threshold="iqr")
        assert isinstance(str_result.data(), pl.DataFrame)

        # Use tuple shorthand
        tuple_result = result2.with_threshold(cluster_threshold=("zscore", 2.0))
        assert isinstance(tuple_result.data(), pl.DataFrame)


@pytest.mark.required
class TestOutliersMultiDataset:
    """Tests for multi-dataset outlier detection via evaluate(data, *other)."""

    def test_evaluate_multi_dataset_basic(self):
        """Evaluate with two datasets produces multi-dataset output."""
        data1 = np.random.random((50, 3, 16, 16))
        data2 = np.random.random((50, 3, 16, 16))
        data2[0] = 1.0  # Make one image an outlier

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(data1, data2)

        assert isinstance(result.data(), pl.DataFrame)
        assert "dataset_index" in result.data().columns
        assert "item_index" in result.data().columns

    def test_evaluate_multi_dataset_three_datasets(self):
        """Evaluate with three datasets."""
        data1 = np.random.random((30, 3, 16, 16))
        data2 = np.random.random((30, 3, 16, 16))
        data3 = np.random.random((30, 3, 16, 16))

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(data1, data2, data3)

        assert "dataset_index" in result.data().columns
        if len(result) > 0:
            ds_indices = result.data()["dataset_index"].unique().to_list()
            assert all(ds in [0, 1, 2] for ds in ds_indices)

    def test_evaluate_multi_dataset_outlier_detected(self):
        """Multi-dataset evaluate detects outlier in second dataset."""
        rng = np.random.default_rng(42)
        data1 = rng.random((50, 1, 16, 16)) * 0.5
        data2 = rng.random((50, 1, 16, 16)) * 0.5
        data2[0] = 1.0  # Extreme outlier

        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=ZScoreThreshold(2.0))
        result = outliers.evaluate(data1, data2)

        assert "dataset_index" in result.data().columns
        # The outlier should be in dataset 1, item 0
        ds1 = result.data().filter(pl.col("dataset_index") == 1)
        assert 0 in ds1["item_index"].to_list()

    def test_evaluate_multi_dataset_local_indices(self):
        """Multi-dataset item_index values are local to each dataset."""
        data1 = np.random.random((20, 3, 16, 16))
        data2 = np.random.random((30, 3, 16, 16))
        data2[0] = 1.0

        outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold=ZScoreThreshold(2.0))
        result = outliers.evaluate(data1, data2)

        if len(result) > 0:
            ds0 = result.data().filter(pl.col("dataset_index") == 0)
            ds1 = result.data().filter(pl.col("dataset_index") == 1)
            # Item indices should be local: < dataset size
            if ds0.shape[0] > 0:
                assert all(idx < 20 for idx in ds0["item_index"].to_list())
            if ds1.shape[0] > 0:
                assert all(idx < 30 for idx in ds1["item_index"].to_list())

    def test_evaluate_multi_dataset_with_threshold(self):
        """with_threshold works on multi-dataset evaluate output."""
        data1 = np.random.random((50, 3, 16, 16))
        data2 = np.random.random((50, 3, 16, 16))
        data2[0] = 1.0

        outliers = Outliers(flags=ImageStats.PIXEL)
        result = outliers.evaluate(data1, data2)

        strict = result.with_threshold(1.5)
        lenient = result.with_threshold(10.0)
        assert isinstance(strict.data(), pl.DataFrame)
        assert isinstance(lenient.data(), pl.DataFrame)
        assert len(strict) >= len(lenient)


@pytest.mark.required
class TestOutliersPropertySingleDataset:
    """Tests for the outliers property on single-dataset output."""

    def test_outliers_property_returns_dict(self):
        """outliers property returns dict[int, list[str]] for single-dataset."""
        data = np.random.random((50, 3, 16, 16))
        data[0] = 1.0  # Make outlier

        detector = Outliers(flags=ImageStats.PIXEL, outlier_threshold=ZScoreThreshold(2.0))
        result = detector.evaluate(data)

        outliers_map = result.outliers
        assert isinstance(outliers_map, dict)
        for item_idx, metrics in outliers_map.items():
            assert isinstance(item_idx, int)
            assert isinstance(metrics, list)
            assert all(isinstance(m, str) for m in metrics)

    def test_outliers_property_contains_detected_item(self):
        """outliers property includes the known outlier."""
        rng = np.random.default_rng(42)
        data = rng.random((50, 1, 16, 16)) * 0.5
        data[0] = 1.0

        detector = Outliers(flags=ImageStats.PIXEL, outlier_threshold=ZScoreThreshold(2.0))
        result = detector.evaluate(data)

        outliers_map = result.outliers
        assert 0 in outliers_map
        assert len(outliers_map[0]) > 0

    def test_outliers_property_empty_when_no_outliers(self):
        """outliers property returns empty dict when no outliers detected."""
        data = np.ones((20, 3, 16, 16)) * 0.5
        data += np.random.default_rng(0).random(data.shape) * 0.001

        detector = Outliers(flags=ImageStats.PIXEL, outlier_threshold=ZScoreThreshold(5.0))
        result = detector.evaluate(data)

        outliers_map = result.outliers
        assert isinstance(outliers_map, dict)
        assert len(outliers_map) == 0


@pytest.mark.required
class TestOutliersPropertyMultiDataset:
    """Tests for the outliers property on multi-dataset output."""

    def test_outliers_property_returns_nested_dict(self):
        """outliers property returns dict[int, dict[int, list[str]]] for multi-dataset."""
        data1 = np.random.random((50, 3, 16, 16))
        data2 = np.random.random((50, 3, 16, 16))
        data2[0] = 1.0

        detector = Outliers(flags=ImageStats.PIXEL, outlier_threshold=ZScoreThreshold(2.0))
        result = detector.evaluate(data1, data2)

        outliers_map = result.outliers
        assert isinstance(outliers_map, dict)
        for ds_idx, items in outliers_map.items():
            assert isinstance(ds_idx, int)
            assert isinstance(items, dict)
            for item_idx, metrics in items.items():
                assert isinstance(item_idx, int)
                assert isinstance(metrics, list)
                assert all(isinstance(m, str) for m in metrics)

    def test_outliers_property_contains_detected_item_in_dataset(self):
        """outliers property includes the known outlier in the correct dataset."""
        rng = np.random.default_rng(42)
        data1 = rng.random((50, 1, 16, 16)) * 0.5
        data2 = rng.random((50, 1, 16, 16)) * 0.5
        data2[0] = 1.0

        detector = Outliers(flags=ImageStats.PIXEL, outlier_threshold=ZScoreThreshold(2.0))
        result = detector.evaluate(data1, data2)

        outliers_map = result.outliers
        # The outlier should be in dataset 1 (index 1), item 0
        assert 1 in outliers_map
        assert 0 in outliers_map[1]
        assert len(outliers_map[1][0]) > 0

    def test_outliers_property_from_stats_multi(self):
        """outliers property works for multi-dataset from_stats."""
        data1 = np.zeros((50, 3, 16, 16))
        data2 = np.zeros((50, 3, 16, 16))
        data2[0] = 1.0

        stats1 = compute_stats(data1, stats=ImageStats.PIXEL)
        stats2 = compute_stats(data2, stats=ImageStats.PIXEL)

        detector = Outliers()
        result = detector.from_stats([stats1, stats2])

        outliers_map = result.outliers
        assert isinstance(outliers_map, dict)
        for ds_idx, items in outliers_map.items():
            assert isinstance(ds_idx, int)
            assert isinstance(items, dict)

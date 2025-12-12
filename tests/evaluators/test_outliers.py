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
            {"image_id": 1, "metric_name": "a", "metric_value": 1.0},
            {"image_id": 1, "metric_name": "b", "metric_value": 1.0},
            {"image_id": 3, "metric_name": "a", "metric_value": 1.0},
            {"image_id": 3, "metric_name": "b", "metric_value": 1.0},
            {"image_id": 5, "metric_name": "a", "metric_value": 1.0},
            {"image_id": 5, "metric_name": "b", "metric_value": 1.0},
        ]
    )
    outlier2 = pl.DataFrame(
        [
            {"image_id": 2, "metric_name": "a", "metric_value": 2.0},
            {"image_id": 2, "metric_name": "d", "metric_value": 2.0},
            {"image_id": 6, "metric_name": "a", "metric_value": 1.0},
            {"image_id": 6, "metric_name": "d", "metric_value": 1.0},
            {"image_id": 7, "metric_name": "a", "metric_value": 0.5},
            {"image_id": 7, "metric_name": "c", "metric_value": 0.5},
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
        assert all(count == 3 for count in result["count"].to_list())

        # Should be sorted by count (descending) then metric_name
        assert result["metric_name"].to_list() == ["a", "b"]

    def test_aggregate_by_metric_different_counts(self):
        """Test aggregate_by_metric with varying counts per metric"""
        df = pl.DataFrame(
            [
                {"image_id": 1, "metric_name": "contrast", "metric_value": 1.0},
                {"image_id": 2, "metric_name": "contrast", "metric_value": 1.0},
                {"image_id": 3, "metric_name": "depth", "metric_value": 1.0},
                {"image_id": 4, "metric_name": "skew", "metric_value": 1.0},
                {"image_id": 5, "metric_name": "skew", "metric_value": 1.0},
                {"image_id": 6, "metric_name": "skew", "metric_value": 1.0},
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_metric()

        # Should have 3 metrics
        assert result.shape[0] == 3

        # Check sorted order: skew (3), contrast (2), depth (1)
        assert result["metric_name"].to_list() == ["skew", "contrast", "depth"]
        assert result["count"].to_list() == [3, 2, 1]

    def test_aggregate_by_metric_raises_on_list(self):
        """Test aggregate_by_metric raises error for multiple datasets"""
        output = OutliersOutput([self.outlier, self.outlier2])
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            output.aggregate_by_metric()

    def test_aggregate_by_class(self):
        """Test aggregate_by_class returns correct pivot table"""
        metadata = make_mock_metadata(self.lstat)

        # Create outliers DataFrame matching the test data
        # image_id: 0,3,7=horse(0), 1,4,6,9=dog(1), 2,5,8=mule(2)
        df = pl.DataFrame(
            [
                {"image_id": 0, "metric_name": "contrast", "metric_value": 1.0},  # horse
                {"image_id": 1, "metric_name": "contrast", "metric_value": 1.0},  # dog
                {"image_id": 1, "metric_name": "depth", "metric_value": 1.0},  # dog
                {"image_id": 2, "metric_name": "depth", "metric_value": 1.0},  # mule
                {"image_id": 3, "metric_name": "skew", "metric_value": 1.0},  # horse
                {"image_id": 4, "metric_name": "skew", "metric_value": 1.0},  # dog
                {"image_id": 6, "metric_name": "contrast", "metric_value": 1.0},  # dog
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

    def test_aggregate_by_image(self):
        """Test aggregate_by_image returns correct pivot table"""
        # Create test data with known structure
        df = pl.DataFrame(
            [
                {"image_id": 0, "metric_name": "contrast", "metric_value": 1.0},
                {"image_id": 0, "metric_name": "depth", "metric_value": 1.0},
                {"image_id": 0, "metric_name": "skew", "metric_value": 1.0},
                {"image_id": 1, "metric_name": "contrast", "metric_value": 1.0},
                {"image_id": 2, "metric_name": "depth", "metric_value": 1.0},
                {"image_id": 2, "metric_name": "skew", "metric_value": 1.0},
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_image()

        # Check shape: 3 images, 4 columns (image_id + 3 metrics + Total)
        assert result.shape == (3, 5)

        # Check column names
        assert "image_id" in result.columns
        assert "Total" in result.columns
        assert set(result.columns) == {"image_id", "contrast", "depth", "skew", "Total"}

        # Verify image_id sorted by Total
        assert result["Total"].to_list() == [3, 2, 1]
        assert result["image_id"].to_list() == [0, 2, 1]

        # Check data types
        assert result["image_id"].dtype == pl.Int64
        for col in ["contrast", "depth", "skew", "Total"]:
            assert result[col].dtype == pl.UInt32

        # Verify binary indicators (0 or 1)
        for col in ["contrast", "depth", "skew"]:
            values = result[col].to_list()
            assert all(v in [0, 1] for v in values)

    def test_aggregate_by_image_sparse_metrics(self):
        """Test aggregate_by_image with images having different metrics"""
        df = pl.DataFrame(
            [
                {"image_id": 0, "metric_name": "a", "metric_value": 1.0},
                {"image_id": 0, "metric_name": "b", "metric_value": 1.0},
                {"image_id": 1, "metric_name": "c", "metric_value": 1.0},
                {"image_id": 2, "metric_name": "a", "metric_value": 1.0},
                {"image_id": 2, "metric_name": "c", "metric_value": 1.0},
            ]
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_image()

        # Check shape: 3 images, 5 columns (image_id + 3 metrics + Total)
        assert result.shape == (3, 5)

        # Check that missing combinations are 0
        # Image 0: has a, b (not c)
        row_0 = result.filter(pl.col("image_id") == 0)
        assert row_0["a"][0] == 1
        assert row_0["b"][0] == 1
        assert row_0["c"][0] == 0
        assert row_0["Total"][0] == 2

        # Image 1: has c (not a, b)
        row_1 = result.filter(pl.col("image_id") == 1)
        assert row_1["a"][0] == 0
        assert row_1["b"][0] == 0
        assert row_1["c"][0] == 1
        assert row_1["Total"][0] == 1

        # Image 2: has a, c (not b)
        row_2 = result.filter(pl.col("image_id") == 2)
        assert row_2["a"][0] == 1
        assert row_2["b"][0] == 0
        assert row_2["c"][0] == 1
        assert row_2["Total"][0] == 2

    def test_aggregate_by_image_empty(self):
        """Test aggregate_by_image with empty DataFrame"""
        df = pl.DataFrame(
            schema={"image_id": pl.Int64, "metric_name": pl.Categorical("lexical"), "metric_value": pl.Float64}
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_image()

        # Should return empty DataFrame with just image_id and Total columns
        assert result.shape[0] == 0
        assert "image_id" in result.columns
        assert "Total" in result.columns

    def test_aggregate_by_image_raises_on_list(self):
        """Test aggregate_by_image raises error for multiple datasets"""
        output = OutliersOutput([self.outlier, self.outlier2])
        with pytest.raises(ValueError, match="only works with output from a single dataset"):
            output.aggregate_by_image()

    def test_aggregate_by_metric_empty(self):
        """Test aggregate_by_metric with empty DataFrame"""
        df = pl.DataFrame(
            schema={"image_id": pl.Int64, "metric_name": pl.Categorical("lexical"), "metric_value": pl.Float64}
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_metric()

        # Should return empty DataFrame with correct schema
        assert result.shape[0] == 0
        assert "metric_name" in result.columns
        assert "count" in result.columns
        assert result["metric_name"].dtype == pl.Categorical("lexical")
        assert result["count"].dtype == pl.UInt32

    def test_aggregate_by_class_empty(self):
        """Test aggregate_by_class with empty DataFrame"""
        metadata = make_mock_metadata(self.lstat)

        df = pl.DataFrame(
            schema={"image_id": pl.Int64, "metric_name": pl.Categorical("lexical"), "metric_value": pl.Float64}
        )
        output = OutliersOutput(df)
        result = output.aggregate_by_class(metadata)

        # Should return empty DataFrame with correct schema
        assert result.shape[0] == 0
        assert "class_name" in result.columns
        assert "Total" in result.columns
        assert result["class_name"].dtype == pl.Categorical("lexical")
        assert result["Total"].dtype == pl.UInt32

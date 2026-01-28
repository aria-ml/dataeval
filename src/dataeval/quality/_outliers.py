__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._helpers import _get_index2label, _get_item_indices
from dataeval.config import EPSILON
from dataeval.core._calculate import CalculationResult, calculate
from dataeval.core._clusterer import ClusterResult, ClusterStats, cluster, compute_cluster_stats
from dataeval.flags import ImageStats
from dataeval.protocols import ArrayLike, Dataset, FeatureExtractor, Metadata
from dataeval.quality._results import StatsMap, combine_results, get_dataset_step_from_idx
from dataeval.types import ArrayND, ClusterConfigMixin, Evaluator, EvaluatorConfig, Output, SourceIndex, set_metadata
from dataeval.utils.arrays import flatten_samples, to_numpy

DEFAULT_OUTLIERS_FLAGS = ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
DEFAULT_OUTLIERS_OUTLIER_METHOD: Literal["zscore", "modzscore", "iqr"] = "modzscore"
DEFAULT_OUTLIERS_OUTLIER_THRESHOLD: float | None = None
DEFAULT_OUTLIERS_CLUSTER_THRESHOLD = 2.5

TDataFrame = TypeVar("TDataFrame", pl.DataFrame, Sequence[pl.DataFrame])


@dataclass(frozen=True)
class OutliersOutput(Output[TDataFrame]):
    """
    Output class for :class:`.Outliers` lint detector.

    Attributes
    ----------
    issues : pl.DataFrame | Sequence[pl.DataFrame]
        DataFrame of outlier issues with columns:
        - item_id: int - Index of the outlier image
        - target_id: int | None - Index of the target/detection within the image (None for image-level outliers).
        This column is omitted when all outliers are image-level (all target_id values would be None).
        - metric_name: str - Name of the metric that flagged this image/target
        - metric_value: float - Value of the metric for this image/target

    - For a single dataset, a single DataFrame
    - For multiple stats outputs, a sequence of DataFrames
    """

    issues: TDataFrame

    def data(self) -> TDataFrame:
        """Returns the underlying DataFrame(s)."""
        return self.issues

    def __len__(self) -> int:
        if isinstance(self.issues, pl.DataFrame):
            # Use target_id if present, otherwise just item_id
            cols = ["item_id", "target_id"] if "target_id" in self.issues.columns else ["item_id"]
            return self.issues.select(cols).n_unique()
        return sum(
            df.select(["item_id", "target_id"] if "target_id" in df.columns else ["item_id"]).n_unique()
            for df in self.issues
        )

    def aggregate_by_class(self, metadata: Metadata) -> pl.DataFrame:
        """
        Returns a Polars DataFrame summarizing outliers per class and metric.

        Creates a pivot table showing the count of outlier images for each combination
        of class and metric. Includes a Total row showing the total number of
        outliers per metric across all classes, and a Total column showing the total number
        of outliers per class across all metrics.

        Parameters
        ----------
        metadata : Metadata
            Metadata object containing class labels and image-to-class mappings for the dataset.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - class_name: cat - Name of the class
            - <metric_name>: int - Count of outliers for each metric (one column per metric)
            - Total: int - Total outlier count for the class across all metrics
            The last row is "Total" showing the sum across all classes for each metric.
            Rows are sorted by Total in descending order (excluding the Total row).

        Raises
        ------
        ValueError
            If the issues contain multiple DataFrames (from multiple datasets).

        Examples
        --------
        >>> from dataeval import Metadata
        >>> from dataeval.flags import ImageStats
        >>> from dataeval.quality import Outliers

        >>> outliers = Outliers(flags=ImageStats.VISUAL)
        >>> results = outliers.evaluate(dataset)
        >>> metadata = Metadata(dataset)
        >>> summary = results.aggregate_by_class(metadata)
        >>> summary
        shape: (4, 6)
        ┌────────────┬────────────┬──────────┬──────────┬───────────┬───────┐
        │ class_name ┆ brightness ┆ contrast ┆ darkness ┆ sharpness ┆ Total │
        │ ---        ┆ ---        ┆ ---      ┆ ---      ┆ ---       ┆ ---   │
        │ cat        ┆ u32        ┆ u32      ┆ u32      ┆ u32       ┆ u32   │
        ╞════════════╪════════════╪══════════╪══════════╪═══════════╪═══════╡
        │ plane      ┆ 5          ┆ 5        ┆ 6        ┆ 5         ┆ 21    │
        │ person     ┆ 5          ┆ 5        ┆ 5        ┆ 5         ┆ 20    │
        │ boat       ┆ 2          ┆ 2        ┆ 2        ┆ 2         ┆ 8     │
        │ Total      ┆ 12         ┆ 12       ┆ 13       ┆ 12        ┆ 49    │
        └────────────┴────────────┴──────────┴──────────┴───────────┴───────┘
        """
        # Handle the case where self.issues might be a list of DataFrames
        if not isinstance(self.issues, pl.DataFrame):
            raise ValueError("Aggregation by class only works with output from a single dataset.")

        schema: Any = {"class_name": pl.Categorical("lexical"), "Total": pl.UInt32}

        # Handle empty DataFrame case
        if self.issues.shape[0] == 0:
            return pl.DataFrame(schema=schema)

        item_ids = _get_item_indices(metadata)
        index2label = _get_index2label(metadata)
        class_names = [index2label[label] for label in metadata.class_labels]

        labels_df = pl.DataFrame({"item_id": item_ids, "class_name": class_names})

        # Join the Issues with the Labels
        joined_df = self.issues.join(labels_df, on="item_id", how="left")

        # Create the Summary Pivot (classes as rows, metrics as columns)
        summary_df = (
            joined_df.group_by(["class_name", "metric_name"])
            .len()  # Count occurrences
            .pivot(on="metric_name", index="class_name", values="len")
            .fill_null(0)
        )

        # Get metric columns (all columns except class_name)
        metric_cols = sorted([col for col in summary_df.columns if col != "class_name"])

        # Add a Total column (sum across all metrics for each class)
        if metric_cols:
            summary_df = summary_df.with_columns(pl.sum_horizontal(metric_cols).cast(pl.UInt32).alias("Total"))
        else:
            summary_df = summary_df.with_columns(pl.lit(0, dtype=pl.UInt32).alias("Total"))

        column_order = ["class_name"] + metric_cols + ["Total"]

        # Sort by Total in descending order
        summary_df = summary_df.select(column_order).sort(["Total", "class_name"], descending=[True, False])

        # Create a Total row (sum across all classes for each metric)
        total_row = (
            summary_df.select(pl.col(metric_cols + ["Total"]).sum())
            .with_columns(pl.lit("Total").alias("class_name"))
            .select(column_order)
        )

        # Concatenate the summary with the total row
        return pl.concat([summary_df, total_row]).cast(schema)

    def aggregate_by_metric(self) -> pl.DataFrame:
        """
        Returns a Polars DataFrame summarizing outlier counts per metric.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - metric_name: str - Name of the metric
            - Total: int - Number of images flagged by this metric

        Examples
        --------
        >>> outliers = Outliers(flags=ImageStats.PIXEL)
        >>> results = outliers.evaluate(dataset)
        >>> summary = results.aggregate_by_metric()
        >>> summary
        shape: (6, 2)
        ┌─────────────┬───────┐
        │ metric_name ┆ Total │
        │ ---         ┆ ---   │
        │ cat         ┆ u32   │
        ╞═════════════╪═══════╡
        │ zeros       ┆ 8     │
        │ entropy     ┆ 7     │
        │ mean        ┆ 4     │
        │ std         ┆ 4     │
        │ var         ┆ 4     │
        │ kurtosis    ┆ 1     │
        └─────────────┴───────┘
        """
        # Handle the case where self.issues might be a list of DataFrames
        if not isinstance(self.issues, pl.DataFrame):
            raise ValueError("Aggregation by metric only works with output from a single dataset.")

        # Handle empty DataFrame case
        if self.issues.shape[0] == 0:
            return pl.DataFrame(schema={"metric_name": pl.Categorical("lexical"), "Total": pl.UInt32})

        # Group by metric_name and count unique images
        return (
            self.issues.group_by("metric_name")
            .agg(pl.col("item_id").n_unique().alias("Total"))
            .sort(["Total", "metric_name"], descending=[True, False])
        )

    def aggregate_by_item(self) -> pl.DataFrame:
        """
        Returns a Polars DataFrame summarizing outliers per item (item_id, target_id pair) and metric.

        Creates a pivot table showing whether each item is flagged by each metric (1 if flagged, 0 if not).
        Includes a Total column showing the total number of metrics that flagged each item.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - item_id: int - Image identifier
            - target_id: int or None - Target identifier (Only with per_target outliers)
            - <metric_name>: int - Binary indicator (1 or 0) for each metric
            - count: int - Total number of metrics that flagged this item

        Raises
        ------
        ValueError
            If the issues contain multiple DataFrames (from multiple datasets).

        Examples
        --------
        >>> outliers = Outliers()
        >>> results = outliers.evaluate(dataset)
        >>> summary = results.aggregate_by_item()
        >>> summary
        shape: (20, 13)
        ┌─────────┬───────────┬────────────┬──────────┬───┬─────┬─────┬───────┬───────┐
        │ item_id ┆ target_id ┆ brightness ┆ contrast ┆ … ┆ std ┆ var ┆ zeros ┆ Total │
        │ ---     ┆ ---       ┆ ---        ┆ ---      ┆   ┆ --- ┆ --- ┆ ---   ┆ ---   │
        │ i64     ┆ i64       ┆ u32        ┆ u32      ┆   ┆ u32 ┆ u32 ┆ u32   ┆ u32   │
        ╞═════════╪═══════════╪════════════╪══════════╪═══╪═════╪═════╪═══════╪═══════╡
        │ 0       ┆ null      ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 1     ┆ 1     │
        │ 2       ┆ null      ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 1     ┆ 1     │
        │ 7       ┆ null      ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 7       ┆ 0         ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 11      ┆ null      ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ …       ┆ …         ┆ …          ┆ …        ┆ … ┆ …   ┆ …   ┆ …     ┆ …     │
        │ 34      ┆ null      ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 1     ┆ 1     │
        │ 36      ┆ 2         ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 0     ┆ 2     │
        │ 38      ┆ null      ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 1     ┆ 1     │
        │ 40      ┆ null      ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 1     ┆ 1     │
        │ 41      ┆ null      ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 1     ┆ 1     │
        └─────────┴───────────┴────────────┴──────────┴───┴─────┴─────┴───────┴───────┘
        """
        # Handle the case where self.issues might be a list of DataFrames
        if not isinstance(self.issues, pl.DataFrame):
            raise ValueError("Aggregation by item only works with output from a single dataset.")

        # Check if target_id column exists
        has_target_id = "target_id" in self.issues.columns

        index_cols = ["item_id", "target_id"] if has_target_id else ["item_id"]

        # Build schema for known types
        schema: Any = dict.fromkeys(index_cols, pl.Int64) | {"Total": pl.UInt32}

        # Handle empty DataFrame case
        if self.issues.shape[0] == 0:
            return pl.DataFrame(schema=schema)

        # Create a binary indicator for each (item_id, [target_id,] metric_name) combination
        # Group by item_id, [target_id,] and metric_name, then pivot

        grouped = (
            self.issues.group_by(index_cols + ["metric_name"])
            .agg(pl.len().alias("Total"))  # Count occurrences (should be 1 per combination)
            .with_columns(pl.lit(1, dtype=pl.UInt32).alias("flagged"))  # Create binary indicator
        )

        # Note: Polars 1.0.0 pivot cannot handle null values in index columns, so we use a placeholder
        TEMP_NULL_PLACEHOLDER = -1

        # Replace null target_id with placeholder before pivot (if target_id exists)
        if has_target_id:
            grouped = grouped.with_columns(pl.col("target_id").fill_null(TEMP_NULL_PLACEHOLDER))

        pivoted = grouped.pivot(on="metric_name", index=index_cols, values="flagged")

        # Get metric columns
        metric_cols = sorted([col for col in pivoted.columns if col not in index_cols])

        # Build expressions for columns
        expressions = []
        if has_target_id:
            expressions.append(
                pl.when(pl.col("target_id") == TEMP_NULL_PLACEHOLDER)
                .then(None)
                .otherwise(pl.col("target_id"))
                .alias("target_id")
            )

        if metric_cols:
            expressions.extend([pl.col(metric_cols).fill_null(0), pl.sum_horizontal(metric_cols).alias("Total")])
        else:
            expressions.append(pl.lit(0).alias("Total"))

        column_order = index_cols + metric_cols + ["Total"]

        return pivoted.with_columns(expressions).select(column_order).cast(schema).sort(index_cols)


def _get_zscore_mask(values: NDArray[np.float64], threshold: float | None) -> NDArray[np.bool_] | None:
    threshold = threshold if threshold is not None else 3.0
    std_val = np.nanstd(values)
    if std_val > EPSILON:
        mean_val = np.nanmean(values)
        abs_diff = np.abs(values - mean_val)
        return (abs_diff / std_val) > threshold
    return None


def _get_modzscore_mask(values: NDArray[np.float64], threshold: float | None) -> NDArray[np.bool_] | None:
    threshold = threshold if threshold is not None else 3.5
    median_val = np.nanmedian(values)
    abs_diff = np.abs(values - median_val)
    m_abs_diff = np.nanmedian(abs_diff)
    m_abs_diff = np.nanmean(abs_diff) if m_abs_diff <= EPSILON else m_abs_diff
    if m_abs_diff > EPSILON:
        mod_z_score = 0.6745 * abs_diff / m_abs_diff
        return mod_z_score > threshold
    return None


def _get_iqr_mask(values: NDArray[np.float64], threshold: float | None) -> NDArray[np.bool_] | None:
    threshold = threshold if threshold is not None else 1.5
    qrt = np.nanpercentile(values, q=(25, 75), method="midpoint")
    iqr_val = qrt[1] - qrt[0]
    if iqr_val > EPSILON:
        iqr_threshold = iqr_val * threshold
        return (values < (qrt[0] - iqr_threshold)) | (values > (qrt[1] + iqr_threshold))
    return None


def _get_outlier_mask(
    values: NDArray[Any], method: Literal["zscore", "modzscore", "iqr"], threshold: float | None
) -> NDArray[np.bool_]:
    if len(values) == 0:
        return np.array([], dtype=bool)

    nan_mask = np.isnan(values)

    if np.all(nan_mask):
        outliers = None
    elif method == "zscore":
        outliers = _get_zscore_mask(values.astype(np.float64), threshold)
    elif method == "modzscore":
        outliers = _get_modzscore_mask(values.astype(np.float64), threshold)
    elif method == "iqr":
        outliers = _get_iqr_mask(values.astype(np.float64), threshold)
    else:
        raise ValueError("Outlier method must be 'zscore' 'modzscore' or 'iqr'.")

    # If outliers were found, return the mask with NaN values set to False, otherwise return all False
    return outliers & ~nan_mask if outliers is not None else np.full(values.shape, False, dtype=bool)


class Outliers(Evaluator):
    r"""
    Calculates statistical outliers of a dataset using various statistical tests applied to each image.

    Supports two complementary detection methods:

    1. **Image statistics-based**: Computes pixel-level statistics (brightness, contrast, etc.)
       and flags images with unusual values using statistical methods (zscore, modzscore, iqr).
    2. **Cluster-based**: Uses embeddings from a neural network to cluster images and identifies
       outliers based on distance from cluster centers in embedding space.

    Both methods can be used together or independently based on the ``flags`` parameter.

    Parameters
    ----------
    flags : ImageStats, default ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
        Statistics to compute for image statistics-based outlier detection. Set to
        ``ImageStats.NONE`` to skip image statistics and use only cluster-based detection
        (requires ``feature_extractor``).
    outlier_method : ["modzscore" | "zscore" | "iqr"], default "modzscore"
        Statistical method used to identify outliers from image statistics.
    outlier_threshold : float, optional
        Threshold value for the given ``outlier_method``, above which data is considered an
        outlier. Uses method-specific default if None.
    feature_extractor : FeatureExtractor, optional
        Feature extractor for cluster-based outlier detection. When provided, embeddings
        are extracted and clustered to find semantic/visual outliers in embedding space.
        Common extractors include :class:`~dataeval.Embeddings`.
    cluster_threshold : float, default 2.5
        Number of standard deviations from cluster center beyond which a point is
        considered an outlier. Only used when ``feature_extractor`` is provided.
        Higher values are more permissive (fewer outliers).
    cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
        Clustering algorithm for cluster-based detection.
    n_clusters : int, optional
        Expected number of clusters. For HDBSCAN, this is a hint that adjusts
        min_cluster_size. For KMeans, this is the exact number of clusters.
    config : Outliers.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Attributes
    ----------
    stats : CalculationResult
        Statistics computed during the last evaluate() call.
        Contains dimension, pixel, and/or visual statistics based on the flags.
    flags : ImageStats
        Statistics to compute for outlier detection.
    outlier_method : Literal["zscore", "modzscore", "iqr"]
        Statistical method used to identify outliers.
    outlier_threshold : float | None
        Threshold value for the outlier method.
    feature_extractor : FeatureExtractor | None
        Feature extractor for cluster-based detection.
    cluster_threshold : float
        Threshold for cluster-based outlier detection.
    cluster_algorithm : Literal["kmeans", "hdbscan"]
        Clustering algorithm to use.
    n_clusters : int | None
        Expected number of clusters.

    See Also
    --------
    :term:`Duplicates`

    Notes
    -----
    **Image Statistics Methods:**

    - zscore: Based on difference from mean. Default threshold: 3.
      Z score = :math:`|x_i - \mu| / \sigma`
    - modzscore: Based on difference from median (robust to outliers). Default threshold: 3.5.
      Modified z score = :math:`0.6745 * |x_i - x̃| / MAD`
    - iqr: Based on interquartile range. Default threshold: 1.5.
      Outliers are outside :math:`[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]`

    **Cluster-based Detection:**

    Uses adaptive distance-based detection that accounts for varying cluster densities.
    Points are flagged as outliers if their distance from the nearest cluster center
    exceeds ``cluster_threshold`` standard deviations from the cluster's mean distance.

    Examples
    --------
    Basic image statistics-based outlier detection:

    >>> outliers = Outliers()
    >>> result = outliers.evaluate(dataset)

    Specifying an outlier method:

    >>> outliers = Outliers(outlier_method="iqr")

    Cluster-based detection with embeddings:

    >>> from dataeval import Embeddings
    >>> extractor = Embeddings(encoder=encoder)
    >>> outliers = Outliers(flags=ImageStats.NONE, feature_extractor=extractor)
    >>> result = outliers.evaluate(train_ds)  # Only cluster_distance metric

    Using configuration:

    >>> config = Outliers.Config(outlier_method="zscore", outlier_threshold=2.5)
    >>> outliers = Outliers(config=config)
    """

    class Config(EvaluatorConfig, ClusterConfigMixin):
        """
        Configuration for Outliers detector.

        Attributes
        ----------
        flags : ImageStats, default ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
            Statistics to compute for image statistics-based outlier detection.
        outlier_method : {"zscore", "modzscore", "iqr"}, default "modzscore"
            Statistical method used to identify outliers.
        outlier_threshold : float or None, default None
            Threshold value for the outlier method.
        cluster_threshold : float, default 2.5
            Number of standard deviations from cluster center for cluster-based detection.
        cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
            Clustering algorithm for cluster-based detection.
        n_clusters : int or None, default None
            Expected number of clusters.
        """

        flags: ImageStats = DEFAULT_OUTLIERS_FLAGS
        outlier_method: Literal["zscore", "modzscore", "iqr"] = DEFAULT_OUTLIERS_OUTLIER_METHOD
        outlier_threshold: float | None = DEFAULT_OUTLIERS_OUTLIER_THRESHOLD
        cluster_threshold: float = DEFAULT_OUTLIERS_CLUSTER_THRESHOLD

    stats: CalculationResult
    flags: ImageStats
    outlier_method: Literal["zscore", "modzscore", "iqr"]
    outlier_threshold: float | None
    cluster_threshold: float
    cluster_algorithm: Literal["kmeans", "hdbscan"]
    n_clusters: int | None
    config: Config
    feature_extractor: FeatureExtractor | None

    def __init__(
        self,
        flags: ImageStats | None = None,
        outlier_method: Literal["zscore", "modzscore", "iqr"] | None = None,
        outlier_threshold: float | None = None,
        cluster_threshold: float | None = None,
        cluster_algorithm: Literal["kmeans", "hdbscan"] | None = None,
        n_clusters: int | None = None,
        config: Config | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        super().__init__(locals())
        self.feature_extractor = feature_extractor

    def _get_outliers(self, stats: StatsMap, source_index: Sequence[SourceIndex]) -> pl.DataFrame:
        item_ids: list[int] = []
        target_ids: list[int | None] = []
        metric_names: list[str] = []
        metric_values: list[float] = []

        # Pre-compute masks for image-level vs target-level once (avoid recomputing per metric)
        if len(source_index) > 0:
            is_image_level = np.array([src_idx.target is None for src_idx in source_index], dtype=bool)
            is_target_level = ~is_image_level

            for stat, values in stats.items():
                if values.ndim == 1:
                    # Use boolean indexing instead of list comprehensions
                    image_level_mask_idx = is_image_level
                    target_level_mask_idx = is_target_level

                    # Calculate outliers separately for image-level metrics
                    if np.any(image_level_mask_idx):
                        image_level_values = values[image_level_mask_idx]
                        image_level_outlier_mask = _get_outlier_mask(
                            image_level_values.astype(np.float64), self.outlier_method, self.outlier_threshold
                        )

                        if np.any(image_level_outlier_mask):
                            # Get indices in original source_index where outliers were found
                            image_indices = np.flatnonzero(image_level_mask_idx)
                            outlier_indices = image_indices[image_level_outlier_mask]

                            # Batch append using list extension
                            item_ids.extend(source_index[idx].item for idx in outlier_indices)
                            target_ids.extend(source_index[idx].target for idx in outlier_indices)
                            n_outliers = len(outlier_indices)
                            metric_names.extend([stat] * n_outliers)
                            metric_values.extend(values[outlier_indices].tolist())

                    # Calculate outliers separately for target-level metrics
                    if np.any(target_level_mask_idx):
                        target_level_values = values[target_level_mask_idx]
                        target_level_outlier_mask = _get_outlier_mask(
                            target_level_values.astype(np.float64), self.outlier_method, self.outlier_threshold
                        )

                        if np.any(target_level_outlier_mask):
                            # Get indices in original source_index where outliers were found
                            target_indices = np.flatnonzero(target_level_mask_idx)
                            outlier_indices = target_indices[target_level_outlier_mask]

                            # Batch append using list extension
                            item_ids.extend(source_index[idx].item for idx in outlier_indices)
                            target_ids.extend(source_index[idx].target for idx in outlier_indices)
                            n_outliers = len(outlier_indices)
                            metric_names.extend([stat] * n_outliers)
                            metric_values.extend(values[outlier_indices].tolist())

        if not item_ids:
            return pl.DataFrame(
                schema={
                    "item_id": pl.Int64,
                    "target_id": pl.Int64,
                    "metric_name": pl.Categorical("lexical"),
                    "metric_value": pl.Float64,
                }
            )

        return pl.DataFrame(
            {
                "item_id": pl.Series(item_ids, dtype=pl.Int64),
                "target_id": pl.Series(target_ids, dtype=pl.Int64),
                "metric_name": pl.Series(metric_names, dtype=pl.Categorical("lexical")),
                "metric_value": pl.Series(metric_values, dtype=pl.Float64),
            }
        ).sort(["item_id", "target_id", "metric_name"])

    @overload
    def from_stats(self, stats: CalculationResult) -> OutliersOutput[pl.DataFrame]: ...

    @overload
    def from_stats(self, stats: Sequence[CalculationResult]) -> OutliersOutput[list[pl.DataFrame]]: ...

    @set_metadata(state=["outlier_method", "outlier_threshold"])
    def from_stats(
        self, stats: CalculationResult | Sequence[CalculationResult]
    ) -> OutliersOutput[pl.DataFrame] | OutliersOutput[list[pl.DataFrame]]:
        """
        Returns indices of Outliers with the issues identified for each.

        Parameters
        ----------
        stats : CalculationResult | Sequence[CalculationResult]
            The output(s) from calculate() with ImageStats.DIMENSION, PIXEL, or VISUAL flags

        Returns
        -------
        OutliersOutput
            Output class containing a DataFrame of outlier issues with columns:
            - item_id: int - Index of the outlier image
            - target_id: int | None - Index of the target within the image (None for image-level outliers)
            - metric_name: str - Name of the metric that flagged this image/target
            - metric_value: float - Value of the metric for this image/target

        Example
        -------
        Evaluate the dataset using pre-computed stats:

        >>> from dataeval.core import calculate
        >>> from dataeval.flags import ImageStats
        >>> stats = calculate(images, stats=ImageStats.PIXEL)
        >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=2.5)
        >>> results = outliers.from_stats(stats)
        >>> results.issues.head(10)
        shape: (10, 3)
        ┌─────────┬─────────────┬──────────────┐
        │ item_id ┆ metric_name ┆ metric_value │
        │ ---     ┆ ---         ┆ ---          │
        │ i64     ┆ cat         ┆ f64          │
        ╞═════════╪═════════════╪══════════════╡
        │ 7       ┆ entropy     ┆ 0.0          │
        │ 7       ┆ mean        ┆ 0.97998      │
        │ 7       ┆ std         ┆ 0.0          │
        │ 7       ┆ var         ┆ 0.0          │
        │ 8       ┆ skew        ┆ 0.062317     │
        │ 11      ┆ entropy     ┆ 0.0          │
        │ 11      ┆ mean        ┆ 0.97998      │
        │ 11      ┆ std         ┆ 0.0          │
        │ 11      ┆ var         ┆ 0.0          │
        │ 18      ┆ entropy     ┆ 0.0          │
        └─────────┴─────────────┴──────────────┘
        """
        combined_stats, dataset_steps = combine_results(stats)

        # Combine source_index from all stats
        if isinstance(stats, Sequence):
            combined_source_index: list[SourceIndex] = []
            for stat in stats:
                combined_source_index.extend(stat["source_index"])
        else:
            combined_source_index = list(stats["source_index"])

        outliers_df = self._get_outliers(combined_stats, combined_source_index)

        if not isinstance(stats, Sequence):
            # Drop target_id column if all values are None
            if "target_id" in outliers_df.columns and outliers_df["target_id"].null_count() == len(outliers_df):
                outliers_df = outliers_df.drop("target_id")
            return OutliersOutput(outliers_df)

        # Split results back to individual datasets
        output_list: list[pl.DataFrame] = []
        for dataset_idx in range(len(stats)):
            # Filter rows that belong to this dataset
            dataset_item_ids: list[int] = []
            dataset_target_ids: list[int | None] = []
            dataset_metric_names: list[str] = []
            dataset_metric_values: list[float] = []

            for row in outliers_df.iter_rows(named=True):
                k, v = get_dataset_step_from_idx(row["item_id"], dataset_steps)
                if k == dataset_idx:
                    dataset_item_ids.append(v)
                    dataset_target_ids.append(row["target_id"])
                    dataset_metric_names.append(row["metric_name"])
                    dataset_metric_values.append(row["metric_value"])

            if dataset_item_ids:
                dataset_df = pl.DataFrame(
                    {
                        "item_id": pl.Series(dataset_item_ids, dtype=pl.Int64),
                        "target_id": pl.Series(dataset_target_ids, dtype=pl.Int64),
                        "metric_name": pl.Series(dataset_metric_names, dtype=pl.Categorical("lexical")),
                        "metric_value": pl.Series(dataset_metric_values, dtype=pl.Float64),
                    }
                ).sort(["item_id", "target_id", "metric_name"], descending=[False, False, False])

                # Drop target_id column if all values are None
                if "target_id" in dataset_df.columns and dataset_df["target_id"].null_count() == len(dataset_df):
                    dataset_df = dataset_df.drop("target_id")

                output_list.append(dataset_df)
            else:
                output_list.append(
                    pl.DataFrame(
                        schema={
                            "item_id": pl.Int64,
                            "target_id": pl.Int64,
                            "metric_name": pl.Categorical("lexical"),
                            "metric_value": pl.Float64,
                        }
                    )
                )

        return OutliersOutput(output_list)

    @set_metadata(state=["outlier_threshold"])
    def from_clusters(
        self,
        embeddings: ArrayND[float],
        cluster_result: ClusterResult,
        threshold: float | None = None,
    ) -> OutliersOutput[pl.DataFrame]:
        """
        Find outliers using cluster-based adaptive distance detection.

        Identifies outliers based on their distance from cluster centers in
        embedding space. Points that are unusually far from their nearest
        cluster center are flagged as outliers. This method is particularly
        effective for finding semantic or visual outliers in image embeddings.

        Parameters
        ----------
        embeddings : ArrayND[float]
            The embedding vectors used for clustering, shape (n_samples, n_features).
            Should be the same embeddings passed to the cluster() function.
        cluster_result : ClusterResult
            Clustering results from the cluster() function, containing cluster
            assignments and related metadata.
        threshold : float, default=2.5
            Number of standard deviations beyond cluster mean to use for outlier
            threshold. Higher values are more permissive (fewer outliers), lower
            values are stricter (more outliers). Typical range: 1.5-3.5.

        Returns
        -------
        OutliersOutput[IndexIssueMap]
            Output containing outlier indices and their issue details. Each outlier
            includes:
            - 'cluster_distance': the distance from the cluster mean
            - 'std_devs': the number of standard deviations from the mean

        See Also
        --------
        dataeval.core.cluster : Function to compute clusters from embeddings
        dataeval.core.compute_cluster_stats : Computes statistics for adaptive detection
        from_stats : Find outliers from pre-computed image statistics
        evaluate : Find outliers by computing statistics from images

        Notes
        -----
        This method uses adaptive distance-based outlier detection that accounts
        for varying cluster densities. It significantly reduces false outliers
        compared to using HDBSCAN's binary -1 labels, especially for image
        embeddings with varying density distributions.

        The threshold parameter allows experimentation with different sensitivity
        levels without recomputing clusters. Recommended values:
        - 1.5-2.0: Very strict (many outliers)
        - 2.5: Balanced (default)
        - 3.0-3.5: Permissive (fewer outliers)
        """
        # Convert embeddings to numpy array and flatten if needed
        embeddings_array = flatten_samples(to_numpy(embeddings))

        # Compute cluster statistics
        cluster_stats = compute_cluster_stats(
            embeddings=embeddings_array,
            cluster_labels=cluster_result["clusters"],
        )

        # Find outliers using adaptive method
        outlier_issues = self._find_outliers_adaptive(
            cluster_stats=cluster_stats,
            threshold_std=threshold or self.outlier_threshold or 2.5,
        )

        return OutliersOutput(outlier_issues)

    def _find_outliers_adaptive(
        self,
        cluster_stats: ClusterStats,
        threshold_std: float,
    ) -> pl.DataFrame:
        """
        Find outliers using pre-calculated cluster statistics.

        This method uses pre-calculated cluster centers and distance statistics to identify
        outliers. Points are considered outliers if they are further than the
        threshold distance from their nearest cluster center.

        Parameters
        ----------
        cluster_stats : dict
            Pre-calculated cluster centers, distance statistics, and nearest cluster indices.
            Should contain keys: 'distances', 'nearest_cluster_idx', 'cluster_distances_mean',
            'cluster_distances_std'.
        threshold_std : float
            Number of standard deviations beyond cluster mean to use for threshold.
            Higher values are more permissive (fewer outliers), lower values are
            stricter (more outliers).

        Returns
        -------
        pl.DataFrame
            DataFrame with outlier details containing columns:
            - item_id: int - Index of the outlier
            - metric_name: str - Always "cluster_distance"
            - metric_value: float - Distance in std dev from cluster mean
        """
        # Get pre-calculated distances and nearest cluster indices
        min_distances = cluster_stats["distances"]
        nearest_cluster_idx = cluster_stats["nearest_cluster_idx"]
        cluster_distances_mean = cluster_stats["cluster_distances_mean"]
        cluster_distances_std = cluster_stats["cluster_distances_std"]

        # Compute thresholds on-the-fly based on the provided threshold_std
        thresholds = cluster_distances_mean + threshold_std * cluster_distances_std

        # Get the threshold for each point's nearest cluster
        nearest_thresholds = thresholds[nearest_cluster_idx]

        # Points are outliers if their distance exceeds the threshold of their nearest cluster
        is_outlier = min_distances > nearest_thresholds

        # Build the result DataFrame with issue details
        outlier_indices = np.nonzero(is_outlier)[0]

        if len(outlier_indices) == 0:
            return pl.DataFrame(
                schema={
                    "item_id": pl.Int64,
                    "metric_name": pl.Categorical("lexical"),
                    "metric_value": pl.Float64,
                }
            )

        item_ids: list[int] = []
        metric_values: list[float] = []

        for idx in outlier_indices:
            cluster_idx = nearest_cluster_idx[idx]
            distance = float(min_distances[idx])
            mean = float(cluster_distances_mean[cluster_idx])
            std = float(cluster_distances_std[cluster_idx])

            # Calculate number of standard deviations from mean
            std_devs = (distance - mean) / std if std > EPSILON else 0.0

            item_ids.append(int(idx))
            metric_values.append(std_devs)

        # Cluster-based detection is always image-level, so we don't include target_id
        return pl.DataFrame(
            {
                "item_id": pl.Series(item_ids, dtype=pl.Int64),
                "metric_name": pl.Series(["cluster_distance"] * len(item_ids), dtype=pl.Categorical("lexical")),
                "metric_value": pl.Series(metric_values, dtype=pl.Float64),
            }
        ).sort(["item_id", "metric_name"], descending=[False, False])

    @set_metadata(
        state=[
            "flags",
            "outlier_method",
            "outlier_threshold",
            "cluster_threshold",
            "cluster_algorithm",
            "n_clusters",
        ]
    )
    def evaluate(
        self,
        data: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
        *,
        per_image: bool = True,
        per_target: bool = True,
    ) -> OutliersOutput[pl.DataFrame]:
        """
        Returns indices of Outliers with the issues identified for each.

        Computes outliers using image statistics and/or cluster-based detection,
        depending on configuration. When both methods are enabled, results are
        combined into a single DataFrame.

        Parameters
        ----------
        data : Dataset[ArrayLike] or Dataset[tuple[ArrayLike, Any, Any]]
            Dataset of images in array format. Can be image-only dataset
            or dataset with additional tuple elements (labels, metadata).
            Images should be in standard array format (C, H, W).
        per_image : bool, default True
            Whether to compute statistics for full items (images/videos).
            When True, item-level outliers will be detected.
        per_target : bool, default True
            Whether to compute statistics for individual targets/detections.
            When True and targets are present, target-level outliers will be detected.
            Has no effect for datasets without targets or for cluster-based detection.

        Returns
        -------
        OutliersOutput
            Output class containing a DataFrame of outlier issues with columns:

            - item_id: int - Index of the outlier image
            - target_id: int | None - Index of the target within the image
              (None for image-level outliers, omitted if all are image-level)
            - metric_name: str - Name of the metric that flagged this image/target.
              Includes "cluster_distance" when feature_extractor is provided.
            - metric_value: float - Value of the metric for this image/target.
              For cluster_distance, this is the number of std devs from cluster mean.

        Raises
        ------
        ValueError
            If ``flags`` is ``ImageStats.NONE`` and no ``feature_extractor`` is provided.
            If both ``per_image`` and ``per_target`` are False.

        Examples
        --------
        Basic outlier detection:

        >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=2.5)
        >>> results = outliers.evaluate(images)
        >>> results.issues.head(10)
        shape: (10, 3)
        ┌─────────┬─────────────┬──────────────┐
        │ item_id ┆ metric_name ┆ metric_value │
        │ ---     ┆ ---         ┆ ---          │
        │ i64     ┆ cat         ┆ f64          │
        ╞═════════╪═════════════╪══════════════╡
        │ 7       ┆ brightness  ┆ 0.97998      │
        │ 7       ┆ contrast    ┆ 0.0          │
        │ 7       ┆ darkness    ┆ 0.97998      │
        │ 7       ┆ entropy     ┆ 0.0          │
        │ 7       ┆ mean        ┆ 0.97998      │
        │ 7       ┆ sharpness   ┆ 0.0          │
        │ 7       ┆ std         ┆ 0.0          │
        │ 7       ┆ var         ┆ 0.0          │
        │ 8       ┆ skew        ┆ 0.062317     │
        │ 11      ┆ brightness  ┆ 0.97998      │
        └─────────┴─────────────┴──────────────┘

        Cluster-based detection with embeddings:

        >>> from dataeval import Embeddings
        >>> extractor = Embeddings(encoder=encoder)
        >>> outliers = Outliers(flags=ImageStats.NONE, feature_extractor=extractor)
        >>> results = outliers.evaluate(train_ds)
        """
        # Validate parameters
        if self.flags == ImageStats.NONE and self.feature_extractor is None:
            raise ValueError("Either flags must not be ImageStats.NONE or feature_extractor must be provided.")

        if not (per_image or per_target):
            raise ValueError("At least one of per_image or per_target must be True.")

        outliers_dfs: list[pl.DataFrame] = []

        # Image statistics-based outlier detection
        if self.flags != ImageStats.NONE:
            self.stats = calculate(data, None, stats=self.flags, per_image=per_image, per_target=per_target)
            stats_outliers = self._get_outliers(self.stats["stats"], self.stats["source_index"])
            outliers_dfs.append(stats_outliers)

        # Cluster-based outlier detection
        if self.feature_extractor is not None:
            # Extract embeddings
            embeddings = self.feature_extractor(data)
            embeddings_array = flatten_samples(to_numpy(embeddings))

            # Cluster the embeddings
            cluster_result = cluster(
                embeddings_array,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            # Compute cluster statistics
            cluster_stats = compute_cluster_stats(
                embeddings=embeddings_array,
                cluster_labels=cluster_result["clusters"],
            )

            # Find cluster-based outliers
            cluster_outliers = self._find_outliers_adaptive(
                cluster_stats=cluster_stats,
                threshold_std=self.cluster_threshold,
            )
            outliers_dfs.append(cluster_outliers)

        # Merge results
        if len(outliers_dfs) == 0:
            # This shouldn't happen due to validation above, but handle gracefully
            outliers = pl.DataFrame(
                schema={
                    "item_id": pl.Int64,
                    "target_id": pl.Int64,
                    "metric_name": pl.Categorical("lexical"),
                    "metric_value": pl.Float64,
                }
            )
        elif len(outliers_dfs) == 1:
            outliers = outliers_dfs[0]
        else:
            # Ensure both DataFrames have target_id column for concatenation
            normalized_dfs: list[pl.DataFrame] = []
            for df in outliers_dfs:
                if "target_id" not in df.columns:
                    df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("target_id"))
                normalized_dfs.append(df)
            outliers = pl.concat(normalized_dfs).sort(["item_id", "target_id", "metric_name"])

        # Drop target_id column if there are no target-level stats (all target_id values are None)
        # This happens when per_target=False or when the dataset has no bounding boxes
        if "target_id" in outliers.columns and outliers["target_id"].null_count() == len(outliers):
            outliers = outliers.drop("target_id")

        return OutliersOutput(outliers)

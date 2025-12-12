from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.config import EPSILON
from dataeval.core._calculate import CalculationResult, calculate
from dataeval.core._clusterer import ClusterResult, ClusterStats, compute_cluster_stats
from dataeval.core.flags import ImageStats
from dataeval.data import Metadata
from dataeval.data._images import Images
from dataeval.protocols import ArrayLike, Dataset
from dataeval.types import ArrayND, Output, set_metadata
from dataeval.utils._array import flatten, to_numpy
from dataeval.utils._stats import StatsMap, combine_results, get_dataset_step_from_idx

TDataFrame = TypeVar("TDataFrame", pl.DataFrame, Sequence[pl.DataFrame])


@dataclass(frozen=True)
class OutliersOutput(Output[TDataFrame]):
    """
    Output class for :class:`.Outliers` lint detector.

    Attributes
    ----------
    issues : pl.DataFrame | Sequence[pl.DataFrame]
        DataFrame of outlier issues with columns:
        - image_id: int - Index of the outlier image
        - metric_name: str - Name of the metric that flagged this image
        - metric_value: float - Value of the metric for this image

    - For a single dataset, a single DataFrame
    - For multiple stats outputs, a sequence of DataFrames
    """

    issues: TDataFrame

    def data(self) -> TDataFrame:
        """Returns the underlying DataFrame(s)."""
        return self.issues

    def __len__(self) -> int:
        if isinstance(self.issues, pl.DataFrame):
            return self.issues["image_id"].n_unique()
        return sum(df["image_id"].n_unique() for df in self.issues)

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
        >>> outliers = Outliers()
        >>> results = outliers.evaluate(dataset)
        >>> metadata = Metadata(dataset)
        >>> summary = results.aggregate_by_class(metadata)
        >>> summary
        shape: (4, 10)
        ┌────────────┬──────────┬───────┬──────────┬───┬──────┬───────┬───────┬───────┐
        │ class_name ┆ contrast ┆ depth ┆ kurtosis ┆ … ┆ skew ┆ width ┆ zeros ┆ Total │
        │ ---        ┆ ---      ┆ ---   ┆ ---      ┆   ┆ ---  ┆ ---   ┆ ---   ┆ ---   │
        │ cat        ┆ u32      ┆ u32   ┆ u32      ┆   ┆ u32  ┆ u32   ┆ u32   ┆ u32   │
        ╞════════════╪══════════╪═══════╪══════════╪═══╪══════╪═══════╪═══════╪═══════╡
        │ chicken    ┆ 2        ┆ 1     ┆ 2        ┆ … ┆ 2    ┆ 1     ┆ 1     ┆ 11    │
        │ cow        ┆ 0        ┆ 0     ┆ 1        ┆ … ┆ 1    ┆ 1     ┆ 0     ┆ 5     │
        │ pig        ┆ 1        ┆ 1     ┆ 1        ┆ … ┆ 1    ┆ 0     ┆ 1     ┆ 5     │
        │ Total      ┆ 3        ┆ 2     ┆ 4        ┆ … ┆ 4    ┆ 2     ┆ 2     ┆ 21    │
        └────────────┴──────────┴───────┴──────────┴───┴──────┴───────┴───────┴───────┘
        """
        # Handle the case where self.issues might be a list of DataFrames
        if not isinstance(self.issues, pl.DataFrame):
            raise ValueError("Aggregation by class only works with output from a single dataset.")

        # Handle empty DataFrame case
        if self.issues.shape[0] == 0:
            return pl.DataFrame(
                {"class_name": pl.Series([], dtype=pl.Categorical("lexical")), "Total": pl.Series([], dtype=pl.UInt32)}
            )

        # Create mapping: image_index -> class_name
        mapping_data = {
            "image_id": metadata.item_indices.tolist(),
            "class_name": [metadata.index2label[label] for label in metadata.class_labels],
        }
        labels_df = pl.DataFrame(mapping_data)

        # Join the Issues with the Labels
        joined_df = self.issues.join(labels_df, on="image_id", how="left")

        # Create the Summary Pivot (classes as rows, metrics as columns)
        summary_df = (
            joined_df.group_by(["class_name", "metric_name"])
            .len()  # Count occurrences
            .pivot(on="metric_name", index="class_name", values="len")
            .fill_null(0)  # Replace NaNs with 0 for cleaner viewing
        )

        # Get metric columns (all columns except class_name)
        metric_cols = sorted([col for col in summary_df.columns if col != "class_name"])

        # Add a Total column (sum across all metrics for each class)
        if metric_cols:
            summary_df = summary_df.with_columns(pl.sum_horizontal(metric_cols).alias("Total"))
        else:
            summary_df = summary_df.with_columns(pl.lit(0, dtype=pl.UInt32).alias("Total"))

        # Sort by Total in descending order
        summary_df = summary_df.sort(["Total", "class_name"], descending=[True, False])

        # Create a Total row (sum across all classes for each metric)
        total_row_data = {"class_name": pl.Series(["Total"], dtype=pl.String)}
        for metric_col in metric_cols:
            total_row_data[metric_col] = pl.Series([summary_df[metric_col].sum()], dtype=pl.UInt32)
        total_row_data["Total"] = pl.Series([summary_df["Total"].sum()], dtype=pl.UInt32)

        total_row = pl.DataFrame(total_row_data)

        # Concatenate the summary with the total row
        column_order = ["class_name"] + metric_cols + ["Total"]
        return pl.concat([summary_df.select(column_order), total_row], how="vertical").cast(
            {"class_name": pl.Categorical("lexical")}
        )

    def aggregate_by_metric(self) -> pl.DataFrame:
        """
        Returns a Polars DataFrame summarizing outlier counts per metric.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - metric_name: str - Name of the metric
            - count: int - Number of images flagged by this metric

        Examples
        --------
        >>> outliers = Outliers()
        >>> results = outliers.evaluate(dataset)
        >>> summary = results.aggregate_by_metric()
        >>> summary
        shape: (8, 2)
        ┌─────────────┬───────┐
        │ metric_name ┆ count │
        │ ---         ┆ ---   │
        │ cat         ┆ u32   │
        ╞═════════════╪═══════╡
        │ contrast    ┆ 2     │
        │ kurtosis    ┆ 2     │
        │ skew        ┆ 2     │
        │ depth       ┆ 1     │
        │ sharpness   ┆ 1     │
        │ size        ┆ 1     │
        │ width       ┆ 1     │
        │ zeros       ┆ 1     │
        └─────────────┴───────┘
        """
        # Handle the case where self.issues might be a list of DataFrames
        if not isinstance(self.issues, pl.DataFrame):
            raise ValueError("Aggregation by metric only works with output from a single dataset.")

        # Handle empty DataFrame case
        if self.issues.shape[0] == 0:
            return pl.DataFrame(
                {"metric_name": pl.Series([], dtype=pl.Categorical("lexical")), "count": pl.Series([], dtype=pl.UInt32)}
            )

        # Group by metric_name and count unique images
        return (
            self.issues.group_by("metric_name")
            .agg(pl.col("image_id").n_unique().alias("count"))
            .sort(["count", "metric_name"], descending=[True, False])
        )

    def aggregate_by_image(self) -> pl.DataFrame:
        """
        Returns a Polars DataFrame summarizing outliers per image and metric.

        Creates a pivot table showing whether each image is flagged by each metric (1 if flagged, 0 if not).
        Includes a Total column showing the total number of metrics that flagged each image.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - image_id: int - Image identifier
            - <metric_name>: int - Binary indicator (1 or 0) for each metric
            - Total: int - Total number of metrics that flagged this image
            Rows are sorted by Total in descending order, then by image_id.

        Raises
        ------
        ValueError
            If the issues contain multiple DataFrames (from multiple datasets).

        Examples
        --------
        >>> outliers = Outliers()
        >>> results = outliers.evaluate(dataset)
        >>> summary = results.aggregate_by_image()
        >>> summary
        shape: (3, 10)
        ┌──────────┬──────────┬───────┬──────────┬───┬──────┬───────┬───────┬───────┐
        │ image_id ┆ contrast ┆ depth ┆ kurtosis ┆ … ┆ skew ┆ width ┆ zeros ┆ Total │
        │ ---      ┆ ---      ┆ ---   ┆ ---      ┆   ┆ ---  ┆ ---   ┆ ---   ┆ ---   │
        │ i64      ┆ u32      ┆ u32   ┆ u32      ┆   ┆ u32  ┆ u32   ┆ u32   ┆ u32   │
        ╞══════════╪══════════╪═══════╪══════════╪═══╪══════╪═══════╪═══════╪═══════╡
        │ 0        ┆ 1        ┆ 1     ┆ 1        ┆ … ┆ 1    ┆ 0     ┆ 1     ┆ 5     │
        │ 4        ┆ 0        ┆ 0     ┆ 1        ┆ … ┆ 1    ┆ 1     ┆ 0     ┆ 5     │
        │ 1        ┆ 1        ┆ 0     ┆ 0        ┆ … ┆ 0    ┆ 0     ┆ 0     ┆ 1     │
        └──────────┴──────────┴───────┴──────────┴───┴──────┴───────┴───────┴───────┘
        """
        # Handle the case where self.issues might be a list of DataFrames
        if not isinstance(self.issues, pl.DataFrame):
            raise ValueError("Aggregation by image only works with output from a single dataset.")

        # Handle empty DataFrame case
        if self.issues.shape[0] == 0:
            return pl.DataFrame({"image_id": pl.Series([], dtype=pl.Int64), "Total": pl.Series([], dtype=pl.UInt32)})

        # Create a binary indicator for each image-metric combination
        # Group by image_id and metric_name, then pivot
        summary_df = (
            self.issues.group_by(["image_id", "metric_name"])
            .agg(pl.len().alias("count"))  # Count occurrences (should be 1 per combination)
            .with_columns(pl.lit(1, dtype=pl.UInt32).alias("flagged"))  # Create binary indicator
            .pivot(on="metric_name", index="image_id", values="flagged")
            .fill_null(0)  # Replace NaNs with 0 for images not flagged by certain metrics
        )

        # Get metric columns (all columns except image_id)
        metric_cols = sorted([col for col in summary_df.columns if col != "image_id"])

        # Cast metric columns to UInt32 and add a Total column (sum across all metrics for each image)
        if metric_cols:
            summary_df = summary_df.with_columns([pl.col(col).cast(pl.UInt32) for col in metric_cols]).with_columns(
                pl.sum_horizontal(metric_cols).cast(pl.UInt32).alias("Total")
            )
        else:
            summary_df = summary_df.with_columns(pl.lit(0, dtype=pl.UInt32).alias("Total"))

        # Sort by Total, then by image_id
        summary_df = summary_df.sort(["Total", "image_id"], descending=[True, False])

        # Return with proper column ordering
        column_order = ["image_id"] + metric_cols + ["Total"]
        return summary_df.select(column_order)


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


class Outliers:
    r"""
    Calculates statistical outliers of a dataset using various statistical tests applied to each image.

    Parameters
    ----------
    use_dimension : bool, default True
        If True, use dimension statistics to identify outliers
    use_pixel : bool, default True
        If True, use pixel statistics to identify outliers
    use_visual : bool, default True
        If True, use visual statistics to identify outliers
    outlier_method : ["modzscore" | "zscore" | "iqr"], optional - default "modzscore"
        Statistical method used to identify outliers
    outlier_threshold : float, optional - default None
        Threshold value for the given ``outlier_method``, above which data is considered an
        outlier - uses method specific default if `None`

    Attributes
    ----------
    stats : CalculationResult
        Statistics computed during the last evaluate() call.
        Contains dimension, pixel, and/or visual statistics based on the use_* flags.
    use_dimension : bool
        Whether to use dimension statistics for outlier detection
    use_pixel : bool
        Whether to use pixel statistics for outlier detection
    use_visual : bool
        Whether to use visual statistics for outlier detection
    outlier_method : Literal["zscore", "modzscore", "iqr"]
        Statistical method used to identify outliers
    outlier_threshold : float | None
        Threshold value for the outlier method

    See Also
    --------
    :term:`Duplicates`

    Notes
    -----
    There are 3 different statistical methods:

    - zscore
    - modzscore
    - iqr

    | The z score method is based on the difference between the data point and the mean of the data.
        The default threshold value for `zscore` is 3.
    | Z score = :math:`|x_i - \mu| / \sigma`

    | The modified z score method is based on the difference between the data point and the median of the data.
        The default threshold value for `modzscore` is 3.5.
    | Modified z score = :math:`0.6745 * |x_i - x̃| / MAD`, where :math:`MAD` is the median absolute deviation

    | The interquartile range method is based on the difference between the data point and
        the difference between the 75th and 25th qartile. The default threshold value for `iqr` is 1.5.
    | Interquartile range = :math:`threshold * (Q_3 - Q_1)`

    Examples
    --------
    Initialize the Outliers class:

    >>> outliers = Outliers()

    Specifying an outlier method:

    >>> outliers = Outliers(outlier_method="iqr")

    Specifying an outlier method and threshold:

    >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=3.5)
    """

    def __init__(
        self,
        use_dimension: bool = True,
        use_pixel: bool = True,
        use_visual: bool = True,
        outlier_method: Literal["zscore", "modzscore", "iqr"] = "modzscore",
        outlier_threshold: float | None = None,
    ) -> None:
        self.stats: CalculationResult
        self.use_dimension = use_dimension
        self.use_pixel = use_pixel
        self.use_visual = use_visual
        self.outlier_method: Literal["zscore", "modzscore", "iqr"] = outlier_method
        self.outlier_threshold = outlier_threshold

    def _get_outliers(self, stats: StatsMap) -> pl.DataFrame:
        image_ids: list[int] = []
        metric_names: list[str] = []
        metric_values: list[float] = []

        for stat, values in stats.items():
            if values.ndim == 1:
                mask = _get_outlier_mask(values.astype(np.float64), self.outlier_method, self.outlier_threshold)
                indices = np.flatnonzero(mask)
                outlier_values = values[mask]

                image_ids.extend(indices.tolist())
                metric_names.extend([stat] * len(indices))
                metric_values.extend(outlier_values.tolist())

        if not image_ids:
            return pl.DataFrame(
                schema={"image_id": pl.Int64, "metric_name": pl.Categorical("lexical"), "metric_value": pl.Float64}
            )

        return pl.DataFrame(
            {
                "image_id": pl.Series(image_ids, dtype=pl.Int64),
                "metric_name": pl.Series(metric_names, dtype=pl.Categorical("lexical")),
                "metric_value": pl.Series(metric_values, dtype=pl.Float64),
            }
        ).sort(["image_id", "metric_name"], descending=[False, False])

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
            - image_id: int - Index of the outlier image
            - metric_name: str - Name of the metric that flagged this image
            - metric_value: float - Value of the metric for this image

        Example
        -------
        Evaluate the dataset:

        >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=3.5)
        >>> results = outliers.from_stats([stats1, stats2])
        >>> len(results)
        2
        >>> results.issues[0]
        shape: (6, 3)
        ┌──────────┬─────────────┬──────────────┐
        │ image_id ┆ metric_name ┆ metric_value │
        │ ---      ┆ ---         ┆ ---          │
        │ i64      ┆ cat         ┆ f64          │
        ╞══════════╪═════════════╪══════════════╡
        │ 10       ┆ entropy     ┆ 0.212769     │
        │ 10       ┆ zeros       ┆ 0.054932     │
        │ 12       ┆ entropy     ┆ 0.212769     │
        │ 12       ┆ std         ┆ 0.00536      │
        │ 12       ┆ var         ┆ 0.000029     │
        │ 12       ┆ zeros       ┆ 0.054932     │
        └──────────┴─────────────┴──────────────┘
        """
        combined_stats, dataset_steps = combine_results(stats)
        outliers_df = self._get_outliers(combined_stats)

        if not isinstance(stats, Sequence):
            return OutliersOutput(outliers_df)

        # Split results back to individual datasets
        output_list: list[pl.DataFrame] = []
        for dataset_idx in range(len(stats)):
            # Filter rows that belong to this dataset
            dataset_image_ids: list[int] = []
            dataset_metric_names: list[str] = []
            dataset_metric_values: list[float] = []

            for row in outliers_df.iter_rows(named=True):
                k, v = get_dataset_step_from_idx(row["image_id"], dataset_steps)
                if k == dataset_idx:
                    dataset_image_ids.append(v)
                    dataset_metric_names.append(row["metric_name"])
                    dataset_metric_values.append(row["metric_value"])

            if dataset_image_ids:
                output_list.append(
                    pl.DataFrame(
                        {
                            "image_id": pl.Series(dataset_image_ids, dtype=pl.Int64),
                            "metric_name": pl.Series(dataset_metric_names, dtype=pl.Categorical("lexical")),
                            "metric_value": pl.Series(dataset_metric_values, dtype=pl.Float64),
                        }
                    ).sort(["image_id", "metric_name"], descending=[False, False])
                )
            else:
                output_list.append(
                    pl.DataFrame(
                        schema={
                            "image_id": pl.Int64,
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
        embeddings_array = flatten(to_numpy(embeddings))

        # Compute cluster statistics
        cluster_stats = compute_cluster_stats(
            embeddings=embeddings_array,
            clusters=cluster_result["clusters"],
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
            - image_id: int - Index of the outlier
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
                schema={"image_id": pl.Int64, "metric_name": pl.Categorical("lexical"), "metric_value": pl.Float64}
            )

        image_ids: list[int] = []
        metric_values: list[float] = []

        for idx in outlier_indices:
            cluster_idx = nearest_cluster_idx[idx]
            distance = float(min_distances[idx])
            mean = float(cluster_distances_mean[cluster_idx])
            std = float(cluster_distances_std[cluster_idx])

            # Calculate number of standard deviations from mean
            std_devs = (distance - mean) / std if std > EPSILON else 0.0

            image_ids.append(int(idx))
            metric_values.append(std_devs)

        return pl.DataFrame(
            {
                "image_id": pl.Series(image_ids, dtype=pl.Int64),
                "metric_name": pl.Series(["cluster_distance"] * len(image_ids), dtype=pl.Categorical("lexical")),
                "metric_value": pl.Series(metric_values, dtype=pl.Float64),
            }
        ).sort(["image_id", "metric_name"], descending=[False, False])

    @set_metadata(state=["use_dimension", "use_pixel", "use_visual", "outlier_method", "outlier_threshold"])
    def evaluate(self, data: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]) -> OutliersOutput[pl.DataFrame]:
        """
        Returns indices of Outliers with the issues identified for each.

        Computes statistical outliers by calculating dimension, pixel, and/or
        visual statistics for the dataset, then applying the configured outlier
        detection method. Stores computed statistics in the stats attribute.

        Parameters
        ----------
        data : Dataset[ArrayLike] or Dataset[tuple[ArrayLike, Any, Any]]
            Dataset of images in array format. Can be image-only dataset
            or dataset with additional tuple elements (labels, metadata).
            Images should be in standard array format (C, H, W).

        Returns
        -------
        OutliersOutput
            Output class containing the indices of outliers and a dictionary showing
            the issues and calculated values for the given index.

        Examples
        --------
        Basic outlier detection:

        >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=3.5)
        >>> results = outliers.evaluate(outlier_images)
        >>> results.issues
        shape: (9, 3)
        ┌──────────┬─────────────┬──────────────┐
        │ image_id ┆ metric_name ┆ metric_value │
        │ ---      ┆ ---         ┆ ---          │
        │ i64      ┆ cat         ┆ f64          │
        ╞══════════╪═════════════╪══════════════╡
        │ 10       ┆ contrast    ┆ 1.25         │
        │ 10       ┆ entropy     ┆ 0.212769     │
        │ 10       ┆ zeros       ┆ 0.054932     │
        │ 12       ┆ contrast    ┆ 1.25         │
        │ 12       ┆ entropy     ┆ 0.212769     │
        │ 12       ┆ sharpness   ┆ 1.509766     │
        │ 12       ┆ std         ┆ 0.00536      │
        │ 12       ┆ var         ┆ 0.000029     │
        │ 12       ┆ zeros       ┆ 0.054932     │
        └──────────┴─────────────┴──────────────┘

        Access computed statistics for reuse:

        >>> saved_stats = outliers.stats
        """
        if not (self.use_dimension or self.use_pixel or self.use_visual):
            raise ValueError("At least one of use_dimension, use_pixel or use_visual must be True.")

        # Build flags for requested statistics
        flags = ImageStats(0)  # Start with no flags
        if self.use_dimension:
            flags |= ImageStats.DIMENSION
        if self.use_pixel:
            flags |= ImageStats.PIXEL
        if self.use_visual:
            flags |= ImageStats.VISUAL

        images = Images(data) if isinstance(data, Dataset) else data

        self.stats: CalculationResult = calculate(images, None, flags)
        outliers = self._get_outliers(self.stats["stats"])
        return OutliersOutput(outliers)

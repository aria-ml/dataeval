__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval import Embeddings
from dataeval._helpers import _get_index2label, _get_item_indices
from dataeval.core import (
    ClusterResult,
    ClusterStats,
    StatsResult,
    cluster,
    combine_stats_results,
    compute_cluster_stats,
    compute_stats,
)
from dataeval.flags import ImageStats
from dataeval.protocols import ArrayLike, Dataset, FeatureExtractor, MetadataLike, Threshold, ThresholdLike
from dataeval.quality._shared import add_dataset_index, drop_null_index_columns
from dataeval.types import (
    ArrayND,
    ClusterConfigMixin,
    DataFrameOutput,
    Evaluator,
    EvaluatorConfig,
    SourceIndex,
    StatsMap,
    set_metadata,
)
from dataeval.utils._internal import EPSILON, flatten_samples, to_numpy
from dataeval.utils.thresholds import AdaptiveThreshold, ZScoreThreshold, resolve_threshold

DEFAULT_OUTLIERS_FLAGS = ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
DEFAULT_OUTLIERS_CLUSTER_THRESHOLD: Threshold = ZScoreThreshold(upper_multiplier=2.5, lower_multiplier=None)
DEFAULT_OUTLIERS_OUTLIER_THRESHOLD: Threshold = AdaptiveThreshold()

OutlierReasons = Sequence[str]
SingleOutliersMap = Mapping[int, Sequence[str]]
SingleTargetOutliersMap = Mapping[SourceIndex, Sequence[str]]
MultiOutliersMap = Mapping[int, Mapping[int, Sequence[str]]]
MultiTargetOutliersMap = Mapping[int, Mapping[SourceIndex, Sequence[str]]]
TOutliers = TypeVar("TOutliers", SingleOutliersMap, SingleTargetOutliersMap, MultiOutliersMap, MultiTargetOutliersMap)


class OutliersOutput(DataFrameOutput, Generic[TOutliers]):
    """
    Output class for :class:`.Outliers` lint detector.

    DataFrame of outlier issues with columns:

    - dataset_index: int - Index of the originating dataset (only present for multi-dataset output)
    - item_index: int - Index of the outlier item (local to each dataset)
    - target_index: int | None - Index of the target/detection within the item (None for item-level outliers).
      This column is omitted when all outliers are item-level (all target_index values would be None).
    - channel_index: int | None - Index of the image channel (None for aggregated stats).
      This column is omitted when all stats are aggregated across channels.
    - metric_name: str - Name of the metric that flagged this image/target
    - metric_value: float - Value of the metric for this image/target

    Attributes
    ----------
    calculation_results : StatsResult or Sequence[StatsResult] or None
        The original calculation result(s) from :func:`compute_stats`. Used internally
        for re-detection via :meth:`classwise`, :meth:`itemwise`, and :meth:`with_threshold`.
    outlier_threshold : ThresholdLike or dict or None
        Threshold configuration used to detect outliers. Preserved across
        re-detection calls unless overridden via :meth:`with_threshold`.
    cluster_stats : ClusterStats or None
        Pre-computed cluster statistics for cluster-based outlier re-detection
        via :meth:`with_threshold`.
    cluster_threshold : ThresholdLike or None
        Threshold configuration used for cluster-based outlier detection.
        Preserved across re-detection calls unless overridden via
        :meth:`with_threshold`.
    dataset_steps : list[int] or None
        Cumulative dataset boundaries for multi-dataset index remapping.
        None for single-dataset output.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        calculation_results: StatsResult | Sequence[StatsResult] | None = None,
        outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None = None,
        cluster_stats: ClusterStats | None = None,
        cluster_threshold: ThresholdLike | None = None,
        dataset_steps: Sequence[int] | None = None,
    ) -> None:
        super().__init__(data)
        self.calculation_results = calculation_results
        self.outlier_threshold = outlier_threshold
        self.cluster_stats = cluster_stats
        self.cluster_threshold = cluster_threshold
        self.dataset_steps = dataset_steps

    @property
    def _combined(self) -> tuple[StatsMap, list[SourceIndex], list[int]] | None:
        """Extract combined stats, source_index, and dataset_steps from stored calculation_results."""
        if self.calculation_results is None:
            return None
        return combine_stats_results(self.calculation_results)

    def __len__(self) -> int:
        cols = ["item_index"]
        if "dataset_index" in self.data().columns:
            cols.insert(0, "dataset_index")
        if "target_index" in self.data().columns:
            cols.append("target_index")
        return self.data().select(cols).n_unique()

    # ------------------------------------------------------------------
    # Convenience accessor
    # ------------------------------------------------------------------

    @property
    def outliers(self) -> TOutliers:  # noqa: C901
        """Outlier items as a mapping of index to flagged metric names.

        When ``per_target=False``:

            - Single-dataset: ``dict[int, list[str]]`` keyed by ``item_index``
            - Multi-dataset: ``dict[int, dict[int, list[str]]]`` outer key is ``dataset_index``

        When ``per_target=True``:

            - Single-dataset: ``dict[SourceIndex, list[str]]`` keyed by :class:`SourceIndex`
            - Multi-dataset: ``dict[int, dict[SourceIndex, list[str]]]`` outer key is ``dataset_index``
        """
        df = self.data()
        is_cross = "dataset_index" in df.columns
        has_targets = "target_index" in df.columns and df["target_index"].null_count() < len(df)

        if is_cross:
            if has_targets:
                result_cross_target: dict[int, dict[SourceIndex, list[str]]] = {}
                for row in df.iter_rows(named=True):
                    si = SourceIndex(row["item_index"], row["target_index"])
                    result_cross_target.setdefault(row["dataset_index"], {}).setdefault(si, []).append(
                        row["metric_name"]
                    )
                return result_cross_target  # type: ignore[return-value]
            result_cross: dict[int, dict[int, list[str]]] = {}
            for row in df.iter_rows(named=True):
                result_cross.setdefault(row["dataset_index"], {}).setdefault(row["item_index"], []).append(
                    row["metric_name"]
                )
            return result_cross  # type: ignore[return-value]

        if has_targets:
            result_target: dict[SourceIndex, list[str]] = {}
            for row in df.iter_rows(named=True):
                si = SourceIndex(row["item_index"], row["target_index"])
                result_target.setdefault(si, []).append(row["metric_name"])
            return result_target  # type: ignore[return-value]

        result_single: dict[int, list[str]] = {}
        for row in df.iter_rows(named=True):
            result_single.setdefault(row["item_index"], []).append(row["metric_name"])
        return result_single  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

    def aggregate_by_class(self, metadata: MetadataLike) -> pl.DataFrame:
        """
        Return a Polars DataFrame summarizing outliers per class and metric.

        Creates a pivot table showing the count of outlier images for each combination
        of class and metric. Includes a Total row showing the total number of
        outliers per metric across all classes, and a Total column showing the total number
        of outliers per class across all metrics.

        Parameters
        ----------
        metadata : MetadataLike
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

        >>> outliers = Outliers(flags=ImageStats.VISUAL, outlier_threshold="modzscore")
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
        │ person     ┆ 2          ┆ 2        ┆ 2        ┆ 2         ┆ 8     │
        │ plane      ┆ 2          ┆ 2        ┆ 2        ┆ 2         ┆ 8     │
        │ boat       ┆ 1          ┆ 1        ┆ 1        ┆ 1         ┆ 4     │
        │ Total      ┆ 5          ┆ 5        ┆ 5        ┆ 5         ┆ 20    │
        └────────────┴────────────┴──────────┴──────────┴───────────┴───────┘
        """
        if "dataset_index" in self.data().columns:
            raise ValueError("Aggregation by class only works with output from a single dataset.")

        schema: Any = {"class_name": pl.Categorical("lexical"), "Total": pl.UInt32}

        # Handle empty DataFrame case
        if self.data().shape[0] == 0:
            return pl.DataFrame(schema=schema)

        item_ids = _get_item_indices(metadata)
        index2label = _get_index2label(metadata)
        class_names = [index2label[label] for label in metadata.class_labels]

        labels_df = pl.DataFrame({"item_index": item_ids, "class_name": class_names})

        # Join the Issues with the Labels
        joined_df = self.data().join(labels_df, on="item_index", how="left")

        # Create the Summary Pivot (classes as rows, metrics as columns)
        summary_df = (
            joined_df
            .group_by(["class_name", "metric_name"])
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
            summary_df
            .select(pl.col(metric_cols + ["Total"]).sum())
            .with_columns(pl.lit("Total").alias("class_name"))
            .select(column_order)
        )

        # Concatenate the summary with the total row
        return pl.concat([summary_df, total_row]).cast(schema)

    def aggregate_by_metric(self) -> pl.DataFrame:
        """
        Return a Polars DataFrame summarizing outlier counts per metric.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            - metric_name: str - Name of the metric
            - Total: int - Number of images flagged by this metric

        Examples
        --------
        >>> outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold="zscore")
        >>> results = outliers.evaluate(dataset)
        >>> summary = results.aggregate_by_metric()
        >>> summary
        shape: (4, 2)
        ┌─────────────┬───────┐
        │ metric_name ┆ Total │
        │ ---         ┆ ---   │
        │ cat         ┆ u32   │
        ╞═════════════╪═══════╡
        │ entropy     ┆ 4     │
        │ mean        ┆ 4     │
        │ std         ┆ 4     │
        │ var         ┆ 4     │
        └─────────────┴───────┘
        """
        if "dataset_index" in self.data().columns:
            raise ValueError("Aggregation by metric only works with output from a single dataset.")

        # Handle empty DataFrame case
        if self.data().shape[0] == 0:
            return pl.DataFrame(schema={"metric_name": pl.Categorical("lexical"), "Total": pl.UInt32})

        # Group by metric_name and count unique images
        return (
            self
            .data()
            .group_by("metric_name")
            .agg(pl.col("item_index").n_unique().alias("Total"))
            .sort(["Total", "metric_name"], descending=[True, False])
        )

    def aggregate_by_item(self) -> pl.DataFrame:  # noqa: C901
        """
        Return a Polars DataFrame summarizing outliers per item (item_index, target_index pair) and metric.

        Creates a pivot table showing whether each item is flagged by each metric (1 if flagged, 0 if not).
        Includes a Total column showing the total number of metrics that flagged each item.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            - item_index: int - Item identifier
            - target_index: int or None - Target identifier (Only with per_target outliers)
            - <metric_name>: int - Binary indicator (1 or 0) for each metric
            - count: int - Total number of metrics that flagged this item

        Raises
        ------
        ValueError
            If the issues contain multiple DataFrames (from multiple datasets).

        Examples
        --------
        >>> outliers = Outliers(outlier_threshold=("modzscore", 3.0))
        >>> results = outliers.evaluate(dataset, per_target=True)
        >>> summary = results.aggregate_by_item()
        >>> summary.head(10)
        shape: (10, 14)
        ┌────────────┬──────────────┬────────────┬──────────┬───┬─────┬─────┬───────┬───────┐
        │ item_index ┆ target_index ┆ brightness ┆ contrast ┆ … ┆ std ┆ var ┆ zeros ┆ Total │
        │ ---        ┆ ---          ┆ ---        ┆ ---      ┆   ┆ --- ┆ --- ┆ ---   ┆ ---   │
        │ i64        ┆ i64          ┆ u32        ┆ u32      ┆   ┆ u32 ┆ u32 ┆ u32   ┆ u32   │
        ╞════════════╪══════════════╪════════════╪══════════╪═══╪═════╪═════╪═══════╪═══════╡
        │ 0          ┆ null         ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 1     ┆ 1     │
        │ 2          ┆ null         ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 1     ┆ 1     │
        │ 7          ┆ null         ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 7          ┆ 0            ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 11         ┆ null         ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 11         ┆ 0            ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 18         ┆ null         ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 18         ┆ 0            ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 18         ┆ 1            ┆ 1          ┆ 1        ┆ … ┆ 1   ┆ 1   ┆ 0     ┆ 8     │
        │ 19         ┆ 0            ┆ 0          ┆ 0        ┆ … ┆ 0   ┆ 0   ┆ 0     ┆ 2     │
        └────────────┴──────────────┴────────────┴──────────┴───┴─────┴─────┴───────┴───────┘
        """
        if "dataset_index" in self.data().columns:
            raise ValueError("Aggregation by item only works with output from a single dataset.")

        # Check if target_index column exists
        has_target_id = "target_index" in self.data().columns

        index_cols = ["item_index", "target_index"] if has_target_id else ["item_index"]

        # Build schema for known types
        schema: Any = dict.fromkeys(index_cols, pl.Int64) | {"Total": pl.UInt32}

        # Handle empty DataFrame case
        if self.data().shape[0] == 0:
            return pl.DataFrame(schema=schema)

        # Create a binary indicator for each (item_index, [target_index,] metric_name) combination
        # Group by item_index, [target_index,] and metric_name, then pivot

        grouped = (
            self
            .data()
            .group_by(index_cols + ["metric_name"])
            .agg(pl.len().alias("Total"))  # Count occurrences (should be 1 per combination)
            .with_columns(pl.lit(1, dtype=pl.UInt32).alias("flagged"))  # Create binary indicator
        )

        # Note: Polars 1.0.0 pivot cannot handle null values in index columns, so we use a placeholder
        temp_null_placeholder = -1

        # Replace null target_index with placeholder before pivot (if target_index exists)
        if has_target_id:
            grouped = grouped.with_columns(pl.col("target_index").fill_null(temp_null_placeholder))

        pivoted = grouped.pivot(on="metric_name", index=index_cols, values="flagged")

        # Get metric columns
        metric_cols = sorted([col for col in pivoted.columns if col not in index_cols])

        # Build expressions for columns
        expressions = []
        if has_target_id:
            expressions.append(
                pl
                .when(pl.col("target_index") == temp_null_placeholder)
                .then(None)
                .otherwise(pl.col("target_index"))
                .alias("target_index"),
            )

        if metric_cols:
            expressions.extend([pl.col(metric_cols).fill_null(0), pl.sum_horizontal(metric_cols).alias("Total")])
        else:
            expressions.append(pl.lit(0).alias("Total"))

        column_order = index_cols + metric_cols + ["Total"]

        return pivoted.with_columns(expressions).select(column_order).cast(schema).sort(index_cols)

    _UNSET = object()  # sentinel for distinguishing "not provided" from None

    def _redetect(  # noqa: C901
        self,
        class_ids: NDArray[np.intp] | None = None,
        outlier_threshold: Any = _UNSET,
        cluster_threshold: Any = _UNSET,
    ) -> Self:
        """Re-run outlier detection with optional class grouping or threshold override.

        Raises
        ------
        ValueError
            If this output was not created from an evaluation with stored
            statistics or cluster stats.
        """
        if self.calculation_results is None and self.cluster_stats is None:
            raise ValueError(
                "Re-detection requires statistics stored from evaluate(), from_stats(), or from_clusters()."
            )

        outlier_dfs: list[pl.DataFrame] = []

        # Stats-based re-detection
        threshold = self.outlier_threshold if outlier_threshold is self._UNSET else outlier_threshold
        if self.calculation_results is not None:
            combined = self._combined
            if combined is None:
                raise ValueError("Combined stats or source_index is None, cannot re-detect outliers.")
            stats_map, source_index, ds_steps = combined

            stats_df = _detect_outliers(stats_map, source_index, threshold, class_ids)

            if ds_steps:
                stats_df = add_dataset_index(stats_df, ds_steps)

            outlier_dfs.append(stats_df)

        # Cluster-based re-detection
        ct = self.cluster_threshold if cluster_threshold is self._UNSET else cluster_threshold
        resolved_ct = resolve_threshold(ct) if ct is not None else DEFAULT_OUTLIERS_CLUSTER_THRESHOLD
        if self.cluster_stats is not None:
            cluster_df = Outliers._find_outliers_adaptive(self.cluster_stats, resolved_ct)
            if self.dataset_steps:
                cluster_df = add_dataset_index(cluster_df, self.dataset_steps)
            outlier_dfs.append(cluster_df)

        merged = Outliers._merge_outlier_dfs(outlier_dfs)
        merged = drop_null_index_columns(merged, ["target_index", "channel_index"])

        return OutliersOutput(  # type: ignore[return-value]
            merged,
            calculation_results=self.calculation_results,
            outlier_threshold=threshold,
            cluster_stats=self.cluster_stats,
            cluster_threshold=ct,
            dataset_steps=self.dataset_steps,
        )

    def classwise(self, metadata: MetadataLike) -> Self:
        """Re-detect outliers using per-class thresholds.

        Computes outlier thresholds within each class separately rather than
        globally. This catches within-class anomalies that global detection misses,
        and avoids false positives where a sample is only unusual because its class
        is inherently different.

        For image classification datasets, each image is assigned to its class.
        For object detection datasets, target-level stats use the target's class,
        while image-level stats for images with multiple classes fall back to
        global detection.

        Parameters
        ----------
        metadata : MetadataLike
            Metadata object containing class labels.

        Returns
        -------
        OutliersOutput
            New output with per-class detected outliers. Supports all the same
            aggregation methods (``aggregate_by_class``, ``aggregate_by_metric``, etc.).

        Raises
        ------
        ValueError
            If this output was not created from an evaluation with stored statistics.

        Examples
        --------
        >>> outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold="modzscore")
        >>> result = outliers.evaluate(dataset)
        >>> classwise_result = result.classwise(metadata)
        >>> classwise_result.aggregate_by_class(metadata)
        shape: (5, 9)
        ┌────────────┬─────────┬──────────┬──────┬───┬─────┬─────┬───────┬───────┐
        │ class_name ┆ entropy ┆ kurtosis ┆ mean ┆ … ┆ std ┆ var ┆ zeros ┆ Total │
        │ ---        ┆ ---     ┆ ---      ┆ ---  ┆   ┆ --- ┆ --- ┆ ---   ┆ ---   │
        │ cat        ┆ u32     ┆ u32      ┆ u32  ┆   ┆ u32 ┆ u32 ┆ u32   ┆ u32   │
        ╞════════════╪═════════╪══════════╪══════╪═══╪═════╪═════╪═══════╪═══════╡
        │ plane      ┆ 2       ┆ 0        ┆ 3    ┆ … ┆ 2   ┆ 2   ┆ 4     ┆ 13    │
        │ person     ┆ 2       ┆ 1        ┆ 2    ┆ … ┆ 2   ┆ 2   ┆ 1     ┆ 11    │
        │ boat       ┆ 1       ┆ 0        ┆ 1    ┆ … ┆ 1   ┆ 1   ┆ 4     ┆ 8     │
        │ car        ┆ 0       ┆ 0        ┆ 0    ┆ … ┆ 0   ┆ 0   ┆ 4     ┆ 4     │
        │ Total      ┆ 5       ┆ 1        ┆ 6    ┆ … ┆ 5   ┆ 5   ┆ 13    ┆ 36    │
        └────────────┴─────────┴──────────┴──────┴───┴─────┴─────┴───────┴───────┘
        """
        combined = self._combined
        if combined is None:
            raise ValueError("classwise requires statistics stored from evaluate() or from_stats().")
        _, source_index, _ = combined
        class_ids = _build_class_ids(source_index, metadata)
        return self._redetect(class_ids)

    def itemwise(self) -> Self:
        """Re-detect outliers using global thresholds (across all items).

        This is the inverse of :meth:`classwise` -- it re-runs detection without
        per-class grouping, using the full dataset distribution for thresholds.

        Returns
        -------
        OutliersOutput
            New output with globally detected outliers.

        Raises
        ------
        ValueError
            If this output was not created from an evaluation with stored statistics.

        Examples
        --------
        >>> outliers = Outliers(flags=ImageStats.PIXEL)
        >>> result = outliers.evaluate(dataset, per_class=True, metadata=metadata)
        >>> global_result = result.itemwise()
        """
        return self._redetect()

    def with_threshold(
        self,
        outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None = _UNSET,  # type: ignore[arg-type]
        cluster_threshold: ThresholdLike | None = _UNSET,  # type: ignore[arg-type]
    ) -> Self:
        """Re-detect outliers using a different threshold configuration.

        Re-runs detection on the stored statistics with the new threshold,
        without recomputing the underlying image statistics. This enables
        quick sensitivity analysis and threshold experimentation.

        Can be chained with :meth:`classwise` for per-class detection
        with a different threshold.

        Parameters
        ----------
        outlier_threshold : ThresholdLike, dict, or None
            New threshold configuration for stats-based outliers. Accepts
            the same formats as :class:`Outliers`:

            - ``None``: ``AdaptiveThreshold(3.5)`` (Double-MAD with asymmetric bounds)
            - ``float``: symmetric multiplier for modified z-score
            - ``str``: named threshold type (``"zscore"``, ``"iqr"``, etc.)
            - ``tuple``: named threshold with bounds, e.g. ``("zscore", 2.5)``
            - :class:`~dataeval.utils.thresholds.Threshold`: fully configured
              threshold
            - ``Mapping[str, ThresholdLike]``: per-metric thresholds
        cluster_threshold : ThresholdLike or None
            New threshold configuration for cluster-based outlier detection.
            Accepts the same formats as ``outlier_threshold``. Only applies
            when cluster statistics are stored from :meth:`evaluate` or
            :meth:`~Outliers.from_clusters`.

        Returns
        -------
        OutliersOutput
            New output with re-detected outliers using the new threshold.

        Raises
        ------
        ValueError
            If no arguments are provided, or if this output was not created
            from an evaluation with stored statistics or cluster stats.

        Examples
        --------
        >>> outliers = Outliers(flags=ImageStats.PIXEL)
        >>> result = outliers.evaluate(dataset)

        Loosen the threshold:

        >>> lenient = result.with_threshold(4.0)

        Switch to IQR method:

        >>> iqr_result = result.with_threshold("iqr")

        Per-metric overrides:

        >>> custom = result.with_threshold({"brightness": 2.0, "contrast": ("zscore", 3.0)})

        Chain with classwise:

        >>> per_class = result.classwise(metadata).with_threshold(2.0)

        Adjust cluster threshold:

        >>> strict_clusters = result.with_threshold(cluster_threshold=1.5)
        """
        if outlier_threshold is self._UNSET and cluster_threshold is self._UNSET:
            raise ValueError("At least one of outlier_threshold or cluster_threshold must be provided.")
        return self._redetect(outlier_threshold=outlier_threshold, cluster_threshold=cluster_threshold)


# Convenience type aliases for parameterized output
SingleOutliersOutput = OutliersOutput[SingleOutliersMap]
SingleTargetOutliersOutput = OutliersOutput[SingleTargetOutliersMap]
MultiOutliersOutput = OutliersOutput[MultiOutliersMap]
MultiTargetOutliersOutput = OutliersOutput[MultiTargetOutliersMap]


def _get_outlier_mask(  # noqa: C901
    values: NDArray[Any],
    threshold: Threshold,
) -> NDArray[np.bool_]:
    """Compute outlier boolean mask using a Threshold object.

    Parameters
    ----------
    values : NDArray
        1D array of metric values.
    threshold : Threshold
        Threshold instance to compute bounds.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask where True indicates an outlier.
    """
    if len(values) == 0:
        return np.array([], dtype=bool)

    nan_mask = np.isnan(values)

    if np.all(nan_mask):
        return np.full(values.shape, False, dtype=bool)

    float_values = values.astype(np.float64)
    lower, upper = threshold(float_values)

    # If both bounds are None, the threshold could not be computed (e.g., zero variance)
    if lower is None and upper is None:
        return np.full(values.shape, False, dtype=bool)

    outlier_mask = np.full(values.shape, False, dtype=bool)
    if lower is not None:
        outlier_mask |= float_values < lower
    if upper is not None:
        outlier_mask |= float_values > upper

    # NaN values are never outliers
    return outlier_mask & ~nan_mask


def _build_class_ids(  # noqa: C901
    source_index: Sequence[SourceIndex],
    metadata: MetadataLike,
) -> NDArray[np.intp]:
    """Map each source_index entry to a class ID for per-class outlier detection.

    For image-level entries (target is None):
        - If all targets in the image share one class, assign that class.
        - If the image has targets of multiple classes, assign -1 (global fallback).
    For target-level entries (target is not None):
        - Look up the target's class via the metadata mapping.

    Parameters
    ----------
    source_index : Sequence[SourceIndex]
        List of source index entries from compute_stats.
    metadata : MetadataLike
        Metadata object with class_labels and optional item_indices.

    Returns
    -------
    NDArray[np.intp]
        Array of class IDs aligned with source_index, -1 for unclassifiable entries.
    """
    item_indices = _get_item_indices(metadata)
    class_labels = metadata.class_labels
    n = len(source_index)
    class_ids = np.full(n, -1, dtype=np.intp)

    # Build mapping: image_id -> list of global indices into class_labels
    image_to_label_indices: dict[int, list[int]] = {}
    for k, img_idx in enumerate(item_indices):
        image_to_label_indices.setdefault(int(img_idx), []).append(k)

    # Precompute unique classes per image
    image_to_unique_classes: dict[int, set[int]] = {}
    for img_idx, label_indices in image_to_label_indices.items():
        image_to_unique_classes[img_idx] = {int(class_labels[k]) for k in label_indices}

    for i, si in enumerate(source_index):
        if si.target is None:
            # Image-level: assign class only if image has a single unique class
            classes = image_to_unique_classes.get(si.item, set())
            if len(classes) == 1:
                class_ids[i] = next(iter(classes))
        else:
            # Target-level: look up the specific target's class
            label_indices = image_to_label_indices.get(si.item, [])
            if si.target < len(label_indices):
                class_ids[i] = int(class_labels[label_indices[si.target]])

    return class_ids


def _resolve_outlier_threshold(
    outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None,
) -> Threshold | Mapping[str, Threshold]:
    """Eagerly resolve an outlier_threshold value so that None is replaced with the default."""
    if isinstance(outlier_threshold, Mapping):
        return {k: resolve_threshold(v) for k, v in outlier_threshold.items()}
    if outlier_threshold is not None:
        return resolve_threshold(outlier_threshold)
    return resolve_threshold(None)


def _resolve_metric_threshold(
    outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None,
    metric_name: str,
) -> Threshold:
    """Resolve the Threshold object for a given metric name.

    Priority:
    1. If outlier_threshold is a dict and metric_name is in it, use that entry.
    2. If outlier_threshold is a non-dict ThresholdLike, use it for all metrics.
    3. Otherwise, use default AdaptiveThreshold (Double-MAD with asymmetric bounds).
    """
    if isinstance(outlier_threshold, Mapping):
        value = outlier_threshold.get(metric_name)
        if value is not None:
            return resolve_threshold(value)
    elif outlier_threshold is not None:
        return resolve_threshold(outlier_threshold)
    return DEFAULT_OUTLIERS_OUTLIER_THRESHOLD


def _compute_outlier_mask(
    level_values: NDArray[Any],
    threshold: Threshold,
    level_mask: NDArray[np.bool_],
    class_ids: NDArray[np.intp] | None,
) -> NDArray[np.bool_]:
    """Compute outlier mask for one level, optionally grouped by class.

    Parameters
    ----------
    level_values : NDArray
        Metric values for entries at this level (image or target).
    threshold : Threshold
        Threshold instance to compute bounds.
    level_mask : NDArray[np.bool_]
        Boolean mask selecting this level from the full source_index.
    class_ids : NDArray[np.intp] or None
        Per-entry class IDs aligned with source_index, or None for global detection.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask over level_values where True indicates an outlier.
    """
    if class_ids is None:
        return _get_outlier_mask(level_values.astype(np.float64), threshold)

    level_class_ids = class_ids[level_mask]
    outlier_mask = np.zeros(len(level_values), dtype=bool)

    # Global threshold for unclassifiable entries (class_id == -1)
    unclassifiable = level_class_ids == -1
    if np.any(unclassifiable):
        outlier_mask[unclassifiable] = _get_outlier_mask(level_values[unclassifiable].astype(np.float64), threshold)

    # Per-class threshold for classifiable entries
    classifiable = level_class_ids >= 0
    if np.any(classifiable):
        for cls in np.unique(level_class_ids[classifiable]):
            cls_mask = level_class_ids == cls
            cls_outliers = _get_outlier_mask(level_values[cls_mask].astype(np.float64), threshold)
            outlier_mask[cls_mask] = cls_outliers

    return outlier_mask


def _detect_outliers(  # noqa: C901
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None,
    class_ids: NDArray[np.intp] | None = None,
) -> pl.DataFrame:
    """Detect outliers from pre-computed statistics, optionally per-class.

    Parameters
    ----------
    stats : StatsMap
        Dictionary mapping metric names to value arrays.
    source_index : Sequence[SourceIndex]
        Source index entries aligned with stat values.
    outlier_threshold : ThresholdLike or dict or None
        Threshold configuration for outlier detection.
    class_ids : NDArray[np.intp] or None
        Per-entry class IDs for per-class detection, or None for global.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: item_index, target_index, metric_name, metric_value.
    """
    item_ids: list[int] = []
    target_ids: list[int | None] = []
    channel_ids: list[int | None] = []
    metric_names: list[str] = []
    metric_values: list[float] = []

    if len(source_index) > 0:
        is_image_level = np.array([src_idx.target is None for src_idx in source_index], dtype=bool)
        is_target_level = ~is_image_level

        for stat, values in stats.items():
            if values.ndim == 1 and np.issubdtype(values.dtype, np.number):
                threshold = _resolve_metric_threshold(outlier_threshold, stat)

                for level_mask in (is_image_level, is_target_level):
                    if not np.any(level_mask):
                        continue

                    level_values = values[level_mask]
                    level_indices = np.flatnonzero(level_mask)
                    outlier_mask = _compute_outlier_mask(level_values, threshold, level_mask, class_ids)

                    if np.any(outlier_mask):
                        outlier_indices = level_indices[outlier_mask]
                        item_ids.extend(source_index[idx].item for idx in outlier_indices)
                        target_ids.extend(source_index[idx].target for idx in outlier_indices)
                        channel_ids.extend(source_index[idx].channel for idx in outlier_indices)
                        metric_names.extend([stat] * len(outlier_indices))
                        metric_values.extend(values[outlier_indices].tolist())

    if not item_ids:
        return pl.DataFrame(
            schema={
                "item_index": pl.Int64,
                "target_index": pl.Int64,
                "channel_index": pl.Int64,
                "metric_name": pl.Categorical("lexical"),
                "metric_value": pl.Float64,
            },
        )

    df = pl.DataFrame(
        {
            "item_index": pl.Series(item_ids, dtype=pl.Int64),
            "target_index": pl.Series(target_ids, dtype=pl.Int64),
            "channel_index": pl.Series(channel_ids, dtype=pl.Int64),
            "metric_name": pl.Series(metric_names, dtype=pl.Categorical("lexical")),
            "metric_value": pl.Series(metric_values, dtype=pl.Float64),
        },
    )

    return df.sort(["item_index", "target_index", "metric_name"])


class Outliers(Evaluator):
    r"""
    Computes statistical outliers of a dataset using various statistical tests applied to each image.

    Supports two complementary detection methods:

    1. **Image statistics-based**: Computes pixel-level statistics (brightness, contrast, etc.)
       and flags images with unusual values using configurable :class:`~dataeval.utils.thresholds.Threshold`
       objects.
    2. **Cluster-based**: Uses embeddings from a neural network to cluster images and identifies
       outliers based on distance from cluster centers in embedding space.

    Both methods can be used together or independently based on the ``flags`` parameter.

    Parameters
    ----------
    flags : ImageStats, default ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
        Statistics to compute for image statistics-based outlier detection. Set to
        ``ImageStats.NONE`` to skip image statistics and use only cluster-based detection
        (requires ``extractor``).
    outlier_threshold : ThresholdLike, dict, or None, default None
        Threshold configuration for image statistics-based outlier detection.

        - ``None``: uses ``AdaptiveThreshold()`` with default multiplier (3.5),
          which computes both z-score and modified z-score bounds and takes the
          wider (more lenient) bound on each side.
        - ``float``: symmetric multiplier for the default method (modified z-score
          via ``resolve_threshold``)
        - ``str``: named threshold type (e.g., ``"zscore"``, ``"iqr"``,
          ``"adaptive"``) with defaults
        - ``tuple[float | None, float | None]``: asymmetric ``(lower, upper)`` multipliers
        - ``tuple[str, ThresholdBounds]``: named threshold with bounds, e.g.
          ``("zscore", 2.5)`` or ``("iqr", (1.0, 3.0))``
        - :class:`~dataeval.utils.thresholds.Threshold`: a fully configured threshold
          (e.g., ``ZScoreThreshold``, ``IQRThreshold``, ``ConstantThreshold``,
          ``AdaptiveThreshold``)
        - ``Mapping[str, ThresholdLike]``: per-metric thresholds keyed by metric name.
          Metrics not in the dict use the default (``AdaptiveThreshold()``).
    extractor : FeatureExtractor, optional
        Feature extractor for cluster-based outlier detection. When provided, embeddings
        are extracted and clustered to find semantic/visual outliers in embedding space.
    cluster_threshold : ThresholdLike or None, default None
        Threshold configuration for cluster-based outlier detection. When None,
        defaults to ``ZScoreThreshold(upper_multiplier=2.5)``.
        Accepts the same formats as ``outlier_threshold``.
        Only used when ``extractor`` is provided.
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
    stats : StatsResult
        Statistics computed during the last evaluate() call.
        Contains dimension, pixel, and/or visual statistics based on the flags.
    flags : ImageStats
        Statistics to compute for outlier detection.
    outlier_threshold : ThresholdLike | Mapping[str, ThresholdLike] | None
        Threshold configuration for outlier detection.
    extractor : FeatureExtractor | None
        Feature extractor for cluster-based detection.
    cluster_threshold : ThresholdLike | None
        Threshold configuration for cluster-based detection.
    cluster_algorithm : Literal["kmeans", "hdbscan"]
        Clustering algorithm to use.
    n_clusters : int | None
        Expected number of clusters.

    See Also
    --------
    :term:`Duplicates`

    Notes
    -----
    **Threshold Methods:**

    - ``AdaptiveThreshold`` (default): Uses tail-weighted Double-MAD (separate MAD for
      data below and above the median) with automatic multiplier scaling for heavy
      tails to produce asymmetric bounds. Default multiplier: 3.0.
    - ``ModifiedZScoreThreshold``: Based on median absolute deviation. Default multiplier: 3.5.
      Modified z score = :math:`0.6745 * |x_i - x̃| / MAD`
    - ``ZScoreThreshold``: Based on standard deviation from mean. Default multiplier: 3.
      Z score = :math:`|x_i - \mu| / \sigma`
    - ``IQRThreshold``: Based on interquartile range. Default multiplier: 1.5.
      Outliers are outside :math:`[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]`
    - ``ConstantThreshold``: Hard lower/upper bounds (data-independent).

    All threshold types support asymmetric lower/upper multipliers via
    ``lower_multiplier`` and ``upper_multiplier`` parameters.

    **Cluster-based Detection:**

    Uses adaptive distance-based detection that accounts for varying cluster densities.
    A :class:`~dataeval.utils.thresholds.Threshold` is applied per-cluster to the distance
    distribution (default: ``ZScoreThreshold(upper_multiplier=2.5)``), and points whose
    distance exceeds the upper bound are flagged as outliers.

    Examples
    --------
    Basic image statistics-based outlier detection (default: modified z-score):

    >>> outliers = Outliers()
    >>> result = outliers.evaluate(dataset)

    Using a specific threshold method:

    >>> from dataeval.utils.thresholds import ZScoreThreshold
    >>> outliers = Outliers(outlier_threshold=ZScoreThreshold(2.5))

    Asymmetric thresholds (stricter on lower, lenient on upper):

    >>> from dataeval.utils.thresholds import IQRThreshold
    >>> outliers = Outliers(outlier_threshold=IQRThreshold(lower_multiplier=1.0, upper_multiplier=3.0))

    Hard bounds:

    >>> from dataeval.utils.thresholds import ConstantThreshold
    >>> outliers = Outliers(outlier_threshold=ConstantThreshold(lower=0.1, upper=0.9))

    Named threshold type with bounds (no need to import threshold classes):

    >>> outliers = Outliers(outlier_threshold="iqr")
    >>> outliers = Outliers(outlier_threshold=("zscore", 2.5))
    >>> outliers = Outliers(outlier_threshold=("modzscore", (1.0, 3.0)))

    Named threshold type with bounds and limits  (no need to import threshold classes):

    >>> outliers = Outliers(outlier_threshold=("zscore", 4.0, (0.0, 1.0)))

    Per-metric thresholds:

    >>> outliers = Outliers(outlier_threshold={"mean": 2.0, "brightness": ("zscore", 2.0)})

    Cluster-based detection with embeddings:

    >>> from dataeval.extractors import FlattenExtractor

    >>> outliers = Outliers(flags=ImageStats.NONE, extractor=FlattenExtractor(), cluster_threshold=2.0)
    >>> result = outliers.evaluate(train_ds)  # Only cluster_distance metric

    Using configuration:

    >>> config = Outliers.Config(outlier_threshold=2.5)
    >>> outliers = Outliers(config=config)
    """

    class Config(EvaluatorConfig, ClusterConfigMixin):
        """
        Configuration for Outliers detector.

        Attributes
        ----------
        flags : ImageStats, default ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
            Statistics to compute for image statistics-based outlier detection.
        outlier_threshold : ThresholdLike | Mapping[str, ThresholdLike] | None, default None
            Threshold configuration. When None, uses ``AdaptiveThreshold(3.5)``
            (Double-MAD with asymmetric bounds). See :class:`Outliers` for full description.
        cluster_threshold : ThresholdLike or None, default None
            Threshold configuration for cluster-based detection. When None,
            defaults to ``ZScoreThreshold(upper_multiplier=2.5)``.
        extractor : FeatureExtractor or None, default None
            Feature extractor for cluster-based outlier detection.
        batch_size : int or None, default None
            Batch size for feature extraction during cluster-based detection. If None, uses DataEval
            default. Must be set by either parameter or global default if extractor is provided.
        cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
            Clustering algorithm for cluster-based detection.
        n_clusters : int or None, default None
            Expected number of clusters.
        """

        flags: ImageStats = DEFAULT_OUTLIERS_FLAGS
        outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None = None
        cluster_threshold: ThresholdLike | None = None

    stats: StatsResult
    flags: ImageStats
    outlier_threshold: Threshold | Mapping[str, Threshold]
    extractor: FeatureExtractor | None
    batch_size: int | None
    cluster_algorithm: Literal["kmeans", "hdbscan"]
    cluster_threshold: ThresholdLike | None
    n_clusters: int | None
    config: Config

    def __init__(
        self,
        flags: ImageStats | None = None,
        outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None = None,
        extractor: FeatureExtractor | None = None,
        batch_size: int | None = None,
        cluster_algorithm: Literal["kmeans", "hdbscan"] | None = None,
        cluster_threshold: ThresholdLike | None = None,
        n_clusters: int | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())
        self.outlier_threshold = _resolve_outlier_threshold(self.outlier_threshold)

    def _get_outliers(
        self,
        stats: StatsMap,
        source_index: Sequence[SourceIndex],
        class_ids: NDArray[np.intp] | None = None,
    ) -> pl.DataFrame:
        return _detect_outliers(stats, source_index, self.outlier_threshold, class_ids)

    @overload
    def from_stats(
        self,
        stats: StatsResult,
        *,
        per_image: bool = True,
        per_target: Literal[False] = ...,
    ) -> SingleOutliersOutput: ...

    @overload
    def from_stats(
        self,
        stats: StatsResult,
        *,
        per_image: bool = True,
        per_target: Literal[True],
    ) -> SingleTargetOutliersOutput: ...

    @overload
    def from_stats(
        self,
        stats: Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: Literal[False] = ...,
    ) -> MultiOutliersOutput: ...

    @overload
    def from_stats(
        self,
        stats: Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: Literal[True],
    ) -> MultiTargetOutliersOutput: ...

    @set_metadata(state=["outlier_threshold"])
    def from_stats(
        self,
        stats: StatsResult | Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: bool = False,
    ) -> SingleOutliersOutput | SingleTargetOutliersOutput | MultiOutliersOutput | MultiTargetOutliersOutput:
        """
        Return indices of Outliers with the issues identified for each.

        Parameters
        ----------
        stats : StatsResult | Sequence[StatsResult]
            The output(s) from compute_stats() with ImageStats.DIMENSION, PIXEL, or VISUAL flags
        per_image : bool, default True
            Whether to include item-level (image) outliers in the results.
        per_target : bool, default False
            Whether to include target-level outliers in the results.
            When True, the ``.outliers`` accessor uses :class:`SourceIndex` keys;
            when False, it uses plain ``int`` item indices.

        Returns
        -------
        OutliersOutput
            Output class containing a DataFrame of outlier issues with columns:
            - item_index: int - Index of the outlier item
            - target_index: int | None - Index of the target within the item (None for item-level outliers)
            - metric_name: str - Name of the metric that flagged this item/target
            - metric_value: float - Value of the metric for this item/target

            For multiple datasets, a ``dataset_index`` column identifies the originating dataset
            and ``item_index`` values are local to each dataset.

        Example
        -------
        Evaluate the dataset using pre-computed stats:

        >>> from dataeval.core import compute_stats
        >>> from dataeval.flags import ImageStats
        >>> from dataeval.utils.thresholds import ZScoreThreshold

        >>> stats = compute_stats(images, stats=ImageStats.PIXEL)
        >>> outliers = Outliers(outlier_threshold=ZScoreThreshold(2.5))
        >>> results = outliers.from_stats(stats)
        >>> results.head(10)
        shape: (10, 3)
        ┌────────────┬─────────────┬──────────────┐
        │ item_index ┆ metric_name ┆ metric_value │
        │ ---        ┆ ---         ┆ ---          │
        │ i64        ┆ cat         ┆ f64          │
        ╞════════════╪═════════════╪══════════════╡
        │ 7          ┆ entropy     ┆ 0.0          │
        │ 7          ┆ mean        ┆ 0.98         │
        │ 7          ┆ std         ┆ 0.0          │
        │ 7          ┆ var         ┆ 0.0          │
        │ 8          ┆ skew        ┆ 0.062311     │
        │ 11         ┆ entropy     ┆ 0.0          │
        │ 11         ┆ mean        ┆ 0.98         │
        │ 11         ┆ std         ┆ 0.0          │
        │ 11         ┆ var         ┆ 0.0          │
        │ 18         ┆ entropy     ┆ 0.0          │
        └────────────┴─────────────┴──────────────┘
        """
        combined_stats, combined_source_index, dataset_steps = combine_stats_results(stats)

        # Filter source_index based on per_image/per_target flags
        if not (per_image and per_target):
            mask = np.array([
                (per_image and si.target is None) or (per_target and si.target is not None)
                for si in combined_source_index
            ])
            indices = np.flatnonzero(mask)
            combined_source_index = [combined_source_index[i] for i in indices]
            combined_stats = {k: v[indices] for k, v in combined_stats.items()}

        outliers_df = self._get_outliers(combined_stats, combined_source_index)
        outliers_df = drop_null_index_columns(outliers_df, ["target_index", "channel_index"])

        if dataset_steps:
            outliers_df = add_dataset_index(outliers_df, dataset_steps)

        return OutliersOutput(
            outliers_df,
            calculation_results=stats,
            outlier_threshold=self.outlier_threshold,
        )

    @set_metadata(state=["cluster_threshold", "cluster_algorithm", "n_clusters"])
    def from_clusters(
        self,
        embeddings: ArrayND[float],
        cluster_result: ClusterResult,
        cluster_threshold: ThresholdLike | None = None,
    ) -> SingleOutliersOutput:
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
        cluster_threshold : ThresholdLike or None, default None
            Threshold configuration for cluster-based outlier detection.
            Accepts the same formats as ``outlier_threshold``. When None,
            uses the detector's configured ``cluster_threshold``.

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
        levels and methods without recomputing clusters.
        """
        # Convert embeddings to numpy array and flatten if needed
        embeddings_array = flatten_samples(to_numpy(embeddings))

        # Compute cluster statistics
        cs = compute_cluster_stats(
            embeddings=embeddings_array,
            cluster_labels=cluster_result["clusters"],
        )

        ct = cluster_threshold if cluster_threshold is not None else self.cluster_threshold
        resolved_ct = resolve_threshold(ct) if ct is not None else DEFAULT_OUTLIERS_CLUSTER_THRESHOLD

        # Find outliers using adaptive method
        outlier_issues = self._find_outliers_adaptive(cluster_stats=cs, threshold=resolved_ct)

        return OutliersOutput(outlier_issues, cluster_stats=cs, cluster_threshold=ct)

    @staticmethod
    def _merge_outlier_dfs(outliers_dfs: Sequence[pl.DataFrame]) -> pl.DataFrame:  # noqa: C901
        """Merge a list of outlier DataFrames into one, normalizing columns."""
        if len(outliers_dfs) == 0:
            return pl.DataFrame(
                schema={
                    "item_index": pl.Int64,
                    "target_index": pl.Int64,
                    "metric_name": pl.Categorical("lexical"),
                    "metric_value": pl.Float64,
                },
            )
        if len(outliers_dfs) == 1:
            return outliers_dfs[0]

        # Determine which optional columns are present in any DataFrame
        has_target_id = any("target_index" in df.columns for df in outliers_dfs)
        has_channel_id = any("channel_index" in df.columns for df in outliers_dfs)

        column_order = ["item_index"]
        if has_target_id:
            column_order.append("target_index")
        if has_channel_id:
            column_order.append("channel_index")
        column_order.extend(["metric_name", "metric_value"])

        normalized_dfs: list[pl.DataFrame] = []
        for df in outliers_dfs:
            if has_target_id and "target_index" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("target_index"))
            if has_channel_id and "channel_index" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("channel_index"))
            normalized_dfs.append(df.select(column_order))
        return pl.concat(normalized_dfs).sort(["item_index", "target_index", "metric_name"])

    @staticmethod
    def _find_outliers_adaptive(
        cluster_stats: ClusterStats,
        threshold: Threshold,
    ) -> pl.DataFrame:
        """
        Find outliers using pre-calculated cluster statistics.

        This method uses pre-calculated cluster centers and distance statistics to identify
        outliers. A :class:`Threshold` is applied per-cluster to the distance distribution,
        and points whose distance exceeds the upper bound are flagged as outliers.

        Parameters
        ----------
        cluster_stats : dict
            Pre-calculated cluster centers, distance statistics, and nearest cluster indices.
            Should contain keys: 'distances', 'nearest_cluster_idx', 'cluster_distances_mean',
            'cluster_distances_std'.
        threshold : Threshold
            Threshold instance to compute per-cluster upper bounds from distance arrays.

        Returns
        -------
        pl.DataFrame
            DataFrame with outlier details containing columns:
            - item_index: int - Index of the outlier
            - metric_name: str - Always "cluster_distance"
            - metric_value: float - Distance in std dev from cluster mean
        """
        # Get pre-calculated distances and nearest cluster indices
        min_distances = cluster_stats["distances"]
        nearest_cluster_idx = cluster_stats["nearest_cluster_idx"]
        cluster_distances_mean = cluster_stats["cluster_distances_mean"]
        cluster_distances_std = cluster_stats["cluster_distances_std"]

        # Compute per-cluster upper bounds using the threshold
        unique_clusters = np.unique(nearest_cluster_idx[nearest_cluster_idx >= 0])
        is_outlier = np.full(len(min_distances), False, dtype=bool)

        for cluster_id in unique_clusters:
            mask = nearest_cluster_idx == cluster_id
            cluster_distances = min_distances[mask]
            _, upper = threshold(cluster_distances)
            if upper is not None:
                is_outlier[mask] = cluster_distances > upper

        # Build the result DataFrame with issue details
        outlier_indices = np.nonzero(is_outlier)[0]

        if len(outlier_indices) == 0:
            return pl.DataFrame(
                schema={
                    "item_index": pl.Int64,
                    "metric_name": pl.Categorical("lexical"),
                    "metric_value": pl.Float64,
                },
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

        # Cluster-based detection is always item-level, so we don't include target_index
        return pl.DataFrame(
            {
                "item_index": pl.Series(item_ids, dtype=pl.Int64),
                "metric_name": pl.Series(["cluster_distance"] * len(item_ids), dtype=pl.Categorical("lexical")),
                "metric_value": pl.Series(metric_values, dtype=pl.Float64),
            },
        ).sort(["item_index", "metric_name"], descending=[False, False])

    def _get_cluster_outliers(
        self,
        data: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    ) -> tuple[pl.DataFrame, ClusterStats]:
        """Extract embeddings, cluster them, and return cluster-based outlier DataFrame and stats."""
        embeddings = Embeddings(data, self.extractor, batch_size=self.batch_size)

        cluster_result = cluster(
            embeddings,
            algorithm=self.cluster_algorithm,
            n_clusters=self.n_clusters,
        )

        cs = compute_cluster_stats(
            embeddings=np.asarray(embeddings),
            cluster_labels=cluster_result["clusters"],
        )

        resolved_ct = (
            resolve_threshold(self.cluster_threshold)
            if self.cluster_threshold is not None
            else DEFAULT_OUTLIERS_CLUSTER_THRESHOLD
        )
        return self._find_outliers_adaptive(cluster_stats=cs, threshold=resolved_ct), cs

    _DatasetInput = Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]

    @overload
    def evaluate(  # pyright: ignore[reportOverlappingOverload]
        self,
        data: _DatasetInput,
        *,
        per_image: bool = True,
        per_target: Literal[False] = ...,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
    ) -> SingleOutliersOutput: ...

    @overload
    def evaluate(  # pyright: ignore[reportOverlappingOverload]
        self,
        data: _DatasetInput,
        *,
        per_image: bool = True,
        per_target: Literal[True],
        per_class: bool = False,
        metadata: MetadataLike | None = None,
    ) -> SingleTargetOutliersOutput: ...

    @overload
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        per_image: bool = True,
        per_target: Literal[False] = ...,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
    ) -> MultiOutliersOutput: ...

    @overload
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        per_image: bool = True,
        per_target: Literal[True],
        per_class: bool = False,
        metadata: MetadataLike | None = None,
    ) -> MultiTargetOutliersOutput: ...

    @set_metadata(
        state=[
            "flags",
            "outlier_threshold",
            "cluster_threshold",
            "cluster_algorithm",
            "n_clusters",
        ],
    )
    def evaluate(
        self,
        data: _DatasetInput,
        *other: _DatasetInput,
        per_image: bool = True,
        per_target: bool = False,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
    ) -> SingleOutliersOutput | SingleTargetOutliersOutput | MultiOutliersOutput | MultiTargetOutliersOutput:
        """
        Return indices of Outliers with the issues identified for each.

        Computes outliers using image statistics and/or cluster-based detection,
        depending on configuration. When both methods are enabled, results are
        combined into a single DataFrame. Supports single or multiple datasets.

        Parameters
        ----------
        data : Dataset
            A dataset of images.
        *other : Dataset
            Additional datasets for cross-dataset outlier detection.
        per_image : bool, default True
            Whether to compute statistics for full items (images/videos).
            When True, item-level outliers will be detected.
        per_target : bool, default False
            Whether to compute statistics for individual targets/detections.
            When True, the ``.outliers`` accessor uses :class:`SourceIndex` keys;
            when False, it uses plain ``int`` item indices.
            Has no effect for datasets without targets or for cluster-based detection.
        per_class : bool, default False
            Whether to compute outlier thresholds within each class separately,
            rather than globally across the entire dataset. When True, ``metadata``
            must be provided. Only applies to image statistics-based detection,
            not cluster-based detection.
        metadata : MetadataLike or None, default None
            Metadata object containing class labels. Required when ``per_class=True``.

        Returns
        -------
        SingleOutliersOutput or MultiOutliersOutput
            Output class containing a DataFrame of outlier issues with columns:

            - item_index: int - Index of the outlier item
            - target_index: int | None - Index of the target within the item
              (None for item-level outliers, omitted if all are item-level)
            - metric_name: str - Name of the metric that flagged this item/target.
              Includes "cluster_distance" when extractor is provided.
            - metric_value: float - Value of the metric for this item/target.
              For cluster_distance, this is the number of std devs from cluster mean.

            For multi-dataset input, includes a ``dataset_index`` column.

        Raises
        ------
        ValueError
            If ``flags`` is ``ImageStats.NONE`` and no ``extractor`` is provided.
            If both ``per_image`` and ``per_target`` are False.
            If ``per_class`` is True and ``metadata`` is None.

        Examples
        --------
        Basic outlier detection:

        >>> outliers = Outliers(outlier_threshold=2.5)
        >>> results = outliers.evaluate(images)
        >>> results.head(6)
        shape: (6, 3)
        ┌────────────┬─────────────┬──────────────┐
        │ item_index ┆ metric_name ┆ metric_value │
        │ ---        ┆ ---         ┆ ---          │
        │ i64        ┆ cat         ┆ f64          │
        ╞════════════╪═════════════╪══════════════╡
        │ 0          ┆ zeros       ┆ 0.000081     │
        │ 2          ┆ zeros       ┆ 0.000081     │
        │ 7          ┆ brightness  ┆ 0.98         │
        │ 7          ┆ contrast    ┆ 0.0          │
        │ 7          ┆ darkness    ┆ 0.98         │
        │ 7          ┆ entropy     ┆ 0.0          │
        └────────────┴─────────────┴──────────────┘

        Evaluate two or more datasets (cross-dataset detection):

        >>> outliers = Outliers()
        >>> results = outliers.evaluate(train_ds, test_ds)
        >>> results = outliers.evaluate(train_ds_area1, train_ds_area2, train_ds_area3, test_ds)  # or more
        """
        if other:
            return self._evaluate_multi(
                [data, *other],
                per_image=per_image,
                per_target=per_target,
                per_class=per_class,
                metadata=metadata,
            )

        return self._evaluate_single(
            data,
            per_image=per_image,
            per_target=per_target,
            per_class=per_class,
            metadata=metadata,
        )

    def _validate_evaluate_inputs(
        self, per_image: bool, per_target: bool, per_class: bool, metadata: MetadataLike | None
    ) -> None:
        """Validate common inputs for single and multi-dataset evaluation."""
        if self.flags == ImageStats.NONE and self.extractor is None:
            raise ValueError("Either flags must not be ImageStats.NONE or extractor must be provided.")
        if not (per_image or per_target):
            raise ValueError("At least one of per_image or per_target must be True.")
        if per_class and metadata is None:
            raise ValueError("metadata must be provided when per_class=True.")

    def _evaluate_single(
        self,
        data: _DatasetInput,
        *,
        per_image: bool = True,
        per_target: bool = True,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
    ) -> SingleOutliersOutput:
        """Single-dataset evaluate implementation."""
        self._validate_evaluate_inputs(per_image, per_target, per_class, metadata)

        outliers_dfs: list[pl.DataFrame] = []
        stats_result: StatsResult | None = None
        stored_cluster_stats: ClusterStats | None = None

        if self.flags != ImageStats.NONE:
            self.stats = compute_stats(
                data, stats=self.flags, per_image=per_image, per_target=per_target, normalize_pixel_values=True
            )
            stats_result = self.stats

            class_ids: NDArray[np.intp] | None = None
            if per_class and metadata is not None:
                class_ids = _build_class_ids(self.stats["source_index"], metadata)

            outliers_dfs.append(self._get_outliers(self.stats["stats"], self.stats["source_index"], class_ids))

        if self.extractor is not None:
            cluster_df, stored_cluster_stats = self._get_cluster_outliers(data)
            outliers_dfs.append(cluster_df)

        return OutliersOutput(  # type: ignore[return-value]
            drop_null_index_columns(self._merge_outlier_dfs(outliers_dfs), ["target_index", "channel_index"]),
            calculation_results=stats_result,
            outlier_threshold=self.outlier_threshold,
            cluster_stats=stored_cluster_stats,
            cluster_threshold=self.cluster_threshold,
        )

    def _evaluate_multi(  # noqa: C901
        self,
        datasets: Sequence[_DatasetInput],
        *,
        per_image: bool = True,
        per_target: bool = True,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
    ) -> MultiOutliersOutput:
        """Multi-dataset evaluate: compute stats per dataset, then combine."""
        self._validate_evaluate_inputs(per_image, per_target, per_class, metadata)

        # Compute dataset_steps from dataset lengths (needed for index remapping)
        dataset_steps: list[int] = []
        cumulative = 0
        for ds in datasets:
            cumulative += len(ds)
            dataset_steps.append(cumulative)

        # Stats-based: compute stats per dataset, then combine
        stats_results: list[StatsResult] = []
        if self.flags != ImageStats.NONE:
            stats_results = [
                compute_stats(
                    ds, stats=self.flags, per_image=per_image, per_target=per_target, normalize_pixel_values=True
                )
                for ds in datasets
            ]
            self.stats = stats_results[-1]

        outliers_dfs: list[pl.DataFrame] = []

        if stats_results:
            combined_stats, combined_source_index, _ = combine_stats_results(stats_results)

            class_ids: NDArray[np.intp] | None = None
            if per_class and metadata is not None:
                class_ids = _build_class_ids(combined_source_index, metadata)

            outliers_df = self._get_outliers(combined_stats, combined_source_index, class_ids)
            outliers_df = drop_null_index_columns(outliers_df, ["target_index", "channel_index"])
            outliers_df = add_dataset_index(outliers_df, dataset_steps)
            outliers_dfs.append(outliers_df)

        # Cluster-based: early fusion — combine all images, cluster globally
        stored_cluster_stats: ClusterStats | None = None
        if self.extractor is not None:
            all_images = [item[0] if isinstance(item, tuple) else item for ds in datasets for item in ds]
            embeddings = Embeddings(all_images, self.extractor)

            cluster_result = cluster(
                embeddings,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            stored_cluster_stats = compute_cluster_stats(
                embeddings=np.asarray(embeddings),
                cluster_labels=cluster_result["clusters"],
            )

            resolved_ct = (
                resolve_threshold(self.cluster_threshold)
                if self.cluster_threshold is not None
                else DEFAULT_OUTLIERS_CLUSTER_THRESHOLD
            )
            cluster_df = self._find_outliers_adaptive(stored_cluster_stats, resolved_ct)
            cluster_df = add_dataset_index(cluster_df, dataset_steps)
            outliers_dfs.append(cluster_df)

        return OutliersOutput(  # type: ignore[return-value]
            drop_null_index_columns(self._merge_outlier_dfs(outliers_dfs), ["target_index", "channel_index"]),
            calculation_results=stats_results if stats_results else None,
            outlier_threshold=self.outlier_threshold,
            cluster_stats=stored_cluster_stats,
            cluster_threshold=self.cluster_threshold,
            dataset_steps=dataset_steps,
        )

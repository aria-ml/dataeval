from __future__ import annotations

__all__ = []

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dataeval.config import EPSILON
from dataeval.core._calculate import CalculationResult, calculate
from dataeval.core._clusterer import ClusterResult, ClusterStats, compute_cluster_stats
from dataeval.core._label_stats import LabelStatsResult
from dataeval.core.flags import ImageStats
from dataeval.data._images import Images
from dataeval.protocols import ArrayLike, Dataset
from dataeval.types import ArrayND, DictOutput, set_metadata
from dataeval.utils._array import flatten, to_numpy
from dataeval.utils._stats import StatsMap, combine_results, get_dataset_step_from_idx

IndexIssueMap: TypeAlias = Mapping[int, Mapping[str, float]]
TIndexIssueMap = TypeVar("TIndexIssueMap", IndexIssueMap, Sequence[IndexIssueMap])


def _reorganize_by_class_and_metric(
    result: IndexIssueMap, lstats: LabelStatsResult
) -> tuple[Mapping[str, Sequence[int]], Mapping[str | None, Mapping[str, int]]]:
    """Flip result from grouping by image to grouping by class and metric"""
    metrics: dict[str, list[int]] = {}

    # Create class_wise dict using index2label mapping and empty images
    index2label = lstats["index2label"]
    class_wise: dict[str | None, dict[str, int]] = {label: {} for label in index2label.values()}

    # Add empty images if they exist
    if lstats["empty_image_count"] > 0:
        class_wise[None] = {}

    # Group metrics and calculate class-wise counts
    for img, group in result.items():
        for extreme in group:
            metrics.setdefault(extreme, []).append(img)

            # Check if image is in empty images
            if img in lstats["empty_image_indices"]:
                class_wise[None][extreme] = class_wise[None].get(extreme, 0) + 1
            else:
                # Check regular class indices
                for class_idx, images in lstats["image_indices_per_class"].items():
                    if img in images:
                        class_name = index2label.get(class_idx, str(class_idx))
                        class_wise[class_name][extreme] = class_wise[class_name].get(extreme, 0) + 1

    return metrics, class_wise


def _create_table(
    metrics: Mapping[str, Sequence[int]], class_wise: Mapping[str | None, Mapping[str, int]]
) -> Sequence[str]:
    """Create table for displaying the results"""
    max_class_length = max(len(str(label)) for label in class_wise)
    max_class_length = max(max_class_length, len("Class"), 5)

    # Calculate actual totals to determine proper column width
    totals = []
    for class_cat, results in class_wise.items():
        total = sum(results.get(group, 0) for group in metrics)
        totals.append(total)

    # Single width calculation for both header and content
    max_total_width = max(len("Total"), max(len(str(total)) for total in totals), 5)

    # Calculate group column widths (single width for both header and content)
    group_widths = {}
    for group in sorted(metrics):
        # Find max width needed for this group's data
        max_data_width = max(len(str(results.get(group, 0))) for results in class_wise.values())
        base_width = max(len(str(group)), max_data_width)
        group_widths[group] = max(base_width, 5)

    table_header = " | ".join(
        [f"{'Class':>{max_class_length}}"]
        + [f"{group:^{group_widths[group]}}" for group in sorted(metrics)]
        + [f"{'Total':^{max_total_width}}"]
    )

    table_rows: Sequence[str] = []

    for class_cat, results in class_wise.items():
        table_value = [f"{class_cat:>{max_class_length}}"]
        total = 0
        for group in sorted(metrics):
            count = results.get(group, 0)
            table_value.append(f"{count:^{group_widths[group]}}")
            total += count
        table_value.append(f"{total:^{max_total_width}}")
        table_rows.append(" | ".join(table_value))

    return [table_header] + table_rows


def _create_pandas_dataframe(
    class_wise: Mapping[str | None, Mapping[str, int]],
) -> Sequence[Mapping[str, str | int | None]]:
    """Create data for pandas dataframe"""
    data = []
    for label, metrics_dict in class_wise.items():
        row: dict[str, str | int | None] = {"Class": label}
        total = sum(metrics_dict.values())
        row.update(metrics_dict)  # Add metric counts
        row["Total"] = total
        data.append(row)
    return data


@dataclass(frozen=True)
class OutliersOutput(DictOutput, Generic[TIndexIssueMap]):
    """
    Output class for :class:`.Outliers` lint detector.

    Attributes
    ----------
    issues : Mapping[int, Mapping[str, float]] | Sequence[Mapping[int, Mapping[str, float]]]
        Indices of image Outliers with their associated issue type and calculated values.

    - For a single dataset, a dictionary containing the indices of outliers and
      a dictionary showing the issues and calculated values for the given index.
    - For multiple stats outputs, a list of dictionaries containing the indices of
      outliers and their associated issues and calculated values.
    """

    issues: TIndexIssueMap

    def __len__(self) -> int:
        if isinstance(self.issues, Mapping):
            return len(self.issues)
        return sum(len(d) for d in self.issues)

    def to_table(self, labelstats: LabelStatsResult) -> str:
        """
        Formats the outlier output results as a table.

        Parameters
        ----------
        labelstats : LabelStatsResult
            Output of :func:`dataeval.core.label_stats`

        Returns
        -------
        str
        """
        if isinstance(self.issues, Mapping):
            metrics, classwise = _reorganize_by_class_and_metric(self.issues, labelstats)
            listed_table = _create_table(metrics, classwise)
            table = "\n".join(listed_table)
        else:
            outertable = []
            for d in self.issues:
                metrics, classwise = _reorganize_by_class_and_metric(d, labelstats)
                listed_table = _create_table(metrics, classwise)
                str_table = "\n".join(listed_table)
                outertable.append(str_table)
            table = "\n\n".join(outertable)
        return table

    def to_dataframe(self, labelstats: LabelStatsResult) -> pd.DataFrame:
        """
        Exports the outliers output results to a pandas DataFrame.

        Parameters
        ----------
        labelstats : LabelStatsResult
            Output of :func:`dataeval.core.label_stats`

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        This method requires `pandas <https://pandas.pydata.org/>`_ to be installed.
        """
        if isinstance(self.issues, Mapping):
            _, classwise = _reorganize_by_class_and_metric(self.issues, labelstats)
            data = _create_pandas_dataframe(classwise)
            df = pd.DataFrame(data)
        else:
            df_list = []
            for i, d in enumerate(self.issues):
                _, classwise = _reorganize_by_class_and_metric(d, labelstats)
                data = _create_pandas_dataframe(classwise)
                single_df = pd.DataFrame(data)
                single_df["Dataset"] = i
                df_list.append(single_df)
            df = pd.concat(df_list)
        return df


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

    def _get_outliers(self, stats: StatsMap) -> dict[int, dict[str, float]]:
        flagged_images: dict[int, dict[str, float]] = {}
        for stat, values in stats.items():
            if values.ndim == 1:
                mask = _get_outlier_mask(values.astype(np.float64), self.outlier_method, self.outlier_threshold)
                indices = np.flatnonzero(mask)
                for i, value in zip(indices, values[mask]):
                    flagged_images.setdefault(int(i), {})[stat] = value

        return {k: dict(sorted(v.items())) for k, v in sorted(flagged_images.items())}

    @overload
    def from_stats(self, stats: CalculationResult) -> OutliersOutput[IndexIssueMap]: ...

    @overload
    def from_stats(self, stats: Sequence[CalculationResult]) -> OutliersOutput[list[IndexIssueMap]]: ...

    @set_metadata(state=["outlier_method", "outlier_threshold"])
    def from_stats(
        self, stats: CalculationResult | Sequence[CalculationResult]
    ) -> OutliersOutput[IndexIssueMap] | OutliersOutput[list[IndexIssueMap]]:
        """
        Returns indices of Outliers with the issues identified for each.

        Parameters
        ----------
        stats : CalculationResult | Sequence[CalculationResult]
            The output(s) from calculate() with ImageStats.DIMENSION, PIXEL, or VISUAL flags

        Returns
        -------
        OutliersOutput
            Output class containing the indices of outliers and a dictionary showing
            the issues and calculated values for the given index.

        Example
        -------
        Evaluate the dataset:

        >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=3.5)
        >>> results = outliers.from_stats([stats1, stats2])
        >>> len(results)
        2
        >>> results.issues[0]
        {10: {'entropy': 0.2128, 'zeros': 0.05493}, 12: {'entropy': 0.2128, 'std': 0.00536, 'var': 2.87e-05, 'zeros': 0.05493}}
        >>> results.issues[1]
        {}
        """  # noqa: E501
        combined_stats, dataset_steps = combine_results(stats)
        outliers = self._get_outliers(combined_stats)

        if not isinstance(stats, Sequence):
            return OutliersOutput(outliers)

        # Split results back to individual datasets
        output_list: list[dict[int, dict[str, float]]] = [{} for _ in stats]
        for idx, issue in outliers.items():
            k, v = get_dataset_step_from_idx(idx, dataset_steps)
            output_list[k][v] = issue

        return OutliersOutput(output_list)

    @set_metadata(state=["outlier_threshold"])
    def from_clusters(
        self,
        embeddings: ArrayND[float],
        cluster_result: ClusterResult,
        threshold: float | None = None,
    ) -> OutliersOutput[IndexIssueMap]:
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
    ) -> dict[int, dict[str, float]]:
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
        dict[int, dict[str, float]]
            Dictionary mapping outlier indices to their issue details.
            Each issue dict contains:
            - 'cluster_distance': distance in std dev from cluster mean
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

        # Build the result dictionary with issue details
        outlier_indices = np.nonzero(is_outlier)[0]
        flagged_images: dict[int, dict[str, float]] = {}

        for idx in outlier_indices:
            cluster_idx = nearest_cluster_idx[idx]
            distance = float(min_distances[idx])
            mean = float(cluster_distances_mean[cluster_idx])
            std = float(cluster_distances_std[cluster_idx])

            # Calculate number of standard deviations from mean
            std_devs = (distance - mean) / std if std > EPSILON else 0.0

            flagged_images[int(idx)] = {
                "cluster_distance": std_devs,
            }

        return flagged_images

    @set_metadata(state=["use_dimension", "use_pixel", "use_visual", "outlier_method", "outlier_threshold"])
    def evaluate(self, data: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]) -> OutliersOutput[IndexIssueMap]:
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
        >>> list(results.issues)
        [10, 12]
        >>> results.issues[10]
        {'contrast': 1.25, 'entropy': 0.2128, 'zeros': 0.05493}

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

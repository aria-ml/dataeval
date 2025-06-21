from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from dataeval.config import EPSILON
from dataeval.data._images import Images
from dataeval.metrics.stats._base import combine_stats, get_dataset_step_from_idx
from dataeval.metrics.stats._imagestats import imagestats
from dataeval.outputs import DimensionStatsOutput, ImageStatsOutput, OutliersOutput, PixelStatsOutput, VisualStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.outputs._linters import IndexIssueMap, OutlierStatsOutput
from dataeval.outputs._stats import BASE_ATTRS
from dataeval.typing import ArrayLike, Dataset


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
    outlier_method : ["modzscore" | "zscore" | "iqr"], optional - default "modzscore"
        Statistical method used to identify outliers
    outlier_threshold : float, optional - default None
        Threshold value for the given ``outlier_method``, above which data is considered an outlier.
        Uses method specific default if `None`

    Attributes
    ----------
    stats : tuple[DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput]
        Various stats output classes that hold the value of each metric for each image

    See Also
    --------
    :term:`Duplicates`

    Note
    ----
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
        self.stats: ImageStatsOutput
        self.use_dimension = use_dimension
        self.use_pixel = use_pixel
        self.use_visual = use_visual
        self.outlier_method: Literal["zscore", "modzscore", "iqr"] = outlier_method
        self.outlier_threshold = outlier_threshold

    def _get_outliers(self, stats: dict) -> dict[int, dict[str, float]]:
        flagged_images: dict[int, dict[str, float]] = {}
        for stat, values in stats.items():
            if stat in BASE_ATTRS:
                continue
            if values.ndim == 1:
                mask = _get_outlier_mask(values.astype(np.float64), self.outlier_method, self.outlier_threshold)
                indices = np.flatnonzero(mask)
                for i, value in zip(indices, values[mask]):
                    flagged_images.setdefault(int(i), {}).update({stat: value})

        return dict(sorted(flagged_images.items()))

    @overload
    def from_stats(self, stats: OutlierStatsOutput | ImageStatsOutput) -> OutliersOutput[IndexIssueMap]: ...

    @overload
    def from_stats(self, stats: Sequence[OutlierStatsOutput]) -> OutliersOutput[list[IndexIssueMap]]: ...

    @set_metadata(state=["outlier_method", "outlier_threshold"])
    def from_stats(
        self, stats: OutlierStatsOutput | ImageStatsOutput | Sequence[OutlierStatsOutput]
    ) -> OutliersOutput[IndexIssueMap] | OutliersOutput[list[IndexIssueMap]]:
        """
        Returns indices of Outliers with the issues identified for each.

        Parameters
        ----------
        stats : OutlierStatsOutput | ImageStatsOutput | Sequence[OutlierStatsOutput]
            The output(s) from a dimensionstats, pixelstats, or visualstats metric
            analysis or an aggregate ImageStatsOutput

        Returns
        -------
        OutliersOutput
            Output class containing the indices of outliers and a dictionary showing
            the issues and calculated values for the given index.

        See Also
        --------
        dimensionstats
        pixelstats
        visualstats

        Example
        -------
        Evaluate the dataset:

        >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=3.5)
        >>> results = outliers.from_stats([stats1, stats2])
        >>> len(results)
        2
        >>> results.issues[0]
        {10: {'entropy': 0.2128}, 12: {'std': 0.00536, 'var': 2.87e-05, 'entropy': 0.2128}}
        >>> results.issues[1]
        {}
        """
        if isinstance(stats, ImageStatsOutput | DimensionStatsOutput | PixelStatsOutput | VisualStatsOutput):
            return OutliersOutput(self._get_outliers(stats.data()))

        if not isinstance(stats, Sequence):
            raise TypeError(
                "Invalid stats output type; only use output from dimensionstats, pixelstats or visualstats."
            )

        stats_map: dict[type, list[int]] = {}
        for i, stats_output in enumerate(stats):
            if not isinstance(
                stats_output, ImageStatsOutput | DimensionStatsOutput | PixelStatsOutput | VisualStatsOutput
            ):
                raise TypeError(
                    "Invalid stats output type; only use output from dimensionstats, pixelstats or visualstats."
                )
            stats_map.setdefault(type(stats_output), []).append(i)

        output_list: list[dict[int, dict[str, float]]] = [{} for _ in stats]
        for _, indices in stats_map.items():
            substats, dataset_steps = combine_stats([stats[i] for i in indices])
            outliers = self._get_outliers(substats.data())
            for idx, issue in outliers.items():
                k, v = get_dataset_step_from_idx(idx, dataset_steps)
                output_list[indices[k]][v] = issue

        return OutliersOutput(output_list)

    @set_metadata(state=["use_dimension", "use_pixel", "use_visual", "outlier_method", "outlier_threshold"])
    def evaluate(self, data: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]) -> OutliersOutput[IndexIssueMap]:
        """
        Returns indices of Outliers with the issues identified for each

        Parameters
        ----------
        data : Iterable[Array], shape - (C, H, W)
            A dataset of images in an Array format

        Returns
        -------
        OutliersOutput
            Output class containing the indices of outliers and a dictionary showing
            the issues and calculated values for the given index.

        Example
        -------
        Evaluate the dataset:

        >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=3.5)
        >>> results = outliers.evaluate(outlier_images)
        >>> list(results.issues)
        [10, 12]
        >>> results.issues[10]
        {'contrast': 1.25, 'zeros': 0.05493, 'entropy': 0.2128}
        """
        images = Images(data) if isinstance(data, Dataset) else data
        self.stats = imagestats(images)
        outliers = self._get_outliers(self.stats.data())
        return OutliersOutput(outliers)

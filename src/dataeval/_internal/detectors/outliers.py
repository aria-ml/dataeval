from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Literal, NamedTuple, Sequence, TypeVar, Union, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.detectors.merged_stats import combine_stats, get_dataset_step_from_idx
from dataeval._internal.metrics.stats import (
    INDEX_MAP,
    DimensionStatsOutput,
    PixelStatsOutput,
    VisualStatsOutput,
    dimensionstats,
    pixelstats,
    visualstats,
)
from dataeval._internal.output import OutputMetadata, set_metadata

IndexIssueMap = dict[int, dict[str, float]]
OutlierStatsOutput = Union[DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput]
TIndexIssueMap = TypeVar("TIndexIssueMap", IndexIssueMap, list[IndexIssueMap])


class StatsOutputs(NamedTuple):
    """
    This class represents the outputs of various stats functions against a single
    dataset, such that each index across all stat outputs are representative of
    the same source image.  Modifying or mixing outputs will result in inaccurate
    outlier calculations if not created correctly.

    Attributes
    ----------
    dimensionstats : DimensionStatsOutput or None
    pixelstats: PixelStatsOutput or None
    visualstats: VisualStatsOutput or None
    """

    dimensionstats: DimensionStatsOutput | None
    pixelstats: PixelStatsOutput | None
    visualstats: VisualStatsOutput | None


@dataclass(frozen=True)
class OutliersOutput(Generic[TIndexIssueMap], OutputMetadata):
    """
    Attributes
    ----------
    issues : dict[int, dict[str, float]] | list[dict[int, dict[str, float]]]
        Indices of image outliers with their associated issue type and calculated values.

    - For a single dataset, a dictionary containing the indices of outliers and
      a dictionary showing the issues and calculated values for the given index.
    - For multiple stats outputs, a list of dictionaries containing the indices of
      outliers and their associated issues and calculated values.
    """

    issues: TIndexIssueMap

    def __len__(self):
        if isinstance(self.issues, dict):
            return len(self.issues)
        else:
            return sum(len(d) for d in self.issues)


def _get_outlier_mask(
    values: NDArray, method: Literal["zscore", "modzscore", "iqr"], threshold: float | None
) -> NDArray:
    values = values.astype(np.float64)
    if method == "zscore":
        threshold = threshold if threshold else 3.0
        std = np.std(values)
        abs_diff = np.abs(values - np.mean(values))
        return (abs_diff / std) > threshold
    elif method == "modzscore":
        threshold = threshold if threshold else 3.5
        abs_diff = np.abs(values - np.median(values))
        med_abs_diff = np.median(abs_diff) if np.median(abs_diff) != 0 else np.mean(abs_diff)
        mod_z_score = 0.6745 * abs_diff / med_abs_diff
        return mod_z_score > threshold
    elif method == "iqr":
        threshold = threshold if threshold else 1.5
        qrt = np.percentile(values, q=(25, 75), method="midpoint")
        iqr = (qrt[1] - qrt[0]) * threshold
        return (values < (qrt[0] - iqr)) | (values > (qrt[1] + iqr))
    else:
        raise ValueError("Outlier method must be 'zscore' 'modzscore' or 'iqr'.")


class Outliers:
    r"""
    Calculates statistical outliers of a dataset using various statistical tests applied to each image

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
    Duplicates

    Notes
    ------
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

    >>> outliers = Outliers(outlier_method="zscore", outlier_threshold=2.75)
    """

    def __init__(
        self,
        use_dimension: bool = True,
        use_pixel: bool = True,
        use_visual: bool = True,
        outlier_method: Literal["zscore", "modzscore", "iqr"] = "modzscore",
        outlier_threshold: float | None = None,
    ):
        self.stats: StatsOutputs
        self.use_dimension = use_dimension
        self.use_pixel = use_pixel
        self.use_visual = use_visual
        self.outlier_method: Literal["zscore", "modzscore", "iqr"] = outlier_method
        self.outlier_threshold = outlier_threshold

    def _get_outliers(self, stats: dict) -> dict[int, dict[str, float]]:
        flagged_images: dict[int, dict[str, float]] = {}
        for stat, values in stats.items():
            if stat == INDEX_MAP:
                continue
            if values.ndim == 1 and np.std(values) != 0:
                mask = _get_outlier_mask(values, self.outlier_method, self.outlier_threshold)
                indices = np.flatnonzero(mask)
                for i, value in zip(indices, values[mask]):
                    flagged_images.setdefault(i, {}).update({stat: np.round(value, 2)})

        return dict(sorted(flagged_images.items()))

    @overload
    def from_stats(self, stats: StatsOutputs | OutlierStatsOutput) -> OutliersOutput[IndexIssueMap]: ...

    @overload
    def from_stats(self, stats: Sequence[OutlierStatsOutput]) -> OutliersOutput[list[IndexIssueMap]]: ...

    @set_metadata("dataeval.detectors", ["outlier_method", "outlier_threshold"])
    def from_stats(self, stats: StatsOutputs | OutlierStatsOutput | Sequence[OutlierStatsOutput]) -> OutliersOutput:
        """
        Returns indices of outliers with the issues identified for each

        Parameters
        ----------
        stats : StatsOutputs | OutlierStatsOutput | Sequence[OutlierStatsOutput]
            The output(s) from an dimensionstats, pixelstats, or visualstats metric analysis

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

        >>> outliers.from_stats([stats1, stats2])
        OutliersOutput(issues=[{10: {'std': 0.0, 'entropy': 0.21}, 12: {'std': 0.01, 'entropy': 0.21}}, {}])
        """
        if isinstance(stats, StatsOutputs):
            outliers = self._get_outliers({k: v for o in stats if o is not None for k, v in o.dict().items()})
            return OutliersOutput(outliers)

        if isinstance(stats, (DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput)):
            return OutliersOutput(self._get_outliers(stats.dict()))

        if not isinstance(stats, Sequence):
            raise TypeError(
                "Invalid stats output type; only use output from dimensionstats, pixelstats or visualstats."
            )

        stats_map: dict[type, list[int]] = {}
        for i, stats_output in enumerate(stats):
            if not isinstance(stats_output, (DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput)):
                raise TypeError(
                    "Invalid stats output type; only use output from dimensionstats, pixelstats or visualstats."
                )
            stats_map.setdefault(type(stats_output), []).append(i)

        output_list: list[dict[int, dict[str, float]]] = [{} for _ in stats]
        for _, indices in stats_map.items():
            substats, dataset_steps = combine_stats([stats[i] for i in indices])
            outliers = self._get_outliers(substats.dict())
            for idx, issue in outliers.items():
                k, v = get_dataset_step_from_idx(idx, dataset_steps)
                output_list[indices[k]][v] = issue

        return OutliersOutput(output_list)

    @set_metadata(
        "dataeval.detectors",
        [
            "use_dimension",
            "use_pixel",
            "use_visual",
            "outlier_method",
            "outlier_threshold",
        ],
    )
    def evaluate(self, data: Iterable[ArrayLike]) -> OutliersOutput[IndexIssueMap]:
        """
        Returns indices of outliers with the issues identified for each

        Parameters
        ----------
        data : Iterable[ArrayLike], shape - (C, H, W)
            A dataset of images in an ArrayLike format

        Returns
        -------
        OutliersOutput
            Output class containing the indices of outliers and a dictionary showing
            the issues and calculated values for the given index.

        Example
        -------
        Evaluate the dataset:

        >>> outliers.evaluate(images)
        OutliersOutput(issues={10: {'std': 0.0, 'entropy': 0.21, 'blurriness': 1.26, 'contrast': 1.25, 'zeros': 0.05}, 12: {'std': 0.01, 'entropy': 0.21, 'blurriness': 1.51, 'contrast': 1.25, 'zeros': 0.05}})
        """  # noqa: E501
        self.stats = StatsOutputs(
            dimensionstats(data) if self.use_dimension else None,
            pixelstats(data) if self.use_pixel else None,
            visualstats(data) if self.use_visual else None,
        )
        outliers = self._get_outliers({k: v for o in self.stats if o is not None for k, v in o.dict().items()})
        return OutliersOutput(outliers)

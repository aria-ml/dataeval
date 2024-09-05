from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.flags import verify_supported
from dataeval._internal.output import OutputMetadata, set_metadata
from dataeval.flags import ImageStat
from dataeval.metrics import imagestats


@dataclass(frozen=True)
class LinterOutput(OutputMetadata):
    """
    Attributes
    ----------
    issues : Dict[int, Dict[str, float]]
        Dictionary containing the indices of outliers and a dictionary showing
        the issues and calculated values for the given index.
    """

    issues: Dict[int, Dict[str, float]]


def _get_outlier_mask(
    values: NDArray, method: Literal["zscore", "modzscore", "iqr"], threshold: Optional[float]
) -> NDArray:
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


class Linter:
    r"""
    Calculates statistical outliers of a dataset using various statistical tests applied to each image

    Parameters
    ----------
    flags : ImageStat, default ImageStat.ALL_PROPERTIES | ImageStat.ALL_VISUALS
        Metric(s) to calculate for each image - calculates all metrics if None
        Only supports ImageStat.ALL_STATS
    outlier_method : ["modzscore" | "zscore" | "iqr"], optional - default "modzscore"
        Statistical method used to identify outliers
    outlier_threshold : float, optional - default None
        Threshold value for the given ``outlier_method``, above which data is considered an outlier.
        Uses method specific default if `None`

    Attributes
    ----------
    stats : Dict[str, Any]
        Dictionary to hold the value of each metric for each image

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
    Initialize the Linter class:

    >>> lint = Linter()

    Specifying specific metrics to analyze:

    >>> lint = Linter(flags=ImageStat.SIZE | ImageStat.ALL_VISUALS)

    Specifying an outlier method:

    >>> lint = Linter(outlier_method="iqr")

    Specifying an outlier method and threshold:

    >>> lint = Linter(outlier_method="zscore", outlier_threshold=2.5)
    """

    def __init__(
        self,
        flags: ImageStat = ImageStat.ALL_PROPERTIES | ImageStat.ALL_VISUALS,
        outlier_method: Literal["zscore", "modzscore", "iqr"] = "modzscore",
        outlier_threshold: Optional[float] = None,
    ):
        verify_supported(flags, ImageStat.ALL_STATS)
        self.flags = flags
        self.outlier_method: Literal["zscore", "modzscore", "iqr"] = outlier_method
        self.outlier_threshold = outlier_threshold

    def _get_outliers(self) -> dict:
        flagged_images = {}
        stats_dict = self.stats.dict()
        for stat, values in stats_dict.items():
            if not isinstance(values, np.ndarray):
                continue

            if values.ndim == 1 and np.std(values) != 0:
                mask = _get_outlier_mask(values, self.outlier_method, self.outlier_threshold)
                indices = np.flatnonzero(mask)
                for i, value in zip(indices, values[mask]):
                    flagged_images.setdefault(i, {}).update({stat: np.round(value, 2)})

        return dict(sorted(flagged_images.items()))

    @set_metadata("dataeval.detectors", ["flags", "outlier_method", "outlier_threshold"])
    def evaluate(self, images: Iterable[ArrayLike]) -> LinterOutput:
        """
        Returns indices of outliers with the issues identified for each

        Parameters
        ----------
        images : Iterable[ArrayLike], shape - (N, C, H, W)
            A dataset in an ArrayLike format.
            Function expects the data to have 3 dimensions, CxHxW.

        Returns
        -------
        LinterOutput
            Output class containing the indices of outliers and a dictionary showing
            the issues and calculated values for the given index.

        Example
        -------
        Evaluate the dataset:

        >>> lint.evaluate(images)
        LinterOutput(issues={18: {'brightness': 0.78}, 25: {'brightness': 0.98}})
        """
        self.stats = imagestats(images, self.flags)
        return LinterOutput(self._get_outliers())

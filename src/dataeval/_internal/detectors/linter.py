from typing import Literal, Optional, Sequence, Union

import numpy as np

from dataeval._internal.flags import ImageProperty, ImageVisuals, LinterFlags
from dataeval._internal.metrics.stats import ImageStats


def _get_outlier_mask(
    values: np.ndarray, method: Literal["zscore", "modzscore", "iqr"], threshold: Optional[float]
) -> np.ndarray:
    if method == "zscore":
        threshold = threshold if threshold else 3.0
        std = np.std(values)
        abs_diff = np.abs(values - np.mean(values))
        return (abs_diff / std) > threshold
    elif method == "modzscore":
        threshold = threshold if threshold else 3.5
        abs_diff = np.abs(values - np.median(values))
        med_abs_diff = np.median(abs_diff)
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
    """
    Calculates statistical outliers of a dataset using various statistical tests applied to each image

    There are 3 different statistical methods:

    - zscore
    - modzscore
    - iqr

    The default statistical method is `modzscore`.

    The z score method is based on the difference between the data point and the mean of the data.
    The default threshold value for `zscore` is 3.
    Z score = \|x_i - mu| / sigma

    The modified z score method is based on the difference between the data point and the median of the data.
    The default threshold value for `modzscore` is 3.5.
    Modified z score = 0.6745 * \|x_i - x̃| / MAD, where MAD is the median absolute deviation

    The interquartile range method is based on the difference between the data point and
    the difference between the 75th and 25th qartile.
    The default threshold value for `iqr` is 1.5.
    Interquartile range $= threshold * (Q_3 - Q_1)$
    """

    def __init__(
        self,
        images: np.ndarray,
        flags: Optional[Union[LinterFlags, Sequence[LinterFlags]]] = None,
        outlier_method: Literal["zscore", "modzscore", "iqr"] = "modzscore",
        outlier_threshold: Optional[float] = None,
    ):
        flags = flags if flags is not None else (ImageProperty.ALL, ImageVisuals.ALL)
        self.stats = ImageStats(flags)
        self.images = images
        self.outlier_method: Literal["zscore", "modzscore", "iqr"] = outlier_method
        self.outlier_threshold = outlier_threshold

    def _get_outliers(self) -> dict:
        flagged_images = {}

        for stat, values in self.results.items():
            if not isinstance(values, np.ndarray):
                continue

            if values.ndim == 1 and np.std(values) != 0:
                mask = _get_outlier_mask(values, self.outlier_method, self.outlier_threshold)
                indices = np.flatnonzero(mask)
                for i, value in zip(indices, values[mask]):
                    flagged_images.setdefault(i, {}).update({stat: np.round(value, 2)})

        return dict(sorted(flagged_images.items()))

    def evaluate(self) -> dict:
        """
        Returns indices of outliers with and the issues identified for each

        Returns
        -------
        Dict[int, Dict[str, float]]
            Dictionary containing the indices of outliers and a dictionary issues and calculated values
        """
        self.stats.reset()
        self.stats.update(self.images)
        self.results = self.stats.compute()
        return self._get_outliers()

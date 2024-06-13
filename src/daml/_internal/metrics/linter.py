from typing import Literal, Optional, Sequence, Union

import numpy as np

from daml._internal.metrics.flags import ImageProperty, ImageVisuals, LinterFlags
from daml._internal.metrics.stats import ImageStats


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
        abs_diff = np.abs(values - np.mean(values))
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
    Calculates statistical outliers of a dataset using various statistical
    tests applied to each image
    """

    def __init__(
        self,
        images: np.ndarray,
        flags: Optional[Union[LinterFlags, Sequence[LinterFlags]]] = None,
    ):
        flags = flags if flags is not None else (ImageProperty.ALL, ImageVisuals.ALL)
        self.stats = ImageStats(flags)
        self.images = images

    def _get_outliers(
        self,
        outlier_method: Literal["zscore", "modzscore", "iqr"] = "modzscore",
        outlier_threshold: Optional[float] = None,
    ) -> dict:
        flagged_images = {}

        for stat, values in self.results.items():
            if not isinstance(values, np.ndarray):
                continue

            if values.ndim == 1 and np.std(values) != 0:
                mask = _get_outlier_mask(values, outlier_method, outlier_threshold)
                indices = np.flatnonzero(mask)
                for i, value in zip(indices, values[mask]):
                    flagged_images.setdefault(i, {}).update({stat: np.round(value, 2)})

        return dict(sorted(flagged_images.items()))

    def evaluate(self) -> dict:
        self.stats.reset()
        self.stats.update(self.images)
        self.results = self.stats.compute()
        return self._get_outliers()

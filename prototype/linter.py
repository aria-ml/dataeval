from typing import Literal, Optional

import numpy as np

from daml._internal.metrics.hash import pchash, xxhash
from daml._internal.metrics.stats import DatasetStats


def calculate_mask(
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
        raise TypeError("Outlier method must be 'zscore' 'modzscore' or 'iqr'.")


class Linter:
    def __init__(self, images: np.ndarray):
        print("Processing images...")
        self.ds = DatasetStats(images)
        print("Hashing images...")
        self.hashes = []
        for image in images:
            self.hashes.append({"xxhash": xxhash(image), "fzhash": pchash(image)})
        for image in images[0:100]:
            self.hashes.append({"xxhash": xxhash(image), "fzhash": pchash(image)})

    def get_outliers(
        self,
        outlier_method: Literal["zscore", "modzscore", "iqr"] = "modzscore",
        outlier_threshold: Optional[float] = None,
    ) -> dict:
        flagged_images = {}

        for stat, values in self.ds.get_image_stats().items():
            if values.ndim == 1 and np.std(values) != 0:
                mask = calculate_mask(values, outlier_method, outlier_threshold)
                indices = np.flatnonzero(mask)
                for i, value in zip(indices, values[mask]):
                    flagged_images.setdefault(i, {}).update({stat: np.round(value, 2)})

        return dict(sorted(flagged_images.items()))

    def get_duplicates(self) -> dict:
        exact = {}
        near = {}
        for i, hashes in enumerate(self.hashes):
            exact.setdefault(hashes["xxhash"], []).append(i)
            near.setdefault(hashes["fzhash"], []).append(i)

        return {
            "exact": {k: v for k, v in exact.items() if len(v) > 1},
            "near": {k: v for k, v in near.items() if len(v) > 1},
        }

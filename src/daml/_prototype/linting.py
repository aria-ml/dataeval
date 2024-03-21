from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import kurtosis, skew

from daml._internal.metrics.base import EvaluateMixin

# Comments from John regarding implementation
"""
images is constrained to be uniform w/h.  Probably a reasonable assumption for now, but
there are datasets with non-uniform image sizes.

Flattening all images out together makes sense for what's there.
I wonder if there will be value in per-image statistics down the road, too.

May want more than 10 bins on the histogram given typical image sizes

size = height * width : size might be an ambiguous variable name.
Maybe pixel_area or something?

in _evaluate_boxes, does NxPx4 imply the same number of objects in each image?
If so, we might need to rethink how we store them.

on boxes_data = np.concatenate(...(line 97), would expand_dims solve this more cleanly?
Or would it be easier to just catch the case where the ndim of the input data is 2 or 4?
"""


class Linting(EvaluateMixin):
    """
    Basic Image and Label Statistics

    Parameters
    ----------
    images : np.ndarray
        A numpy array of n_samples of images either (H, W) or (C, H, W).

    labels : np.ndarray
        A numpy array of n_samples of class labels with M unique classes.

    boxes : np.ndarray
        A numpy array of n_samples of object boxes with P objects per image
        (n_samples, P, H, W)
    """

    def __init__(
        self,
        images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.boxes = boxes

    def evaluate(self) -> Dict[str, float]:
        """
        Returns
        -------
        Dict[str, float]

        """
        results: Dict[str, Any] = {}
        if self.images:
            img_stats = self._evaluate_images()
            results["images"] = img_stats
        if self.labels:
            label_stats = self._evaluate_labels()
            results["labels"] = label_stats
        if self.boxes:
            box_stats = self._evaluate_boxes()
            results["boxes"] = box_stats

        return results

    def _evaluate_images_or_boxes(self, data: np.ndarray) -> Dict[str, Any]:
        if data.ndim == 3:  # Assuming (C, H, W)
            data = np.expand_dims(
                data, axis=0
            )  # Convert to (N, C, H, W) for consistency
        n_samples, channels, height, width = data.shape

        pixel_values = data.reshape(-1, channels)
        size = height * width
        aspect_ratio = width / height

        stats = {
            "size": size,
            "aspect_ratio": aspect_ratio,
            "num_channels": channels,
            "pixel_value_mean": np.mean(pixel_values, axis=0),
            "pixel_value_median": np.median(pixel_values, axis=0),
            "pixel_value_max": np.max(pixel_values, axis=0),
            "pixel_value_min": np.min(pixel_values, axis=0),
            "pixel_value_variance": np.var(pixel_values, axis=0),
            "pixel_value_skew": skew(pixel_values, axis=0),
            "pixel_value_kurtosis": kurtosis(pixel_values, axis=0),
            "pixel_value_histogram": [
                np.histogram(pixel_values[:, i], bins=10)[0].tolist()
                for i in range(channels)
            ],
        }

        return stats

    def _evaluate_images(self) -> Dict[str, Any]:
        if self.images is None:
            return {}

        return self._evaluate_images_or_boxes(self.images)

    def _evaluate_boxes(self) -> Dict[str, Any]:
        if self.boxes is None:
            return {}

        # Assuming boxes is a (N, P, 4) array where each box is defined by
        # (x_min, y_min, x_max, y_max)

        # Convert boxes to an array of dimensions for the sake of statistical analysis
        box_dimensions = (
            self.boxes[:, :, 2:] - self.boxes[:, :, :2]
        )  # (N, P, 2) where 2 corresponds to (width, height)
        box_dimensions = box_dimensions.reshape(-1, 2)  # Flatten to (N*P, 2)
        boxes_data = np.concatenate(
            [box_dimensions, np.ones((len(box_dimensions), 1))], axis=-1
        )  # Add a dummy channel dimension
        return self._evaluate_images_or_boxes(boxes_data)

    def _evaluate_labels(self) -> Dict[str, Any]:
        if self.labels is None:
            return {}

        label_counts = Counter(self.labels.flatten())
        return {"label_counts": dict(label_counts)}

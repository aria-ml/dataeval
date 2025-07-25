from __future__ import annotations

__all__ = []

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from dataeval.core._nullmodel import (
    BinaryClassMetricFunction,
    ConfusionMatrix,
    estimate_accuracy,
    estimate_false_positive_rate,
    estimate_multiclass_accuracy,
    estimate_precision,
    estimate_true_positive_rate,
    get_confusion_matrix,
    reduce_macro,
    reduce_micro,
)
from dataeval.outputs._base import set_metadata
from dataeval.outputs._estimators import NullModelMetrics, NullModelMetricsOutput
from dataeval.typing import ArrayLike
from dataeval.utils._array import as_numpy

BASE_METRICS: dict[str, BinaryClassMetricFunction] = {
    "precision": estimate_precision,
    "recall": estimate_true_positive_rate,
    "false_positive_rate": estimate_false_positive_rate,
}

BINARY_ONLY_METRICS: dict[str, BinaryClassMetricFunction] = {
    "accuracy": estimate_accuracy,
}

AVERAGES: dict[str, Callable[[BinaryClassMetricFunction, Sequence[ConfusionMatrix]], np.float64]] = {
    "micro": reduce_micro,
    "macro": reduce_macro,
}


class NullModelMetricsCalculator:
    """Unified calculator for both binary and multiclass cases."""

    def __init__(self, test_labels: ArrayLike, train_labels: ArrayLike | None) -> None:
        self.test_probs, self.train_probs = self._prepare_inputs(test_labels, train_labels)
        self.classes = np.nonzero(self.test_probs)[0]
        self.is_multiclass = len(self.classes) > 2 or np.count_nonzero(self.train_probs) > 2

    def _prepare_inputs(
        self, test_labels: ArrayLike, train_labels: ArrayLike | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate inputs and prepare probability distributions."""
        # Convert ArrayLike inputs to NumPy arrays
        test_np = as_numpy(test_labels)
        train_np = as_numpy(train_labels)

        if not test_np.size:
            raise ValueError("Empty or null test labels provided")

        # Calculate class probabilities
        test_class_probs = np.bincount(test_np) / test_np.size
        train_class_probs = (np.bincount(train_np) / train_np.size) if train_np.size else train_np

        # Align probability arrays
        test_classes = len(test_class_probs)
        train_classes = len(train_class_probs)
        max_classes = max(test_classes, train_classes)
        if test_classes < max_classes:
            test_class_probs = np.pad(test_class_probs, (0, max_classes - test_classes))
        if train_classes and train_classes < max_classes:
            train_class_probs = np.pad(train_class_probs, (0, max_classes - train_classes))
            train_class_probs = train_class_probs / np.sum(train_class_probs)

        return test_class_probs, train_class_probs

    def calculate(self) -> NullModelMetricsOutput:
        """Calculate metrics for all available null models."""
        uniform_probs = np.where(self.test_probs != 0, 1.0 / len(self.classes), 0.0)
        uniform_metrics = self._calculate_metrics(uniform_probs)

        if self.train_probs.size:
            dominant_probs = np.zeros_like(self.train_probs)
            dominant_probs[np.argmax(self.train_probs)] = 1.0
            dominant_metrics = self._calculate_metrics(dominant_probs)
            proportional_metrics = self._calculate_metrics(self.train_probs)

            return NullModelMetricsOutput(
                uniform_random=uniform_metrics,
                dominant_class=dominant_metrics,
                proportional_random=proportional_metrics,
            )

        return NullModelMetricsOutput(uniform_random=uniform_metrics)

    def _calculate_metrics(self, prediction_probs: NDArray[np.float64]) -> NullModelMetrics:
        """Calculate metrics for a given prediction probability distribution."""
        # Calculate confusion matrices for each class (one-vs-rest)
        confusion_matrices: list[ConfusionMatrix] = []
        for class_idx in self.classes:
            pred_1vr = np.array([prediction_probs[class_idx], 1 - prediction_probs[class_idx]])
            test_1vr = np.array([self.test_probs[class_idx], 1 - self.test_probs[class_idx]])
            confusion_matrices.append(get_confusion_matrix(test_1vr, pred_1vr))

        # Calculate binary metrics
        metrics: dict[str, float] = {}
        metric_map = BASE_METRICS if self.is_multiclass else BASE_METRICS | BINARY_ONLY_METRICS

        for metric_name, metric_fn in metric_map.items():
            for avg_name, avg_fn in AVERAGES.items():
                metrics[f"{metric_name}_{avg_name}"] = avg_fn(metric_fn, confusion_matrices)

        # Add multiclass specific metrics
        if self.is_multiclass:
            metrics["multiclass_accuracy"] = estimate_multiclass_accuracy(self.test_probs, prediction_probs)

        return NullModelMetrics(**metrics)


@set_metadata
def null_model_metrics(test_labels: ArrayLike, train_labels: ArrayLike | None = None) -> NullModelMetricsOutput:
    """
    Calculate null model metrics (dummy classifiers metrics) for given class distributions.

    This function calculates benchmark performance metrics for random classifiers on the training and testing labels
    based on the class distributions.

    Null models to be evaluated:

    - Uniform Random: Classifier applies equal probability to each class
    - Dominant Class: Classifier will choose the most frequent class in the training set (requires training labels)
    - Proportional Random: Classifier applies distribution probabilities from training set (requires training labels)

    The calculated metrics are to be used as a lower-bound performance baseline for model evaluation.

    Parameters
    ----------
    test_labels : ArrayLike
        Class distribution from test set. Each index is the integer representation of the associated class label,
        e.g. [0, 1, 1, 2, 3].
    train_labels : ArrayLike | None, default None
        Class distribution from training set. Each index is the integer representation of the associated class label,
        e.g. [0, 1, 1, 2, 3]. When None, skips calculating class frequencies and does not report metrics for the
        dominant class and proportional random models.

    Raises
    ------
    ValueError
        If test_labels is None or empty

    Returns
    -------
    NullModelMetricsOutput
        Output class mapping null model metrics with null models
    """
    return NullModelMetricsCalculator(test_labels, train_labels).calculate()

__all__ = []

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired

from dataeval.protocols import ArrayLike
from dataeval.types import Array1D
from dataeval.utils.arrays import as_numpy

_logger = logging.getLogger(__name__)

ConfusionMatrix = tuple[np.floating[Any], np.floating[Any], np.floating[Any], np.floating[Any]]
BinaryClassMetricFunction = Callable[[ConfusionMatrix], np.float64]


class NullModelMetrics(TypedDict):
    """
    Per-model results for null-model metrics.

    Attributes
    ----------
    precision_macro : float
        Macro-averaged precision across all classes
    precision_micro : float
        Micro-averaged precision across all classes
    recall_macro : float
        Macro-averaged recall across all classes
    recall_micro : float
        Micro-averaged recall across all classes
    false_positive_rate_macro : float
        Macro-averaged false positive rate across all classes
    false_positive_rate_micro : float
        Micro-averaged false positive rate across all classes
    accuracy_macro : float, optional
        Macro-averaged accuracy (only for binary classification)
    accuracy_micro : float, optional
        Micro-averaged accuracy (only for binary classification)
    multiclass_accuracy : float, optional
        Multiclass accuracy (only for multiclass classification)
    """

    precision_macro: float
    precision_micro: float
    recall_macro: float
    recall_micro: float
    false_positive_rate_macro: float
    false_positive_rate_micro: float
    accuracy_macro: NotRequired[float]
    accuracy_micro: NotRequired[float]
    multiclass_accuracy: NotRequired[float]


class NullModelMetricsResult(TypedDict):
    """
    Result mapping for null model metrics evaluation.

    Attributes
    ----------
    uniform_random : NullModelMetrics
        Metrics for uniform random classifier (assigns equal probability to each class)
    dominant_class : NullModelMetrics, optional
        Metrics for dominant class classifier (always predicts most frequent training class)
    proportional_random : NullModelMetrics, optional
        Metrics for proportional random classifier (samples from training class distribution)
    """

    uniform_random: NullModelMetrics
    dominant_class: NotRequired[NullModelMetrics]
    proportional_random: NotRequired[NullModelMetrics]


def nullmodel_accuracy(
    class_prob: Array1D[float], model_prob: Array1D[float], *, multiclass: bool = False
) -> np.float64:
    """
    Calculates accuracy from binary classification results.

    Parameters
    ----------
    class_prob : Array1D[float]
        Class-wise probabilities for the test set. Can be a 1D list, or array-like object.
    model_prob : Array1D[float]
        Probability distribution for given null model. Can be a 1D list, or array-like object.
    multiclass : bool, default False
        Whether to calculate multiclass accuracy

    Returns
    -------
    np.float64
        Calculated accuracy for binary classification
    """
    return (
        _calculate_multiclass_accuracy(class_prob, model_prob)
        if multiclass
        else _calculate_accuracy(_to_confusion_matrix(class_prob, model_prob))
    )


def _calculate_accuracy(counts: ConfusionMatrix) -> np.float64:
    tp, _, tn, fn = counts
    return np.float64(tp + tn) / np.sum(counts, dtype=np.float64) if tp + fn > 0 else np.float64(0)


def _calculate_multiclass_accuracy(class_prob: Array1D[float], model_prob: Array1D[float]) -> np.float64:
    return np.dot(
        as_numpy(model_prob, dtype=np.float64, required_ndim=1), as_numpy(class_prob, dtype=np.float64, required_ndim=1)
    )


def nullmodel_precision(class_prob: Array1D[float], model_prob: Array1D[float]) -> np.float64:
    """
    Calculates precision from binary classification results.

    Parameters
    ----------
    class_prob : Array1D[float]
        Class-wise probabilities for the test set. Can be a 1D list, or array-like object.
    model_prob : Array1D[float]
        Probability distribution for given null model. Can be a 1D list, or array-like object.

    Returns
    -------
    np.float64
        Calculated precision for binary classification
    """
    return _calculate_precision(_to_confusion_matrix(class_prob, model_prob))


def _calculate_precision(counts: ConfusionMatrix) -> np.float64:
    tp, fp, _, fn = counts
    if (tp + fp) == 0:
        if fn > 0:
            return np.float64(0)
        return np.float64(1)
    return np.float64(tp / (tp + fp))


def nullmodel_recall(class_prob: Array1D[float], model_prob: Array1D[float]) -> np.float64:
    """
    Calculates recall (True Positive Rate) from binary classification results.

    Parameters
    ----------
    class_prob : Array1D[float]
        Class-wise probabilities for the test set. Can be a 1D list, or array-like object.
    model_prob : Array1D[float]
        Probability distribution for given null model. Can be a 1D list, or array-like object.

    Returns
    -------
    np.float64
        Calculated True Positive Rate for binary classification
    """
    return _calculate_recall(_to_confusion_matrix(class_prob, model_prob))


def _calculate_recall(counts: ConfusionMatrix) -> np.float64:
    tp, fp, _, fn = counts
    if (tp + fn) == 0:
        if fp > 0:
            return np.float64(0)
        return np.float64(1)
    return np.float64(tp / (tp + fn))


def nullmodel_fpr(class_prob: Array1D[float], model_prob: Array1D[float]) -> np.float64:
    """
    Calculates FPR (False Positive Rate) from binary classification results.

    Parameters
    ----------
    class_prob : Array1D[float]
        Class-wise probabilities for the test set. Can be a 1D list, or array-like object.
    model_prob : Array1D[float]
        Probability distribution for given null model. Can be a 1D list, or array-like object.

    Returns
    -------
    np.float64
        Estimated False Positive Rate for binary classification
    """
    return _calculate_fpr(_to_confusion_matrix(class_prob, model_prob))


def _calculate_fpr(counts: ConfusionMatrix) -> np.float64:
    _, fp, tn, _ = counts
    return np.float64(fp / (fp + tn)) if fp > 0 else np.float64(0)


def _to_confusion_matrix(class_prob: Array1D[float], pred_prob: Array1D[float]) -> ConfusionMatrix:
    """
    Calculates confusion matrix values from class probabilities and null model probabilities.

    Parameters
    ----------
    class_prob : Array1D[float]
        A "One-vs-Rest" array [1x2] representation of class probabilities, and its complement
    pred_prob : Array1D[float]
        A "One-vs-Rest" array [1x2] representation of given null model probabilities, and its complement

    Returns
    -------
    ConfusionMatrix
        Calculated confusion matrix values [True Positive, False Positive, True Negative, False Negative]
    """
    confusion_matrix = np.outer(
        as_numpy(class_prob, dtype=np.float64, required_ndim=1), as_numpy(pred_prob, dtype=np.float64, required_ndim=1)
    )
    return confusion_matrix[0, 0], confusion_matrix[1, 0], confusion_matrix[1, 1], confusion_matrix[0, 1]


def _reduce_micro(method: BinaryClassMetricFunction, counts: Sequence[ConfusionMatrix]) -> np.float64:
    """
    Micro-averaging for multiclass classification metric.

    Reduces measures by first summing classification outcomes and then performing given metric calculation.

    Parameters
    ----------
    method : BinaryClassMetricFunction
        Metric-calculating method to perform on summed data
    counts : Sequence[ConfusionMatrix]
        2D array of classification results for each class
    Returns
    -------
    np.float64
        Calculated metric reduced with micro-averaging
    """
    return method(np.sum(counts, axis=0))


def _reduce_macro(method: BinaryClassMetricFunction, counts: Sequence[ConfusionMatrix]) -> np.float64:
    """
    Macro-averaging for multiclass classification metric.

    Reduces measures by performing metric-calculating method on each class, then averaging results.

    Parameters
    ----------
    method : BinaryClassMetricFunction
        Metric-calculating method to perform on summed data
    counts : Sequence[ConfusionMatrix]
        2D array of classification results for each class
    Returns
    -------
    np.float64
        Calculated metric reduced with macro-averaging
    """
    return np.mean([method(c) for c in counts], dtype=np.float64)


# Metric configurations for null model calculations
_BASE_METRICS: dict[str, BinaryClassMetricFunction] = {
    "precision": _calculate_precision,
    "recall": _calculate_recall,
    "false_positive_rate": _calculate_fpr,
}

_BINARY_ONLY_METRICS: dict[str, BinaryClassMetricFunction] = {
    "accuracy": _calculate_accuracy,
}

_AVERAGES: dict[str, Callable[[BinaryClassMetricFunction, Sequence[ConfusionMatrix]], np.float64]] = {
    "micro": _reduce_micro,
    "macro": _reduce_macro,
}


def _prepare_probability_distributions(
    test_labels: ArrayLike, train_labels: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Validate inputs and prepare probability distributions from label arrays.

    Parameters
    ----------
    test_labels : ArrayLike
        Test set class labels
    train_labels : ArrayLike | None
        Training set class labels, or None

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Test and train class probability distributions (aligned to same number of classes)

    Raises
    ------
    ValueError
        If test_labels is empty or null
    """
    # Convert ArrayLike inputs to NumPy arrays
    test_np = as_numpy(test_labels)
    train_np = as_numpy(train_labels)

    if not test_np.size:
        raise ValueError("Empty or null test labels provided")

    # Calculate class probabilities
    test_class_probs = np.bincount(test_np) / test_np.size
    train_class_probs = (np.bincount(train_np) / train_np.size) if train_np.size else train_np

    # Align probability arrays to have the same number of classes
    test_classes = len(test_class_probs)
    train_classes = len(train_class_probs)
    max_classes = max(test_classes, train_classes)

    if test_classes < max_classes:
        test_class_probs = np.pad(test_class_probs, (0, max_classes - test_classes))
    if train_classes and train_classes < max_classes:
        train_class_probs = np.pad(train_class_probs, (0, max_classes - train_classes))
        train_class_probs = train_class_probs / np.sum(train_class_probs)

    return test_class_probs, train_class_probs


def _calculate_null_model_metrics(
    test_probs: NDArray[np.float64],
    prediction_probs: NDArray[np.float64],
    is_multiclass: bool | np.bool_,
    classes: NDArray[np.intp],
) -> NullModelMetrics:
    """
    Calculate metrics for a given null model prediction probability distribution.

    Parameters
    ----------
    test_probs : NDArray[np.float64]
        Test set class probabilities
    prediction_probs : NDArray[np.float64]
        Null model prediction probabilities
    is_multiclass : bool
        Whether this is a multiclass problem
    classes : NDArray[np.intp]
        Array of class indices present in test set

    Returns
    -------
    NullModelMetrics
        Calculated metrics for the null model
    """
    # Calculate confusion matrices for each class (one-vs-rest)
    confusion_matrices: list[ConfusionMatrix] = []
    for class_idx in classes:
        pred_1vr = np.array([prediction_probs[class_idx], 1 - prediction_probs[class_idx]])
        test_1vr = np.array([test_probs[class_idx], 1 - test_probs[class_idx]])
        confusion_matrices.append(_to_confusion_matrix(test_1vr, pred_1vr))

    # Calculate binary metrics
    result: NullModelMetrics = {
        "precision_macro": float(_AVERAGES["macro"](_BASE_METRICS["precision"], confusion_matrices)),
        "precision_micro": float(_AVERAGES["micro"](_BASE_METRICS["precision"], confusion_matrices)),
        "recall_macro": float(_AVERAGES["macro"](_BASE_METRICS["recall"], confusion_matrices)),
        "recall_micro": float(_AVERAGES["micro"](_BASE_METRICS["recall"], confusion_matrices)),
        "false_positive_rate_macro": float(
            _AVERAGES["macro"](_BASE_METRICS["false_positive_rate"], confusion_matrices)
        ),
        "false_positive_rate_micro": float(
            _AVERAGES["micro"](_BASE_METRICS["false_positive_rate"], confusion_matrices)
        ),
    }

    # Add binary-specific metrics
    if not is_multiclass:
        result["accuracy_macro"] = float(_AVERAGES["macro"](_BINARY_ONLY_METRICS["accuracy"], confusion_matrices))
        result["accuracy_micro"] = float(_AVERAGES["micro"](_BINARY_ONLY_METRICS["accuracy"], confusion_matrices))

    # Add multiclass specific metrics
    if is_multiclass:
        result["multiclass_accuracy"] = float(_calculate_multiclass_accuracy(test_probs, prediction_probs))

    return result


def nullmodel_metrics(test_labels: ArrayLike, train_labels: ArrayLike | None = None) -> NullModelMetricsResult:
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
    NullModelMetricsResult
        Result mapping containing metrics for each null model strategy:

        - uniform_random: `NullModelMetrics` for uniform random classifier
        - dominant_class: `NullModelMetrics` for dominant class classifier (if train_labels provided)
        - proportional_random: `NullModelMetrics` for proportional random classifier (if train_labels provided)

    Notes
    -----
    The NullModelMetrics returned in each map value are a mapping of:

    - precision_macro: float - Macro-averaged precision across all classes
    - precision_micro: float - Micro-averaged precision across all classes
    - recall_macro: float - Macro-averaged recall across all classes
    - recall_micro: float - Micro-averaged recall across all classes
    - false_positive_rate_macro: float - Macro-averaged false positive rate across all classes
    - false_positive_rate_micro: float - Micro-averaged false positive rate across all classes
    - accuracy_macro: float - Macro-averaged accuracy (only for binary classification)
    - accuracy_micro: float - Micro-averaged accuracy (only for binary classification)
    - multiclass_accuracy: float - Multiclass accuracy (only for multiclass classification)
    """
    # Prepare probability distributions
    test_probs, train_probs = _prepare_probability_distributions(test_labels, train_labels)

    # Determine which classes are present and if this is multiclass
    classes = np.nonzero(test_probs)[0]
    is_multiclass = len(classes) > 2 or np.count_nonzero(train_probs) > 2

    # Calculate uniform random null model (always available)
    uniform_probs = np.where(test_probs != 0, 1.0 / len(classes), 0.0)
    uniform_metrics = _calculate_null_model_metrics(test_probs, uniform_probs, is_multiclass, classes)

    # Calculate train-based null models if training labels provided
    if train_probs.size:
        # Dominant class: all probability on most frequent class
        dominant_probs = np.zeros_like(train_probs)
        dominant_probs[np.argmax(train_probs)] = 1.0
        dominant_metrics = _calculate_null_model_metrics(test_probs, dominant_probs, is_multiclass, classes)

        # Proportional random: use training distribution
        proportional_metrics = _calculate_null_model_metrics(test_probs, train_probs, is_multiclass, classes)

        return {
            "uniform_random": uniform_metrics,
            "dominant_class": dominant_metrics,
            "proportional_random": proportional_metrics,
        }

    return {"uniform_random": uniform_metrics}

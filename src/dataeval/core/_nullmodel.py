from __future__ import annotations

__all__ = []

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

ConfusionMatrix = tuple[np.floating[Any], np.floating[Any], np.floating[Any], np.floating[Any]]
BinaryClassMetricFunction = Callable[[ConfusionMatrix], np.float64]


def nullmodel_accuracy(
    class_prob: NDArray[np.floating[Any]], model_prob: NDArray[np.floating[Any]], *, multiclass: bool = False
) -> np.float64:
    """
    Calculates accuracy from binary classification results.

    Parameters
    ----------
    class_prob : NDArray[np.floating]
        Class-wise probabilities for the test set
    model_prob : NDArray[np.floating]
        Probability distribution for given null model
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


def _calculate_multiclass_accuracy(
    class_prob: NDArray[np.floating[Any]], model_prob: NDArray[np.floating[Any]]
) -> np.float64:
    return np.dot(model_prob, class_prob)


def nullmodel_precision(class_prob: NDArray[np.floating[Any]], model_prob: NDArray[np.floating[Any]]) -> np.float64:
    """
    Calculates precision from binary classification results.

    Parameters
    ----------
    class_prob : NDArray[np.floating]
        Class-wise probabilities for the test set
    model_prob : NDArray[np.floating]
        Probability distribution for given null model

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


def nullmodel_recall(class_prob: NDArray[np.floating[Any]], model_prob: NDArray[np.floating[Any]]) -> np.float64:
    """
    Calculates recall (True Positive Rate) from binary classification results.

    Parameters
    ----------
    class_prob : NDArray[np.floating]
        Class-wise probabilities for the test set
    model_prob : NDArray[np.floating]
        Probability distribution for given null model

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


def nullmodel_fpr(class_prob: NDArray[np.floating[Any]], model_prob: NDArray[np.floating[Any]]) -> np.float64:
    """
    Calculates FPR (False Positive Rate) from binary classification results.

    Parameters
    ----------
    class_prob : NDArray[np.floating]
        Class-wise probabilities for the test set
    model_prob : NDArray[np.floating]
        Probability distribution for given null model

    Returns
    -------
    np.float64
        Estimated False Positive Rate for binary classification
    """
    return _calculate_fpr(_to_confusion_matrix(class_prob, model_prob))


def _calculate_fpr(counts: ConfusionMatrix) -> np.float64:
    _, fp, tn, _ = counts
    return np.float64(fp / (fp + tn)) if fp > 0 else np.float64(0)


def _to_confusion_matrix(
    class_prob: NDArray[np.floating[Any]], pred_prob: NDArray[np.floating[Any]]
) -> ConfusionMatrix:
    """
    Calculates confusion matrix values from class probabilities and null model probabilities.

    Parameters
    ----------
    class_prob : NDArray[np.floating]
        A "One-vs-Rest" array [1x2] representation of class probabilities, and its complement
    pred_prob : NDArray[np.floating]
        A "One-vs-Rest" array [1x2] representation of given null model probabilities, and its complement

    Returns
    -------
    ConfusionMatrix
        Calculated confusion matrix values [True Positive, False Positive, True Negative, False Negative]
    """
    confusion_matrix = np.outer(class_prob, pred_prob)
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

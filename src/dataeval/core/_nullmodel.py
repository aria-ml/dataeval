__all__ = []

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired

from dataeval.protocols import ArrayLike
from dataeval.types import Array1D
from dataeval.utils._internal import as_numpy
from dataeval.utils.preprocessing import compute_iou


def _estimate_hit_probabilities(
    gt_boxes: NDArray[np.float64],
    image_size: tuple[int, int],
    generator_func: Callable[[int, tuple[int, int]], NDArray[np.float64]],
    n_samples: int = 5000,
) -> NDArray[np.float64]:
    """
    Estimate the probability that a single random box hits each ground truth box.

    Parameters
    ----------
    gt_boxes : NDArray[np.float64]
        Ground truth boxes for a single image in XYXY format
    image_size : tuple[int, int]
        Image size as (height, width)
    generator_func : Callable
        Function that generates n random boxes for the given image size
    n_samples : int, default 5000
        Number of samples to use for estimation

    Returns
    -------
    NDArray[np.float64]
        Array of probabilities, one per ground truth box
    """
    if len(gt_boxes) == 0:
        return np.array([], dtype=np.float64)

    # Generate samples
    pred_boxes = generator_func(n_samples, image_size)

    # Compute IoU matrix (n_samples, n_gt)
    ious = compute_iou(pred_boxes, gt_boxes)

    # Hit probability is fraction of samples with IoU > 0.5
    return np.mean(ious > 0.5, axis=0)


def _calculate_localization_ap(
    hit_probs: NDArray[np.float64],
    class_probabilities: NDArray[np.float64],
    gt_labels: NDArray[np.intp],
) -> float:
    """
    Calculate Average Precision for localization given hit probabilities.

    Sweeps a curve by varying the number of predicted boxes K.

    Parameters
    ----------
    hit_probs : NDArray[np.float64]
        Probability of localization hit for each GT box in the test set
    class_probabilities : NDArray[np.float64]
        Probability distribution over classes for the null model
    gt_labels : NDArray[np.intp]
        Ground truth labels for each box in the test set

    Returns
    -------
    float
        Computed Average Precision
    """
    if len(hit_probs) == 0:
        return 0.0

    # K values to sweep (number of boxes predicted per image)
    # Using a logarithmic-ish scale to cover 1 to 5000
    k_values = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])

    precisions = []
    recalls = []

    total_gt = len(hit_probs)

    for k in k_values:
        # For each GT box j, probability of hit at least once among K predictions
        # P(hit_j) = 1 - (1 - p_j * P(class_j))^K
        # where p_j is localization hit prob and P(class_j) is null model class prob for GT class
        p_match = hit_probs * class_probabilities[gt_labels]
        prob_recalled = 1 - (1 - p_match) ** k

        expected_recall = np.sum(prob_recalled) / total_gt
        # Expected Precision: TP / K.
        # Since each prediction matches at most one GT, and assuming hits are rare
        # TP is effectively the number of recalled GTs (as long as TP <= K)
        expected_tp = np.sum(prob_recalled)
        expected_precision = expected_tp / k

        recalls.append(expected_recall)
        precisions.append(min(1.0, expected_precision))

    # Add endpoints for AP calculation
    recalls = np.array([0.0] + recalls + [1.0])
    precisions = np.array([1.0] + precisions + [0.0])

    # Ensure recall is monotonic
    # (Expected recall IS monotonic with K, but precision might not be strictly decreasing)
    # Use standard VOC-style AP (area under PR curve)
    # Sort by recall
    idx = np.argsort(recalls)
    recalls = recalls[idx]
    precisions = precisions[idx]

    # Compute AP using trapezoidal rule or step integration
    # Standard AP is typically the area under the precision-recall curve
    # Use np.trapezoid (modern replacement for np.trapz in NumPy 2.0+)
    return float(np.trapezoid(precisions, recalls))


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
    localization_ap: NotRequired[float]


class NullModelMetricsResult(TypedDict):
    """
    Type definition for null model metrics output.

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
    class_prob: Array1D[float],
    model_prob: Array1D[float],
    *,
    multiclass: bool = False,
) -> np.float64:
    """
    Compute accuracy from binary classification results.

    Parameters
    ----------
    class_prob : Array1D[float]
        Class-wise probabilities for the test set. Can be a 1D list, or array-like object.
    model_prob : Array1D[float]
        Probability distribution for given null model. Can be a 1D list, or array-like object.
    multiclass : bool, default False
        Whether to compute multiclass accuracy

    Returns
    -------
    np.float64
        Computed accuracy for binary classification
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
        as_numpy(model_prob, dtype=np.float64, required_ndim=1),
        as_numpy(class_prob, dtype=np.float64, required_ndim=1),
    )


def _generate_uniform_random_boxes(n: int, image_size: tuple[int, int]) -> NDArray[np.float64]:
    """
    Generate n uniform random boxes within the given image size.

    Parameters
    ----------
    n : int
        Number of boxes to generate
    image_size : tuple[int, int]
        Image size as (height, width)

    Returns
    -------
    NDArray[np.float64]
        Generated boxes in XYXY format with shape (n, 4)
    """
    h, w = image_size
    # Generate random coordinates
    x = np.random.uniform(0, w, size=(n, 2))
    y = np.random.uniform(0, h, size=(n, 2))

    # Ensure x0 < x1 and y0 < y1
    x0 = np.min(x, axis=1)
    x1 = np.max(x, axis=1)
    y0 = np.min(y, axis=1)
    y1 = np.max(y, axis=1)

    return np.stack([x0, y0, x1, y1], axis=1)


def _generate_proportional_random_boxes(
    n: int,
    image_size: tuple[int, int],
    train_boxes: NDArray[np.float64],
    train_image_sizes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Generate n random boxes sampled from the training set box distribution.

    Samples center_x, center_y, width, and height independently from the training set.

    Parameters
    ----------
    n : int
        Number of boxes to generate
    image_size : tuple[int, int]
        Target image size as (height, width)
    train_boxes : NDArray[np.float64]
        Training set boxes in XYXY format
    train_image_sizes : NDArray[np.float64]
        Training set image sizes as (height, width) for each box

    Returns
    -------
    NDArray[np.float64]
        Generated boxes in XYXY format
    """
    h_target, w_target = image_size

    # 1. Convert training boxes to CXCYWH normalized to [0, 1]
    train_w = train_boxes[:, 2] - train_boxes[:, 0]
    train_h = train_boxes[:, 3] - train_boxes[:, 1]
    train_cx = train_boxes[:, 0] + train_w / 2
    train_cy = train_boxes[:, 1] + train_h / 2

    norm_cx = train_cx / train_image_sizes[:, 1]
    norm_cy = train_cy / train_image_sizes[:, 0]
    norm_w = train_w / train_image_sizes[:, 1]
    norm_h = train_h / train_image_sizes[:, 0]

    # 2. Sample components independently
    gen_cx_norm = np.random.choice(norm_cx, size=n, replace=True)
    gen_cy_norm = np.random.choice(norm_cy, size=n, replace=True)
    gen_w_norm = np.random.choice(norm_w, size=n, replace=True)
    gen_h_norm = np.random.choice(norm_h, size=n, replace=True)

    # 3. Scale to target image size
    gen_cx = gen_cx_norm * w_target
    gen_cy = gen_cy_norm * h_target
    gen_w = gen_w_norm * w_target
    gen_h = gen_h_norm * h_target

    # 4. Convert back to XYXY
    gen_x0 = gen_cx - gen_w / 2
    gen_y0 = gen_cy - gen_h / 2
    gen_x1 = gen_cx + gen_w / 2
    gen_y1 = gen_cy + gen_h / 2

    return np.stack([gen_x0, gen_y0, gen_x1, gen_y1], axis=1)


def _generate_modal_boxes(
    n: int,
    image_size: tuple[int, int],
    train_boxes: NDArray[np.float64],
    train_image_sizes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Generate n boxes using the modal dimensions from the training set.

    Parameters
    ----------
    n : int
        Number of boxes to generate
    image_size : tuple[int, int]
        Target image size as (height, width)
    train_boxes : NDArray[np.float64]
        Training set boxes in XYXY format
    train_image_sizes : NDArray[np.float64]
        Training set image sizes as (height, width)

    Returns
    -------
    NDArray[np.float64]
        Generated boxes in XYXY format
    """
    h_target, w_target = image_size

    # Calculate widths and heights in training set
    train_widths = train_boxes[:, 2] - train_boxes[:, 0]
    train_heights = train_boxes[:, 3] - train_boxes[:, 1]

    # Normalize by training image sizes
    norm_widths = train_widths / train_image_sizes[:, 1]
    norm_heights = train_heights / train_image_sizes[:, 0]

    # Find "mode" - since these are floats, we can use a histogram or just round them
    # For now, let's just take the median as a robust "representative" size
    # or we could bin them.
    # The requirement said "modal boxes (?)".
    # Let's use a simple binning to find the mode of (width, height) pairs.
    # Binning to 1% resolution
    bins_w = np.round(norm_widths * 100)
    bins_h = np.round(norm_heights * 100)
    bins = np.stack([bins_w, bins_h], axis=1)

    unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
    modal_bin = unique_bins[np.argmax(counts)]

    mode_w_norm = modal_bin[0] / 100.0
    mode_h_norm = modal_bin[1] / 100.0

    mode_w = mode_w_norm * w_target
    mode_h = mode_h_norm * h_target

    # Place boxes distributed over a grid that spans the image bounds
    # Determine nx, ny such that nx * ny >= n and nx/ny roughly matches image aspect ratio
    aspect = w_target / h_target
    nx = max(1, int(np.round(np.sqrt(n * aspect))))
    ny = max(1, int(np.ceil(n / nx)))

    x_coords = np.linspace(0, max(0, w_target - mode_w), nx)
    y_coords = np.linspace(0, max(0, h_target - mode_h), ny)

    xv, yv = np.meshgrid(x_coords, y_coords)
    gen_x0 = xv.flatten()
    gen_y0 = yv.flatten()

    # Shuffle the grid indices to ensure any subset (first n) spans the image
    indices = np.arange(len(gen_x0))
    np.random.shuffle(indices)

    indices = indices[:n]
    gen_x0 = gen_x0[indices]
    gen_y0 = gen_y0[indices]
    gen_x1 = gen_x0 + mode_w
    gen_y1 = gen_y0 + mode_h

    return np.stack([gen_x0, gen_y0, gen_x1, gen_y1], axis=1)


def nullmodel_precision(class_prob: Array1D[float], model_prob: Array1D[float]) -> np.float64:
    """
    Compute precision from binary classification results.

    Parameters
    ----------
    class_prob : Array1D[float]
        Class-wise probabilities for the test set. Can be a 1D list, or array-like object.
    model_prob : Array1D[float]
        Probability distribution for given null model. Can be a 1D list, or array-like object.

    Returns
    -------
    np.float64
        Computed precision for binary classification
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
    Compute recall (True Positive Rate) from binary classification results.

    Parameters
    ----------
    class_prob : Array1D[float]
        Class-wise probabilities for the test set. Can be a 1D list, or array-like object.
    model_prob : Array1D[float]
        Probability distribution for given null model. Can be a 1D list, or array-like object.

    Returns
    -------
    np.float64
        Computed True Positive Rate for binary classification
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
    Compute FPR (False Positive Rate) from binary classification results.

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
    Compute confusion matrix values from class probabilities and null model probabilities.

    Parameters
    ----------
    class_prob : Array1D[float]
        A "One-vs-Rest" array [1x2] representation of class probabilities, and its complement
    pred_prob : Array1D[float]
        A "One-vs-Rest" array [1x2] representation of given null model probabilities, and its complement

    Returns
    -------
    ConfusionMatrix
        Computed confusion matrix values [True Positive, False Positive, True Negative, False Negative]
    """
    confusion_matrix = np.outer(
        as_numpy(class_prob, dtype=np.float64, required_ndim=1),
        as_numpy(pred_prob, dtype=np.float64, required_ndim=1),
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
        Computed metric reduced with micro-averaging
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
        Computed metric reduced with macro-averaging
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
    test_labels: ArrayLike,
    train_labels: ArrayLike | None,
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
    Compute metrics for a given null model prediction probability distribution.

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
        Computed metrics for the null model
    """
    # Compute confusion matrices for each class (one-vs-rest)
    confusion_matrices: list[ConfusionMatrix] = []
    for class_idx in classes:
        pred_1vr = np.array([prediction_probs[class_idx], 1 - prediction_probs[class_idx]])
        test_1vr = np.array([test_probs[class_idx], 1 - test_probs[class_idx]])
        confusion_matrices.append(_to_confusion_matrix(test_1vr, pred_1vr))

    # Compute binary metrics
    result: NullModelMetrics = {
        "precision_macro": float(_AVERAGES["macro"](_BASE_METRICS["precision"], confusion_matrices)),
        "precision_micro": float(_AVERAGES["micro"](_BASE_METRICS["precision"], confusion_matrices)),
        "recall_macro": float(_AVERAGES["macro"](_BASE_METRICS["recall"], confusion_matrices)),
        "recall_micro": float(_AVERAGES["micro"](_BASE_METRICS["recall"], confusion_matrices)),
        "false_positive_rate_macro": float(
            _AVERAGES["macro"](_BASE_METRICS["false_positive_rate"], confusion_matrices),
        ),
        "false_positive_rate_micro": float(
            _AVERAGES["micro"](_BASE_METRICS["false_positive_rate"], confusion_matrices),
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
    Compute null model metrics (dummy classifiers metrics) for given class distributions.

    This function computes benchmark performance metrics for random classifiers on the training and testing labels
    based on the class distributions.

    Null models to be evaluated:

    - Uniform Random: Classifier applies equal probability to each class
    - Dominant Class: Classifier will choose the most frequent class in the training set (requires training labels)
    - Proportional Random: Classifier applies distribution probabilities from training set (requires training labels)

    The computed metrics are to be used as a lower-bound performance baseline for model evaluation.

    Parameters
    ----------
    test_labels : ArrayLike
        Class distribution from test set. Each index is the integer representation of the associated class label,
        e.g. [0, 1, 1, 2, 3].
    train_labels : ArrayLike | None, default None
        Class distribution from training set. Each index is the integer representation of the associated class label,
        e.g. [0, 1, 1, 2, 3]. When None, skips calculating class frequencies and does not report metrics for the
        dominant class and proportional random models.

    Returns
    -------
    NullModelMetricsResult
        Result mapping containing metrics for each null model strategy:

        - uniform_random: `NullModelMetrics` for uniform random classifier
        - dominant_class: `NullModelMetrics` for dominant class classifier (if train_labels provided)
        - proportional_random: `NullModelMetrics` for proportional random classifier (if train_labels provided)

    Raises
    ------
    ValueError
        If test_labels is None or empty

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

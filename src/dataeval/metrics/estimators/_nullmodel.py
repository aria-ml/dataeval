from collections import defaultdict
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from dataeval.outputs import NullModelMetricsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike
from dataeval.utils._array import as_numpy


def _estimate_accuracy(counts: NDArray[np.floating]) -> np.float64:
    """
    Calculates accuracy from binary classification results

    Parameters
    ----------
    counts : NDArray[np.floating]
        True positives, false positives, true negatives, false negatives

    Returns
    -------
    np.float64
        Calculated accuracy for binary classification
    """
    tp, _, tn, fn = counts
    return np.float64(tp + tn) / np.sum(counts, dtype=np.float64) if tp + fn > 0 else np.float64(0)


def _estimate_ber(counts: NDArray[np.floating]) -> np.float64:
    """
    Calculates BER from binary classification results

    Parameters
    ----------
    counts : NDArray[np.floating]
        True positives, false positives, true negatives, false negatives

    Returns
    -------
    np.float64
        Calculated Bayes error rate for binary classification
    """
    return 1 - _estimate_accuracy(counts)


def _estimate_multiclass_accuracy(class_prob_tst: NDArray[np.floating], model_prob: NDArray[np.floating]) -> np.float64:
    """
    Calculates accuracy from multiclass results

    Parameters
    ----------
    class_prob_tst: NDArray[np.floating]
        Class-wise probabilities for the testing set
    model_prob: NDArray[np.floating]
        Probability distribution for given null model
    Returns
    -------
    np.float64
        Calculated accuracy for multiclass classification
    """
    return np.dot(model_prob, class_prob_tst)


def _estimate_multiclass_ber(class_prob_tst: NDArray[np.floating], model_prob: NDArray[np.floating]) -> np.float64:
    """
    Calculates BER from multiclass results

    Parameters
    ----------
    class_prob_tst: NDArray[np.floating]
        Class-wise probabilities for the testing set
    model_prob: NDArray[np.floating]
        Probability distribution for given null model
    Returns
    -------
    np.float64
        Calculated Bayes error rate for multiclass classification
    """
    return 1 - _estimate_multiclass_accuracy(class_prob_tst, model_prob)


def _estimate_precision(counts: NDArray[np.floating]) -> np.float64:
    """
    Estimates precision from binary classification results

    Parameters
    ----------
    counts : NDArray[np.floating]
        True positives, false positives, true negatives, false negatives

    Returns
    -------
    np.float64
        Calculated precision for binary classification
    """
    tp, fp, _, fn = counts
    if (tp + fp) == 0:
        if fn > 0:
            return np.float64(0)
        return np.float64(1)
    return tp / (tp + fp)


def _estimate_recall(counts: NDArray[np.floating]) -> np.float64:
    """
    Estimates recall from binary classification results

    Parameters
    ----------
    counts : NDArray[np.floating]
        True positives, false positives, true negatives, false negatives

    Returns
    -------
    np.float64
        Calculated recall for binary classification
    """
    tp, fp, _, fn = counts
    if (tp + fn) == 0:
        if fp > 0:
            return np.float64(0)
        return np.float64(1)
    return tp / (tp + fn)


def _estimate_true_positive_rate(counts: NDArray[np.floating]) -> np.float64:
    """
    Estimates True Positive Rate from binary classification results

    Parameters
    ----------
    counts : NDArray[np.floating]
        True positives, false positives, true negatives, false negatives

    Returns
    -------
    np.float64
        Estimated True Positive Rate for binary classification
    """
    return _estimate_recall(counts)


def _estimate_false_positive_rate(counts: NDArray[np.floating]) -> np.float64:
    _, fp, tn, _ = counts
    """
    Estimates False Positive Rate from binary classification results

    Parameters
    ----------
    counts : NDArray[np.floating]
        True positives, false positives, true negatives, false negatives

    Returns
    -------
    np.float64
        Estimated False Positive Rate for binary classification
    """
    return fp / (fp + tn) if fp > 0 else np.float64(0)


def _get_tpfptnfn(
    class_prob: NDArray[np.floating], pred_prob: NDArray[np.floating]
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Calculates confusion matrix values from class probabilities and null model probabilities

    Parameters
    ----------
    class_prob : NDArray[np.floating]
        A "One-vs-Rest" array [1x2] representation of class probabilities, and its complement
    pred_prob : NDArray[np.floating]
        A "One-vs-Rest" array [1x2] representation of given null model probabilities, and its complement
    Returns
    -------
    tuple[np.float64]
        Calculated confusion matrix values [True Positive, False Postivie, True Negative, False Negative]
    """
    confusion_matrix = np.outer(class_prob, pred_prob)
    return confusion_matrix[0, 0], confusion_matrix[1, 0], confusion_matrix[1, 1], confusion_matrix[0, 1]


def _reduce_micro(method: Callable, counts: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Reduces measures by first summing classification outcomes and then performing given metric calculation

    Parameters
    ----------
    method: Callable
        Metric-calculating method to perform on summed data
    counts: NDArray[np.floating]
        2D array of classification results for each class
    Returns
    -------
    NDArray[np.float64]
        Calculated metric reduced with micro-averaging
    """
    return method(np.sum(counts, axis=0))


def _reduce_macro(method: Callable, counts: NDArray[np.floating]) -> np.float64:
    """
    Reduces measures by performing metric-calculating method on each class, then averaging results

    Parameters
    ----------
    method: Callable
        Metric-calculating method to perform on summed data
    counts: NDArray[np.floating]
        2D array of classification results for each class
    Returns
    -------
    NDArray[np.float64]
        Calculated metric reduced with macro-averaging
    """
    return np.mean([method(c) for c in counts], dtype=np.float64)


@set_metadata
def null_model_metrics(test_labels: ArrayLike, train_labels: ArrayLike | None = None) -> NullModelMetricsOutput:
    """
    Calculate null model metrics (dummy classifiers metrics) for given class distributions

    This function calculates benchmark performance metrics for random classifiers on the training and testing labels
    based on the class distributions.

    Null models to be evaluated:
    - Uniform Random:
        Classifier applies equal probability to each class
    - Dominant Class:
        Classifier will choose the most frequent class in the training set (requires both training and testing labels)
    - Proportional Random:
        Classifier applies distribution probabilities from training set (requires both training and testing labels)

    The calculated metrics are to be used as a lower-bound performance baseline for model evaluation.

    Parameters
    ----------
    test_labels: ArrayLike
        Class distribution from training set. Each index is the integer representation of the associated class label,
        e.g. [0, 1, 1, 2, 3]
    train_labels: ArrayLike | None, default None
        Class distribution from test set. Each index is the integer representation of the associated class label,
        e.g. [0, 1, 1, 2, 3]. When None, skips calculating class frequencies
        and does not report metrics for the dominant class and proportional random models.

    Raises
    ------
    ValueError
        If test_labels is None or empty

    Returns
    -------
    NullModelMetricsOutput
        Dictionaries mapping null model metrics with null models
    """
    test_labels_as_np = as_numpy(test_labels)
    train_labels_as_np = as_numpy(train_labels)

    # validate test labels are not shapeless and empty
    if not (test_labels_as_np.ndim and test_labels_as_np.size):
        raise ValueError("Empty or null test labels provided")

    test_class_prob = np.bincount(test_labels_as_np) / test_labels_as_np.size
    num_classes = np.count_nonzero(test_class_prob)
    is_multiclass = num_classes > 2
    prediction_probs = {}

    if train_labels_as_np.ndim and train_labels_as_np.size:
        train_class_prob = np.bincount(train_labels_as_np) / train_labels_as_np.size
        # if either train or test set is non-binary, it is a multiclass problem
        is_multiclass |= np.count_nonzero(train_class_prob) > 2
        test_len = len(test_class_prob)
        train_len = len(train_class_prob)
        if test_len > train_len:
            train_class_prob = np.pad(train_class_prob, (0, test_len - train_len), "constant")
        elif train_len > test_len:
            test_class_prob = np.pad(test_class_prob, (0, train_len - test_len), "constant")
        train_class_prob = np.array(train_class_prob) / np.sum(train_class_prob)
        dom_pred = np.zeros(len(train_class_prob))
        dom_pred[np.argmax(train_class_prob)] = 1
        prediction_probs["dominant_class"] = dom_pred
        prediction_probs["proportional_random"] = train_class_prob

    # uniform random is only model available without training labels
    prediction_probs["uniform_random"] = [1 / num_classes if prob != 0 else 0 for prob in test_class_prob]

    # Base metric set: always run
    metrics: dict[str, Callable] = {
        "precision": _estimate_precision,
        "recall": _estimate_recall,
        "true_positive_rate": _estimate_true_positive_rate,
        "false_positive_rate": _estimate_false_positive_rate,
    }
    # Multiclass specific metrics
    if is_multiclass:
        task_metrics = {
            "multiclass_accuracy": _estimate_multiclass_accuracy,
            "multiclass_ber": _estimate_multiclass_ber,
        }
    # Binary specific metrics
    else:
        task_metrics = {
            "accuracy": _estimate_accuracy,
            "ber": _estimate_ber,
        }
    # Add specific task metrics
    metrics.update(task_metrics)

    classes = np.nonzero(test_class_prob)[0]
    results_by_model = defaultdict(dict)
    results_by_metric = defaultdict(dict)
    for model_name, probs in prediction_probs.items():
        for metric, method in metrics.items():
            if "multiclass" in metric:
                # metrics that use class probabilities, e.g. multiclass_accuracy
                measure = method(test_class_prob, probs)
                results_by_model[model_name][metric] = measure
                results_by_metric[metric][model_name] = measure
                continue
            counts = np.zeros((num_classes, 4))
            for index, class_index in enumerate(classes):
                prediction_probs_1vR = np.array([probs[class_index], 1 - probs[class_index]])
                class_probs_1vR = np.array([test_class_prob[class_index], 1 - test_class_prob[class_index]])
                counts[index, :] = _get_tpfptnfn(class_probs_1vR, prediction_probs_1vR)
            micro = _reduce_micro(method, counts)
            macro = _reduce_macro(method, counts)

            results_by_model[model_name][f"{metric}_micro"] = micro
            results_by_model[model_name][f"{metric}_macro"] = macro
            results_by_metric[f"{metric}_micro"][model_name] = micro
            results_by_metric[f"{metric}_macro"][model_name] = macro

    return NullModelMetricsOutput(results_by_metric, results_by_model)

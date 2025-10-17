from __future__ import annotations

__all__ = []

from dataclasses import dataclass

from dataeval.outputs._base import Output


@dataclass(frozen=True)
class BEROutput(Output):
    """
    Output class for :func:`.ber` estimator metric.

    Attributes
    ----------
    ber : float
        The upper bounds of the :term:`Bayes error rate<Bayes Error Rate (BER)>`
    ber_lower : float
        The lower bounds of the Bayes Error Rate
    """

    ber: float
    ber_lower: float


@dataclass(frozen=True)
class DivergenceOutput(Output):
    """
    Output class for :func:`.divergence` estimator metric.

    Attributes
    ----------
    divergence : float
        :term:`Divergence` value calculated between 2 datasets ranging between 0.0 and 1.0
    errors : int
        The number of differing edges between the datasets
    """

    divergence: float
    errors: int


@dataclass(frozen=True)
class UAPOutput(Output):
    """
    Output class for :func:`.uap` estimator metric.

    Attributes
    ----------
    uap : float
        The empirical mean precision estimate
    """

    uap: float


@dataclass(frozen=True)
class NullModelMetrics:
    """
    Per-model results for null-model metrics

    Attributes
    ----------
    precision_macro : float
    precision_micro : float
    recall_macro : float
    recall_micro : float
    false_positive_rate_macro : float
    false_positive_rate_micro : float
    accuracy_macro : float or None
    accuracy_micro : float or None
    multiclass_accuracy : float or None
    """

    precision_macro: float
    precision_micro: float
    recall_macro: float
    recall_micro: float
    false_positive_rate_macro: float
    false_positive_rate_micro: float
    accuracy_macro: float | None = None
    accuracy_micro: float | None = None
    multiclass_accuracy: float | None = None


@dataclass(frozen=True)
class NullModelMetricsOutput(Output):
    """
    Output class for null-model metrics

    Attributes
    ----------
    uniform_random : NullModelMetrics
    dominant_class : NullModelMetrics or None
    proportional_random : NullModelMetrics or None
    """

    uniform_random: NullModelMetrics
    dominant_class: NullModelMetrics | None = None
    proportional_random: NullModelMetrics | None = None

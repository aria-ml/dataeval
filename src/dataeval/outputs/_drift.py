from __future__ import annotations

__all__ = []

import contextlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure

from dataeval.detectors.drift._nml._result import Metric, PerMetricResult
from dataeval.outputs._base import Output


@dataclass(frozen=True)
class DriftBaseOutput(Output):
    """
    Base output class for Drift Detector classes
    """

    drifted: bool
    threshold: float
    p_val: float
    distance: float


@dataclass(frozen=True)
class DriftMMDOutput(DriftBaseOutput):
    """
    Output class for :class:`.DriftMMD` :term:`drift<Drift>` detector.

    Attributes
    ----------
    drifted : bool
        Drift prediction for the images
    threshold : float
        :term:`P-Value` used for significance of the permutation test
    p_val : float
        P-value obtained from the permutation test
    distance : float
        MMD^2 between the reference and test set
    distance_threshold : float
        MMD^2 threshold above which drift is flagged
    """

    # drifted: bool
    # threshold: float
    # p_val: float
    # distance: float
    distance_threshold: float


@dataclass(frozen=True)
class DriftOutput(DriftBaseOutput):
    """
    Output class for :class:`.DriftCVM`, :class:`.DriftKS`, and :class:`.DriftUncertainty` drift detectors.

    Attributes
    ----------
    drifted : bool
        :term:`Drift` prediction for the images
    threshold : float
        Threshold after multivariate correction if needed
    p_val : float
        Instance-level p-value
    distance : float
        Instance-level distance
    feature_drift : NDArray
        Feature-level array of images detected to have drifted
    feature_threshold : float
        Feature-level threshold to determine drift
    p_vals : NDArray
        Feature-level p-values
    distances : NDArray
        Feature-level distances
    """

    # drifted: bool
    # threshold: float
    # p_val: float
    # distance: float
    feature_drift: NDArray[np.bool_]
    feature_threshold: float
    p_vals: NDArray[np.float32]
    distances: NDArray[np.float32]


class DriftMVDCOutput(PerMetricResult):
    """Class wrapping the results of the classifier for drift detection and providing plotting functionality."""

    def __init__(self, results_data: pd.DataFrame) -> None:
        """Initialize a DomainClassifierCalculator results object.

        Parameters
        ----------
        results_data : pd.DataFrame
            Results data returned by a DomainClassifierCalculator.
        """
        metric = Metric(display_name="Domain Classifier", column_name="domain_classifier_auroc")
        super().__init__(results_data, [metric])

    def plot(self) -> Figure:
        """
        Render the roc_auc metric over the train/test data in relation to the threshold.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(dpi=300)
        resdf = self.to_dataframe()
        xticks = np.arange(resdf.shape[0])
        trndf = resdf[resdf["chunk"]["period"] == "reference"]
        tstdf = resdf[resdf["chunk"]["period"] == "analysis"]
        # Get local indices for drift markers
        driftx = np.where(resdf["domain_classifier_auroc"]["alert"].values)  # type: ignore | dataframe
        if np.size(driftx) > 2:
            ax.plot(resdf.index, resdf["domain_classifier_auroc"]["upper_threshold"], "r--", label="thr_up")
            ax.plot(resdf.index, resdf["domain_classifier_auroc"]["lower_threshold"], "r--", label="thr_low")
            ax.plot(trndf.index, trndf["domain_classifier_auroc"]["value"], "b", label="train")
            ax.plot(tstdf.index, tstdf["domain_classifier_auroc"]["value"], "g", label="test")
            ax.plot(
                resdf.index.values[driftx],  # type: ignore | dataframe
                resdf["domain_classifier_auroc"]["value"].values[driftx],  # type: ignore | dataframe
                "dm",
                markersize=3,
                label="drift",
            )
            ax.set_xticks(xticks)
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.legend(loc="lower left", fontsize=6)
            ax.set_title("Domain Classifier, Drift Detection", fontsize=8)
            ax.set_ylabel("ROC AUC", fontsize=7)
            ax.set_xlabel("Chunk Index", fontsize=7)
            ax.set_ylim((0.0, 1.1))
        return fig

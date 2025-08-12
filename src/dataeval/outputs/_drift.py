from __future__ import annotations

__all__ = []

import contextlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure

from dataeval.outputs._base import Output
from dataeval.outputs._result import Metric, PerMetricResult


@dataclass(frozen=True)
class DriftBaseOutput(Output):
    """Base output class for drift detector classes.

    Provides common fields returned by all drift detection methods, containing
    instance-level drift predictions and summary statistics. Subclasses extend
    this with detector-specific additional fields.

    Attributes
    ----------
    drifted : bool
        Whether drift was detected in the analyzed data. True indicates
        significant drift from reference distribution.
    threshold : float
        Significance threshold used for drift detection, typically between 0 and 1.
        For multivariate methods, this is the corrected threshold after
        Bonferroni or FDR correction.
    p_val : float
        Instance-level p-value from statistical test, between 0 and 1.
        For univariate methods, this is the mean p-value across all features.
    distance : float
        Instance-level test statistic or distance metric, always >= 0.
        For univariate methods, this is the mean distance across all features.
        Higher values indicate greater deviation from reference distribution.
    """

    drifted: bool
    threshold: float
    p_val: float
    distance: float


@dataclass(frozen=True)
class DriftMMDOutput(DriftBaseOutput):
    """
    Output class for :class:`.DriftMMD` (Maximum Mean Discrepancy) drift detector.

    Extends :class:`.DriftBaseOutput` with MMD-specific distance threshold information.
    Used by MMD-based drift detectors that compare kernel embeddings between
    reference and test distributions.

    Attributes
    ----------
    drifted : bool
        Whether drift was detected based on MMD permutation test.
    threshold : float
        P-value threshold used for significance of the permutation test.
    p_val : float
        P-value obtained from the MMD permutation test, between 0 and 1.
    distance : float
        Squared Maximum Mean Discrepancy between reference and test set.
        Always >= 0, with higher values indicating greater distributional difference.
    distance_threshold : float
        Squared Maximum Mean Discrepancy threshold above which drift is flagged, always >= 0.
        Determined from permutation test at specified significance level.

    Notes
    -----
    MMD uses kernel methods to compare distributions in reproducing kernel
    Hilbert spaces, making it effective for high-dimensional data like images.
    """

    distance_threshold: float


@dataclass(frozen=True)
class DriftOutput(DriftBaseOutput):
    """Output class for univariate drift detectors.

    Extends :class:`.DriftBaseOutput` with feature-level (per-pixel) drift information.
    Used by Kolmogorov-Smirnov, Cramér-von Mises, and uncertainty-based
    drift detectors that analyze each feature independently.

    Attributes
    ----------
    drifted : bool
        Overall drift prediction after multivariate correction.
    threshold : float
        Corrected threshold after Bonferroni or FDR correction for multiple testing.
    p_val : float
        Mean p-value across all features, between 0 and 1.
        For descriptive purposes only; individual feature p-values are used
        for drift detection decisions. Can appear high even when drifted=True
        if only a subset of features show drift.
    distance : float
        Mean test statistic across all features, always >= 0.
    feature_drift : NDArray[bool]
        Boolean array indicating which features (pixels) show drift.
        Shape matches the number of features in the input data.
    feature_threshold : float
        Uncorrected p-value threshold used for individual feature testing.
        Typically the original p_val before multivariate correction.
    p_vals : NDArray[np.float32]
        P-values for each feature, all values between 0 and 1.
        Shape matches the number of features in the input data.
    distances : NDArray[np.float32]
        Test statistics for each feature, all values >= 0.
        Shape matches the number of features in the input data.

    Notes
    -----
    Feature-level analysis enables identification of specific pixels or regions
    that contribute most to detected drift, useful for interpretability.
    """

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

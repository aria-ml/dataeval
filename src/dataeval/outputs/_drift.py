from __future__ import annotations

__all__ = []

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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

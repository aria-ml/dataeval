"""
:term:`Drift` detectors identify if the statistical properties of the data has changed.
"""

from dataeval import _IS_TORCH_AVAILABLE
from dataeval.detectors.drift import updates
from dataeval.detectors.drift.base import DriftOutput
from dataeval.detectors.drift.cvm import DriftCVM
from dataeval.detectors.drift.ks import DriftKS

__all__ = ["DriftCVM", "DriftKS", "DriftOutput", "updates"]

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval.detectors.drift.mmd import DriftMMD, DriftMMDOutput
    from dataeval.detectors.drift.torch import preprocess_drift
    from dataeval.detectors.drift.uncertainty import DriftUncertainty

    __all__ += ["DriftMMD", "DriftMMDOutput", "DriftUncertainty", "preprocess_drift"]

del _IS_TORCH_AVAILABLE

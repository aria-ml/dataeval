"""
Drift detectors identify if the statistical properties of the data has changed.
"""

from dataeval import _IS_TORCH_AVAILABLE
from dataeval._internal.detectors.drift.base import DriftOutput
from dataeval._internal.detectors.drift.cvm import DriftCVM
from dataeval._internal.detectors.drift.ks import DriftKS

from . import updates

__all__ = ["DriftCVM", "DriftKS", "DriftOutput", "updates"]

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval._internal.detectors.drift.mmd import DriftMMD, DriftMMDOutput
    from dataeval._internal.detectors.drift.torch import preprocess_drift
    from dataeval._internal.detectors.drift.uncertainty import DriftUncertainty

    from . import kernels

    __all__ += ["DriftMMD", "DriftMMDOutput", "DriftUncertainty", "kernels", "preprocess_drift"]

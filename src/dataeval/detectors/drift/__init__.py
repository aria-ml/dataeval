from dataeval import _IS_TORCH_AVAILABLE
from dataeval._internal.detectors.drift.cvm import DriftCVM
from dataeval._internal.detectors.drift.ks import DriftKS

from . import updates

__all__ = ["DriftCVM", "DriftKS", "updates"]

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval._internal.detectors.drift.mmd import DriftMMD
    from dataeval._internal.detectors.drift.torch import preprocess_drift
    from dataeval._internal.detectors.drift.uncertainty import DriftUncertainty

    from . import kernels

    __all__ += ["DriftMMD", "DriftUncertainty", "kernels", "preprocess_drift"]

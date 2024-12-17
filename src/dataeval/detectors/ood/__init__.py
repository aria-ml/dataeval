"""
Out-of-distribution (OOD)` detectors identify data that is different from the data used to train a particular model.
"""

from dataeval import _IS_TORCH_AVAILABLE
from dataeval.detectors.ood.base import OODOutput, OODScoreOutput

__all__ = ["OODOutput", "OODScoreOutput"]

if _IS_TORCH_AVAILABLE:
    from dataeval.detectors.ood.ae_torch import OOD_AE

    __all__ += ["OOD_AE"]

del _IS_TORCH_AVAILABLE

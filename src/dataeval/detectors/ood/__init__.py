"""
Out-of-distribution (OOD)` detectors identify data that is different from the data used to train a particular model.
"""

from dataeval import _IS_TENSORFLOW_AVAILABLE, _IS_TORCH_AVAILABLE
from dataeval.detectors.ood.base import OODOutput, OODScoreOutput

__all__ = ["OODOutput", "OODScoreOutput"]

if _IS_TENSORFLOW_AVAILABLE:
    from dataeval.detectors.ood.ae import OOD_AE
    from dataeval.detectors.ood.aegmm import OOD_AEGMM
    from dataeval.detectors.ood.llr import OOD_LLR
    from dataeval.detectors.ood.vae import OOD_VAE
    from dataeval.detectors.ood.vaegmm import OOD_VAEGMM

    __all__ += ["OOD_AE", "OOD_AEGMM", "OOD_LLR", "OOD_VAE", "OOD_VAEGMM"]

elif _IS_TORCH_AVAILABLE:
    from dataeval.detectors.ood.ae_torch import OOD_AE

    __all__ += ["OOD_AE", "OODOutput"]

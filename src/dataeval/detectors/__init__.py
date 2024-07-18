from importlib.util import find_spec

from dataeval._internal.detectors.clusterer import Clusterer
from dataeval._internal.detectors.drift.base import LastSeenUpdate, ReservoirSamplingUpdate
from dataeval._internal.detectors.drift.cvm import DriftCVM
from dataeval._internal.detectors.drift.ks import DriftKS
from dataeval._internal.detectors.duplicates import Duplicates
from dataeval._internal.detectors.linter import Linter

__all__ = ["Clusterer", "Duplicates", "Linter", "DriftCVM", "DriftKS", "LastSeenUpdate", "ReservoirSamplingUpdate"]

if find_spec("torch") is not None:  # pragma: no cover
    from dataeval._internal.detectors.drift.mmd import DriftMMD
    from dataeval._internal.detectors.drift.torch import GaussianRBF, preprocess_drift
    from dataeval._internal.detectors.drift.uncertainty import DriftUncertainty

    __all__ += ["DriftMMD", "GaussianRBF", "DriftUncertainty", "preprocess_drift"]

if find_spec("tensorflow") is not None and find_spec("tensorflow_probability") is not None:  # pragma: no cover
    from dataeval._internal.detectors.ood.ae import OOD_AE
    from dataeval._internal.detectors.ood.aegmm import OOD_AEGMM
    from dataeval._internal.detectors.ood.base import OODScore
    from dataeval._internal.detectors.ood.llr import OOD_LLR
    from dataeval._internal.detectors.ood.vae import OOD_VAE
    from dataeval._internal.detectors.ood.vaegmm import OOD_VAEGMM

    __all__ += ["OOD_AE", "OOD_AEGMM", "OOD_LLR", "OODScore", "OOD_VAE", "OOD_VAEGMM"]

del find_spec

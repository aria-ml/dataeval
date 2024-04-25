from importlib.util import find_spec

from daml._internal.metrics.drift.base import LastSeenUpdate, ReservoirSamplingUpdate
from daml._internal.metrics.drift.cvm import CVMDrift
from daml._internal.metrics.drift.ks import KSDrift

__all__ = ["CVMDrift", "KSDrift", "LastSeenUpdate", "ReservoirSamplingUpdate"]

if find_spec("torch") is not None:  # pragma: no cover
    from daml._internal.metrics.drift.mmd import MMDDrift
    from daml._internal.metrics.drift.torch import GaussianRBF, preprocess_drift
    from daml._internal.metrics.drift.uncertainty import UncertaintyDrift

    __all__ += ["MMDDrift", "GaussianRBF", "UncertaintyDrift", "preprocess_drift"]

del find_spec

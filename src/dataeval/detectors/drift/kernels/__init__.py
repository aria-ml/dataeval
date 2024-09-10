from dataeval import _IS_TORCH_AVAILABLE

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval._internal.detectors.drift.torch import GaussianRBF

    __all__ = ["GaussianRBF"]

from dataclasses import dataclass
from typing import Generic, TypeVar

TGMMData = TypeVar("TGMMData")


@dataclass
class GaussianMixtureModelParams(Generic[TGMMData]):
    """
    phi : TGMMData
        Mixture component distribution weights.
    mu : TGMMData
        Mixture means.
    cov : TGMMData
        Mixture covariance.
    L : TGMMData
        Cholesky decomposition of `cov`.
    log_det_cov : TGMMData
        Log of the determinant of `cov`.
    """

    phi: TGMMData
    mu: TGMMData
    cov: TGMMData
    L: TGMMData
    log_det_cov: TGMMData

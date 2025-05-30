from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from dataeval.detectors.drift._nml._chunk import CountBasedChunker, SizeBasedChunker
from dataeval.detectors.drift._nml._domainclassifier import DomainClassifierCalculator
from dataeval.detectors.drift._nml._thresholds import ConstantThreshold
from dataeval.outputs._drift import DriftMVDCOutput
from dataeval.utils._array import flatten


class DriftMVDC:
    """Multivariant Domain Classifier

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-validation (CV) folds.
    chunk_size : int or None, default None
        Number of samples in a chunk used in CV, will get one metric & prediction per chunk.
    chunk_count : int or None, default None
        Number of total chunks used in CV, will get one metric & prediction per chunk.
    threshold : Tuple[float, float], default (0.45, 0.65)
        (lower, upper) metric bounds on roc_auc for identifying :term:`drift<Drift>`.
    """

    def __init__(
        self,
        n_folds: int = 5,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        threshold: tuple[float, float] = (0.45, 0.65),
    ) -> None:
        self.threshold: tuple[float, float] = max(0.0, min(threshold)), min(1.0, max(threshold))
        chunker = (
            CountBasedChunker(10 if chunk_count is None else chunk_count)
            if chunk_size is None
            else SizeBasedChunker(chunk_size)
        )
        self._calc = DomainClassifierCalculator(
            cv_folds_num=n_folds,
            chunker=chunker,
            threshold=ConstantThreshold(lower=self.threshold[0], upper=self.threshold[1]),
        )

    def fit(self, x_ref: ArrayLike) -> DriftMVDC:
        """
        Fit the domain classifier on the training dataframe

        Parameters
        ----------
        x_ref : ArrayLike
            Reference data with dim[n_samples, n_features].

        Returns
        -------
        DriftMVDC

        """
        # for 1D input, assume that is 1 sample: dim[1,n_features]
        self.x_ref: pd.DataFrame = pd.DataFrame(flatten(np.atleast_2d(np.asarray(x_ref))))
        self.n_features: int = self.x_ref.shape[-1]
        self._calc.fit(self.x_ref)
        return self

    def predict(self, x: ArrayLike) -> DriftMVDCOutput:
        """
        Perform :term:`inference<Inference>` on the test dataframe

        Parameters
        ----------
        x : ArrayLike
            Test (analysis) data with dim[n_samples, n_features].

        Returns
        -------
        DomainClassifierDriftResult
        """
        self.x_test: pd.DataFrame = pd.DataFrame(flatten(np.atleast_2d(np.asarray(x))))
        if self.x_test.shape[-1] != self.n_features:
            raise ValueError("Reference and test embeddings have different number of features")

        return self._calc.calculate(self.x_test)

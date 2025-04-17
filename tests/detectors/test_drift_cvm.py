"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.detectors.drift._cvm import DriftCVM
from dataeval.utils.data._embeddings import Embeddings

np.random.seed(0)


@pytest.mark.required
class TestCVMDrift:
    n, n_test = 500, 200

    def get_embeddings(self, n: int = 100, n_features: int = 10, value: float | None = None) -> Embeddings:
        arr = (
            np.random.random((n, n_features)).astype(np.float32)
            if value is None
            else np.full(shape=(n, n_features), fill_value=value, dtype=np.float32)
        )
        mock = MagicMock(spec=Embeddings)
        mock.__getitem__.side_effect = lambda idx: arr[idx]
        mock.__len__.return_value = n
        mock.to_numpy.return_value = arr
        return mock

    def test_cvmdrift(self):
        # Reference data
        ref_emb = self.get_embeddings(self.n, value=0.0)

        # Instantiate detector
        cd = DriftCVM(data=ref_emb, p_val=0.05, correction="fdr")

        # Test predict on reference data
        preds = cd.predict(ref_emb)
        assert not preds.drifted and (preds.p_vals >= cd.p_val).any()

        # Test predict on heavily drifted data
        x = self.get_embeddings(self.n_test, value=0.5)
        preds = cd.predict(x)
        assert preds.drifted
        assert preds.distances.min() >= 0.0

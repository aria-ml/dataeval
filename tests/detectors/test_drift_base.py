"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from itertools import product
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dataeval.detectors.drift._base import BaseDriftUnivariate
from dataeval.detectors.drift.updates import LastSeenUpdate, ReservoirSamplingUpdate
from dataeval.typing import Dataset
from dataeval.utils.data._embeddings import Embeddings


@pytest.mark.required
class TestUpdateReference:
    n = [3, 50]
    n_features = [1, 10]
    update_method = [LastSeenUpdate, ReservoirSamplingUpdate]
    tests_update = list(product(n, n_features, update_method))
    n_tests_update = len(tests_update)

    @pytest.fixture(scope="class")
    def update_params(self, request):
        return self.tests_update[request.param]

    @pytest.mark.parametrize("update_params", list(range(n_tests_update)), indirect=True)
    def test_update_reference(self, update_params):
        n, n_features, update_method = update_params
        n_ref = np.random.randint(1, n)
        n_test = np.random.randint(1, 2 * n)
        X_ref = np.random.rand(n_ref * n_features).reshape(n_ref, n_features)
        X = np.random.rand(n_test * n_features).reshape(n_test, n_features)
        update_method = update_method(n)
        X_ref_new = update_method(X_ref, X, n)

        assert X_ref_new.shape[0] <= n
        if isinstance(update_method, LastSeenUpdate):
            assert (X_ref_new[-1] == X[-1]).all()


@pytest.mark.required
class TestBaseDrift:
    model = torch.nn.Identity()
    batch_size = 10
    device = torch.device("cpu")

    def get_dataset(self, n: int = 100, n_features: int = 10) -> Dataset:
        mock = MagicMock(spec=Dataset)
        mock._selection = list(range(n))
        mock.__len__.return_value = n
        mock.__getitem__.return_value = np.random.random(n_features), np.zeros(10), {}
        return mock

    def get_embeddings(self, n: int = 100, n_features: int = 10) -> Embeddings:
        return Embeddings(
            self.get_dataset(n, n_features), batch_size=self.batch_size, model=self.model, device=self.device
        )

    def test_base_init_update_x_ref_valueerror(self):
        with pytest.raises(ValueError):
            BaseDriftUnivariate(self.get_embeddings(1), update_strategy="invalid")  # type: ignore

    def test_base_init_correction_valueerror(self):
        with pytest.raises(ValueError):
            BaseDriftUnivariate(self.get_embeddings(1), n_features=2, correction="invalid")  # type: ignore

    def test_base_init_infer_n_features(self):
        base = BaseDriftUnivariate(self.get_embeddings(1))
        assert base.n_features == 10

    def test_base_init_set_n_features(self):
        base = BaseDriftUnivariate(self.get_embeddings(1), n_features=1)
        assert base.n_features == 1

    def test_base_predict_correction_valueerror(self):
        base = BaseDriftUnivariate(self.get_embeddings(1))
        mock_score = MagicMock()
        mock_score.return_value = (np.array(0.5), np.array(0.5))
        base.score = mock_score
        base.correction = "invalid"  # type: ignore
        with pytest.raises(ValueError):
            base.predict(np.empty([]))

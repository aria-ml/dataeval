"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from itertools import product
from typing import Any, Literal

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataeval.detectors.drift.torch import GaussianRBF, _squared_pairwise_distance, mmd2_from_kernel_matrix
from dataeval.utils.torch.internal import get_device, predict_batch


def id_fn(x):
    return x


class TestPredictBatch:
    n, n_features, n_classes, latent_dim = 100, 10, 5, 2
    x = np.zeros((n, n_features), dtype=np.float32)
    t = torch.zeros((n, n_features), dtype=torch.float32)

    class MyModel(nn.Module):
        n_features, n_classes = 10, 5

        def __init__(self, output: Literal["single", "multi", "numpy", "list", "unsupported"] = "single"):
            super().__init__()
            self.dense = nn.Linear(self.n_features, self.n_classes)
            self.output = output

        def forward(self, x: torch.Tensor) -> Any:
            out = self.dense(x)
            if self.output == "multi":
                return out, out
            elif self.output == "numpy":
                return out.numpy()
            elif self.output == "list":
                return [out, out]
            elif self.output == "unsupported":
                return {1, 2, 3}
            return out

    AutoEncoder = nn.Sequential(nn.Linear(n_features, latent_dim), nn.Linear(latent_dim, n_features))

    # model, batch size, dtype, preprocessing function
    tests_predict = [
        (x, MyModel("single"), 2, np.float32, None),
        (x, MyModel("single"), int(1e10), np.float32, None),
        (x, MyModel("single"), int(1e10), torch.float32, None),
        (x, MyModel("single"), int(1e10), np.float32, id_fn),
        (t, MyModel("single"), int(1e10), np.float32, id_fn),
        (x, MyModel("multi"), int(1e10), torch.float32, None),
        (t, MyModel("multi"), 2, torch.float32, None),
        (x, MyModel("numpy"), int(1e10), np.float32, None),
        (t, MyModel("numpy"), int(1e10), np.float32, None),
        (x, MyModel("list"), 2, np.float32, None),
        (t, MyModel("list"), int(1e10), torch.float32, None),
        (x, AutoEncoder, 2, np.float32, None),
        (x, AutoEncoder, int(1e10), np.float32, None),
        (t, AutoEncoder, int(1e10), np.float32, None),
        (x, AutoEncoder, int(1e10), torch.float32, None),
        (t, AutoEncoder, int(1e10), torch.float32, None),
        (x, id_fn, 2, np.float32, None),
        (x, id_fn, 2, torch.float32, None),
        (x, id_fn, 2, np.float32, id_fn),
        (t, id_fn, 2, torch.float32, None),
    ]
    n_tests = len(tests_predict)

    @pytest.fixture
    def params(self, request):
        return self.tests_predict[request.param]

    @pytest.mark.parametrize("params", list(range(n_tests)), indirect=True)
    def test_predict_batch(self, params):
        x, model, batch_size, dtype, preprocess_fn = params
        preds = predict_batch(
            x,
            model,
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            dtype=dtype,
            device=get_device("cpu"),
        )
        if isinstance(preds, tuple):
            preds = preds[0]
        assert preds.dtype == dtype
        if isinstance(model, nn.Sequential) or hasattr(model, "__name__") and model.__name__ == "id_fn":
            assert preds.shape == self.x.shape
        elif isinstance(model, nn.Module):
            assert preds.shape == (self.n, self.n_classes)

    def test_predict_batch_unsupported_model(self):
        with pytest.raises(TypeError):
            predict_batch(self.x, self.MyModel("unsupported"), device=get_device("cpu"))


class TestSquaredPairwiseDistance:
    n_features = [2, 5]
    n_instances = [(100, 100), (100, 75)]
    tests_pairwise = list(product(n_features, n_instances))
    n_tests_pairwise = len(tests_pairwise)

    @pytest.fixture
    def pairwise_params(self, request):
        return self.tests_pairwise[request.param]

    @pytest.mark.parametrize("pairwise_params", list(range(n_tests_pairwise)), indirect=True)
    def test_pairwise(self, pairwise_params):
        n_features, n_instances = pairwise_params
        xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
        np.random.seed(0)
        x = torch.from_numpy(np.random.random(xshape).astype("float32"))
        y = torch.from_numpy(np.random.random(yshape).astype("float32"))

        dist_xx = _squared_pairwise_distance(x, x).numpy()
        dist_xy = _squared_pairwise_distance(x, y).numpy()

        assert dist_xx.shape == (xshape[0], xshape[0])
        assert dist_xy.shape == n_instances
        np.testing.assert_almost_equal(dist_xx.trace(), 0.0, decimal=5)


class TestMMDKernelMatrix:
    n_features = [2, 5]
    n_instances = [(100, 100), (100, 75)]
    batch_size = [1, 5]
    tests_bckm = list(product(n_features, n_instances, batch_size))
    n_tests_bckm = len(tests_bckm)

    n = [10, 100]
    m = [10, 100]
    permute = [True, False]
    zero_diag = [True, False]
    tests_mmd_from_kernel_matrix = list(product(n, m, permute, zero_diag))
    n_tests_mmd_from_kernel_matrix = len(tests_mmd_from_kernel_matrix)

    @pytest.fixture
    def mmd_from_kernel_matrix_params(self, request):
        return self.tests_mmd_from_kernel_matrix[request.param]

    @pytest.mark.parametrize(
        "mmd_from_kernel_matrix_params",
        list(range(n_tests_mmd_from_kernel_matrix)),
        indirect=True,
    )
    def test_mmd_from_kernel_matrix(self, mmd_from_kernel_matrix_params):
        n, m, permute, zero_diag = mmd_from_kernel_matrix_params
        n_tot = n + m
        shape = (n_tot, n_tot)
        kernel_mat = np.random.uniform(0, 1, size=shape)
        kernel_mat_2 = kernel_mat.copy()
        kernel_mat_2[-m:, :-m] = 1.0
        kernel_mat_2[:-m, -m:] = 1.0
        kernel_mat = torch.from_numpy(kernel_mat)
        kernel_mat_2 = torch.from_numpy(kernel_mat_2)
        if not zero_diag:
            kernel_mat -= torch.diag(kernel_mat.diag())
            kernel_mat_2 -= torch.diag(kernel_mat_2.diag())
        mmd = mmd2_from_kernel_matrix(kernel_mat, m, permute=permute, zero_diag=zero_diag)
        mmd_2 = mmd2_from_kernel_matrix(kernel_mat_2, m, permute=permute, zero_diag=zero_diag)
        if not permute:
            assert mmd_2.numpy() < mmd.numpy()


@pytest.mark.parametrize("device", [None, torch.device("cpu"), "cpu", "gpu", "cuda", "random"])
def test_drift_get_device(device):
    assert isinstance(get_device(device), torch.device)


def test_gaussianrbf_forward_valueerror():
    g = GaussianRBF(trainable=True)
    with pytest.raises(ValueError):
        g.forward(np.zeros((2, 2)), np.zeros((2, 2)), infer_sigma=True)

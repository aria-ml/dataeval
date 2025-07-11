"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
import torch
import torch.nn as nn
from maite_datasets import to_image_classification_dataset

from dataeval.config import get_device
from dataeval.data._embeddings import Embeddings
from dataeval.detectors.drift._mmd import DriftMMD, GaussianRBF, _squared_pairwise_distance, mmd2_from_kernel_matrix
from dataeval.detectors.drift.updates import LastSeenUpdate, ReservoirSamplingUpdate


class HiddenOutput(nn.Module):
    def __init__(
        self,
        model: nn.Module | nn.Sequential,
        layer: int = -1,
        flatten: bool = False,
    ) -> None:
        super().__init__()
        layers = list(model.children())[:layer]
        if flatten:
            layers += [nn.Flatten()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MyModel(nn.Module):
    def __init__(self, n_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.dense0 = nn.Flatten()
        self.dense1 = nn.Linear(math.prod(n_shape), 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(self.dense0(x)))
        return self.dense2(x)


def get_embeddings(n: int, n_shape: tuple[int, int, int], n_classes: int, model: nn.Module) -> Embeddings:
    images = np.random.randn(n * math.prod(n_shape)).reshape(n, *n_shape).astype(np.float32)
    dataset = to_image_classification_dataset(images, np.random.randint(n_classes, size=n).tolist(), None, None)
    return Embeddings(dataset=dataset, model=model, batch_size=n, device="cpu")


@pytest.mark.required
class TestMMDDrift:
    n, n_hidden, n_classes = 100, 10, 5
    n_shape = (1, 16, 16)
    model = [nn.Identity(), HiddenOutput(MyModel(n_shape), layer=-1)]
    update_strategy = [LastSeenUpdate(750), ReservoirSamplingUpdate(750), None]
    n_permutations = [10]
    sigma = [np.array([[1, 0], [0, 1]]), None]
    tests_mmddrift = list(product(model, n_permutations, update_strategy, sigma))
    n_tests = len(tests_mmddrift)

    @pytest.fixture(scope="class")
    def mmd_params(self, request):
        return self.tests_mmddrift[request.param]

    @pytest.mark.parametrize("mmd_params", list(range(n_tests)), indirect=True)
    def test_mmd(self, mmd_params):
        model, n_permutations, update_strategy, sigma = mmd_params

        np.random.seed(0)
        torch.manual_seed(0)
        x_ref = get_embeddings(self.n, self.n_shape, self.n_classes, model)
        cd = DriftMMD(
            data=x_ref,
            p_val=0.05,
            update_strategy=update_strategy,
            sigma=sigma,
            n_permutations=n_permutations,
            device="cpu",
        )
        preds = cd.predict(x_ref)
        assert not preds.drifted and preds.p_val >= cd.p_val
        if isinstance(update_strategy, dict):
            k = list(update_strategy.keys())[0]
            assert cd.n == self.n + self.n
            assert cd._data.shape[0] == min(update_strategy[k], self.n + self.n)  # type: ignore

        preds = cd.predict(get_embeddings(self.n, self.n_shape, self.n_classes, model))
        if preds.drifted:
            assert preds.p_val < preds.threshold == cd.p_val
            assert preds.distance > preds.distance_threshold
        else:
            assert preds.p_val >= preds.threshold == cd.p_val
            assert preds.distance <= preds.distance_threshold


@pytest.mark.required
class TestSquaredPairwiseDistance:
    n_features = [2, 5]
    n_instances = [(100, 100), (100, 75)]
    tests_pairwise = list(product(n_features, n_instances))
    n_tests_pairwise = len(tests_pairwise)

    @pytest.fixture(scope="class")
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


@pytest.mark.required
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

    @pytest.fixture(scope="class")
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


@pytest.mark.required
@pytest.mark.parametrize("device", [None, torch.device("cpu"), "cpu", "cuda"])
def test_drift_get_device(device):
    assert isinstance(get_device(device), torch.device)


@pytest.mark.required
def test_gaussianrbf_forward_valueerror():
    g = GaussianRBF(trainable=True)
    with pytest.raises(ValueError):
        g.forward(np.zeros((2, 2)), np.zeros((2, 2)), infer_sigma=True)

"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from functools import partial
from itertools import product
from typing import Callable, List, Union

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataeval._internal.detectors.drift.base import (
    LastSeenUpdate,
    ReservoirSamplingUpdate,
)
from dataeval.detectors import DriftMMD, preprocess_drift

n, n_hidden, n_classes = 500, 10, 5


class HiddenOutput(nn.Module):
    def __init__(
        self,
        model: Union[nn.Module, nn.Sequential],
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
    def __init__(self, n_features: int):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        return self.dense2(x)


# test List[Any] inputs to the detector
def preprocess_list(x: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(x, axis=0)


class TestMMDDrift:
    n_features = [10]
    preprocess = [
        (None, None),
        (preprocess_drift, {"model": HiddenOutput, "layer": -1}),
    ]
    update_x_ref = [LastSeenUpdate(750), ReservoirSamplingUpdate(750), None]
    n_permutations = [10]
    sigma = [np.array([[1, 0], [0, 1]]), None]
    configure_kernel_from_x_ref = [True, False]
    tests_mmddrift = list(
        product(
            n_features,
            preprocess,
            n_permutations,
            update_x_ref,
            sigma,
            configure_kernel_from_x_ref,
        )
    )
    n_tests = len(tests_mmddrift)

    @pytest.fixture
    def mmd_params(self, request):
        return self.tests_mmddrift[request.param]

    @pytest.mark.parametrize("mmd_params", list(range(n_tests)), indirect=True)
    def test_mmd(self, mmd_params):
        (
            n_features,
            preprocess,
            n_permutations,
            update_x_ref,
            sigma,
            configure_kernel_from_x_ref,
        ) = mmd_params

        np.random.seed(0)
        torch.manual_seed(0)

        x_ref = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
        preprocess_fn, preprocess_kwargs = preprocess
        if (
            isinstance(preprocess_fn, Callable)
            and "layer" in list(preprocess_kwargs.keys())
            and preprocess_kwargs["model"].__name__ == "HiddenOutput"
        ):
            model = MyModel(n_features)
            layer = preprocess_kwargs["layer"]
            preprocess_fn = partial(
                preprocess_fn,
                model=HiddenOutput(model=model, layer=layer),
                device="cpu",
            )
        else:
            preprocess_fn = None

        cd = DriftMMD(
            x_ref=x_ref,
            p_val=0.05,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            sigma=sigma,
            configure_kernel_from_x_ref=configure_kernel_from_x_ref,
            n_permutations=n_permutations,
            device="cpu",
        )
        x = x_ref.copy()
        preds = cd.predict(x)
        assert not preds.is_drift and preds.p_val >= cd.p_val
        if isinstance(update_x_ref, dict):
            k = list(update_x_ref.keys())[0]
            assert cd.n == len(x) + len(x_ref)
            assert cd.x_ref.shape[0] == min(update_x_ref[k], len(x) + len(x_ref))  # type: ignore

        x_h1 = np.random.randn(n * n_features).reshape(n, n_features).astype(np.float32)
        preds = cd.predict(x_h1)
        if preds.is_drift:
            assert preds.p_val < preds.threshold == cd.p_val
            assert preds.distance > preds.distance_threshold
        else:
            assert preds.p_val >= preds.threshold == cd.p_val
            assert preds.distance <= preds.distance_threshold


def test_mmd_init_preprocess_fn_valueerror():
    with pytest.raises(ValueError):
        DriftMMD([], preprocess_fn="NotCallable")  # type: ignore

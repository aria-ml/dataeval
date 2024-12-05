"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from itertools import product

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits

from dataeval.detectors.ood.ae_torch import OOD_AE
from dataeval.utils.torch.models import AE

threshold_perc = [90.0]
loss_fn = [torch.nn.MSELoss(), None]
ood_type = ["instance", "feature"]

tests = list(product(threshold_perc, loss_fn, ood_type))[:-1]
n_tests = len(tests)
input_shape = (1, 8, 8)


# load iris data
@pytest.fixture
def x_ref() -> np.ndarray:
    X, y = load_digits(return_X_y=True)
    assert isinstance(X, np.ndarray)
    X = X.astype(np.float32)
    X = X.reshape(X.shape[0], *input_shape)
    return X


@pytest.fixture
def ae_params(request):
    return tests[request.param]


@pytest.mark.parametrize("ae_params", list(range(n_tests)), indirect=True)
def test_ae(ae_params, x_ref):
    # OutlierAE parameters
    threshold_perc, loss_fn, ood_type = ae_params

    # init OutlierAE
    ae = OOD_AE(AE(input_shape=input_shape))

    # fit OutlierAE, infer threshold and compute scores
    ae.fit(x_ref, threshold_perc=threshold_perc, loss_fn=loss_fn, epochs=1, verbose=True)
    iscore = ae._ref_score.instance_score
    perc_score = 100 * (iscore < ae._threshold_score()).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = ae.predict(x_ref, ood_type=ood_type)
    scores = ae._threshold_score(ood_type)

    if ood_type == "instance":
        assert od_preds.is_ood.shape == (x_ref.shape[0],)
        assert od_preds.is_ood.sum() == (od_preds.instance_score > scores).sum()
    elif ood_type == "feature":
        assert od_preds.is_ood.shape == x_ref.shape
        assert od_preds.feature_score is not None
        assert od_preds.feature_score.shape == x_ref.shape
        assert od_preds.is_ood.sum() == (od_preds.feature_score > scores).sum()

    assert od_preds.instance_score.shape == (x_ref.shape[0],)

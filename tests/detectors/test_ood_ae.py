"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.datasets import load_digits

from dataeval.detectors.ood.ae import OOD_AE
from dataeval.utils.torch.models import AE

input_shape = (1, 8, 8)


# load iris data
@pytest.fixture(scope="module")
def x_ref() -> np.ndarray:
    X, y = load_digits(return_X_y=True)
    assert isinstance(X, np.ndarray)
    X = X.astype(np.float32)
    X = X.reshape(X.shape[0], *input_shape)
    return X


@pytest.mark.parametrize("ood_type", ["instance", "feature"])
def test_ae(ood_type, x_ref):
    # OutlierAE parameters
    threshold_perc = 90.0

    # init OutlierAE
    ae = OOD_AE(AE(input_shape=input_shape))

    # fit OutlierAE, infer threshold and compute scores
    ae.fit(x_ref, threshold_perc=threshold_perc, epochs=1, verbose=True)
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


@patch("dataeval.detectors.ood.ae.OODBase.fit")
def test_custom_loss_fn(mock_fit, x_ref):
    mock_loss_fn = MagicMock()
    ae = OOD_AE(AE(input_shape=input_shape))
    ae.fit(x_ref, 0.0, mock_loss_fn)
    assert isinstance(mock_fit.call_args_list[0][0][2], MagicMock)


@patch("dataeval.detectors.ood.ae.OODBase.fit")
def test_custom_optimizer(mock_fit, x_ref):
    mock_opt = MagicMock()
    ae = OOD_AE(AE(input_shape=input_shape))
    ae.fit(x_ref, 0.0, None, mock_opt)
    assert isinstance(mock_fit.call_args_list[0][0][3], MagicMock)

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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from dataeval.detectors.drift._uncertainty import DriftUncertainty, classifier_uncertainty
from dataeval.detectors.drift.updates import LastSeenUpdate, ReservoirSamplingUpdate


class PtModel(nn.Module):
    def __init__(self, n_features, n_labels, softmax=False, dropout=False):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, n_labels)
        self.dropout = nn.Dropout(0.5) if dropout else lambda x: x
        self.softmax = nn.Softmax() if softmax else lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


def gen_model(n_features, n_labels, softmax=False, dropout=False):
    return PtModel(n_features, n_labels, softmax, dropout)


def id_fn(x: list) -> torch.Tensor:
    return torch.from_numpy(np.concatenate(x, axis=0))


@pytest.mark.required
class TestFunctionalClassifierUncertainty:
    p_val = [0.05]
    n_features = [16]
    n_labels = [3]
    preds_type = ["probs", "logits"]
    update_x_ref = [LastSeenUpdate(1000), ReservoirSamplingUpdate(1000), None]
    test_params = list(
        product(
            p_val,
            n_features,
            n_labels,
            preds_type,
            update_x_ref,
        )
    )
    n_tests = len(test_params)

    @pytest.fixture(scope="class")
    def clfuncdrift_params(self, request):
        return self.test_params[request.param]

    @pytest.mark.parametrize("clfuncdrift_params", list(range(n_tests)), indirect=True)
    def test_clfuncdrift(self, clfuncdrift_params):
        (
            p_val,
            n_features,
            n_labels,
            preds_type,
            update_x_ref,
        ) = clfuncdrift_params

        np.random.seed(0)

        model = gen_model(n_features, n_labels, preds_type == "probs")
        x_ref = np.random.randn(*(500, n_features)).astype(np.float32)
        x_test0 = x_ref.copy()
        x_test1 = np.ones_like(x_ref)

        cd = DriftUncertainty(
            x_ref=x_ref,
            model=model,  # type: ignore
            p_val=p_val,
            update_x_ref=update_x_ref,
            preds_type=preds_type,
            batch_size=10,
            preprocess_batch_fn=None,
            device="cpu",
        )

        preds_0 = cd.predict(x_test0)
        assert cd._detector.n == len(x_test0) + len(x_ref)
        assert not preds_0.is_drift
        assert preds_0.distances >= 0

        preds_1 = cd.predict(x_test1)
        assert cd._detector.n == len(x_test1) + len(x_test0) + len(x_ref)
        assert preds_1.is_drift
        assert preds_1.distances >= 0
        assert preds_0.distances < preds_1.distances


@pytest.mark.required
class TestClassifierUncertainty:
    n, n_features = 100, 10
    shape = (n_features,)
    X_train = np.random.rand(n * n_features).reshape(n, n_features).astype("float32")
    y_train_reg = np.random.rand(n).astype("float32")
    y_train_clf = np.random.choice(2, n)
    X_test = np.random.rand(n * n_features).reshape(n, n_features).astype("float32")

    tests_cu = ["probs", "logits"]
    n_tests_cu = len(tests_cu)

    @pytest.fixture(scope="class")
    def cu_params(self, request):
        return self.tests_cu[request.param]

    @pytest.mark.parametrize("cu_params", list(range(n_tests_cu)), indirect=True)
    def test_classifier_uncertainty(self, cu_params):
        preds_type = cu_params
        clf = LogisticRegression().fit(self.X_train, self.y_train_clf)
        model_fn = clf.predict_log_proba if preds_type == "logits" else clf.predict_proba
        uncertainties = classifier_uncertainty(self.X_test, model_fn, preds_type=preds_type)
        assert uncertainties.shape == (self.X_test.shape[0], 1)

    def test_classifier_uncertainty_notimplementederror(self):
        with pytest.raises(NotImplementedError):
            classifier_uncertainty(np.empty([]), lambda x: x, "invalid")  # type: ignore

    def test_classifier_uncertainty_valueerror(self):
        with pytest.raises(ValueError):
            classifier_uncertainty(np.empty([]), lambda x: 10 + x, "probs")

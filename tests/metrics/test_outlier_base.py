import numpy as np
import pytest

from daml._internal.metrics.outlier.base import BaseGMMOutlier, OutlierScore
from daml._internal.models.tensorflow.autoencoder import AE
from daml._internal.models.tensorflow.utils import create_model

image_shape = (32, 32, 1)
model = create_model(AE, image_shape)


class TestOutlier(BaseGMMOutlier):
    def score(self, X: np.ndarray, batch_size: int = int(1e10)) -> OutlierScore:
        return OutlierScore(np.array([0.0]), np.array([0.0]))


def test_invalid_model_raises_typeerror():
    with pytest.raises(TypeError):
        TestOutlier("invalid")  # type: ignore


def test_invalid_data_raises_typeerror():
    outlier = TestOutlier(model)
    with pytest.raises(TypeError):
        outlier._get_data_info("invalid")  # type: ignore


def test_validate_state_raises_runtimeerror():
    outlier = TestOutlier(model)
    with pytest.raises(RuntimeError):
        outlier._validate_state("invalid")  # type: ignore


def test_validate_raises_runtimeerror():
    outlier = TestOutlier(model)
    outlier._data_info = (image_shape, np.float64)
    with pytest.raises(RuntimeError):
        outlier._validate(np.array([0], dtype=np.int8))  # type: ignore


def test_validate_state_additional_attrs():
    outlier = TestOutlier(model)
    outlier._ref_score = "not none"  # type: ignore
    outlier._threshold_perc = 99.0
    outlier._data_info = (), np.float32
    outlier.gmm_params = "not none"  # type: ignore
    # should pass
    outlier._validate_state(np.array([0.0], dtype=np.float32))
    # should fail
    with pytest.raises(RuntimeError):
        outlier._validate_state(np.array([0.0], dtype=np.float32), ["other_param"])

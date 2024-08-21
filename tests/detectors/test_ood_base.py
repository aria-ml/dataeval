import numpy as np
import pytest

from dataeval._internal.detectors.ood.base import OODGMMBase, OODScore
from dataeval._internal.interop import ArrayLike
from dataeval._internal.models.tensorflow.autoencoder import AE
from dataeval._internal.models.tensorflow.utils import create_model

image_shape = (32, 32, 1)
model = create_model(AE, image_shape)


class MockOutlier(OODGMMBase):
    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScore:
        return OODScore(np.array([0.0]), np.array([0.0]))


def test_invalid_model_raises_typeerror():
    with pytest.raises(TypeError):
        MockOutlier("invalid")  # type: ignore


def test_invalid_data_raises_typeerror():
    outlier = MockOutlier(model)
    with pytest.raises(TypeError):
        outlier._get_data_info("invalid")  # type: ignore


def test_validate_state_raises_runtimeerror():
    outlier = MockOutlier(model)
    with pytest.raises(RuntimeError):
        outlier._validate_state("invalid")  # type: ignore


def test_validate_raises_runtimeerror():
    outlier = MockOutlier(model)
    outlier._data_info = (image_shape, np.float64)
    with pytest.raises(RuntimeError):
        outlier._validate(np.array([0], dtype=np.int8))  # type: ignore


def test_validate_state_additional_attrs():
    outlier = MockOutlier(model)
    outlier._ref_score = "not none"  # type: ignore
    outlier._threshold_perc = 99.0
    outlier._data_info = (), np.float32
    outlier.gmm_params = "not none"  # type: ignore
    # should pass
    outlier._validate_state(np.array([0.0], dtype=np.float32))
    # should fail
    with pytest.raises(RuntimeError):
        outlier._validate_state(np.array([0.0], dtype=np.float32), ["other_param"])

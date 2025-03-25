from typing import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval.detectors.ood.base import OODBaseGMM
from dataeval.detectors.ood.mixin import OODBaseMixin, OODGMMMixin
from dataeval.outputs._ood import OODScoreOutput

image_shape = (32, 32, 1)
model = MagicMock()


class MockOOD(OODGMMMixin, OODBaseMixin[Callable]):
    def _score(self, X: NDArray[np.float32], batch_size: int = int(1e10)) -> OODScoreOutput:
        return OODScoreOutput(np.array([0.0]), np.array([0.0]))


class MockOODGMM(OODBaseGMM):
    def _score(self, X: NDArray[np.float32], batch_size: int = int(1e10)) -> OODScoreOutput:
        return OODScoreOutput(np.array([0.0]), np.array([0.0]))


@pytest.mark.required
def test_invalid_data_raises_typeerror():
    outlier = MockOOD(model)
    with pytest.raises(TypeError):
        outlier._get_data_info("invalid")  # type: ignore


@pytest.mark.required
def test_validate_state_raises_runtimeerror():
    outlier = MockOOD(model)
    with pytest.raises(RuntimeError):
        outlier._validate_state(np.array([]))


@pytest.mark.required
def test_validate_raises_runtimeerror():
    outlier = MockOOD(model)
    outlier._data_info = (image_shape, np.float64)
    with pytest.raises(RuntimeError):
        outlier._validate(np.array([0], dtype=np.int8))  # type: ignore


@pytest.mark.required
def test_validate_state_additional_attrs():
    outlier = MockOOD(model)
    outlier._ref_score = "not none"  # type: ignore
    outlier._threshold_perc = 99.0
    outlier._data_info = (), np.float32
    outlier._gmm_params = "not none"  # type: ignore
    # should pass
    outlier._validate_state(np.array([0.0], dtype=np.float32))


@pytest.mark.required
def test_oodbasegmm_fit():
    outlier = MockOODGMM(lambda _: (1, 1, 1))  # type: ignore
    mock_trainer = patch("dataeval.detectors.ood.base.trainer").start()
    mock_gmm_params = patch("dataeval.detectors.ood.base.gmm_params").start()
    outlier.fit(np.zeros((10, 3, 64, 64)), 0.5, None, None, 1, 1, False)

    assert mock_trainer.called
    assert mock_gmm_params.called


@pytest.mark.required
def test_ood_unit_interval():
    data = np.random.randint(0, 255, size=(10, 3, 16, 16))
    outlier = MockOOD(lambda _: (1, 1, 1))  # type: ignore
    with pytest.raises(ValueError):
        outlier._get_data_info(data)

from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

from daml._internal.metrics.aria.base import _BaseMetric
from daml.datasets import DamlDataset


class MockNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, x):
        return x

    def forward(self, x):
        pass


class MockMetric(_BaseMetric):
    def __init__(
        self,
        encode: bool,
        model: Optional[nn.Module],
        fit: Optional[bool],
        epochs: Optional[int],
    ) -> None:
        super().__init__(
            DamlDataset(np.ndarray([])),
            encode,
            model=model,
            fit=fit,
            epochs=epochs,
        )

    def _evaluate(self) -> int:
        return 0


class TestBaseValidation:
    @pytest.mark.parametrize(
        "name, args, error_expected",
        [
            ("model_wrong_type", (True, "model", True, 3), TypeError),
            ("missing_model", (True, None, True, 3), ValueError),
            ("missing_fit", (True, MockNet(), None, 3), ValueError),
            ("missing_epochs", (True, MockNet(), True, None), ValueError),
            ("no_encode_with_model", (False, MockNet(), True, 3), ValueError),
        ],
    )
    def test_validation(self, name, args, error_expected):
        with pytest.raises(error_expected):
            MockMetric(*args)


class TestBaseEncode:
    @pytest.mark.parametrize(
        "name, metric_args, data, expected_shape",
        [
            (
                "encode_numpy_to_numpy",
                (True, MockNet(), False, None),
                np.ones([1, 32, 32, 3]),
                (1, 3, 32, 32),
            ),
            (
                "encode_torch_to_numpy",
                (True, MockNet(), True, 3),
                torch.ones([1, 3, 32, 32]),
                (1, 3, 32, 32),
            ),
            (
                "no_encode_numpy_to_numpy",
                (False, None, None, None),
                np.ones([1, 32, 32, 3]),
                (1, 32, 32, 3),
            ),
            (
                "no_encode_torch_to_numpy",
                (False, None, None, None),
                torch.ones([1, 3, 32, 32]),
                (1, 3, 32, 32),
            ),
        ],
    )
    def test_encode(self, name, metric_args, data, expected_shape):
        m = MockMetric(*metric_args)
        result = m._encode(data)
        assert result.shape == expected_shape

import numpy as np
import pytest
import torch

from dataeval.interop import to_numpy

array_native = [[0, 1], [2, 3]]
array_expected = np.asarray(array_native)


class TestInteropArrayLike:
    @pytest.mark.parametrize(
        "param, expected",
        (
            (array_native, array_expected),
            (np.array(array_native), array_expected),
            (torch.Tensor(array_native), array_expected),
            (None, np.array([])),
        ),
    )
    def test_to_numpy(self, param, expected):
        np.testing.assert_equal(to_numpy(param), expected)

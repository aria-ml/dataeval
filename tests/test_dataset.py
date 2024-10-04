from itertools import product
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import mnist

corrupt = [
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "shear",
    "scale",
    "rotate",
    "brightness",
    "translate",
    "stripe",
    "fog",
    "spatter",
    "dotted_line",
    "zigzag",
    "canny_edges",
]
tests = list(
    product(["./data"], [True], [False], [-1], [False], [None], [None], [False], [None], corrupt, [(60000, 28, 28)])
)
n_tests = len(tests)


@pytest.fixture
def mnist_params(request):
    return tests[request.param]


class TestMNIST:
    @pytest.mark.parametrize(
        "root, train, download, size, unit_interval, dtype, channels, flatten, normalize, corruption, output",
        [
            ("./data", True, True, -1, False, None, "channels_last", False, None, None, (60000, 28, 28, 1)),
            (
                Path("./data"),
                False,
                False,
                1000,
                True,
                np.float32,
                "channels_first",
                False,
                (0, 1),
                "identity",
                (1000, 1, 28, 28),
            ),
            ("./data", False, False, 5, True, None, None, True, None, None, (5, 784)),
        ],
    )
    def test_MNIST(
        self, root, train, download, size, unit_interval, dtype, channels, flatten, normalize, corruption, output
    ):
        """Unit testing of MNIST class"""
        data, _ = mnist(root, train, download, size, unit_interval, dtype, channels, flatten, normalize, corruption)
        assert data.shape == output

    @pytest.mark.parametrize("mnist_params", list(range(n_tests)), indirect=True)
    def test_MNIST_corrupt(self, mnist_params):
        """Unit testing of MNIST class"""
        root, train, download, size, unit_interval, dtype, channels, flatten, normalize, corruption, output = (
            mnist_params
        )
        data, _ = mnist(root, train, download, size, unit_interval, dtype, channels, flatten, normalize, corruption)
        assert data.shape == output

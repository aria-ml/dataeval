from __future__ import annotations

from typing import Literal

import pytest
from numpy.typing import NDArray

from dataeval._internal.datasets import MNIST


@pytest.fixture
def mnist():
    def _method(
        train: bool = True,
        size: int = 1000,
        unit_normalize: bool = False,
        dtype: type | None = None,
        channels: Literal["channels_first", "channels_last"] | None = None,
        flatten: bool = False,
    ) -> tuple[NDArray, NDArray]:
        dataset = MNIST(
            root="./data/",
            train=train,
            download=True,
            size=size,
            unit_interval=unit_normalize,
            dtype=dtype,
            channels=channels,
            flatten=flatten,
        )

        return dataset.data, dataset.targets

    return _method

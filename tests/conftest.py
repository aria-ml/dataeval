from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.utils.datasets import download_mnist


@pytest.fixture
def mnist():
    def _method(
        size: int = 1000,
        category: Literal["train", "test"] = "train",
        dtype: type | None = None,
        add_channels: Literal["channels_first", "channels_last", "none"] = "none",
        unit_normalize: bool = False,
    ) -> tuple[NDArray, NDArray]:
        path = download_mnist()
        with np.load(path, allow_pickle=True) as fp:
            images, labels = fp["x_" + category][:size], fp["y_" + category][:size]
        if dtype:
            images = images.astype(dtype)

        if add_channels == "channels_last":
            images = images[..., np.newaxis]
        elif add_channels == "channels_first":
            images = images[:, np.newaxis]
        return images / 255 if unit_normalize else images, labels

    return _method

from typing import Literal, Optional, Tuple

import numpy as np
import pytest

from tests.utils.datasets import download_mnist


@pytest.fixture
def mnist():
    def _method(
        size: int = 1000,
        category: Literal["train", "test"] = "train",
        dtype: Optional[type] = None,
        add_channels: Literal["channels_first", "channels_last", "none"] = "none",
        unit_normalize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        path = download_mnist()
        with np.load(path, allow_pickle=True) as fp:
            images, labels = fp["x_" + category][:size], fp["y_" + category][:size]
        if dtype is not None:
            images = images.astype(dtype)

        rescale_factor = 1.0/255 if unit_normalize else 1

        if add_channels == "channels_last":
            images = images[..., np.newaxis] * rescale_factor
        elif add_channels == "channels_first":
            images = images[:, np.newaxis] * rescale_factor
        return images, labels

    return _method

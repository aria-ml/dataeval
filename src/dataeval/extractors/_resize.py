"""Shared CHW image resize helper (IR-3.1-S-4)."""

__all__ = ["resize_chw"]

from typing import Any

import numpy as np
from numpy.typing import NDArray


def resize_chw(image: NDArray[Any], size: tuple[int, int]) -> NDArray[np.floating[Any]]:
    """Bilinearly resize a CHW image to ``(height, width)``."""
    import torch

    height, width = size
    tensor = torch.as_tensor(np.asarray(image)).float()
    if tensor.ndim != 3:
        raise ValueError(f"resize_chw expects CHW images; got shape {tuple(tensor.shape)}")
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).numpy()

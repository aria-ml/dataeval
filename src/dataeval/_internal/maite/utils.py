from typing import Tuple

import numpy as np
import torch

import maite.protocols.image_classification as ic
from maite.protocols import ArrayLike


def arraylike_to_numpy(xp: ArrayLike) -> np.ndarray:
    """Converts ArrayLike objects to numpy"""

    # Must ensure Tensors are not on GPU
    return xp.detach().cpu().numpy() if isinstance(xp, torch.Tensor) else np.asarray(xp)


# TODO: Overload with od.Dataset
# TODO: Check if batching aggregation is faster (e.g. DataLoader)
# TODO: Add verbosity flags (tqdm?)
def extract_to_numpy(dataset: ic.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate over dataset and separate images from labels"""
    images = []
    labels = []

    # (image, label, metadata)
    for image, label, _ in dataset:
        images.append(image)
        labels.append(label)

    return np.asarray(images), np.asarray(labels)

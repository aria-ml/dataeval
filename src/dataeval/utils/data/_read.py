from __future__ import annotations

__all__ = []

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataeval.utils.data.datasets import VOCDetection


def read_dataset(dataset: Dataset[Any]) -> list[list[Any]]:
    """
    Extract information from a dataset at each index into individual lists of each information position.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Input dataset

    Returns
    -------
    List[List[Any]]
        All objects in individual lists based on return position from dataset

    Warning
    -------
    No type checking is done between lists or data inside lists

    See Also
    --------
    torch.utils.data.Dataset

    Examples
    --------
    >>> import numpy as np
    >>> data = np.ones((10, 1, 3, 3))
    >>> labels = np.ones((10,))
    >>> class ICDataset:
    ...     def __init__(self, data, labels):
    ...         self.data = data
    ...         self.labels = labels
    ...
    ...     def __getitem__(self, idx):
    ...         return self.data[idx], self.labels[idx]

    >>> ds = ICDataset(data, labels)

    >>> result = read_dataset(ds)
    >>> len(result)  # images and labels
    2
    >>> np.asarray(result[0]).shape  # images
    (10, 1, 3, 3)
    >>> np.asarray(result[1]).shape  # labels
    (10,)
    """

    ddict: dict[int, list[Any]] = defaultdict(list[Any])

    for data in dataset:
        for i, d in enumerate(data if isinstance(data, tuple) else (data,)):
            ddict[i].append(d)

    return list(ddict.values())


# Reduce overhead cost by not tracking tensor gradients
@torch.no_grad
def batch_voc(
    dataset: VOCDetection, model: nn.Module, batch_size: int = 64, flatten_labels: bool = False
) -> tuple[torch.Tensor, list[str] | list[list[str]]]:
    """
    Iterates through the dataset to generate model embeddings and store labels

    Note
    ----
    Due to a bug with the VOCDetection dataset and DataLoaders,
    the batching is done manually
    """

    model.eval()
    embeddings = []
    images, labels = [], []

    dataloader = DataLoader(dataset)

    for i, (image, targets) in tqdm(enumerate(dataloader), desc="Batching VOC", mininterval=1):
        # Aggregate images -> [image]
        images.append(image[0])
        # Aggregate all objects in an image
        objects: list[dict[str, list[str]]] = targets["annotation"]["object"]

        # Extract only the label from each object
        lbls = [obj["name"][0] for obj in objects]

        # Creates either a 1-D or 2-D array of the labels
        labels.extend(lbls) if flatten_labels else labels.append(lbls)

        if (i + 1) % batch_size == 0:
            outputs = model(torch.stack(images))
            embeddings.append(outputs)
            images = []

    # Add last batch even if not full batch size
    embeddings.append(model(torch.stack(images)))

    return torch.vstack(embeddings).cpu(), labels

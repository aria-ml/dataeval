from __future__ import annotations

from collections import defaultdict
from typing import Any

from torch.utils.data import Dataset


def read_dataset(dataset: Dataset) -> list[list[Any]]:
    """
    Extract information from a dataset at each index into individual lists of each information position

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

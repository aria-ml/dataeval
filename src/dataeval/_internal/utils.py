from __future__ import annotations

from collections import defaultdict
from typing import Any

from torch.utils.data import Dataset


def read_dataset(dataset: Dataset) -> list[list[Any]]:
    """
    Extract information from a dataset at each index into a individual lists of each information position

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

    >>> data = np.ones((10, 3, 3))
    >>> labels = np.ones((10,))
    >>> class ICDataset:
    ...     def __init__(self, data, labels):
    ...         self.data = data
    ...         self.labels = labels

    ...     def __getitem__(self, idx):
    ...         return self.data[idx], self.labels[idx]

    >>> ds = ICDataset(data, labels)

    >>> result = read_dataset(ds)
    >>> assert len(result) == 2
    True
    >>> assert result[0].shape == (10, 3, 3)  # 10 3x3 images
    True
    >>> assert result[1].shape == (10,)  # 10 labels
    True
    """

    ddict: dict[int, list] = defaultdict(list)

    for data in dataset:
        # Convert to tuple if single return (e.g. images only)
        if not isinstance(data, tuple):
            data = (data,)

        for i, d in enumerate(data):
            ddict[i].append(d)

    return list(ddict.values())

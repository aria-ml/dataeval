from typing import List, Tuple

from torch.utils.data import Dataset

from dataeval._internal.interop import ArrayLike


def _validate_getitem(dataset: Dataset, min_length: int):
    """
    Checks if return value is type tuple and has at least min_length ArrayLike values to parse

    Parameters
    ----------
    dataset : Dataset
        Collection of data
    min_length : int
        Minimum number of entries within data that are ArrayLike
    """

    if min_length < 1:
        raise ValueError(f"Minimum tuple length cannot be less than 1, got {min_length}")

    data = next(iter(dataset))
    if not isinstance(data, tuple):
        raise TypeError(f"Expected return type of tuple, got {type(data)}")

    if len(data) < min_length:
        raise ValueError(f"Expected length of {min_length} or more in tuple, got {len(data)}")

    for i in range(min_length):
        if not isinstance(data[i], ArrayLike):
            raise TypeError(f"Expected ArrayLike in return position {i}, got {type(data[i])}")


def read_dataset(dataset: Dataset) -> Tuple[List[ArrayLike], List[ArrayLike]]:
    """
    Extract images and labels from a dataset

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset that returns (images, labels).

    Returns
    -------
    Tuple[List[ArrayLike], [List[ArrayLike]]]
        Separate images and label lists in native backend

    Note
    ----
    This function ignores additional return values in __getitem__ such as metadata.

    This function returns a list in case images have inhomogeneous shapes.

    """
    images = []
    labels = []

    _validate_getitem(dataset=dataset, min_length=2)  # Images, labels

    # Only want first two values, with the assumption (image, label, ...)
    for data in dataset:
        images.append(data[0])
        labels.append(data[1])

    return images, labels

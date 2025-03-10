from __future__ import annotations

__all__ = []

from typing import Generic, Iterator, Sequence, overload

from dataeval.typing import TArray
from dataeval.utils.data.datasets._types import (
    ImageClassificationDataset,
    ObjectDetectionDataset,
    TDatasetMetadata,
)


class Images(Generic[TArray]):
    """
    Collection of image data from a dataset.

    Images are accessed by index or slice and are only loaded on-demand.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access images from.
    """

    def __init__(
        self,
        dataset: ImageClassificationDataset[TArray, TDatasetMetadata]
        | ObjectDetectionDataset[TArray, TDatasetMetadata],
    ) -> None:
        self._dataset = dataset

    def to_list(self) -> Sequence[TArray]:
        """
        Converts entire dataset to a sequence of images.

        Warning
        -------
        Will load the entire dataset and return the images as a
        single sequence of images in memory.

        Returns
        -------
        list[TArray]
        """
        return self[:]

    @overload
    def __getitem__(self, key: slice) -> Sequence[TArray]: ...

    @overload
    def __getitem__(self, key: int) -> TArray: ...

    def __getitem__(self, key: int | slice) -> Sequence[TArray] | TArray:
        if isinstance(key, slice):
            indices = list(range(len(self._dataset))[key])
            return [self._dataset[i][0] for i in indices]
        elif isinstance(key, int):
            return self._dataset[key][0]
        raise TypeError("Invalid argument type.")

    def __iter__(self) -> Iterator[TArray]:
        for i in range(len(self._dataset)):
            yield self._dataset[i][0]

    def __len__(self) -> int:
        return len(self._dataset)

from __future__ import annotations

__all__ = []

from typing import Any, Generic, Iterator, Sequence, TypeVar, overload

from dataeval.typing import Dataset

T = TypeVar("T")


class Images(Generic[T]):
    """
    Collection of image data from a dataset.

    Images are accessed by index or slice and are only loaded on-demand.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access images from.
    """

    def __init__(self, dataset: Dataset[tuple[T, Any, Any]]) -> None:
        self._dataset = dataset

    def to_list(self) -> Sequence[T]:
        """
        Converts entire dataset to a sequence of images.

        Warning
        -------
        Will load the entire dataset and return the images as a
        single sequence of images in memory.

        Returns
        -------
        list[T]
        """
        return self[:]

    @overload
    def __getitem__(self, key: slice | list[int]) -> Sequence[T]: ...

    @overload
    def __getitem__(self, key: int) -> T: ...

    def __getitem__(self, key: int | slice | list[int]) -> Sequence[T] | T:
        if isinstance(key, list):
            return [self._dataset[i][0] for i in key]
        if isinstance(key, slice):
            indices = list(range(len(self._dataset))[key])
            return [self._dataset[i][0] for i in indices]
        elif isinstance(key, int):
            return self._dataset[key][0]
        raise TypeError("Invalid argument type.")

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self._dataset)):
            yield self._dataset[i][0]

    def __len__(self) -> int:
        return len(self._dataset)

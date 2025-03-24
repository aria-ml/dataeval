from __future__ import annotations

__all__ = []

from typing import Any, Generic, Iterator, Sequence, TypeVar, cast, overload

from dataeval.typing import Dataset

T = TypeVar("T")


class Images(Generic[T]):
    """
    Collection of image data from a dataset.

    Images are accessed by index or slice and are only loaded on-demand.

    Parameters
    ----------
    dataset : Dataset[tuple[T, ...]] or Dataset[T]
        Dataset to access images from.
    """

    def __init__(self, dataset: Dataset[tuple[T, Any, Any] | T]) -> None:
        self._is_tuple_datum = isinstance(dataset[0], tuple)
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
    def __getitem__(self, key: int, /) -> T: ...
    @overload
    def __getitem__(self, key: slice, /) -> Sequence[T]: ...

    def __getitem__(self, key: int | slice, /) -> Sequence[T] | T:
        if self._is_tuple_datum:
            dataset = cast(Dataset[tuple[T, Any, Any]], self._dataset)
            if isinstance(key, slice):
                return [dataset[k][0] for k in range(len(self._dataset))[key]]
            elif isinstance(key, int):
                return dataset[key][0]
        else:
            dataset = cast(Dataset[T], self._dataset)
            if isinstance(key, slice):
                return [dataset[k] for k in range(len(self._dataset))[key]]
            elif isinstance(key, int):
                return dataset[key]
        raise TypeError(f"Key must be integers or slices, not {type(key)}")

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self._dataset)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._dataset)

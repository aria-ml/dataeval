from __future__ import annotations

__all__ = []

from typing import TYPE_CHECKING, Any, Generic, Iterator, Sequence, TypeVar, cast, overload

import numpy as np

from dataeval.typing import Array, ArrayLike, Dataset
from dataeval.utils._array import as_numpy, channels_first_to_last

if TYPE_CHECKING:
    from matplotlib.figure import Figure

T = TypeVar("T", Array, ArrayLike)


class Images(Generic[T]):
    """
    Collection of image data from a dataset.

    Images are accessed by index or slice and are only loaded on-demand.

    Parameters
    ----------
    dataset : Dataset[tuple[T, ...]] or Dataset[T]
        Dataset to access images from.
    """

    def __init__(
        self,
        dataset: Dataset[tuple[T, Any, Any] | T],
    ) -> None:
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

    def plot(
        self,
        indices: Sequence[int],
        images_per_row: int = 3,
        figsize: tuple[int, int] = (10, 10),
    ) -> Figure:
        import matplotlib.pyplot as plt

        num_images = len(indices)
        num_rows = (num_images + images_per_row - 1) // images_per_row
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=figsize)
        for i, ax in enumerate(np.asarray(axes).flatten()):
            image = channels_first_to_last(as_numpy(self[i]))
            ax.imshow(image)
            ax.axis("off")
        plt.tight_layout()
        return fig

    @overload
    def __getitem__(self, key: int, /) -> T: ...
    @overload
    def __getitem__(self, key: slice, /) -> Sequence[T]: ...

    def __getitem__(self, key: int | slice, /) -> Sequence[T] | T:
        if isinstance(key, slice):
            return [self._get_image(k) for k in range(len(self._dataset))[key]]
        if hasattr(key, "__int__"):
            return self._get_image(int(key))
        raise TypeError(f"Key must be integers or slices, not {type(key)}")

    def _get_image(self, index: int) -> T:
        if self._is_tuple_datum:
            return cast(Dataset[tuple[T, Any, Any]], self._dataset)[index][0]
        return cast(Dataset[T], self._dataset)[index]

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self._dataset)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._dataset)

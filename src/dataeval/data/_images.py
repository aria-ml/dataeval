from __future__ import annotations

__all__ = []

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

import numpy as np

from dataeval.typing import Array, ArrayLike, Dataset
from dataeval.utils._array import as_numpy, channels_first_to_last

if TYPE_CHECKING:
    from matplotlib.figure import Figure

T = TypeVar("T", Array, ArrayLike)


class Images(Generic[T]):
    """
    Collection of image data from a dataset.

    Images are accessed by index or slice and are loaded on-demand for
    memory-efficient processing of large datasets.

    Parameters
    ----------
    dataset : Dataset[tuple[T, ...]] or Dataset[T]
        Dataset that provides image data for access and visualization.

    Attributes
    ----------
    None
        All dataset access is handled through indexing operations.
    """

    def __init__(
        self,
        dataset: Dataset[tuple[T, Any, Any] | T],
    ) -> None:
        self._is_tuple_datum = isinstance(dataset[0], tuple)
        self._dataset = dataset

    def to_list(self) -> Sequence[T]:
        """
        Convert entire dataset to a sequence of images.

        Load all images from the dataset and return a single sequence
        in memory for batch processing or analysis.

        Returns
        -------
        list[T]
            Complete sequence of all images in the dataset

        Warnings
        --------
        Loading entire dataset into memory can consume significant resources
        for large image collections.
        """
        return self[:]

    def plot(
        self,
        indices: Sequence[int],
        images_per_row: int = 3,
        figsize: tuple[int, int] = (10, 10),
    ) -> Figure:
        """
        Display images in a grid layout.

        Create matplotlib figure showing specified images arranged in a
        grid format for visual inspection and comparison.

        Parameters
        ----------
        indices : Sequence[int]
            Dataset indices of images to display in the plot.
        images_per_row : int, default 3
            Number of images displayed per row in the grid. Default 3 provides a balanced layout
            for most screen sizes.
        figsize : tuple[int, int], default (10, 10)
            Figure dimensions as (width, height) in inches. Default (10, 10)
            accommodates typical grid layouts with readable detail.

        Returns
        -------
        Figure
            Matplotlib figure object containing the image grid display.
        """
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

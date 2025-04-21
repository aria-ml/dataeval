from __future__ import annotations

__all__ = []

import math
from typing import Any, Iterator, Sequence, cast

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataeval.config import DeviceLike, get_device
from dataeval.typing import Array, ArrayLike, Dataset, Transform
from dataeval.utils._array import as_numpy
from dataeval.utils.torch.models import SupportsEncode


class Embeddings:
    """
    Collection of image embeddings from a dataset.

    Embeddings are accessed by index or slice and are only loaded on-demand.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access original images from.
    batch_size : int
        Batch size to use when encoding images.
    transforms : Transform or Sequence[Transform] or None, default None
        Transforms to apply to images before encoding.
    model : torch.nn.Module or None, default None
        Model to use for encoding images.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    cache : bool, default False
        Whether to cache the embeddings in memory.
    verbose : bool, default False
        Whether to print progress bar when encoding images.
    """

    device: torch.device
    batch_size: int
    verbose: bool

    def __init__(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        batch_size: int,
        transforms: Transform[torch.Tensor] | Sequence[Transform[torch.Tensor]] | None = None,
        model: torch.nn.Module | None = None,
        device: DeviceLike | None = None,
        cache: bool = False,
        verbose: bool = False,
    ) -> None:
        self.device = get_device(device)
        self.cache = cache
        self.batch_size = batch_size if batch_size > 0 else 1
        self.verbose = verbose

        self._dataset = dataset
        self._length = len(dataset)
        model = torch.nn.Flatten() if model is None else model
        self._transforms = [transforms] if isinstance(transforms, Transform) else transforms
        self._model = model.to(self.device).eval() if isinstance(model, torch.nn.Module) else model
        self._encoder = model.encode if isinstance(model, SupportsEncode) else model
        self._collate_fn = lambda datum: [torch.as_tensor(d[0] if isinstance(d, tuple) else d) for d in datum]
        self._cached_idx = set()
        self._embeddings: torch.Tensor = torch.empty(())
        self._shallow: bool = False

    def to_tensor(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """
        Converts dataset to embeddings.

        Parameters
        ----------
        indices : Sequence[int] or None, default None
            The indices to convert to embeddings

        Returns
        -------
        torch.Tensor

        Warning
        -------
        Processing large quantities of data can be resource intensive.
        """
        if indices is not None:
            return torch.vstack(list(self._batch(indices))).to(self.device)
        else:
            return self[:]

    def to_numpy(self, indices: Sequence[int] | None = None) -> NDArray[Any]:
        """
        Converts dataset to embeddings as numpy array.

        Parameters
        ----------
        indices : Sequence[int] or None, default None
            The indices to convert to embeddings

        Returns
        -------
        NDArray[Any]

        Warning
        -------
        Processing large quantities of data can be resource intensive.
        """
        return self.to_tensor(indices).cpu().numpy()

    def new(self, dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike]) -> Embeddings:
        """
        Creates a new Embeddings object with the same parameters but a different dataset.

        Parameters
        ----------
        dataset : ImageClassificationDataset or ObjectDetectionDataset
            Dataset to access original images from.

        Returns
        -------
        Embeddings
        """
        return Embeddings(
            dataset, self.batch_size, self._transforms, self._model, self.device, self.cache, self.verbose
        )

    @classmethod
    def from_array(cls, array: ArrayLike, device: DeviceLike | None = None) -> Embeddings:
        """
        Instantiates a shallow Embeddings object using an array.

        Parameters
        ----------
        array : ArrayLike
            The array to convert to embeddings.
        device : DeviceLike or None, default None
            The hardware device to use if specified, otherwise uses the DataEval
            default or torch default.

        Returns
        -------
        Embeddings

        Example
        -------
        >>> import numpy as np
        >>> from dataeval.utils.data._embeddings import Embeddings
        >>> array = np.random.randn(100, 3, 224, 224)
        >>> embeddings = Embeddings.from_array(array)
        >>> print(embeddings.to_tensor().shape)
        torch.Size([100, 3, 224, 224])
        """
        embeddings = Embeddings([], 0, None, None, device, True, False)
        array = array if isinstance(array, Array) else as_numpy(array)
        embeddings._length = len(array)
        embeddings._cached_idx = set(range(len(array)))
        embeddings._embeddings = torch.as_tensor(array).to(get_device(device))
        embeddings._shallow = True
        return embeddings

    def _encode(self, images: list[torch.Tensor]) -> torch.Tensor:
        if self._transforms:
            images = [transform(image) for transform in self._transforms for image in images]
        return self._encoder(torch.stack(images).to(self.device))

    @torch.no_grad()  # Reduce overhead cost by not tracking tensor gradients
    def _batch(self, indices: Sequence[int]) -> Iterator[torch.Tensor]:
        dataset = cast(torch.utils.data.Dataset, self._dataset)
        total_batches = math.ceil(len(indices) / self.batch_size)

        # If not caching, process all indices normally
        if not self.cache:
            for images in tqdm(
                DataLoader(Subset(dataset, indices), self.batch_size, collate_fn=self._collate_fn),
                total=total_batches,
                desc="Batch embedding",
                disable=not self.verbose,
            ):
                yield self._encode(images)
            return

        # If caching, process each batch of indices at a time, preserving original order
        for i in tqdm(range(0, len(indices), self.batch_size), desc="Batch embedding", disable=not self.verbose):
            batch = indices[i : i + self.batch_size]
            uncached = [idx for idx in batch if idx not in self._cached_idx]

            if uncached:
                # Process uncached indices as as single batch
                for images in DataLoader(Subset(dataset, uncached), len(uncached), collate_fn=self._collate_fn):
                    embeddings = self._encode(images)

                    if not self._embeddings.shape:
                        full_shape = (len(self._dataset), *embeddings.shape[1:])
                        self._embeddings = torch.empty(full_shape, dtype=embeddings.dtype, device=self.device)

                    self._embeddings[uncached] = embeddings
                    self._cached_idx.update(uncached)

            yield self._embeddings[batch]

    def __getitem__(self, key: int | slice, /) -> torch.Tensor:
        if not isinstance(key, slice) and not hasattr(key, "__int__"):
            raise TypeError("Invalid argument type.")

        if self._shallow:
            if not self._embeddings.shape:
                raise ValueError("Embeddings not initialized.")
            return self._embeddings[key]

        indices = list(range(len(self._dataset))[key]) if isinstance(key, slice) else [int(key)]
        result = torch.vstack(list(self._batch(indices))).to(self.device)
        return result.squeeze(0) if len(indices) == 1 else result

    def __iter__(self) -> Iterator[torch.Tensor]:
        # process in batches while yielding individual embeddings
        for batch in self._batch(range(self._length)):
            yield from batch

    def __len__(self) -> int:
        return self._length

"""
NumPy-based embedding encoder using simple flattening.
"""

__all__ = []

from collections.abc import Iterator, Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ArrayLike, Dataset
from dataeval.utils.arrays import as_numpy, flatten_samples


class NumpyFlattenEncoder:
    """
    Simple NumPy-based encoder that flattens images.

    No deep learning framework required. Simply flattens each image
    to a 1D vector using the flatten_samples utility. This is useful
    as a baseline or when no model-based feature extraction is needed.

    Parameters
    ----------
    batch_size : int, default 32
        Number of samples to process per batch.

    Example
    -------
    >>> from dataeval.encoders import NumpyFlattenEncoder
    >>> from dataeval import Embeddings
    >>>
    >>> encoder = NumpyFlattenEncoder(batch_size=64)
    >>> embeddings = Embeddings(train_dataset, encoder=encoder)
    >>> result = np.asarray(embeddings)
    """

    def __init__(self, batch_size: int = 32) -> None:
        self._batch_size = max(1, batch_size)

    @property
    def batch_size(self) -> int:
        """Return the batch size used for encoding."""
        return self._batch_size

    def _encode_batch(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        batch_indices: Sequence[int],
    ) -> NDArray[Any]:
        """Encode a single batch of images."""
        batch_images: list[NDArray[Any]] = []
        for idx in batch_indices:
            item = dataset[idx]
            image = item[0] if isinstance(item, tuple) else item
            batch_images.append(as_numpy(image))
        batch_array = np.stack(batch_images)
        return flatten_samples(batch_array)

    @overload
    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: Literal[True],
    ) -> Iterator[tuple[Sequence[int], NDArray[Any]]]: ...

    @overload
    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: Literal[False] = ...,
    ) -> NDArray[Any]: ...

    def encode(
        self,
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        indices: Sequence[int],
        stream: bool = False,
    ) -> Iterator[tuple[Sequence[int], NDArray[Any]]] | NDArray[Any]:
        """
        Flatten images at specified indices to embeddings.

        Parameters
        ----------
        dataset : Dataset
            Dataset providing images to encode.
        indices : Sequence[int]
            Indices of images to encode from the dataset.
        stream : bool, default False
            If True, yields (batch_indices, batch_embeddings) tuples.
            If False, returns all embeddings as a single array.

        Returns
        -------
        NDArray[Any] or Iterator[tuple[Sequence[int], NDArray[Any]]]
            Flattened embeddings array or iterator of batches.
        """

        def _generate() -> Iterator[tuple[Sequence[int], NDArray[Any]]]:
            for batch_start in range(0, len(indices), self._batch_size):
                batch_idx = list(indices[batch_start : batch_start + self._batch_size])
                yield batch_idx, self._encode_batch(dataset, batch_idx)

        if not indices:
            if stream:
                return iter([])
            return np.empty((0,), dtype=np.float32)

        if stream:
            return _generate()

        return np.vstack([emb for _, emb in _generate()])

    def __repr__(self) -> str:
        return f"NumpyFlattenEncoder(batch_size={self._batch_size})"

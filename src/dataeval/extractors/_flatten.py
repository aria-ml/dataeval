"""NumPy-based feature extractor using simple flattening."""

__all__ = []

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Array
from dataeval.types import ReprMixin
from dataeval.utils._internal import as_numpy, flatten_samples, iter_images


class FlattenExtractor(ReprMixin):
    """
    Simple NumPy-based feature extractor that flattens images to 1D vectors.

    No deep learning framework required. Simply flattens each image
    to a 1D vector using the flatten_samples utility. This is useful
    as a baseline or when no model-based feature extraction is needed.

    Implements the :class:`~dataeval.protocols.FeatureExtractor` protocol.

    Example
    -------
    >>> from dataeval.extractors import FlattenExtractor
    >>> from dataeval import Embeddings
    >>>
    >>> extractor = FlattenExtractor()
    >>> embeddings = Embeddings(train_dataset, extractor=extractor)
    >>> result = np.asarray(embeddings)
    """

    def __call__(self, data: Any) -> Array:
        """
        Flatten a batch of images to 1D vectors.

        Parameters
        ----------
        data : Any
            A batch of images to flatten. Accepts an iterable (or sized,
            indexable source) of images where each image is any array-like, or a
            full MAITE dataset whose items are ``(image, target, metadata)``
            tuples -- the image is auto-unwrapped from element 0 of each tuple.
            The input must be a batch/iterable of images: a single un-batched
            ``(C, H, W)`` array is treated as an iterable over its channel axis,
            not as one image.

        Returns
        -------
        Array
            Flattened embeddings array of shape (n_images, flattened_dim).
        """
        batch_images: list[NDArray[Any]] = [as_numpy(img) for img in iter_images(data)]
        if not batch_images:
            return np.empty((0,), dtype=np.float32)
        batch_array = np.stack(batch_images)
        return flatten_samples(batch_array)

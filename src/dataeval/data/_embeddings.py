from __future__ import annotations

__all__ = []

import logging
import math
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import xxhash as xxh
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataeval.config import DeviceLike, get_device
from dataeval.typing import (
    AnnotatedDataset,
    AnnotatedModel,
    Array,
    ArrayLike,
    Dataset,
    Transform,
)
from dataeval.utils._array import as_numpy
from dataeval.utils.torch.models import SupportsEncode

_logger = logging.getLogger(__name__)


class Embeddings:
    """
    Collection of image embeddings from a dataset.

    Embeddings are accessed by index or slice and are loaded on-demand.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access original images from.
    batch_size : int
        Batch size to use when encoding images. When less than 1, automatically sets to 1 for safe processing.
    transforms : Transform or Sequence[Transform] or None, default None
        Image transformationss to apply before encoding. When None, uses raw images without
        preprocessing.
    model : torch.nn.Module or None, default None
        Neural network model that generates embeddings from images. When None, uses Flatten layer for simple
        baseline compatibility with all DataEval tools without requiring pre-trained weights or GPU resources.
    device : DeviceLike or None, default None
        Hardware device for computation. When None, automatically selects DataEval's configured device, falling
        back to PyTorch's default.
    cache : Path, str, or bool, default False
        When True, caches embeddings in memory for faster repeated access.
        When Path or string is provided, persists embeddings to disk for reuse across sessions.
        Default False minimizes memory usage.
    verbose : bool, default False
        When True, displays a progress bar when encoding images. Default False reduces console output
        for cleaner automated workflows.

    Attributes
    ----------
    batch_size : int
        Number of images processed per batch during encoding. Minimum value of 1.
    cache : Path or bool
        Disk path where embeddings are stored, or True when cached in memory.
    device : torch.device
        Hardware device used for tensor computations.
    verbose : bool
        Whether progress information is displayed during operations.
    """

    device: torch.device
    batch_size: int
    verbose: bool

    def __init__(
        self,
        # Technically more permissive than ImageClassificationDataset or ObjectDetectionDataset
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        batch_size: int,
        transforms: Transform[torch.Tensor] | Sequence[Transform[torch.Tensor]] | None = None,
        model: torch.nn.Module | None = None,
        device: DeviceLike | None = None,
        cache: Path | str | bool = False,
        verbose: bool = False,
    ) -> None:
        self.device = get_device(device)
        self.batch_size = batch_size if batch_size > 0 else 1
        self.verbose = verbose

        self._embeddings_only: bool = False
        self._dataset = dataset
        self._transforms = [transforms] if isinstance(transforms, Transform) else transforms
        model = torch.nn.Flatten() if model is None else model
        self._model = model.to(self.device).eval() if isinstance(model, torch.nn.Module) else model
        self._encoder = model.encode if isinstance(model, SupportsEncode) else model
        self._collate_fn = lambda datum: [torch.as_tensor(d[0] if isinstance(d, tuple) else d) for d in datum]
        self._cached_idx: set[int] = set()
        self._embeddings: torch.Tensor = torch.empty(())

        self._cache = cache if isinstance(cache, bool) else self._resolve_path(cache)

    def __hash__(self) -> int:
        if self._embeddings_only:
            bid = as_numpy(self._embeddings).ravel().tobytes()
        else:
            did = self._dataset.metadata["id"] if isinstance(self._dataset, AnnotatedDataset) else str(self._dataset)
            mid = self._model.metadata["id"] if isinstance(self._model, AnnotatedModel) else str(self._model)
            tid = str.join("|", [str(t) for t in self._transforms or []])
            bid = f"{did}{mid}{tid}".encode()

        return int(xxh.xxh3_64_hexdigest(bid), 16)

    @property
    def cache(self) -> Path | bool:
        return self._cache

    @cache.setter
    def cache(self, value: Path | str | bool) -> None:
        if isinstance(value, bool) and not value:
            self._cached_idx = set()
            self._embeddings = torch.empty(())
        elif isinstance(value, Path | str):
            value = self._resolve_path(value)

        if isinstance(value, Path) and value != getattr(self, "_cache", None):
            self._save(value)

        self._cache = value

    def _resolve_path(self, path: Path | str) -> Path:
        if isinstance(path, str):
            path = Path(os.path.abspath(path))
        if isinstance(path, Path) and (path.is_dir() or not path.suffix):
            path = path / f"emb-{hash(self)}.pt"
        return path

    def to_tensor(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """
        Convert dataset items to embedding tensor.

        Process specified dataset indices through the model in batches and
        return concatenated embeddings as a single tensor.

        Parameters
        ----------
        indices : Sequence[int] or None, default None
            Dataset indices to convert to embeddings. When None, processes entire dataset.

        Returns
        -------
        torch.Tensor
            Concatenated embeddings with shape (n_samples, embedding_dim).

        Warnings
        --------
        Processing large datasets can be memory and compute intensive.
        """
        if indices is not None:
            return torch.vstack(list(self._batch(indices))).to(self.device)
        return self[:]

    def to_numpy(self, indices: Sequence[int] | None = None) -> NDArray[Any]:
        """
        Convert dataset items to embedding array.

        Parameters
        ----------
        indices : Sequence[int] or None, default None
            Dataset indices to convert to embeddings. When None, processes entire dataset.

        Returns
        -------
        NDArray[Any]
            Embedding array with shape (n_samples, embedding_dim)

        Warning
        -------
        Processing large datasets can be memory and compute intensive.
        """
        return self.to_tensor(indices).cpu().numpy()

    def new(self, dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike]) -> Embeddings:
        """
        Create new Embeddings instance with a different dataset.

        Generate a new Embeddings object using the same model, transforms,
        and configuration but with a different dataset.

        Parameters
        ----------
        dataset : ImageClassificationDataset or ObjectDetectionDataset
            Dataset that provides images for the new Embeddings instance.

        Returns
        -------
        Embeddings
            New Embeddings object configured identically to the current instance.

        Raises
        ------
        ValueError
            When called on embeddings-only instance that lacks a model.
        """
        if self._embeddings_only:
            raise ValueError("Embeddings object does not have a model.")
        return Embeddings(
            dataset, self.batch_size, self._transforms, self._model, self.device, bool(self.cache), self.verbose
        )

    @classmethod
    def from_array(cls, array: ArrayLike, device: DeviceLike | None = None) -> Embeddings:
        """
        Create Embeddings instance from an existing image array.

        Parameters
        ----------
        array : ArrayLike
            In-memory image data to wrap in an Embeddings object.
        device : DeviceLike or None, default None
            Hardware device for computation. When None, automatically selects DataEval's configured device, falling
            back to PyTorch's default.

        Returns
        -------
        Embeddings

        Example
        -------
        >>> import numpy as np
        >>> from dataeval.data import Embeddings
        >>> array = np.random.randn(100, 3, 224, 224)
        >>> embeddings = Embeddings.from_array(array)
        >>> print(embeddings.to_tensor().shape)
        torch.Size([100, 3, 224, 224])
        """
        embeddings = Embeddings([], 0, None, None, device, True, False)
        array = array if isinstance(array, Array) else as_numpy(array)
        embeddings._cached_idx = set(range(len(array)))
        embeddings._embeddings = torch.as_tensor(array).to(get_device(device))
        embeddings._embeddings_only = True
        return embeddings

    def save(self, path: Path | str) -> None:
        """
        Save embeddings to disk.

        Persist current embeddings to the specified file path for later
        loading and reuse.

        Parameters
        ----------
        path : Path or str
            File path where embeddings will be saved.
        """
        self._save(self._resolve_path(path), True)

    def _save(self, path: Path, force: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._embeddings_only or self.cache and not force:
            embeddings = self._embeddings
            cached_idx = self._cached_idx
        else:
            embeddings = self.to_tensor()
            cached_idx = list(range(len(self)))
        try:
            cache_data = {
                "embeddings": embeddings,
                "cached_indices": cached_idx,
                "device": self.device,
            }
            torch.save(cache_data, path)
            _logger.log(logging.DEBUG, f"Saved embeddings cache from {path}")
        except Exception as e:
            _logger.log(logging.ERROR, f"Failed to save embeddings cache: {e}")
            raise e

    @classmethod
    def load(cls, path: Path | str) -> Embeddings:
        """
        Loads the embeddings from disk.

        Create an Embeddings instance from previously saved embedding data.

        Parameters
        ----------
        path : Path or str
            File path to load embeddings from.

        Returns
        -------
        Embeddings
            Embeddings-only instance containing the loaded data.

        Raises
        ------
        FileNotFoundError
            When the specified file path does not exist.
        Exception
            When file loading or parsing fails.
        """
        emb = Embeddings([], 0)
        path = Path(os.path.abspath(path)) if isinstance(path, str) else path
        if path.exists() and path.is_file():
            try:
                cache_data = torch.load(path, weights_only=False)
                emb._embeddings_only = True
                emb._embeddings = cache_data["embeddings"]
                emb._cached_idx = cache_data["cached_indices"]
                emb.device = cache_data["device"]
                _logger.log(logging.DEBUG, f"Loaded embeddings cache from {path}")
            except Exception as e:
                _logger.log(logging.ERROR, f"Failed to load embeddings cache: {e}")
                raise e
        else:
            raise FileNotFoundError(f"Specified cache file {path} was not found.")

        return emb

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
                        full_shape = (len(self), *embeddings.shape[1:])
                        self._embeddings = torch.empty(full_shape, dtype=embeddings.dtype, device=self.device)

                    self._embeddings[uncached] = embeddings
                    self._cached_idx.update(uncached)

                if isinstance(self.cache, Path):
                    self._save(self.cache)

            yield self._embeddings[batch]

    def __getitem__(self, key: int | slice, /) -> torch.Tensor:
        if not isinstance(key, slice) and not hasattr(key, "__int__"):
            raise TypeError("Invalid argument type.")

        indices = list(range(len(self))[key]) if isinstance(key, slice) else [int(key)]

        if self._embeddings_only:
            if not self._embeddings.shape:
                raise ValueError("Embeddings not initialized.")
            if not set(indices).issubset(self._cached_idx):
                raise ValueError("Unable to generate new embeddings from a shallow instance.")
            return self._embeddings[key]

        result = torch.vstack(list(self._batch(indices))).to(self.device)
        return result.squeeze(0) if len(indices) == 1 else result

    def __iter__(self) -> Iterator[torch.Tensor]:
        # process in batches while yielding individual embeddings
        for batch in self._batch(range(len(self))):
            yield from batch

    def __len__(self) -> int:
        return len(self._embeddings) if self._embeddings_only else len(self._dataset)

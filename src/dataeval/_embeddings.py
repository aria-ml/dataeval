"""
Embeddings class for extracting and managing image embeddings.
"""

__all__ = []

import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import xxhash as xxh
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.encoders import NumpyFlattenEncoder
from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    ArrayLike,
    Dataset,
    EmbeddingEncoder,
    FeatureExtractor,
    ProgressCallback,
)

_logger = logging.getLogger(__name__)


class Embeddings(Array, FeatureExtractor):
    """
    Collection of image embeddings from a dataset.

    Embeddings are accessed by index or slice and are loaded on-demand. For large
    datasets, embeddings are automatically memory-mapped to disk to avoid exceeding
    available memory.

    This class also implements the :class:`~dataeval.protocols.FeatureExtractor` protocol,
    allowing it to be used directly with drift detectors and quality metrics that accept
    feature extractors.

    Parameters
    ----------
    dataset : ImageClassificationDataset, ObjectDetectionDataset, or None, default None
        Dataset to access original images from. When None, creates an unbound instance
        that can be used as a reusable feature extractor. Use :meth:`bind` to attach
        a dataset later, or pass data directly to :meth:`__call__`.
    encoder : EmbeddingEncoder or None, default None
        Encoder for extracting embeddings from images. Handles model inference,
        device management, transforms, and batching. When None, uses
        :class:`~dataeval.encoders.NumpyFlattenEncoder` for simple baseline
        compatibility with all DataEval tools.
    path : Path, str, or None, default None
        File path for memory-mapped storage. When None, caches embeddings in memory only.
        When Path or string is provided, uses memory-mapped storage for large embeddings
        (automatic based on memory_threshold).
    memory_threshold : float, default 0.8
        Fraction of available memory (0-1) that triggers memory-mapped storage. When estimated
        embedding size exceeds this threshold, uses disk-backed memmap instead of in-memory arrays.
        Only applies when path is provided.
    progress_callback : ProgressCallback or None, default None
        Callback to report progress during embedding computation.


    Attributes
    ----------
    memory_threshold : float
        Fraction of available memory (0-1) that triggers memory-mapped storage.

    Example
    -------
    Using with a PyTorch model:

    >>> from dataeval import Embeddings
    >>> from dataeval.encoders import TorchEmbeddingEncoder
    >>>
    >>> embeddings = Embeddings(train_dataset, encoder=encoder)
    >>> train_emb = embeddings[:]
    >>> train_emb.shape
    (40, 32)

    Using with default flattening (no model):

    >>> # Uses NumpyFlattenEncoder by default
    >>> embeddings = Embeddings(dataset)
    >>> flat_features = np.asarray(embeddings)
    """

    memory_threshold: float

    def __init__(
        self,
        # Technically more permissive than ImageClassificationDataset or ObjectDetectionDataset
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike] | None = None,
        encoder: EmbeddingEncoder | None = None,
        path: Path | str | None = None,
        memory_threshold: float = 0.8,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        # Use NumpyFlattenEncoder as default
        self._encoder = encoder if encoder is not None else NumpyFlattenEncoder()
        self.memory_threshold = max(0.0, min(1.0, memory_threshold))
        self._progress_callback = progress_callback

        self._dataset = dataset
        self._embeddings_only: bool = False

        self._cached_idx: set[int] = set()
        self._embeddings: np.ndarray | np.memmap = np.empty((0,))
        self._use_memmap: bool = False

        self._path = self._resolve_path(path) if path is not None else None
        self._shape: tuple[int, ...] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            if self._dataset is None:
                raise ValueError("Cannot determine shape: no dataset bound. Call bind() first.")
            if len(self._dataset) == 0:
                self._shape = (0,)
            elif self._cached_idx:
                embedding_shape = self[list(self._cached_idx)[0]].shape
                self._shape = tuple([len(self)] + [*embedding_shape])
            else:
                embedding_shape = self[0].shape
                self._shape = tuple([len(self)] + [*embedding_shape])
        return self._shape

    @property
    def is_bound(self) -> bool:
        """Whether this instance is bound to a dataset.

        Returns
        -------
        bool
            True if a dataset is bound, False otherwise.
        """
        return self._dataset is not None

    def bind(self, dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike]) -> Self:
        """Bind this instance to a dataset.

        Attaches a dataset to this Embeddings instance for embedding computation.
        Any previously cached embeddings are cleared.

        Parameters
        ----------
        dataset : ImageClassificationDataset or ObjectDetectionDataset
            Dataset to bind for embedding computation.

        Returns
        -------
        Self
            Returns self for method chaining.

        Raises
        ------
        ValueError
            When called on an embeddings-only instance.

        Example
        -------
        >>> from dataeval import Embeddings
        >>> from dataeval.encoders import TorchEmbeddingEncoder
        >>>
        >>> encoder = TorchEmbeddingEncoder(my_model, batch_size=32)
        >>> extractor = Embeddings(encoder=encoder)
        >>> _ = extractor.bind(train_dataset)
        >>> embeddings = extractor()
        """
        self._dataset = dataset
        # Clear cached state
        self._cached_idx.clear()
        self._embeddings = np.empty((0,))
        self._shape = None
        return self

    def __call__(self, data: Any | None = None) -> Array:
        """Extract embeddings from data.

        Implements the :class:`~dataeval.protocols.FeatureExtractor` protocol,
        allowing this instance to be used directly with drift detectors and
        quality metrics.

        Parameters
        ----------
        data : Any or None, default None
            Dataset to extract embeddings from. If None, uses the bound dataset.

        Returns
        -------
        Array
            Embeddings array of shape (n_samples, embedding_dim).

        Raises
        ------
        ValueError
            If data is None and no dataset is bound.

        Example
        -------
        >>> from dataeval import Embeddings
        >>> from dataeval.encoders import TorchEmbeddingEncoder
        >>>
        >>> encoder = TorchEmbeddingEncoder(my_model, batch_size=32)
        >>> embeddings = Embeddings(train_dataset, encoder=encoder)
        >>>
        >>> # Extract from bound dataset
        >>> train_emb = embeddings()
        >>>
        >>> # Extract from new dataset
        >>> test_emb = embeddings(test_dataset)
        """
        if data is None:
            if self._dataset is None:
                raise ValueError("No dataset bound. Provide data or call bind() first.")
            # Return embeddings for bound dataset
            return self[:]

        # Check if same as bound dataset (by identity)
        if self._dataset is not None and data is self._dataset:
            return self[:]

        # Compute embeddings for new data using this config
        return self.new(data).compute()[:]

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[Any]:
        """
        Implement numpy array protocol while preserving memory-mapped storage.

        This method is called by numpy when converting to array (e.g., np.asarray()).
        For lazy embeddings, this triggers computation of all embeddings.

        Parameters
        ----------
        dtype : data-type or None
            Desired data type. If None or matches existing dtype, preserves memmap.
        copy : bool or None
            - None: No requirement (preserves memmap)
            - False: Must NOT copy (returns view, preserves memmap)
            - True: Must copy (loads memmap into memory)

        Returns
        -------
        NDArray[Any]
            Array view preserving memmap when possible, or in-memory copy when required.

        Raises
        ------
        ValueError
            When copy=False but dtype conversion is requested (unavoidable copy).
        """
        # Trigger computation for lazy embeddings
        # Always compute if not embeddings-only (handles both empty cache and no-cache scenarios)
        arr = self[:]

        # Check if dtype conversion is needed
        needs_conversion = dtype is not None and np.dtype(dtype) != arr.dtype

        if needs_conversion:
            # Dtype conversion always creates a new array (loads memmap into memory)
            if copy is False:
                raise ValueError(f"Cannot avoid copy when converting dtype from {arr.dtype} to {dtype}")
            return arr.astype(dtype)

        # No dtype conversion needed - handle copy parameter
        if copy is True:
            # Explicitly requested copy (loads memmap into memory)
            return np.array(arr, copy=True)

        # copy is False or None - return original (preserves memmap)
        return arr

    def __hash__(self) -> int:
        if self._dataset is None:
            # Unbound instance - hash based on encoder only
            bid = f"unbound:{self._encoder!r}".encode()
        else:
            did = self._dataset.metadata["id"] if isinstance(self._dataset, AnnotatedDataset) else str(self._dataset)
            bid = f"{did}{self._encoder!r}".encode()

        return int(xxh.xxh3_64_hexdigest(bid), 16)

    @property
    def path(self) -> Path | None:
        return self._path

    @path.setter
    def path(self, value: Path | str | None) -> None:
        if value is None:
            # Clear path, but keep in-memory embeddings
            self._path = None
            if isinstance(self._embeddings, np.memmap):
                # Convert memmap to in-memory array
                self._embeddings = np.array(self._embeddings)
                self._use_memmap = False
        else:
            new_path = self._resolve_path(value)
            if new_path != self._path:
                self._path = new_path
                # Save current embeddings to new path
                if self._embeddings.size > 0:
                    self.save(new_path)

    def _resolve_path(self, path: Path | str) -> Path:
        if isinstance(path, str):
            path = Path(os.path.abspath(path))
        if isinstance(path, Path) and (path.is_dir() or not path.suffix):
            path = path / f"emb-{hash(self)}.npy"
        return path

    def _should_use_memmap(self, embedding_shape: tuple[int, ...]) -> bool:
        """Determine if memmap should be used based on estimated size."""
        if self._path is None:
            return False

        if self._dataset is None:
            raise ValueError("No dataset bound. Call bind() first.")

        n_samples = len(self._dataset)
        bytes_per_element = np.dtype(np.float32).itemsize  # Assume float32
        estimated_bytes = n_samples * np.prod(embedding_shape) * bytes_per_element

        available_memory = psutil.virtual_memory().available
        threshold_bytes = available_memory * self.memory_threshold

        use_memmap = bool(estimated_bytes > threshold_bytes)

        if use_memmap:
            _logger.info(
                f"Using memory-mapped storage: estimated size {estimated_bytes / 1e9:.2f}GB "
                f"exceeds {self.memory_threshold * 100:.0f}% of available memory "
                f"({available_memory / 1e9:.2f}GB)"
            )

        return use_memmap

    def _initialize_storage(self, sample_embedding: NDArray[Any]) -> None:
        """Initialize storage backend (in-memory or memmap) based on size."""
        if self._dataset is None:
            raise ValueError("No dataset bound. Call bind() first.")

        n_samples = len(self._dataset)
        embedding_shape = sample_embedding.shape
        full_shape = (n_samples, *embedding_shape)
        dtype = sample_embedding.dtype

        if self._path is not None:
            self._use_memmap = self._should_use_memmap(embedding_shape)

            if self._use_memmap:
                # Create memory-mapped file
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._embeddings = np.memmap(self._path, dtype=dtype, mode="w+", shape=full_shape)
            else:
                # Use in-memory array
                self._embeddings = np.empty(full_shape, dtype=dtype)
        else:
            # In-memory only
            self._embeddings = np.empty(full_shape, dtype=dtype)

    def new(self, dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike]) -> Self:
        """
        Create new Embeddings instance with a different dataset.

        Generate a new Embeddings object using the same encoder and configuration
        but with a different dataset.

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
            When called on embeddings-only instance that lacks an encoder.
        """
        return self.__class__(
            dataset,
            encoder=self._encoder,
            path=self._path,
            memory_threshold=self.memory_threshold,
            progress_callback=self._progress_callback,
        )

    def save(self, path: Path | str | None = None) -> None:
        """
        Compute all embeddings and save to disk.

        Forces computation of all embeddings if not already computed, then
        saves to the specified file path. Progress updates are reported via
        the progress_callback if configured during computation.

        Parameters
        ----------
        path : Path, str, or None, default None
            File path where embeddings will be saved. When None, uses the
            configured path from initialization. Raises ValueError if no
            path is available.

        Raises
        ------
        ValueError
            When no path is specified and instance has no configured path.
        """
        # Determine target path
        if path is not None:
            target_path = self._resolve_path(path)
        elif self._path is not None:
            target_path = self._path
        else:
            raise ValueError("No path specified. Provide a path or initialize Embeddings with a path.")

        # Ensure all embeddings are computed
        self.compute()

        # Save to disk
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self._embeddings, np.memmap):
            # Memmap is already on disk, just flush
            self._embeddings.flush()
            _logger.debug(f"Flushed memmap embeddings to {target_path}")
        else:
            # Save in-memory array to disk
            np.save(target_path, self._embeddings)
            _logger.debug(f"Saved embeddings to {target_path}")

    def compute(self, force: bool = False) -> Self:
        """
        Compute and cache all embeddings.

        Forces evaluation of all lazy embeddings, storing them in memory or
        memmap according to the configured storage strategy. Progress updates
        are reported via the progress_callback if configured.

        Parameters
        ----------
        force : bool, default False
            If True, recomputes all embeddings even if already cached.
            If False, only computes uncached embeddings.

        Returns
        -------
        Embeddings
            Returns self for method chaining.
        """
        if force:
            self._cached_idx.clear()
            self._embeddings = np.empty((0,))

        # Trigger computation of all embeddings via __getitem__
        _ = self[:]

        return self

    def _batch(self, indices: Sequence[int]) -> Iterator[NDArray[Any]]:
        """Process indices in batches using the encoder's streaming interface."""
        if self._dataset is None:
            raise ValueError("No dataset bound. Call bind() first.")

        # Filter to uncached indices
        uncached = [idx for idx in indices if idx not in self._cached_idx]

        if uncached:
            # Validate indices
            out_of_range = set(uncached) - set(range(len(self._dataset)))
            if out_of_range:
                raise IndexError(
                    f"Indices {sorted(out_of_range)} are out of range for dataset of size {len(self._dataset)}"
                )

            # Stream batches from encoder
            processed = 0
            for batch_indices, embeddings in self._encoder.encode(self._dataset, uncached, stream=True):
                # Initialize storage on first batch
                if self._embeddings.size == 0:
                    self._initialize_storage(embeddings[0])

                # Store embeddings
                for i, idx in enumerate(batch_indices):
                    self._embeddings[idx] = embeddings[i]
                self._cached_idx.update(batch_indices)

                # Flush memmap writes (cheap operation)
                if isinstance(self._embeddings, np.memmap):
                    self._embeddings.flush()

                # Report progress
                processed += len(batch_indices)
                if self._progress_callback:
                    self._progress_callback(processed, total=len(uncached))

        # Yield results in batches matching encoder's batch size for consistency
        for batch_start in range(0, len(indices), self._encoder.batch_size):
            batch_indices = list(indices[batch_start : batch_start + self._encoder.batch_size])
            yield self._embeddings[batch_indices]

    def __getitem__(self, key: int | Iterable[int] | slice, /) -> NDArray[Any]:
        """
        Access embeddings by index, indices or slice.

        Returns a view of the memmap when possible (slices/ints),
        and a copy only when necessary (arbitrary list of indices).
        """
        from collections import deque

        # 1. Validation and Index Normalization
        if isinstance(key, (int, np.integer)):
            # Fast path for single integer
            if self._embeddings.size > 0 and int(key) in self._cached_idx:
                return self._embeddings[key]
            indices = [int(key)]

        elif isinstance(key, slice):
            # Resolve slice without expanding to a list (preserves memory)
            # key.indices handles step size and negative indices correctly
            indices = range(*key.indices(len(self)))

        elif isinstance(key, Iterable) and not isinstance(key, str | bytes):
            # Handle arbitrary list of indices
            indices = []
            for k in key:
                if not isinstance(k, int | np.integer):
                    raise TypeError("All indices in the sequence must be integers")
                indices.append(int(k))
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

        # 2. Ensure Cache is Populated
        # If the array is not initialized OR we are missing items, we must compute.
        # We check specific indices only if the global "fully cached" flag is false.
        is_fully_cached = (self._embeddings.size > 0) and (len(self._cached_idx) == len(self))

        if not is_fully_cached:
            # Run _batch purely for side effects (updating self._embeddings).
            # deque(..., maxlen=0) consumes the generator at C-speed without storing results.
            deque(self._batch(indices), maxlen=0)

        # 3. Return Data
        # At this point, self._embeddings is guaranteed to be initialized and populated.

        # Slices return a VIEW of the memmap (Zero-Copy)
        # Advanced indexing (lists) returns a COPY (NumPy limitation)
        key = key if isinstance(key, slice | range | int) else indices
        return self._embeddings[key]

    def __iter__(self) -> Iterator[NDArray[Any]]:
        """Iterate over individual embeddings."""
        for batch in self._batch(range(len(self))):
            yield from batch

    def __len__(self) -> int:
        """Return number of embeddings.

        Raises
        ------
        ValueError
            If no dataset is bound and instance is not embeddings-only.
        """
        if self._dataset is None:
            raise ValueError("Cannot determine length: no dataset bound. Call bind() first.")
        return len(self._dataset)

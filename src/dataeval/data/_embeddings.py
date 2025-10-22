from __future__ import annotations

__all__ = []

import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import psutil
import torch
import xxhash as xxh
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset

from dataeval.config import DeviceLike, get_device
from dataeval.protocols import (
    AnnotatedDataset,
    AnnotatedModel,
    Array,
    ArrayLike,
    Dataset,
    EmbeddingModel,
    Transform,
)
from dataeval.utils._array import as_numpy, flatten
from dataeval.utils._tqdm import tqdm

_logger = logging.getLogger(__name__)


class Embeddings(Array):
    """
    Collection of image embeddings from a dataset.

    Embeddings are accessed by index or slice and are loaded on-demand. For large
    datasets, embeddings are automatically memory-mapped to disk to avoid exceeding
    available memory.

    Parameters
    ----------
    dataset : ImageClassificationDataset or ObjectDetectionDataset
        Dataset to access original images from.
    batch_size : int
        Batch size to use when encoding images. When less than 1, automatically sets to 1 for safe processing.
    transforms : Transform or Sequence[Transform] or None, default None
        Image transformations to apply before encoding. When None, uses raw images without
        preprocessing.
    model : EmbeddingModel or None, default None
        A model such as a PyTorch neural network model that generates embeddings from images. When None, uses
        Flatten layer for simple baseline compatibility with all DataEval tools without requiring pre-trained
        weights or GPU resources.
    layer_name : str or None, default None
        Network layer from which to extract embeddings. When None, uses model output. If specified, extracts
        either the input or output tensors from this layer depending on the value of `use_output`
    use_output : bool, default True
         The relative location to extract intermediate tensors in the model. If true, captures the output
         tensors from `layer_name`. If False, captures the input tensors to `layer_name`. Ignored if `layer_name`
         is None.
    device : DeviceLike or None, default None
        Hardware device for computation. When None, automatically selects DataEval's configured device, falling
        back to PyTorch's default.
    path : Path, str, or None, default None
        File path for memory-mapped storage. When None, caches embeddings in memory only.
        When Path or string is provided, uses memory-mapped storage for large embeddings
        (automatic based on memory_threshold).
    memory_threshold : float, default 0.8
        Fraction of available memory (0-1) that triggers memory-mapped storage. When estimated
        embedding size exceeds this threshold, uses disk-backed memmap instead of in-memory arrays.
        Only applies when path is provided.
    verbose : bool, default False
        When True, displays a progress bar when encoding images. Default False reduces console output
        for cleaner automated workflows.

    Attributes
    ----------
    batch_size : int
        Number of images processed per batch during encoding. Minimum value of 1.
    device : torch.device
        Hardware device used for tensor computations.
    memory_threshold : float
        Fraction of available memory (0-1) that triggers memory-mapped storage.
    verbose : bool
        Whether progress information is displayed during operations.
    """

    device: torch.device
    batch_size: int
    memory_threshold: float
    verbose: bool

    def __init__(
        self,
        # Technically more permissive than ImageClassificationDataset or ObjectDetectionDataset
        dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike],
        batch_size: int,
        transforms: Transform[torch.Tensor] | Iterable[Transform[torch.Tensor]] | None = None,
        model: EmbeddingModel | None = None,
        layer_name: str | None = None,
        use_output: bool = True,
        device: DeviceLike | None = None,
        path: Path | str | None = None,
        memory_threshold: float = 0.8,
        verbose: bool = False,
    ) -> None:
        self.device = get_device(device)
        self.batch_size = batch_size if batch_size > 0 else 1
        self.verbose = verbose
        self.memory_threshold = max(0.0, min(1.0, memory_threshold))

        self._dataset = dataset
        self._transforms = (
            [transforms] if isinstance(transforms, Transform) else [] if transforms is None else list(transforms)
        )
        self._embeddings_only: bool = False

        self.layer_name = layer_name
        self.use_output = use_output
        if isinstance(model, torch.nn.Module) and layer_name is not None:
            self.captured_output: Any = None

            target_layer = self._get_valid_layer_selection(layer_name, model)
            self._use_output = bool(use_output)

            target_layer.register_forward_hook(self._hook_fn)

            if verbose:
                _logger.log(
                    logging.DEBUG, f"Capturing {'output' if use_output else 'input'} data from layer {layer_name}."
                )

        self._model = model.to(self.device).eval() if isinstance(model, torch.nn.Module) else flatten

        self._cached_idx: set[int] = set()
        self._embeddings: np.ndarray | np.memmap = np.empty((0,))
        self._use_memmap: bool = False

        self._path = self._resolve_path(path) if path is not None else None
        self._shape: tuple[int, ...] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            if self._embeddings_only:
                self._shape = tuple(self._embeddings.shape)
            elif len(self._dataset) == 0:
                self._shape = (0,)
            elif self._cached_idx:
                embedding_shape = self[list(self._cached_idx)[0]].shape
                self._shape = tuple([len(self)] + [*embedding_shape])
            else:
                embedding_shape = self[0].shape
                self._shape = tuple([len(self)] + [*embedding_shape])
        return self._shape

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
        arr = self[:] if not self._embeddings_only else self._embeddings

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

    def _hook_fn(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        # Copy the output to avoid computation graph issues
        if self._use_output:
            self.captured_output = output.detach().clone()
        else:
            self.captured_output = inputs[0].detach().clone()

    def _get_valid_layer_selection(self, layer_name: str, model: torch.nn.Module) -> torch.nn.Module:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected PyTorch model (torch.nn.Module), got {type(model).__name__}")

        modules_dict = dict(model.named_modules())

        if layer_name not in modules_dict:
            formatted_layers = "\n".join(f"  {layer}" for layer in modules_dict)
            raise ValueError(f"Invalid layer '{layer_name}'. Available layers are:\n{formatted_layers}")

        return modules_dict[layer_name]

    def __hash__(self) -> int:
        if self._embeddings_only:
            bid = self._embeddings.ravel().tobytes()
        else:
            did = self._dataset.metadata["id"] if isinstance(self._dataset, AnnotatedDataset) else str(self._dataset)
            mid = self._model.metadata["id"] if isinstance(self._model, AnnotatedModel) else str(self._model)
            tid = str.join("|", [str(t) for t in self._transforms])
            bid = f"{did}{mid}{tid}".encode()

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

        n_samples = len(self._dataset)
        bytes_per_element = np.dtype(np.float32).itemsize  # Assume float32
        estimated_bytes = n_samples * np.prod(embedding_shape) * bytes_per_element

        available_memory = psutil.virtual_memory().available
        threshold_bytes = available_memory * self.memory_threshold

        use_memmap = bool(estimated_bytes > threshold_bytes)

        if use_memmap and self.verbose:
            _logger.info(
                f"Using memory-mapped storage: estimated size {estimated_bytes / 1e9:.2f}GB "
                f"exceeds {self.memory_threshold * 100:.0f}% of available memory "
                f"({available_memory / 1e9:.2f}GB)"
            )

        return use_memmap

    def _initialize_storage(self, sample_embedding: NDArray[Any]) -> None:
        """Initialize storage backend (in-memory or memmap) based on size."""
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

    def to_tensor(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """
        Convert embeddings to PyTorch tensor.

        Process specified dataset indices through the model in batches and
        return concatenated embeddings as a single tensor on the configured device.

        Parameters
        ----------
        indices : Sequence[int] or None, default None
            Dataset indices to convert to embeddings. When None, processes entire dataset.

        Returns
        -------
        torch.Tensor
            Concatenated embeddings with shape (n_samples, embedding_dim) on configured device.

        Warnings
        --------
        Processing large datasets can be memory and compute intensive. Consider using
        numpy arrays via `__getitem__` for memory efficiency.
        """
        arr = np.vstack([self[i] for i in indices]) if indices is not None else self[:]
        return torch.from_numpy(arr).to(self.device)

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
            dataset,
            self.batch_size,
            self._transforms,
            self._model,
            self.layer_name,
            self.use_output,
            self.device,
            self._path,
            self.memory_threshold,
            self.verbose,
        )

    @classmethod
    def from_array(cls, array: ArrayLike) -> Embeddings:
        """
        Create Embeddings instance from an existing array.

        Parameters
        ----------
        array : ArrayLike
            In-memory data to wrap in an Embeddings object. Can be a numpy array,
            memmap, or the result of np.load(). Memmap arrays are preserved as-is.

        Returns
        -------
        Embeddings
            Embeddings-only instance containing the provided data.

        Example
        -------
        >>> import numpy as np
        >>> from dataeval.data import Embeddings
        >>> # From in-memory array
        >>> array = np.random.randn(100, 512)
        >>> embeddings = Embeddings.from_array(array)
        >>> tmp_file = tmp_path / "embeddings.npy"
        >>> # From saved file (preserves memmap)
        >>> np.save(tmp_file, array)
        >>> loaded = np.load(tmp_file, mmap_mode="r")
        >>> embeddings = Embeddings.from_array(loaded)
        >>> print(embeddings.shape)
        (100, 512)
        """
        embeddings = Embeddings([], 0, None, None, None, False, None, None, 0.8, False)
        embeddings._embeddings = array if isinstance(array, np.ndarray) else as_numpy(array)
        embeddings._cached_idx = set(range(len(embeddings._embeddings)))
        embeddings._embeddings_only = True
        return embeddings

    @classmethod
    def load(cls, path: Path | str, mmap_mode: Literal["r", "r+", "w+", "c"] | None = None) -> Embeddings:
        """
        Load embeddings from a saved .npy file.

        Parameters
        ----------
        path : Path or str
            File path to the saved .npy file containing embeddings.
        mmap_mode : str or None, default None
            Mode for memory-mapping the file. When None, loads the entire array
            into memory as an ndarray. When specified, uses memory-mapping which
            is more efficient for large files. Valid modes are:
            - 'r': Open existing file for reading only
            - 'r+': Open existing file for reading and writing
            - 'w+' : Open existing file and overwrite
            - 'c': Copy-on-write mode without updating file
            See numpy.load documentation for more details.

        Returns
        -------
        Embeddings
            Embeddings-only instance containing the loaded data.

        Example
        -------
        >>> import numpy as np
        >>> from dataeval.data import Embeddings
        >>> # Save some embeddings
        >>> array = np.random.randn(100, 512)
        >>> tmp_file = tmp_path / "embeddings.npy"
        >>> np.save(tmp_file, array)
        >>> # Load as in-memory array
        >>> embeddings = Embeddings.load(tmp_file)
        >>> # Load as memmap for large files
        >>> embeddings_mmap = Embeddings.load(tmp_file, mmap_mode="r")
        >>> print(embeddings.shape)
        (100, 512)
        """
        file_path = Path(path) if isinstance(path, str) else path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        array = np.load(file_path, mmap_mode=mmap_mode)
        return cls.from_array(array)

    def save(self, path: Path | str | None = None) -> None:
        """
        Compute all embeddings and save to disk.

        Forces computation of all embeddings if not already computed, then
        saves to the specified file path.

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
        if not self._embeddings_only:
            self.compute()

        # Save to disk
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self._embeddings, np.memmap):
            # Memmap is already on disk, just flush
            self._embeddings.flush()
            if self.verbose:
                _logger.debug(f"Flushed memmap embeddings to {target_path}")
        else:
            # Save in-memory array to disk
            np.save(target_path, self._embeddings)
            if self.verbose:
                _logger.debug(f"Saved embeddings to {target_path}")

    def compute(self, force: bool = False) -> Embeddings:
        """
        Compute and cache all embeddings.

        Forces evaluation of all lazy embeddings, storing them in memory or
        memmap according to the configured storage strategy.

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
        if self._embeddings_only:
            return self  # No-op for already-computed embeddings

        if force:
            self._cached_idx.clear()
            self._embeddings = np.empty((0,))

        # Trigger computation of all embeddings via __getitem__
        _ = self[:]

        return self

    def _encode(self, images: list[torch.Tensor]) -> NDArray[Any]:
        """Encode images to embeddings using the model."""
        input_tensor = torch.stack(images).to(self.device)

        with torch.no_grad():
            if self.layer_name:
                _ = self._model(input_tensor)  # Triggers hook
                output = self.captured_output
            else:
                output = self._model(input_tensor)

        return output.cpu().numpy()

    class _TorchDatasetWrapper(torch.utils.data.Dataset[torch.Tensor]):
        """Wrapper for dataset to convert to PyTorch and apply transforms."""

        def __init__(
            self, dataset: Dataset[tuple[ArrayLike, Any, Any]] | Dataset[ArrayLike], transforms: Iterable[Transform]
        ) -> None:
            self._dataset = dataset
            self._transforms = transforms

        def __getitem__(self, index: int) -> torch.Tensor:
            item = self._dataset[index]
            image = torch.as_tensor(item[0] if isinstance(item, tuple) else item)
            for transform in self._transforms:
                image = transform(image)
            return image

    def _batch(self, indices: Sequence[int]) -> Iterator[NDArray[Any]]:
        """Process indices in batches, yielding numpy arrays."""
        dataset = self._TorchDatasetWrapper(self._dataset, self._transforms)

        # Process all indices in batches
        for i in tqdm(range(0, len(indices), self.batch_size), desc="Batch embedding", disable=not self.verbose):
            batch = indices[i : i + self.batch_size]
            uncached = [idx for idx in batch if idx not in self._cached_idx]

            if uncached:
                out_of_range = set(uncached) - set(range(len(self._dataset)))
                if out_of_range:
                    raise IndexError(
                        f"Indices {sorted(out_of_range)} are out of range for dataset of size {len(self._dataset)}"
                    )
                # Process uncached indices
                loader = DataLoader(Subset(dataset, uncached), len(uncached), collate_fn=self._encode)
                for embeddings in loader:
                    # Initialize storage on first batch
                    if self._embeddings.size == 0:
                        self._initialize_storage(embeddings[0])

                    self._embeddings[uncached] = embeddings
                    self._cached_idx.update(uncached)

                    # Flush memmap writes (cheap operation)
                    if isinstance(self._embeddings, np.memmap):
                        self._embeddings.flush()

            yield self._embeddings[batch]

    def __getitem__(self, key: int | Iterable[int] | slice, /) -> NDArray[Any]:
        """
        Access embeddings by index, indices or slice.

        Parameters
        ----------
        key : int, Iterable[int], or slice
            Index, indices or slice to retrieve embeddings.

        Returns
        -------
        NDArray
            Embedding array for the requested indices.

        Raises
        ------
        TypeError
            When key is not an integer, Iterable[int] or slice.
        ValueError
            When trying to generate new embeddings from an embeddings-only instance.
        """
        if not isinstance(key, int | Iterable | slice) and not hasattr(key, "__int__"):
            raise TypeError("Invalid argument type.")

        # Validate and listify Iterable of indices
        if isinstance(key, Iterable):
            listified: list[int] = []
            for k in key:
                if not isinstance(k, int) and not hasattr(k, "__int__"):
                    raise TypeError(f"All indices in the sequence must be integers. Found: {k}")
                listified.append(int(k))
            key = listified

            if any(not isinstance(k, int) and not hasattr(k, "__int__") for k in key):
                raise TypeError("All indices in the sequence must be integers.")

        indices = list(range(len(self))[key]) if isinstance(key, slice) else [int(key)] if isinstance(key, int) else key

        if self._embeddings_only:
            if self._embeddings.size == 0:
                raise ValueError("Embeddings not initialized.")
            if not set(indices).issubset(self._cached_idx):
                raise ValueError("Unable to generate new embeddings from a shallow instance.")
            return self._embeddings[key]

        if not indices:
            return np.empty((0,), dtype=np.float32)

        result = np.vstack(list(self._batch(indices)))
        return result[0] if isinstance(key, int) else result

    def __iter__(self) -> Iterator[NDArray[Any]]:
        """Iterate over individual embeddings."""
        for batch in self._batch(range(len(self))):
            yield from batch

    def __len__(self) -> int:
        """Return number of embeddings."""
        return len(self._embeddings) if self._embeddings_only else len(self._dataset)

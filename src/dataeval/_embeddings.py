"""Embeddings class for extracting and managing image embeddings."""

__all__ = ["Embeddings"]

import os
import re
import shutil
import threading
import weakref
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Any, Self

import numpy as np
import psutil
import torch
import xxhash as xxh
from numpy.typing import NDArray

from dataeval._experimental import experimental
from dataeval._log import get_logger
from dataeval.config import resolve_batch_size
from dataeval.exceptions import NotFittedError, ShapeMismatchError
from dataeval.extractors import FlattenExtractor, TorchLayers
from dataeval.extractors._videotorch import _is_mirrored, _mirror
from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    FeatureExtractor,
    ImageDataset,
    ProgressCallback,
)
from dataeval.utils._internal import read_dataset
from dataeval.utils.data import requires_maite_dataset, unwrap_image

_logger = get_logger(__name__)

# Each file name `_resolve_path` gives, held by the live instance it was given to, and each memmap opened on one. The
# name comes from the dataset's id and the extractor, which two datasets, two views of one source or two crop views
# can share, and a second instance opening the first one's file with mode "w+" changes its rows or, when smaller,
# crashes the next read. A memmap holds its name after its instance is gone, since a view of it, such as the array
# `Embeddings.__call__` returns, still reads the file.
# ponytail: one process only; two processes writing to one directory can still take one name. A lock file per name
# is the upgrade if that case appears.
_held_files: "weakref.WeakValueDictionary[str, Embeddings]" = weakref.WeakValueDictionary()
_mapped_files: "weakref.WeakValueDictionary[str, np.memmap]" = weakref.WeakValueDictionary()
_held_files_lock = threading.Lock()


def _held_by_another(key: str, embeddings: "Embeddings") -> bool:
    """Whether another live instance holds the file `key`, or a memmap other than this instance's store maps it."""
    holder = _held_files.get(key)
    mapped = _mapped_files.get(key)
    return (holder is not None and holder is not embeddings) or (
        mapped is not None and mapped is not embeddings._embeddings
    )


def _exceeds_memory(estimated_bytes: int, memory_threshold: float) -> bool:
    """Whether `estimated_bytes` exceed `memory_threshold` of the memory available now, logging when they do."""
    available_memory = psutil.virtual_memory().available
    use_memmap = bool(estimated_bytes > available_memory * memory_threshold)
    if use_memmap:
        _logger.info(
            f"Using memory-mapped storage: estimated size {estimated_bytes / 1e9:.2f}GB "
            f"exceeds {memory_threshold * 100:.0f}% of available memory "
            f"({available_memory / 1e9:.2f}GB)",
        )
    return use_memmap


def _private_copies(items: Sequence[Any]) -> list[Any]:
    """Give one extractor its own copy of a batch, so that writing into its input cannot change what the next reads.

    A NumPy array is copied and a tensor is cloned. A read-only view would not do: `torch.from_numpy` and
    `torch.as_tensor` ignore NumPy's write flag. Anything else, such as a list, an image object or a video stream,
    passes as it is.
    """
    copies: list[Any] = []
    for item in items:
        if isinstance(item, np.ndarray):
            copies.append(item.copy())
        elif isinstance(item, torch.Tensor):
            copies.append(item.clone())
        else:
            copies.append(item)
    return copies


def _refuse_one_shot(items: Sequence[Any]) -> None:
    """Raise when an item is an iterator, which only the first of several extractors could walk."""
    if any(isinstance(item, Iterator) for item in items):
        raise ValueError(
            "An item is a one-shot stream, an iterator, which only one extractor can walk. Give "
            "Embeddings.compute_multi a dataset whose items can be walked again, such as a list of frames or a stream "
            "that decodes afresh on each walk, or compute each extractor on its own."
        )


def _check_extractors(dataset: Any, extractors: Mapping[str, Any]) -> None:
    """Refuse what `Embeddings.compute_multi` cannot run, before anything is read."""
    if dataset is None or not extractors:
        raise ValueError("Embeddings.compute_multi needs a dataset to read and at least one extractor.")
    for key, extractor in extractors.items():
        _check_extractor(key, extractor)


def _check_extractor(key: Any, extractor: Any) -> None:
    if not isinstance(key, str):
        raise TypeError(
            f"Each key must be a str, and {key!r} is of type {type(key).__name__}. The keys name the returned "
            "Embeddings, and with path the directories they are stored in."
        )
    if isinstance(extractor, Embeddings):
        raise TypeError(
            f"The extractor {key!r} is an Embeddings. Pass its extractor instead, so it is computed in this pass."
        )
    if not callable(extractor) and not isinstance(extractor, TorchLayers):
        raise TypeError(f"The extractor {key!r} is not callable: it is a {type(extractor).__name__}.")


_Groups = dict[str, tuple[TorchLayers, dict[str, str]]]
"""Each `TorchLayers` given to `Embeddings.compute_multi`, by its key, with the child key of each of its layers."""


def _add_way(ways: dict[str, Any], key: str, extractor: Any) -> None:
    if key in ways:
        raise ValueError(f"The key {key!r} is given twice: once directly, and once as a layer of a TorchLayers.")
    ways[key] = extractor


def _expand(extractors: Mapping[str, Any]) -> tuple[dict[str, Any], _Groups]:
    """Give each layer of a `TorchLayers` a key of its own, `<key>/<layer>`, read by a single-layer extractor."""
    ways: dict[str, Any] = {}
    groups: _Groups = {}
    for key, extractor in extractors.items():
        if not isinstance(extractor, TorchLayers):
            _add_way(ways, key, extractor)
            continue
        groups[key] = (extractor, {layer: f"{key}/{layer}" for layer in extractor.layer_names})
        for layer, child_key in groups[key][1].items():
            _add_way(ways, child_key, extractor.solo(layer))
    return ways, groups


def _agreed_batch_size(extractors: Mapping[str, Any], batch_size: int | None) -> int:
    """Resolve each extractor's batch size as it would be alone, and raise unless they agree."""
    sizes = {key: resolve_batch_size(batch_size, getattr(e, "batch_size", None)) for key, e in extractors.items()}
    if len(set(sizes.values())) > 1:
        listed = ", ".join(f"{key!r}: {size}" for key, size in sizes.items())
        raise ValueError(
            f"The extractors resolve to different batch sizes ({listed}). Pass batch_size so that every extractor "
            "sees the same batches."
        )
    return next(iter(sizes.values()))


def _directory_name(key: str) -> str:
    """Return the directory name for one key, with anything but letters, digits, '.', '_' and '-' made '_'."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", key)
    if name in ("", ".", ".."):
        raise ValueError(
            f"The key {key!r} names no directory to store its embeddings in. Use a key with a letter or digit."
        )
    return name


def _root_directory(path: Path | str) -> Path:
    root = Path(os.path.abspath(path))
    if root.exists() and not root.is_dir():
        raise ValueError(
            f"path {str(root)!r} is not a directory. Embeddings.compute_multi stores each extractor's embeddings in "
            "a directory of its own under path."
        )
    return root


def _child_directories(path: Path | str | None, keys: Iterable[str]) -> dict[str, Path | None]:
    """Plan one directory under `path` for each key, without making it: `_place_in_directories` does that.

    Keys that name no directory, the same one, or ones that differ only in case, which are one directory on macOS
    and Windows, are refused. Each child is given its directory as its `path`, so it names its file inside from its
    own hash, which covers the dataset. The key keeps extractors apart, and the hash keeps datasets apart.
    """
    if path is None:
        return dict.fromkeys(keys)
    root = _root_directory(path)
    named: dict[str, str] = {}
    for key in keys:
        name = _directory_name(key)
        other = named.get(name.casefold())
        if other is not None:
            same = "the same directory" if _directory_name(other) == name else "directories that differ only in case"
            raise ValueError(f"The keys {other!r} and {key!r} name {same} under path. Rename one of them.")
        named[name.casefold()] = key
    return {key: root / _directory_name(key) for key in named.values()}


def _place_in_directories(children: Mapping[str, "Embeddings"], directories: Mapping[str, Path | None]) -> None:
    """Make each child's directory and name its file there, once every child has read its dataset's first item."""
    for key, directory in directories.items():
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            children[key]._path = children[key]._resolve_path(directory)


def _memmap_for_all(children: Mapping[str, "Embeddings"], rows: Mapping[str, NDArray[Any]]) -> bool:
    """Whether the children's stores, together, are too large for memory, decided once from their first rows."""
    first = next(iter(children.values()))
    if first._path is None:
        return False
    estimated = sum(child._estimated_bytes(rows[key].shape[1:]) for key, child in children.items())
    return _exceeds_memory(estimated, first.memory_threshold)


def _mirrored(children: Mapping[str, "Embeddings"]) -> dict[str, Any]:
    """Return the video extractors that share one walk of each video, when there are at least two of them."""
    mirrored = {key: child._extractor for key, child in children.items() if _is_mirrored(child._extractor)}
    return mirrored if len(mirrored) > 1 else {}


def _shared_outputs(
    mirrored: Mapping[str, Any], groups: _Groups, indices: Sequence[int], images: Sequence[Any]
) -> dict[str, Any]:
    """Run, once each, what several children share: the walk of the mirrored videos, and each `TorchLayers`."""
    # No batch-wide copy for the mirror: it copies a sliced NumPy or tensor video for each extractor and a window's
    # frames for one with a processor, and any other clip is stacked into a new array. A sliced array of another type
    # is passed as it is, as it is to every extractor.
    outputs: dict[str, Any] = _mirror(mirrored, images, indices) if mirrored else {}
    for entry, (layers, child_keys) in groups.items():
        try:
            read = layers.extract(_private_copies(images))
        except Exception as error:
            error.add_note(f"Raised by the TorchLayers {entry!r} given to Embeddings.compute_multi.")
            raise
        outputs.update({child_key: read[layer] for layer, child_key in child_keys.items()})
    return outputs


def _extract_each(
    children: Mapping[str, "Embeddings"], indices: Sequence[int], images: Sequence[Any], groups: _Groups
) -> dict[str, NDArray[Any]]:
    """Run every child's extractor on its own copies of one batch, naming the extractor in any error it raises.

    Two or more `VideoTorchExtractor`s share one walk of each video, and the layers of one `TorchLayers` share one
    forward pass, so each counts as one reader of a one-shot stream.
    """
    grouped = {child_key for _, child_keys in groups.values() for child_key in child_keys.values()}
    mirrored = _mirrored({key: child for key, child in children.items() if key not in grouped})
    if len(children) - len(grouped) - len(mirrored) + len(groups) + bool(mirrored) > 1:
        _refuse_one_shot(images)
    outputs = _shared_outputs(mirrored, groups, indices, images)
    rows: dict[str, NDArray[Any]] = {}
    for key, child in children.items():
        try:
            raw = outputs[key] if key in outputs else child._extractor(_private_copies(images))
            rows[key] = child._rows(raw, len(images))
        except Exception as error:
            error.add_note(f"Raised by the extractor {key!r} given to Embeddings.compute_multi.")
            raise
    return rows


def _ingest_each(
    children: Mapping[str, "Embeddings"], indices: Sequence[int], rows: Mapping[str, NDArray[Any]], use_memmap: bool
) -> None:
    """Store every child's rows for one batch, naming the extractor in any error, such as rows of a changed width."""
    for key, child in children.items():
        try:
            child._ingest(indices, rows[key], use_memmap=use_memmap)
        except Exception as error:
            error.add_note(f"Raised while storing the rows of the extractor {key!r} given to Embeddings.compute_multi.")
            raise


def _fill_together(
    children: Mapping[str, "Embeddings"], size: int, progress_callback: ProgressCallback | None, groups: _Groups
) -> None:
    """Read each batch once and store every child's rows for it, reporting progress once per batch."""
    first = next(iter(children.values()))
    total = len(first)
    use_memmap: bool | None = None
    with closing(first._read_images(range(total))) as images:
        for start in range(0, total, size):
            indices = list(range(start, min(start + size, total)))
            rows = _extract_each(children, indices, list(islice(images, len(indices))), groups)
            if use_memmap is None:
                use_memmap = _memmap_for_all(children, rows)
            _ingest_each(children, indices, rows, use_memmap)
            if progress_callback:
                progress_callback(indices[-1] + 1, total=total)


class Embeddings(Array, FeatureExtractor):
    """
    Collection of image embeddings from a dataset.

    Embeddings are accessed by index or slice and are loaded on-demand. For large
    datasets, embeddings are automatically memory-mapped to disk to avoid exceeding
    available memory.

    This class also implements the :class:`~dataeval.protocols.FeatureExtractor` protocol,
    allowing it to be used directly with drift detectors and quality metrics that accept
    feature extractors.

    The extractor must return one row for each image it is given, in order. ``Embeddings`` checks the
    row count of every batch and raises when it differs, and raises when the extractor returns something
    other than an array, such as a ``dict``. The count cannot prove that each row belongs to the image
    at the same position: an extractor that returns one row per detection passes whenever the total number
    of detections equals the number of images, even though the rows then belong to different images.
    Extractors that return one row per detection or per class, such as a detection model wrapped in
    :class:`~dataeval.extractors.ScoresExtractor`, are not valid here. Call them directly and give the
    result, or the extractor itself, to a drift detector.

    Parameters
    ----------
    dataset : Dataset, sequence of images, or None, default None
        Source of the original images to embed. Accepts either a MAITE dataset
        (an :class:`ImageClassificationDataset` or :class:`ObjectDetectionDataset`
        whose items are ``(image, target, metadata)`` tuples) or any indexable,
        sized source of bare images -- for example a plain ``list`` of image
        arrays or a single stacked ``ndarray`` of images (validated as
        "image_only"). MAITE tuple items are auto-unwrapped to their image
        (element 0); bare-image sources pass through unchanged. When None,
        creates an unbound instance that can be used as a reusable feature
        extractor. Use :meth:`bind` to attach a dataset later, or pass data
        directly to :meth:`__call__`.
    extractor : FeatureExtractor or None, default None
        Feature extractor for converting images to embeddings. Handles model inference,
        device management, and transforms. When None, uses
        :class:`~dataeval.extractors.FlattenExtractor` for simple baseline
        compatibility with all DataEval tools.
    batch_size : int or None, default None
        I/O chunk size: how many images are loaded from the dataset, encoded, and
        written to storage per step. Resolved via
        :func:`~dataeval.config.resolve_batch_size` as the first set of
        ``batch_size`` (this argument), the extractor's own ``batch_size``, then
        the global default. This is distinct from an extractor's own forward-pass
        (compute) batch size: an extractor with its own ``batch_size`` sub-batches
        each chunk for the model, so the smaller of the two bounds the forward
        pass. Batching does not change the resulting embeddings except for
        differences in the last digits, which come from floating-point rounding.
    path : Path, str, or None, default None
        File path for memory-mapped storage. When None, caches embeddings in memory only.
        When Path or string is provided, uses memory-mapped storage for large embeddings
        (automatic based on memory_threshold). In a directory, the file is named for the
        dataset and the extractor, and is never one that another live instance in this
        process uses.
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
    >>> from dataeval.extractors import TorchExtractor
    >>>
    >>> extractor = TorchExtractor(my_model)
    >>> embeddings = Embeddings(train_dataset, extractor=extractor, batch_size=32)
    >>> train_emb = embeddings[:]
    >>> train_emb.shape
    (40, 32)

    Using with default flattening (no model):

    >>> # Uses FlattenExtractor by default
    >>> embeddings = Embeddings(dataset, batch_size=32)
    >>> flat_features = np.asarray(embeddings)
    """

    memory_threshold: float

    @requires_maite_dataset("dataset", expected="image_only")
    def __init__(
        self,
        # Technically more permissive than ImageClassificationDataset or ObjectDetectionDataset
        dataset: ImageDataset | None = None,
        extractor: FeatureExtractor | None = None,
        batch_size: int | None = None,
        path: Path | str | None = None,
        memory_threshold: float = 0.8,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if extractor is not None and not callable(extractor):
            raise TypeError(
                f"extractor must be callable, and a {type(extractor).__name__} is not. A TorchLayers, which reads "
                "several layers of one model in one forward pass, is given to Embeddings.compute_multi instead."
            )
        self._extractor = extractor if extractor is not None else FlattenExtractor()
        self._batch_size = resolve_batch_size(batch_size, getattr(self._extractor, "batch_size", None))
        self.memory_threshold = max(0.0, min(1.0, memory_threshold))
        self._progress_callback = progress_callback

        self._dataset = dataset

        self._cached_idx: set[int] = set()
        self._embeddings: np.ndarray | np.memmap = np.empty((0,))
        self._use_memmap: bool = False

        self._own_names: dict[str, Path] = {}
        self._path = self._resolve_path(path) if path is not None else None
        self._shape: tuple[int, ...] | None = None

    @property
    def batch_size(self) -> int:
        """Batch size used for embedding computation.

        Returns
        -------
        int
            Number of samples processed per batch.
        """
        return self._batch_size

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            if self._dataset is None:
                raise NotFittedError("Cannot determine shape: no dataset bound. Call bind() first.")
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

    @requires_maite_dataset("dataset", expected="image_only")
    def bind(self, dataset: ImageDataset) -> Self:
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

        Example
        -------
        >>> from dataeval import Embeddings
        >>> from dataeval.extractors import TorchExtractor
        >>>
        >>> extractor = TorchExtractor(my_model)
        >>> emb = Embeddings(extractor=extractor, batch_size=32)
        >>> _ = emb.bind(train_dataset)
        >>> embeddings = emb()
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
        NotFittedError
            If data is None and no dataset is bound.

        Example
        -------
        >>> from dataeval import Embeddings
        >>> from dataeval.extractors import TorchExtractor
        >>>
        >>> extractor = TorchExtractor(my_model)
        >>> embeddings = Embeddings(train_dataset, extractor=extractor, batch_size=32)
        >>>
        >>> # Extract from bound dataset
        >>> train_emb = embeddings()
        >>>
        >>> # Extract from new dataset
        >>> test_emb = embeddings(test_dataset)
        """
        if data is None:
            if self._dataset is None:
                raise NotFittedError("No dataset bound. Provide data or call bind() first.")
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
        # Trigger computation for lazy embeddings, whether the cache is empty or only partly filled
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

    def __repr__(self) -> str:
        extractor = self._extractor.__class__.__name__
        bound = self._dataset is not None
        parts = [f"extractor={extractor}", f"batch_size={self._batch_size}", f"bound={bound}"]
        if bound and self._shape is not None:
            parts.append(f"shape={self._shape}")
        if self._path is not None:
            parts.append(f"path={self._path!r}")
        return f"Embeddings({', '.join(parts)})"

    def __str__(self) -> str:
        info = "is_bound=False" if self._dataset is None else f"n={len(self._dataset)}, cached={len(self._cached_idx)}"
        return f"Embeddings({info}, extractor={self._extractor.__class__.__name__})"

    def __hash__(self) -> int:
        if self._dataset is None:
            # Unbound instance - hash based on extractor only
            bid = f"unbound:{self._extractor!r}".encode()
        else:
            did = self._dataset.metadata["id"] if isinstance(self._dataset, AnnotatedDataset) else str(self._dataset)
            bid = f"{did}{self._extractor!r}".encode()

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
                self._move_to(new_path)

    def _move_to(self, new_path: Path) -> None:
        """Save the rows to `new_path`, then make it this instance's path, and a memmap store's file.

        The rows are saved before the path changes, so a refused save changes nothing.
        """
        if self._embeddings.size > 0:
            self.save(new_path)
        self._path = new_path
        if isinstance(self._embeddings, np.memmap):
            # Move the store onto the copy, so it reads and writes the file `path` names
            self._embeddings = np.lib.format.open_memmap(new_path, mode="r+")
            with _held_files_lock:
                _mapped_files[os.path.realpath(new_path)] = self._embeddings

    def _resolve_path(self, path: Path | str) -> Path:
        if isinstance(path, str):
            path = Path(os.path.abspath(path))
        if isinstance(path, Path) and (path.is_dir() or not path.suffix):
            path = self._own_file(path)
        return path

    def _own_file(self, directory: Path) -> Path:
        """Name this instance's file in `directory`, one that no other live instance in this process holds.

        The name is ``emb-<hash>.npy``, or ``emb-<hash>-<n>.npy`` with the smallest free `n` when another live instance,
        or a memmap still read through a view, holds that name. A name is free again once both are gone, so a rerun
        reuses its file. This instance keeps the name it was given for the same directory and hash.
        """
        stem = f"emb-{hash(self)}"
        candidate = directory / f"{stem}.npy"
        given = os.path.realpath(candidate)
        with _held_files_lock:
            if given in self._own_names:
                return self._own_names[given]
            n = 0
            while _held_by_another(os.path.realpath(candidate), self):
                n += 1
                candidate = directory / f"{stem}-{n}.npy"
            _held_files[os.path.realpath(candidate)] = self
            self._own_names[given] = candidate
        return candidate

    def _should_use_memmap(self, embedding_shape: tuple[int, ...]) -> bool:
        """Determine if memmap should be used based on estimated size."""
        if self._path is None:
            return False
        return _exceeds_memory(self._estimated_bytes(embedding_shape), self.memory_threshold)

    def _estimated_bytes(self, embedding_shape: tuple[int, ...]) -> int:
        """Bytes a store of float32 rows of `embedding_shape` takes for the bound dataset."""
        if self._dataset is None:
            raise NotFittedError("No dataset bound. Call bind() first.")
        return int(len(self._dataset) * np.prod(embedding_shape) * np.dtype(np.float32).itemsize)

    def _initialize_storage(self, sample_embedding: NDArray[Any], use_memmap: bool | None = None) -> None:
        """Initialize storage backend (in-memory or memmap) based on size."""
        if self._dataset is None:
            raise NotFittedError("No dataset bound. Call bind() first.")

        n_samples = len(self._dataset)
        embedding_shape = sample_embedding.shape
        full_shape = (n_samples, *embedding_shape)
        dtype = sample_embedding.dtype

        if self._path is not None:
            self._use_memmap = self._should_use_memmap(embedding_shape) if use_memmap is None else use_memmap

            if self._use_memmap:
                # Create memory-mapped file
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # A memmap with a .npy header, so `np.load(path, mmap_mode="r")` reads the file.
                self._embeddings = np.lib.format.open_memmap(self._path, mode="w+", dtype=dtype, shape=full_shape)
                with _held_files_lock:
                    _mapped_files[os.path.realpath(self._path)] = self._embeddings
            else:
                # Use in-memory array
                self._embeddings = np.empty(full_shape, dtype=dtype)
        else:
            # In-memory only
            self._embeddings = np.empty(full_shape, dtype=dtype)

    def new(self, dataset: ImageDataset) -> Self:
        """
        Create new Embeddings instance with a different dataset.

        Generate a new Embeddings object using the same extractor and configuration
        but with a different dataset. With a path, the new instance stores its embeddings
        in the same directory, in a file named for its own dataset, so it never writes
        to this instance's file.

        Parameters
        ----------
        dataset : ImageClassificationDataset or ObjectDetectionDataset
            Dataset that provides images for the new Embeddings instance.

        Returns
        -------
        Embeddings
            New Embeddings object configured identically to the current instance.
        """
        other = self.__class__(
            dataset,
            extractor=self._extractor,
            batch_size=self._batch_size,
            memory_threshold=self.memory_threshold,
            progress_callback=self._progress_callback,
        )
        if self._path is not None:
            # The directory, not this instance's file: the new instance names a file of its own there, so a memmap
            # here is not overwritten and a saved file is not replaced. Named as a directory even before it exists,
            # since a directory such as "v1.2" would otherwise read as a file name.
            other._path = other._own_file(self._path.parent)
        return other

    @classmethod
    @experimental
    def compute_multi(
        cls,
        dataset: ImageDataset,
        extractors: Mapping[str, FeatureExtractor | TorchLayers],
        *,
        batch_size: int | None = None,
        path: Path | str | None = None,
        memory_threshold: float = 0.8,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Self]:
        """
        Compute the embeddings of several extractors in one pass over a dataset, reading each item once.

        Each batch of `dataset` is read once and given to every extractor in turn, in the order of `extractors`, before
        the next batch is read. Two or more video extractors, and each :class:`~dataeval.extractors.TorchLayers`, run
        first, as the Notes describe. Each
        extractor's embeddings become an ordinary :class:`Embeddings`, equal to the one it gives on its own with the
        same batch size, unless it draws random numbers, which it then draws in another order.

        Parameters
        ----------
        dataset : ImageClassificationDataset or ObjectDetectionDataset
            Dataset that provides the images.
        extractors : Mapping[str, FeatureExtractor | TorchLayers]
            The extractors, each under the key its embeddings are returned under. A
            :class:`~dataeval.extractors.TorchLayers` returns one :class:`Embeddings` for each of its layers, under
            ``"<key>/<layer>"``.
        batch_size : int or None, default None
            Number of items read and embedded at a time. None takes each extractor's own `batch_size`, or else the
            global default, and the extractors must then agree.
        path : Path, str or None, default None
            Directory with one directory under it for each key. The embeddings are stored there as memory-mapped files
            when all of them together exceed `memory_threshold` of the available memory, and are otherwise kept in
            memory until an :class:`Embeddings` is saved. None keeps them in memory. Keys whose directory names
            differ only in case are refused with a path, since macOS and Windows would give them one directory.
        memory_threshold : float, default 0.8
            Share of the available memory that the embeddings of all the extractors may take together before they are
            stored as memory-mapped files under `path`.
        progress_callback : ProgressCallback or None, default None
            Called once per batch with the number of items done and the total. Each returned :class:`Embeddings`
            keeps it, as it keeps the other settings.

        Returns
        -------
        dict[str, Embeddings]
            One computed :class:`Embeddings` for each key, in the order of `extractors`.

        Raises
        ------
        ValueError
            When there is no dataset or no extractor, when the extractors' batch sizes differ and `batch_size` is not
            given, when a key is given twice, when a key names no directory, when two keys name the same directory or
            directories that differ only in case, when `path` is a file, or when an item is a one-shot stream that more
            than one extractor would walk.
        TypeError
            When a key is not a str, or an extractor is an :class:`Embeddings` or is neither callable nor a
            :class:`~dataeval.extractors.TorchLayers`.

        Notes
        -----
        Each extractor gets its own copy of every NumPy image and tensor in a batch, so one that changes its input in
        place cannot change what the next one reads. Any other item, such as a list of frames, an image object or a
        video stream, is passed as it is, and an extractor that changes it in place changes it for the next. An error
        raised by an extractor carries a note that names its key, and nothing is returned.

        Each :class:`~dataeval.extractors.TorchExtractor` runs its own forward pass, also when several share a model;
        a :class:`~dataeval.extractors.TorchLayers` reads all its layers in one forward pass per batch. Every
        extractor's model is held in memory for the whole pass. :func:`~dataeval.config.set_load_threads`
        applies to the reads, and the extractors run on the calling thread. Each embedding is tested equal to the
        extractor's own result on the CPU; a GPU and Windows are untested.

        Two or more :class:`~dataeval.extractors.VideoTorchExtractor` share one walk of each video: each frame is read
        once and given to every one of them, and no more than one clip of it is held for each. A video that is sliced
        into clips rather than walked, such as a tensor, is in memory already, and each of them slices its own copy of
        a NumPy array or a tensor. Any other extractor
        walks the video again, and so does a subclass of :class:`~dataeval.extractors.VideoTorchExtractor` that
        overrides its call. A one-shot stream is refused only when more than one walk of it would be needed.

        This raises :class:`~dataeval.exceptions.ExperimentalWarning` on the first call in a process.

        .. versionadded:: 1.3

        Examples
        --------
        >>> from dataeval import Embeddings
        >>> from dataeval.extractors import FlattenExtractor, TorchExtractor
        >>>
        >>> extractors = {"flat": FlattenExtractor(), "model": TorchExtractor(my_model)}
        >>> children = Embeddings.compute_multi(train_dataset, extractors, batch_size=32)
        >>> list(children)
        ['flat', 'model']
        >>> children["model"].shape
        (40, 32)
        """
        _check_extractors(dataset, extractors)
        size = _agreed_batch_size(extractors, batch_size)
        ways, groups = _expand(extractors)
        directories = _child_directories(path, ways)
        children = {
            key: cls(
                dataset,
                extractor=extractor,
                batch_size=size,
                memory_threshold=memory_threshold,
                progress_callback=progress_callback,
            )
            for key, extractor in ways.items()
        }
        _place_in_directories(children, directories)
        _fill_together(children, size, progress_callback, groups)
        return children

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
            When no path is specified and instance has no configured path, or when the path is the file of another
            Embeddings in this process.
        """
        # Determine target path
        if path is not None:
            target_path = self._resolve_path(path)
        elif self._path is not None:
            target_path = self._path
        else:
            raise ValueError("No path specified. Provide a path or initialize Embeddings with a path.")

        self._refuse_another_file(target_path)

        # Ensure all embeddings are computed
        self.compute()
        self._write(target_path)

    def _refuse_another_file(self, target_path: Path) -> None:
        """Raise when `target_path` is the file of another live instance, which a save there would overwrite."""
        with _held_files_lock:
            held = _held_by_another(os.path.realpath(target_path), self)
        if held:
            raise ValueError(
                f"{target_path} is the file of another Embeddings in this process, and saving there would change its "
                "rows. Save to another file, or to a directory, where each Embeddings gets a file of its own."
            )

    def _write(self, target_path: Path) -> None:
        """Write the computed rows to `target_path` as a .npy file, under the name as given."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self._embeddings, np.memmap):
            # The store is a .npy file already: flush it, and copy it, in chunks, when the target is another file.
            self._embeddings.flush()
            own = str(self._embeddings.filename)
            if os.path.exists(target_path) and os.path.samefile(target_path, own):
                _logger.debug(f"Flushed memmap embeddings to {target_path}")
            else:
                shutil.copyfile(own, target_path)
                _logger.debug(f"Copied memmap embeddings from {own} to {target_path}")
        else:
            # Save in-memory array to disk, through a file object so the name is kept as given, as for a memmap
            with open(target_path, "wb") as file:
                np.save(file, self._embeddings)
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

    def _read_images(self, indices: Sequence[int]) -> Generator[Any, None, None]:
        """Yield the image of each index in order, read as one pass, in threads when ``load_threads`` allows it.

        One pass, not one per batch, so the choice and the first read on the calling thread are made once, and reads
        continue into the next batch while the extractor runs on this one. Close it if the pass stops early.
        """
        if self._dataset is None:
            raise NotFittedError("No dataset bound. Call bind() first.")
        return read_dataset(self._dataset, indices, unwrap_image)

    def _rows(self, raw: Any, expected: int) -> NDArray[Any]:
        """Return the extractor's output as an array with one row for each of ``expected`` images.

        A matching count is the most a shape can prove. It cannot show that each row belongs to the image at the
        same position, so an extractor that returns one row per detection still passes when the totals agree.
        """
        name = type(self._extractor).__name__
        advice = (
            "Embeddings stores one row for each dataset item. An extractor that returns one row per detection or per "
            "class, such as a detection model wrapped in ScoresExtractor, DetectionGeometryExtractor or "
            "ClasswiseUncertaintyExtractor, cannot be wrapped in Embeddings. Call it directly and give the result, or "
            "the extractor itself, to a drift detector."
        )
        if isinstance(raw, Mapping):
            several = (
                "To store several embeddings of each item, such as several layers of one model, give the extractors to "
                "Embeddings.compute_multi, with the layers of one torch model as a TorchLayers."
            )
            raise TypeError(f"{name} returned a {type(raw).__name__}, not an array. {advice} {several}")
        rows = np.asarray(raw)
        if rows.ndim == 0 or len(rows) != expected:
            got = "a single value" if rows.ndim == 0 else f"{len(rows)} rows"
            raise ShapeMismatchError(f"{name} returned {got} for a batch of {expected} images. {advice}")
        return rows

    def _ingest(self, indices: Sequence[int], rows: NDArray[Any], *, use_memmap: bool | None = None) -> None:
        """Store `rows`, already checked by `_rows`, as the embeddings of `indices`.

        The first call initializes the storage: `use_memmap` None decides between memory and a memmap as the
        storage always has, and True or False makes the choice for it.
        """
        if self._embeddings.size == 0:
            self._initialize_storage(rows[0], use_memmap)

        for i, idx in enumerate(indices):
            self._embeddings[idx] = rows[i]
        self._cached_idx.update(indices)

        # Flush memmap writes (cheap operation)
        if isinstance(self._embeddings, np.memmap):
            self._embeddings.flush()

    def _batch(self, indices: Sequence[int]) -> Iterator[NDArray[Any]]:  # noqa: C901
        """Process indices in batches using the extractor."""
        if self._dataset is None:
            raise NotFittedError("No dataset bound. Call bind() first.")

        # Filter to uncached indices
        uncached = [idx for idx in indices if idx not in self._cached_idx]

        if uncached:
            # Validate indices
            out_of_range = set(uncached) - set(range(len(self._dataset)))
            if out_of_range:
                raise IndexError(
                    f"Indices {sorted(out_of_range)} are out of range for dataset of size {len(self._dataset)}",
                )

            # Process in batches
            processed = 0
            with closing(self._read_images(uncached)) as images:
                for batch_start in range(0, len(uncached), self._batch_size):
                    batch_idx = uncached[batch_start : batch_start + self._batch_size]

                    # Fetch images from dataset
                    batch_images = list(islice(images, len(batch_idx)))

                    # Extract features and store them
                    self._ingest(batch_idx, self._rows(self._extractor(batch_images), len(batch_idx)))

                    # Report progress
                    processed += len(batch_idx)
                    if self._progress_callback:
                        self._progress_callback(processed, total=len(uncached))

        # Yield results in batches for consistency
        for batch_start in range(0, len(indices), self._batch_size):
            batch_indices = list(indices[batch_start : batch_start + self._batch_size])
            yield self._embeddings[batch_indices]

    def __getitem__(self, key: int | Iterable[int] | slice, /) -> NDArray[Any]:  # noqa: C901
        """
        Access embeddings by index, indices or slice.

        Returns a view of the memmap when possible (slices/ints),
        and a copy only when necessary (arbitrary list of indices).
        """
        from collections import deque

        # 1. Validation and Index Normalization
        if isinstance(key, int | np.integer):
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
        NotFittedError
            If no dataset is bound.
        """
        if self._dataset is None:
            raise NotFittedError("Cannot determine length: no dataset bound. Call bind() first.")
        return len(self._dataset)

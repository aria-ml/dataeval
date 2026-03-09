"""Utility functions for array conversion and manipulation across different frameworks."""

__all__ = []

import logging
import multiprocessing
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from enum import Enum
from importlib import import_module
from os import cpu_count
from types import ModuleType
from typing import Any, Literal, TypeVar, overload

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from dataeval._log import LogMessage
from dataeval.exceptions import ShapeMismatchError
from dataeval.protocols import Array, SequenceLike

_logger = logging.getLogger(__name__)

EPSILON = 1e-12
MODULE_CACHE = {}


def try_import(module_name: str) -> ModuleType | None:
    if module_name in MODULE_CACHE:
        return MODULE_CACHE[module_name]

    try:
        module = import_module(module_name)
    except ImportError:  # pragma: no cover
        _logger.log(logging.INFO, f"Unable to import {module_name}.")
        module = None

    MODULE_CACHE[module_name] = module
    return module


np_dtype = TypeVar("np_dtype", bound=np.generic)


def opt_as_numpy(
    array: ArrayLike | SequenceLike[Any] | None,
    *,
    dtype: type[np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
) -> NDArray[np_dtype] | None:
    """
    Convert an ArrayLike to Numpy array without copying (if possible), returns None if input is None.

    Parameters
    ----------
    array : ArrayLike or SequenceLike or None
        Input array-like object or None
    dtype : numpy dtype or None, default None
        Desired output dtype
    required_ndim : int or Iterable[int] or None, default None
        Required number of dimensions (or set of valid dimensions)
    required_shape : tuple[int, ...] or None, default None
        Required shape of output

    Returns
    -------
    NDArray or None
        NumPy array or None if input was None
    """
    return opt_to_numpy(array, dtype=dtype, required_ndim=required_ndim, required_shape=required_shape, copy=False)


def opt_to_numpy(
    array: ArrayLike | SequenceLike[Any] | None,
    *,
    dtype: type[np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
    copy: bool = True,
) -> NDArray[np_dtype] | None:
    """
    Convert an ArrayLike to Numpy array, returns None if input is None.

    Parameters
    ----------
    array : ArrayLike or SequenceLike or None
        Input array-like object or None
    dtype : numpy dtype or None, default None
        Desired output dtype
    required_ndim : int or Iterable[int] or None, default None
        Required number of dimensions (or set of valid dimensions)
    required_shape : tuple[int, ...] or None, default None
        Required shape of output
    copy : bool, default True
        Whether to copy the array

    Returns
    -------
    NDArray or None
        NumPy array or None if input was None
    """
    return (
        None
        if array is None
        else to_numpy(array, dtype=dtype, required_ndim=required_ndim, required_shape=required_shape, copy=copy)
    )


def as_numpy(
    array: ArrayLike | SequenceLike[Any] | None,
    *,
    dtype: type[np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
) -> NDArray[np_dtype]:
    """
    Convert an ArrayLike to Numpy array without copying (if possible).

    Parameters
    ----------
    array : ArrayLike or SequenceLike or None
        Input array-like object
    dtype : numpy dtype or None, default None
        Desired output dtype
    required_ndim : int or Iterable[int] or None, default None
        Required number of dimensions (or set of valid dimensions)
    required_shape : tuple[int, ...] or None, default None
        Required shape of output

    Returns
    -------
    NDArray
        NumPy array
    """
    return to_numpy(array, dtype=dtype, required_ndim=required_ndim, required_shape=required_shape, copy=False)


def to_numpy(
    array: ArrayLike | SequenceLike[Any] | None,
    *,
    dtype: type[np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
    copy: bool = True,
) -> NDArray[np_dtype]:
    """
    Convert an ArrayLike to new Numpy array.

    Parameters
    ----------
    array : ArrayLike or SequenceLike or None
        Input array-like object
    dtype : numpy dtype or None, default None
        Desired output dtype
    required_ndim : int or Iterable[int] or None, default None
        Required number of dimensions (or set of valid dimensions)
    required_shape : tuple[int, ...] or None, default None
        Required shape of output
    copy : bool, default True
        Whether to copy the array

    Returns
    -------
    NDArray
        NumPy array

    Raises
    ------
    ShapeMismatchError
        If required_ndim or required_shape constraints are not met
    """
    _array: NDArray[np_dtype] | None = None

    if array is None:
        _array = np.array([], dtype=dtype)
    elif isinstance(array, np.ndarray | np.memmap):
        numpy = array.copy() if copy else array
        _array = numpy.astype(dtype) if dtype is not None else numpy
    elif array.__class__.__module__.startswith("tensorflow"):  # pragma: no cover - removed tf from deps
        tf = try_import("tensorflow")
        if tf and tf.is_tensor(array):
            _logger.log(logging.INFO, "Converting Tensorflow array to NumPy array.")
            numpy = array.numpy().copy() if copy else array.numpy()  # type: ignore
            _array = numpy.astype(dtype) if dtype is not None else numpy
    elif array.__class__.__module__.startswith("torch"):
        torch = try_import("torch")
        if torch and isinstance(array, torch.Tensor):
            _logger.log(logging.INFO, "Converting PyTorch array to NumPy array.")
            numpy = array.detach().cpu().numpy().copy() if copy else array.detach().cpu().numpy()  # type: ignore
            _logger.log(logging.DEBUG, LogMessage(lambda: f"{str(array)} -> {str(numpy)}"))
            _array = numpy.astype(dtype) if dtype is not None else numpy

    # If the array was not converted yet, let numpy create the array directly
    if _array is None:
        _array = np.array(array, dtype=dtype) if copy else np.asarray(array, dtype=dtype)

    required_ndims = (required_ndim,) if isinstance(required_ndim, int) else required_ndim
    if required_ndims is not None and _array.ndim not in required_ndims:
        raise ShapeMismatchError(f"Array has {_array.ndim} dimensions, expected {required_ndim}.")

    if required_shape is not None and _array.shape != required_shape:
        raise ShapeMismatchError(f"Array has shape {_array.shape}, expected {required_shape}.")

    return _array


def to_numpy_iter(iterable: Iterable[ArrayLike]) -> Iterator[NDArray[Any]]:
    """
    Yield an iterator of numpy arrays from an ArrayLike iterable.

    Parameters
    ----------
    iterable : Iterable[ArrayLike]
        Iterable of array-like objects

    Yields
    ------
    NDArray
        NumPy arrays
    """
    for array in iterable:
        yield to_numpy(array)


@overload
def rescale_array(array: NDArray[np_dtype]) -> NDArray[np_dtype]: ...
@overload
def rescale_array(array: torch.Tensor) -> torch.Tensor: ...
def rescale_array(array: Array | NDArray[np_dtype] | torch.Tensor) -> Array | NDArray[np_dtype] | torch.Tensor:
    """
    Rescale an array to the range [0, 1].

    Parameters
    ----------
    array : NDArray or torch.Tensor
        Input array

    Returns
    -------
    NDArray or torch.Tensor
        Rescaled array in range [0, 1]

    Raises
    ------
    TypeError
        If array type is not supported
    """
    if isinstance(array, np.ndarray | torch.Tensor):
        arr_min = array.min()
        arr_max = array.max()
        return (array - arr_min) / (arr_max - arr_min)
    raise TypeError(f"Unsupported type: {type(array)}")


TArray = TypeVar("TArray", Array, np.ndarray, torch.Tensor)


@overload
def ensure_embeddings(
    embeddings: TArray,
    dtype: torch.dtype,
    unit_interval: Literal[True, False, "force"] = False,
) -> torch.Tensor: ...


@overload
def ensure_embeddings(
    embeddings: TArray,
    dtype: type[np_dtype],
    unit_interval: Literal[True, False, "force"] = False,
) -> NDArray[np_dtype]: ...


@overload
def ensure_embeddings(
    embeddings: TArray,
    dtype: None = None,
    unit_interval: Literal[True, False, "force"] = False,
) -> TArray: ...


def ensure_embeddings(
    embeddings: TArray,
    dtype: type[np_dtype] | torch.dtype | None = None,
    unit_interval: Literal[True, False, "force"] = False,
) -> torch.Tensor | NDArray[np_dtype] | TArray:
    """
    Validate the embeddings array and convert it to the specified type.

    Parameters
    ----------
    embeddings : ArrayLike
        Embeddings array
    dtype : numpy dtype or torch dtype or None, default None
        The desired dtype of the output array, None to skip conversion
    unit_interval : bool or "force", default False
        Whether to validate or force the embeddings to unit interval

    Returns
    -------
    torch.Tensor or NDArray or other Array
        Converted embeddings array

    Raises
    ------
    ShapeMismatchError
        If the embeddings array is not 2D
    ShapeMismatchError
        If the embeddings array has a zero dimension
    ValueError
        If the embeddings array is not unit interval [0, 1] (when unit_interval=True)
    """
    if isinstance(dtype, torch.dtype):
        arr = torch.as_tensor(embeddings, dtype=dtype)
    else:
        arr = (
            embeddings.detach().cpu().numpy().astype(dtype)
            if isinstance(embeddings, torch.Tensor)
            else np.asarray(embeddings, dtype=dtype)
        )

    if arr.ndim != 2:
        raise ShapeMismatchError(f"Expected a 2D array, but got a {arr.ndim}D array.")

    if np.prod(arr.shape) == 0:
        raise ShapeMismatchError(f"Array has at least one zero dimension: {arr.shape}.")

    if unit_interval and (arr.min() < 0 or arr.max() > 1):
        if unit_interval == "force":
            _logger.warning("Embeddings are not unit interval [0, 1]. Forcing to unit interval.")
            arr = rescale_array(arr)
        else:
            raise ValueError("Embeddings must be unit interval [0, 1].")

    if dtype is None:
        return embeddings
    return arr


@overload
def flatten_samples(array: torch.Tensor) -> torch.Tensor: ...
@overload
def flatten_samples(array: SequenceLike[Any]) -> NDArray[Any]: ...


def flatten_samples(array: SequenceLike[Any] | torch.Tensor) -> NDArray[Any] | torch.Tensor:
    """
    Flattens input array from (N, ...) to (N, -1) where all samples N have all data in their last dimension.

    Parameters
    ----------
    array : ArrayLike
        Input array with shape (N, ...)

    Returns
    -------
    np.ndarray or torch.Tensor
        Flattened array with shape (N, -1)

    Raises
    ------
    TypeError
        If array type is not supported
    """
    if isinstance(array, torch.Tensor):
        return torch.flatten(array, start_dim=1)
    if isinstance(array, str | bytes):
        raise TypeError(f"Unsupported array type {type(array)}.")
    try:
        nparr = as_numpy(array)
        return nparr.reshape((nparr.shape[0], -1))
    except (TypeError, ValueError) as e:
        raise TypeError(f"Unsupported array type {type(array)}: {e}.") from e


def channels_first_to_last(array: TArray) -> TArray:
    """
    Convert array from channels first to channels last format.

    Parameters
    ----------
    array : ArrayLike
        Input array in CHW format

    Returns
    -------
    ArrayLike
        Converted array in HWC format

    Raises
    ------
    TypeError
        If array type is not supported
    """
    if isinstance(array, np.ndarray):
        return np.transpose(array, (1, 2, 0))
    if isinstance(array, torch.Tensor):
        return torch.permute(array, (1, 2, 0))
    raise TypeError(f"Unsupported array type {type(array)}.")


TYPE_MAP = {int: 0, float: 1, str: 2}


@overload
def simplify_type(data: list[str]) -> list[int] | list[float] | list[str]: ...
@overload
def simplify_type(data: str) -> int | float | str: ...


def simplify_type(data: list[str] | str) -> list[int] | list[float] | list[str] | int | float | str:
    """
    Simplify a value or a list of values to the simplest form possible.

    In preferred order of `int`, `float`, or `string`.

    Parameters
    ----------
    data : list[str] | str
        A list of values or a single value

    Returns
    -------
    list[int | float | str] | int | float | str
        The same values converted to the numerical type if possible
    """
    if not isinstance(data, list):
        try:
            value = float(data)
        except (TypeError, ValueError):
            value = None
        return str(data) if value is None else int(value) if value.is_integer() else value

    converted = []
    max_type = 0
    for value in data:
        value = simplify_type(value)
        max_type = max(max_type, TYPE_MAP.get(type(value), 2))
        converted.append(value)
    for i in range(len(converted)):
        converted[i] = list(TYPE_MAP)[max_type](converted[i])
    return converted


def _get_key_indices(keys: Iterable[tuple[str, ...]]) -> dict[tuple[str, ...], int]:
    """
    Find indices to minimize unique tuple keys.

    Parameters
    ----------
    keys : Iterable[tuple[str, ...]]
        Collection of unique expanded tuple keys

    Returns
    -------
    dict[tuple[str, ...], int]
        Mapping of tuple keys to starting index
    """
    indices = dict.fromkeys(keys, -1)
    ks = list(keys)
    while len(ks) > 0:
        seen: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        for k in ks:
            seen.setdefault(k[indices[k] :], []).append(k)
        ks.clear()
        for sk in seen.values():
            if len(sk) > 1:
                ks.extend(sk)
                for k in sk:
                    indices[k] -= 1
    return indices


class DropReason(Enum):
    INCONSISTENT_KEY = "inconsistent_key"
    INCONSISTENT_SIZE = "inconsistent_size"
    NESTED_LIST = "nested_list"


def sorted_drop_reasons(d: dict[str, set[DropReason]]) -> dict[str, list[str]]:
    return {k: sorted({vv.value for vv in v}) for k, v in sorted(d.items(), key=lambda item: item[1])}


def flatten_dict_inner(
    d: Mapping[str, Any],
    dropped: dict[tuple[str, ...], set[DropReason]],
    parent_keys: tuple[str, ...],
    size: int | None = None,
    nested: bool = False,
) -> tuple[dict[tuple[str, ...], Any], int | None]:
    """
    Recursive internal function for flattening a dictionary.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary to flatten
    dropped: set[tuple[str, ...]]
        Reference to set of dropped keys from the dictionary
    parent_keys : tuple[str, ...]
        Parent keys to the current dictionary being flattened
    size : int or None, default None
        Tracking int for length of lists
    nested : bool, default False
        Tracking if inside a list

    Returns
    -------
    tuple[dict[tuple[str, ...], Any], int | None]
        - [0]: Dictionary of flattened values with the keys reformatted as a
               hierarchical tuple of strings
        - [1]: Size, if any, of the current list of values
    """
    items: dict[tuple[str, ...], Any] = {}
    for k, v in d.items():
        new_keys: tuple[str, ...] = parent_keys + (k,)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, dict):
            fd, size = flatten_dict_inner(v, dropped, new_keys, size=size, nested=nested)
            items.update(fd)
        elif isinstance(v, list | tuple):
            if nested:
                dropped.setdefault(parent_keys + (k,), set()).add(DropReason.NESTED_LIST)
            elif size is not None and size != len(v):
                dropped.setdefault(parent_keys + (k,), set()).add(DropReason.INCONSISTENT_SIZE)
            else:
                size = len(v)
                if all(isinstance(i, dict) for i in v):
                    for sub_dict in v:
                        fd, size = flatten_dict_inner(sub_dict, dropped, new_keys, size=size, nested=True)
                        for fk, fv in fd.items():
                            items.setdefault(fk, []).append(fv)
                else:
                    items[new_keys] = v
        else:
            items[new_keys] = v
    return items, size


@overload
def flatten_metadata(
    d: Mapping[str, Any],
    return_dropped: Literal[True],
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
) -> tuple[dict[str, Any], int, dict[str, list[str]]]: ...


@overload
def flatten_metadata(
    d: Mapping[str, Any],
    return_dropped: Literal[False] = False,
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
) -> tuple[dict[str, Any], int]: ...


def flatten_metadata(
    d: Mapping[str, Any],
    return_dropped: bool = False,
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
):
    """
    Flattens a nested metadata dictionary and converts values to numeric values when possible.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary to flatten
    return_dropped: bool, default False
        Option to return a dictionary of dropped keys and the reason(s) for dropping
    sep : str, default "_"
        String separator to use when concatenating key names
    ignore_lists : bool, default False
        Option to skip expanding lists within metadata
    fully_qualified : bool, default False
        Option to return dictionary keys fully qualified instead of reduced

    Returns
    -------
    dict[str, Any]
        Dictionary of flattened values with the keys reformatted as strings
    int
        Size of the values in the flattened dictionary
    dict[str, list[str]], Optional
        Dictionary containing dropped keys and reason(s) for dropping
    """
    dropped_inner: dict[tuple[str, ...], set[DropReason]] = {}
    expanded, size = flatten_dict_inner(d, dropped=dropped_inner, parent_keys=(), nested=ignore_lists)

    output = {}
    for k, v in expanded.items():
        cv = simplify_type(v)
        if isinstance(cv, list):
            if len(cv) == size:
                output[k] = cv
            else:
                dropped_inner.setdefault(k, set()).add(DropReason.INCONSISTENT_KEY)
        else:
            output[k] = cv if not size else [cv] * size

    if fully_qualified:
        output = {sep.join(k): v for k, v in output.items()}
    else:
        keys = _get_key_indices(output)
        output = {sep.join(k[keys[k] :]): v for k, v in output.items()}

    size = size if size is not None else 1
    dropped = {sep.join(k): v for k, v in dropped_inner.items()}

    if return_dropped:
        return output, size, sorted_drop_reasons(dropped)
    if dropped:
        dropped_items = "\n".join([f"    {k}: {v}" for k, v in sorted_drop_reasons(dropped).items()])
        _logger.warning(f"Metadata entries were dropped:\n{dropped_items}")
    return output, size


def _flatten_for_merge(
    metadatum: Mapping[str, Any],
    ignore_lists: bool,
    fully_qualified: bool,
    targets: int | None,
) -> tuple[dict[str, list[Any]] | dict[str, Any], int, dict[str, list[str]]]:
    flattened, image_repeats, dropped_inner = flatten_metadata(
        metadatum,
        return_dropped=True,
        ignore_lists=ignore_lists,
        fully_qualified=fully_qualified,
    )
    if targets is not None:
        # check for mismatch in targets per image and force ignore_lists
        if not ignore_lists and targets != image_repeats:
            flattened, image_repeats, dropped_inner = flatten_metadata(
                metadatum,
                return_dropped=True,
                ignore_lists=True,
                fully_qualified=fully_qualified,
            )
        if targets != image_repeats:
            flattened = {k: [v] * targets for k, v in flattened.items()}
        image_repeats = targets
    return flattened, image_repeats, dropped_inner


def _merge(
    dicts: list[Mapping[str, Any]],
    ignore_lists: bool,
    fully_qualified: bool,
    targets_per_image: Sequence[int] | None,
) -> tuple[dict[str, list[Any]], dict[str, set[DropReason]], NDArray[np.intp]]:
    merged: dict[str, list[Any]] = {}
    isect: set[str] = set()
    union: set[str] = set()
    image_repeats = np.zeros(len(dicts), dtype=np.intp)
    dropped: dict[str, set[DropReason]] = {}
    for i, d in enumerate(dicts):
        targets = None if targets_per_image is None else targets_per_image[i]
        if targets == 0:
            continue
        flattened, image_repeats[i], dropped_inner = _flatten_for_merge(d, ignore_lists, fully_qualified, targets)
        isect = isect.intersection(flattened.keys()) if isect else set(flattened.keys())
        union.update(flattened.keys())
        for k, v in dropped_inner.items():
            dropped.setdefault(k, set()).update({DropReason(vv) for vv in v})
        for k, v in flattened.items():
            merged.setdefault(k, []).extend(flattened[k]) if isinstance(v, list) else merged.setdefault(k, []).append(v)

    for k in union - isect:
        dropped.setdefault(k, set()).add(DropReason.INCONSISTENT_KEY)

    if image_repeats.sum() == image_repeats.size:
        image_indices = np.arange(image_repeats.size)
    else:
        image_ids = np.arange(image_repeats.size)
        image_data = np.concatenate(
            [np.repeat(image_ids[i], image_repeats[i]) for i in range(image_ids.size)],
            dtype=np.intp,
        )
        _, image_unsorted = np.unique(image_data, return_inverse=True)
        image_indices = np.sort(image_unsorted)

    merged = {k: simplify_type(v) for k, v in merged.items() if k in isect}
    return merged, dropped, image_indices


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[True],
    return_numpy: Literal[False] = False,
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> tuple[dict[str, list[Any]], dict[str, list[str]]]: ...


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[False] = False,
    return_numpy: Literal[False] = False,
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> dict[str, list[Any]]: ...


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[True],
    return_numpy: Literal[True],
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> tuple[dict[str, NDArray[Any]], dict[str, list[str]]]: ...


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[False] = False,
    return_numpy: Literal[True],
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> dict[str, NDArray[Any]]: ...


def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: bool = False,
    return_numpy: bool = False,
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
):
    """
    Merge a collection of metadata dictionaries into a single flattened dictionary.

    Nested dictionaries are flattened, and lists are expanded. Nested lists are
    dropped as the expanding into multiple hierarchical trees is not supported.
    The function adds an internal "_image_index" key to the metadata dictionary
    used by the `Metadata` class.

    Parameters
    ----------
    metadata : Iterable[Mapping[str, Any]]
        Iterable collection of metadata dictionaries to flatten and merge
    return_dropped: bool, default False
        Option to return a dictionary of dropped keys and the reason(s) for dropping
    return_numpy : bool, default False
        Option to return results as lists or NumPy arrays
    ignore_lists : bool, default False
        Option to skip expanding lists within metadata
    fully_qualified : bool, default False
        Option to return dictionary keys full qualified instead of minimized
    targets_per_image : Sequence[int] or None, default None
        Number of targets for each image metadata entry
    image_index_key : str, default "_image_index"
        User provided metadata key which maps the metadata entry to the source image.

    Returns
    -------
    dict[str, list[Any]] | dict[str, NDArray[Any]]
        A single dictionary containing the flattened data as lists or NumPy arrays
    dict[str, list[str]], Optional
        Dictionary containing dropped keys and reason(s) for dropping

    Notes
    -----
    Nested lists of values and inconsistent keys are dropped in the merged
    metadata dictionary

    Example
    -------
    >>> list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4}], "source": "example"}]
    >>> reorganized_metadata, dropped_keys = merge_metadata(list_metadata, return_dropped=True)
    >>> reorganized_metadata
    {'common': [1, 1], 'a': [1, 2], 'b': [3, 4], 'source': ['example', 'example'], '_image_index': [0, 0]}
    >>> dropped_keys
    {'target_c': ['inconsistent_key']}
    """
    dicts: list[Mapping[str, Any]] = list(metadata)

    if targets_per_image is not None and len(dicts) != len(targets_per_image):
        raise ValueError("Number of targets per image must be equal to number of metadata entries.")

    merged, dropped, image_indices = _merge(dicts, ignore_lists, fully_qualified, targets_per_image)

    output: dict[str, Any] = {k: np.asarray(v) for k, v in merged.items()} if return_numpy else merged

    if image_index_key not in output:
        output[image_index_key] = image_indices if return_numpy else image_indices.tolist()

    if return_dropped:
        return output, sorted_drop_reasons(dropped)

    if dropped:
        dropped_items = "\n".join([f"    {k}: {v}" for k, v in sorted_drop_reasons(dropped).items()])
        _logger.warning(f"Metadata entries were dropped:\n{dropped_items}")

    return output


R = TypeVar("R")
T = TypeVar("T")

# fork is fastest (no serialization) and safe on Linux.
# macOS defaults to spawn (fork unsafe with Objective-C runtime).
# Windows only supports spawn.
DEFAULT_CONTEXT: Literal["fork", "spawn"] = "fork" if sys.platform == "linux" else "spawn"


class PoolWrapper:
    """
    Wrap pool executors to allow easy switching between multiprocessing and single-threaded execution.

    Defaults to 'fork' on Linux (fastest, no serialization overhead) and 'spawn' elsewhere.
    Also supports 'threads' for workloads where the GIL is released during computation.
    """

    def __init__(self, processes: int | None, context: Literal["fork", "spawn"] = DEFAULT_CONTEXT) -> None:
        procs = 1 if processes is None else max(1, (cpu_count() or 1) + processes + 1) if processes < 0 else processes
        self._pool = multiprocessing.get_context(context).Pool(procs) if procs > 1 else None

    def imap_unordered(self, func: Callable[[T], R], iterable: Iterable[T]) -> Iterator[R]:
        """Apply `func` to each item in `iterable`, optionally using a pool."""
        return map(func, iterable) if self._pool is None else self._pool.imap_unordered(func, iterable)

    def __enter__(self, *args: Any, **kwargs: Any) -> Self:
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the runtime context and clean up the pool if it was created."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()

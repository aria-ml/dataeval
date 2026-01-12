"""
Utility functions for array conversion and manipulation across different frameworks.
"""

__all__ = [
    "as_numpy",
    "to_numpy",
    "opt_as_numpy",
    "opt_to_numpy",
    "flatten_samples",
    "ensure_embeddings",
    "rescale_array",
]

import logging
from collections.abc import Iterable, Iterator
from importlib import import_module
from types import ModuleType
from typing import Any, Literal, TypeVar, overload

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from dataeval._log import LogMessage
from dataeval.protocols import Array, SequenceLike

_logger = logging.getLogger(__name__)

_MODULE_CACHE = {}

T = TypeVar("T", Array, np.ndarray, torch.Tensor)
_np_dtype = TypeVar("_np_dtype", bound=np.generic)


def _try_import(module_name: str) -> ModuleType | None:
    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    try:
        module = import_module(module_name)
    except ImportError:  # pragma: no cover
        _logger.log(logging.INFO, f"Unable to import {module_name}.")
        module = None

    _MODULE_CACHE[module_name] = module
    return module


def opt_as_numpy(
    array: ArrayLike | SequenceLike[Any] | None,
    *,
    dtype: type[_np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
) -> NDArray[_np_dtype] | None:
    """
    Converts an ArrayLike to Numpy array without copying (if possible), returns None if input is None.

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
    dtype: type[_np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
    copy: bool = True,
) -> NDArray[_np_dtype] | None:
    """
    Converts an ArrayLike to Numpy array, returns None if input is None.

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
    dtype: type[_np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
) -> NDArray[_np_dtype]:
    """
    Converts an ArrayLike to Numpy array without copying (if possible).

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
    dtype: type[_np_dtype] | None = None,
    required_ndim: int | Iterable[int] | None = None,
    required_shape: tuple[int, ...] | None = None,
    copy: bool = True,
) -> NDArray[_np_dtype]:
    """
    Converts an ArrayLike to new Numpy array.

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
    ValueError
        If required_ndim or required_shape constraints are not met
    """
    _array: NDArray[_np_dtype] | None = None

    if array is None:
        _array = np.array([], dtype=dtype)
    elif isinstance(array, np.ndarray | np.memmap):
        _array = array.copy().astype(dtype) if copy else array
    elif array.__class__.__module__.startswith("tensorflow"):  # pragma: no cover - removed tf from deps
        tf = _try_import("tensorflow")
        if tf and tf.is_tensor(array):
            _logger.log(logging.INFO, "Converting Tensorflow array to NumPy array.")
            _array = array.numpy().copy().astype(dtype) if copy else array.numpy().astype(dtype)  # type: ignore
    elif array.__class__.__module__.startswith("torch"):
        torch = _try_import("torch")
        if torch and isinstance(array, torch.Tensor):
            _logger.log(logging.INFO, "Converting PyTorch array to NumPy array.")
            numpy = array.detach().cpu().numpy().copy() if copy else array.detach().cpu().numpy()  # type: ignore
            _logger.log(logging.DEBUG, LogMessage(lambda: f"{str(array)} -> {str(numpy)}"))
            _array = numpy.astype(dtype)

    # If the array was not converted yet, let numpy create the array directly
    if _array is None:
        _array = np.array(array, dtype=dtype) if copy else np.asarray(array, dtype=dtype)

    required_ndims = (required_ndim,) if isinstance(required_ndim, int) else required_ndim
    if required_ndims is not None and _array.ndim not in required_ndims:
        raise ValueError(f"Array has {_array.ndim} dimensions, expected {required_ndim}.")

    if required_shape is not None and _array.shape != required_shape:
        raise ValueError(f"Array has shape {_array.shape}, expected {required_shape}.")

    return _array


def to_numpy_iter(iterable: Iterable[ArrayLike]) -> Iterator[NDArray[Any]]:
    """
    Yields an iterator of numpy arrays from an ArrayLike iterable.

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
def rescale_array(array: NDArray[_np_dtype]) -> NDArray[_np_dtype]: ...
@overload
def rescale_array(array: torch.Tensor) -> torch.Tensor: ...
def rescale_array(array: Array | NDArray[_np_dtype] | torch.Tensor) -> Array | NDArray[_np_dtype] | torch.Tensor:
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


@overload
def ensure_embeddings(
    embeddings: T,
    dtype: torch.dtype,
    unit_interval: Literal[True, False, "force"] = False,
) -> torch.Tensor: ...


@overload
def ensure_embeddings(
    embeddings: T,
    dtype: type[_np_dtype],
    unit_interval: Literal[True, False, "force"] = False,
) -> NDArray[_np_dtype]: ...


@overload
def ensure_embeddings(
    embeddings: T,
    dtype: None = None,
    unit_interval: Literal[True, False, "force"] = False,
) -> T: ...


def ensure_embeddings(
    embeddings: T,
    dtype: type[_np_dtype] | torch.dtype | None = None,
    unit_interval: Literal[True, False, "force"] = False,
) -> torch.Tensor | NDArray[_np_dtype] | T:
    """
    Validates the embeddings array and converts it to the specified type.

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
    torch.Tensor or NDArray or T
        Converted embeddings array

    Raises
    ------
    ValueError
        If the embeddings array is not 2D
    ValueError
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
        raise ValueError(f"Expected a 2D array, but got a {arr.ndim}D array.")

    if np.prod(arr.shape) == 0:
        raise ValueError(f"Array has at least one zero dimension: {arr.shape}.")

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
    try:
        nparr = as_numpy(array)
        return nparr.reshape((nparr.shape[0], -1))
    except Exception as e:
        raise TypeError(f"Unsupported array type {type(array)}: {e}.")


_TArray = TypeVar("_TArray", bound=Array)


def channels_first_to_last(array: _TArray) -> _TArray:
    """
    Converts array from channels first to channels last format.

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

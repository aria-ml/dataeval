from __future__ import annotations

__all__ = []

import logging
import warnings
from importlib import import_module
from types import ModuleType
from typing import Any, Iterable, Iterator, Literal, TypeVar, overload

import numpy as np
import torch
from numpy.typing import NDArray

from dataeval._log import LogMessage
from dataeval.typing import ArrayLike

_logger = logging.getLogger(__name__)

_MODULE_CACHE = {}

T = TypeVar("T", ArrayLike, np.ndarray, torch.Tensor)
_np_dtype = TypeVar("_np_dtype", bound=np.generic)


def _try_import(module_name) -> ModuleType | None:
    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    try:
        module = import_module(module_name)
    except ImportError:  # pragma: no cover
        _logger.log(logging.INFO, f"Unable to import {module_name}.")
        module = None

    _MODULE_CACHE[module_name] = module
    return module


def as_numpy(array: ArrayLike | None) -> NDArray[Any]:
    """Converts an ArrayLike to Numpy array without copying (if possible)"""
    return to_numpy(array, copy=False)


def to_numpy(array: ArrayLike | None, copy: bool = True) -> NDArray[Any]:
    """Converts an ArrayLike to new Numpy array"""
    if array is None:
        return np.ndarray([])

    if isinstance(array, np.ndarray):
        return array.copy() if copy else array

    if array.__class__.__module__.startswith("tensorflow"):  # pragma: no cover - removed tf from deps
        tf = _try_import("tensorflow")
        if tf and tf.is_tensor(array):
            _logger.log(logging.INFO, "Converting Tensorflow array to NumPy array.")
            return array.numpy().copy() if copy else array.numpy()  # type: ignore

    if array.__class__.__module__.startswith("torch"):
        torch = _try_import("torch")
        if torch and isinstance(array, torch.Tensor):
            _logger.log(logging.INFO, "Converting PyTorch array to NumPy array.")
            numpy = array.detach().cpu().numpy().copy() if copy else array.detach().cpu().numpy()  # type: ignore
            _logger.log(logging.DEBUG, LogMessage(lambda: f"{str(array)} -> {str(numpy)}"))
            return numpy

    return np.array(array) if copy else np.asarray(array)


def to_numpy_iter(iterable: Iterable[ArrayLike]) -> Iterator[NDArray[Any]]:
    """Yields an iterator of numpy arrays from an ArrayLike"""
    for array in iterable:
        yield to_numpy(array)


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
    dtype: None,
    unit_interval: Literal[True, False, "force"] = False,
) -> T: ...


def ensure_embeddings(
    embeddings: T,
    dtype: type[_np_dtype] | torch.dtype | None = None,
    unit_interval: Literal[True, False, "force"] = False,
) -> torch.Tensor | NDArray[_np_dtype] | T:
    """
    Validates the embeddings array and converts it to the specified type

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
        Converted embeddings array

    Raises
    ------
    ValueError
        If the embeddings array is not 2D
    ValueError
        If the embeddings array is not unit interval [0, 1]
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

    if unit_interval:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_min < 0 or arr_max > 1:
            if unit_interval == "force":
                warnings.warn("Embeddings are not unit interval [0, 1]. Forcing to unit interval.")
                arr = (arr - arr_min) / (arr_max - arr_min)
            else:
                raise ValueError("Embeddings must be unit interval [0, 1].")

    if dtype is None:
        return embeddings
    else:
        return arr


def flatten(array: ArrayLike) -> NDArray[Any]:
    """
    Flattens input array from (N, ... ) to (N, -1) where all samples N have all data in their last dimension

    Parameters
    ----------
    X : NDArray, shape - (N, ... )
        Input array

    Returns
    -------
    NDArray, shape - (N, -1)
    """
    nparr = as_numpy(array)
    return nparr.reshape((nparr.shape[0], -1))

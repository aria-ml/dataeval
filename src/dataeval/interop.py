from __future__ import annotations

__all__ = ["as_numpy", "to_numpy", "to_numpy_iter"]

from importlib import import_module
from typing import Any, Iterable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MODULE_CACHE = {}


def _try_import(module_name):
    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    try:
        module = import_module(module_name)
    except ImportError:  # pragma: no cover - covered by test_mindeps.py
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

    if array.__class__.__module__.startswith("tensorflow"):
        tf = _try_import("tensorflow")
        if tf and tf.is_tensor(array):
            return array.numpy().copy() if copy else array.numpy()  # type: ignore

    if array.__class__.__module__.startswith("torch"):
        torch = _try_import("torch")
        if torch and isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy().copy() if copy else array.detach().cpu().numpy()  # type: ignore

    return np.array(array, copy=copy)


def to_numpy_iter(iterable: Iterable[ArrayLike]) -> Iterator[NDArray[Any]]:
    """Yields an iterator of numpy arrays from an ArrayLike"""
    for array in iterable:
        yield to_numpy(array)

from __future__ import annotations

from types import ModuleType

from dataeval.logging import LogMessage

__all__ = ["as_numpy", "to_numpy", "to_numpy_iter"]

import logging
from importlib import import_module
from typing import Any, Iterable, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

_logger = logging.getLogger(__name__)

_MODULE_CACHE = {}


def _try_import(module_name) -> ModuleType | None:
    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    try:
        module = import_module(module_name)
    except ImportError:  # pragma: no cover - covered by test_mindeps.py
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

    if array.__class__.__module__.startswith("tensorflow"):
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

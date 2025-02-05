"""Utility functions for interoperability with different array types."""

from __future__ import annotations

__all__ = []

import logging
from importlib import import_module
from types import ModuleType
from typing import Any, Iterable, Iterator, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._log import LogMessage

_logger = logging.getLogger(__name__)

_MODULE_CACHE = {}
TYPE_MAP = {int: 0, float: 1, str: 2}
T = TypeVar("T")


def _try_cast(v: Any, t: type[T]) -> T | None:
    """Casts a value to a type or returns None if unable"""
    try:
        return t(v)  # type: ignore
    except (TypeError, ValueError):
        return None


@overload
def _simplify_type(data: list[str]) -> list[int] | list[float] | list[str]: ...
@overload
def _simplify_type(data: str) -> int | float | str: ...


def _simplify_type(data: list[str] | str) -> list[int] | list[float] | list[str] | int | float | str:
    """
    Simplifies a value or a list of values to the simplest form possible,
    in preferred order of `int`, `float`, or `string`.

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
        value = _try_cast(data, float)
        return str(data) if value is None else int(value) if value.is_integer() else value

    converted = []
    max_type = 0
    for value in data:
        value = _simplify_type(value)
        max_type = max(max_type, TYPE_MAP.get(type(value), 2))
        converted.append(value)
    for i in range(len(converted)):
        converted[i] = list(TYPE_MAP)[max_type](converted[i])
    return converted


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

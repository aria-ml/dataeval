from __future__ import annotations

from importlib import import_module
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

module_cache = {}


def try_import(module_name):
    if module_name in module_cache:
        return module_cache[module_name]

    try:
        module = import_module(module_name)
    except ImportError:  # pragma: no cover - covered by test_mindeps.py
        module = None

    module_cache[module_name] = module
    return module


def to_numpy(array: ArrayLike | None) -> NDArray:
    if array is None:
        return np.ndarray([])

    if isinstance(array, np.ndarray):
        return array

    tf = try_import("tensorflow")
    if tf and tf.is_tensor(array):
        return array.numpy()  # type: ignore

    torch = try_import("torch")
    if torch and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()  # type: ignore

    return np.asarray(array)


def to_numpy_iter(iterable: Iterable[ArrayLike]):
    for array in iterable:
        yield to_numpy(array)

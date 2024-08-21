from importlib import import_module
from typing import Any, Iterable, Optional, runtime_checkable

import numpy as np

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


try:
    from maite.protocols import ArrayLike  # type: ignore
except ImportError:  # pragma: no cover - covered by test_mindeps.py
    from typing import Protocol

    @runtime_checkable
    class ArrayLike(Protocol):
        def __array__(self) -> Any: ...


def to_numpy(array: Optional[ArrayLike]) -> np.ndarray:
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

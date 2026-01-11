"""
Global configuration settings for DataEval.
"""

from __future__ import annotations

__all__ = [
    "get_device",
    "set_device",
    "use_device",
    "get_batch_size",
    "set_batch_size",
    "use_batch_size",
    "get_max_processes",
    "set_max_processes",
    "use_max_processes",
]

from typing import Any

import numpy as np
import torch

from dataeval.protocols import DeviceLike

### GLOBALS ###

_device: torch.device | None = None
_processes: int | None = None
_seed: int | None = None
_batch_size: int | None = None

### CONSTS ###

EPSILON = 1e-12


### CONTEXT MANAGER ###


class _ConfigContextManager:
    """Generic context manager for temporarily overriding configuration values."""

    def __init__(self, global_name: str, value: Any) -> None:
        self._global_name = global_name
        self._value = value
        self._old: Any = None

    def __enter__(self) -> None:
        self._old = globals()[self._global_name]
        globals()[self._global_name] = self._value

    def __exit__(self, *args: tuple[Any, ...]) -> None:
        globals()[self._global_name] = self._old


### FUNCS ###


def _todevice(device: DeviceLike) -> torch.device:
    return torch.device(*device) if isinstance(device, tuple) else torch.device(device)


def set_device(device: DeviceLike | None) -> None:
    """
    Sets the default device to use when executing against a PyTorch backend.

    Parameters
    ----------
    device : DeviceLike or None
        The default device to use. See documentation for more information.

    See Also
    --------
    `torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_
    """
    global _device
    _device = None if device is None else _todevice(device)


def get_device(override: DeviceLike | None = None) -> torch.device:
    """
    Returns the PyTorch device to use.

    Parameters
    ----------
    override : DeviceLike or None, default None
        The user specified override if provided, otherwise returns the default device.

    Returns
    -------
    `torch.device`
    """
    if override is None:
        global _device
        return (
            _device
            if _device is not None
            else torch.get_default_device()
            if hasattr(torch, "get_default_device")
            else torch.device("cpu")
        )
    return _todevice(override)


def use_device(device: DeviceLike) -> _ConfigContextManager:
    return _ConfigContextManager("_device", None if device is None else _todevice(device))


def set_batch_size(batch_size: int | None) -> None:
    """
    Sets the default batch size to use when processing data.

    Parameters
    ----------
    batch_size : int or None
        The default batch size to use. None will unset the global batch size.
    """
    global _batch_size
    _batch_size = batch_size


def get_batch_size(override: int | None = None) -> int:
    """
    Returns the batch size to use.

    Parameters
    ----------
    override : int or None, default None
        The user specified override if provided, otherwise returns the global batch size.

    Returns
    -------
    int
        The batch size to use.

    Raises
    ------
    ValueError
        If no batch size is provided and no global batch size is set.
    ValueError
        If the batch size is less than 1.
    """
    if override is not None:
        return override

    global _batch_size
    if _batch_size is None:
        raise ValueError(
            "No batch_size provided. Either pass batch_size as a parameter to the call "
            "or set a global batch_size using dataeval.config.set_batch_size()."
        )
    if _batch_size < 1:
        raise ValueError("Provided batch_size must be greater than 0.")
    return _batch_size


def use_batch_size(batch_size: int) -> _ConfigContextManager:
    return _ConfigContextManager("_batch_size", batch_size)


def set_max_processes(processes: int | None) -> None:
    """
    Sets the maximum number of worker processes to use when running tasks that support parallel processing.

    Parameters
    ----------
    processes : int or None
        The maximum number of worker processes to use, or -1 to use
        `os.cpu_count <https://docs.python.org/3/library/os.html#os.cpu_count>`_
        to determine the number of worker processes. For negative values less than -1,
        the number of worker processes will be set to `max(1, cpu_count + processes + 1)`.
        None is unset, and defaults to 1 process.

    Raises
    ------
    ValueError
        If `processes` is zero.

    See Also
    --------
    `n_jobs` (scikit-learn): https://scikit-learn.org/stable/glossary.html#term-n_jobs
    """
    if processes == 0:
        raise ValueError("processes cannot be zero; use None to default to CPU count.")
    global _processes
    _processes = processes


def get_max_processes() -> int | None:
    """
    Returns the maximum number of worker processes to use when running tasks that support parallel processing.

    Returns
    -------
    int or None
        The maximum number of worker processes to use. None is unset, and defaults to 1 process.

    See Also
    --------
    `n_jobs` (scikit-learn): https://scikit-learn.org/stable/glossary.html#term-n_jobs
    """
    global _processes
    return _processes


def use_max_processes(processes: int) -> _ConfigContextManager:
    return _ConfigContextManager("_processes", processes)


def set_seed(seed: int | None, all_generators: bool = False) -> None:
    """
    Sets the seed for use by classes that allow for a random state or seed.

    Parameters
    ----------
    seed : int or None
        The seed to use.
    all_generators : bool, default False
        Whether to set the seed for all generators, including NumPy and PyTorch.
    """
    global _seed
    _seed = seed

    if all_generators:
        np.random.seed(seed)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            torch.seed()
            torch.cuda.seed_all()


def get_seed() -> int | None:
    """
    Returns the seed for random state or seed.

    Returns
    -------
    int or None
        The seed to use.
    """
    global _seed
    return _seed

"""
Global configuration settings for DataEval.
"""

from __future__ import annotations

__all__ = ["get_device", "set_device", "get_max_processes", "set_max_processes", "use_max_processes"]

from typing import Any

import numpy as np
import torch

from dataeval.protocols import DeviceLike

### GLOBALS ###

_device: torch.device | None = None
_processes: int | None = None
_seed: int | None = None

### CONSTS ###

EPSILON = 1e-12

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


class MaxProcessesContextManager:
    def __init__(self, processes: int) -> None:
        self._processes = processes

    def __enter__(self) -> None:
        global _processes
        self._old = _processes
        set_max_processes(self._processes)

    def __exit__(self, *args: tuple[Any, ...]) -> None:
        global _processes
        _processes = self._old


def use_max_processes(processes: int) -> MaxProcessesContextManager:
    return MaxProcessesContextManager(processes)


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
        torch.manual_seed(seed)


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

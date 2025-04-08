"""
Global configuration settings for DataEval.
"""

from __future__ import annotations

__all__ = ["get_device", "set_device", "get_max_processes", "set_max_processes", "DeviceLike"]

import sys
from typing import Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy as np
import torch

### GLOBALS ###

_device: torch.device | None = None
_processes: int | None = None
_seed: int | None = None

### CONSTS ###

EPSILON = 1e-10

### TYPES ###

DeviceLike: TypeAlias = Union[int, str, tuple[str, int], torch.device]
"""
Type alias for types that are acceptable for specifying a torch.device.

See Also
--------
`torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_
"""

### FUNCS ###


def _todevice(device: DeviceLike) -> torch.device:
    return torch.device(*device) if isinstance(device, tuple) else torch.device(device)


def set_device(device: DeviceLike) -> None:
    """
    Sets the default device to use when executing against a PyTorch backend.

    Parameters
    ----------
    device : DeviceLike
        The default device to use. See documentation for more information.

    See Also
    --------
    `torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_
    """
    global _device
    _device = _todevice(device)


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
        return torch.get_default_device() if _device is None else _device
    else:
        return _todevice(override)


def set_max_processes(processes: int | None) -> None:
    """
    Sets the maximum number of worker processes to use when running tasks that support parallel processing.

    Parameters
    ----------
    processes : int or None
        The maximum number of worker processes to use, or None to use
        `os.process_cpu_count <https://docs.python.org/3/library/os.html#os.process_cpu_count>`_
        to determine the number of worker processes.
    """
    global _processes
    _processes = processes


def get_max_processes() -> int | None:
    """
    Returns the maximum number of worker processes to use when running tasks that support parallel processing.

    Returns
    -------
    int or None
        The maximum number of worker processes to use, or None to use
        `os.process_cpu_count <https://docs.python.org/3/library/os.html#os.process_cpu_count>`_
        to determine the number of worker processes.
    """
    global _processes
    return _processes


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

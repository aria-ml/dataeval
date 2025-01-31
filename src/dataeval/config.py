"""
Global configuration settings for DataEval.
"""

from __future__ import annotations

__all__ = ["get_device", "set_device", "get_max_processes", "set_max_processes"]

import torch
from torch import device

_device: device | None = None
_processes: int | None = None


def set_device(device: str | device | int) -> None:
    """
    Sets the default device to use when executing against a PyTorch backend.

    Parameters
    ----------
    device : str or int or `torch.device`
        The default device to use. See `torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_
        documentation for more information.
    """
    global _device
    _device = torch.device(device)


def get_device(override: str | device | int | None = None) -> torch.device:
    """
    Returns the PyTorch device to use.

    Parameters
    ----------
    override : str or int or `torch.device` or None, default None
        The user specified override if provided, otherwise returns the default device.

    Returns
    -------
    `torch.device`
    """
    if override is None:
        global _device
        if _device is None:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _device
    else:
        return torch.device(override)


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

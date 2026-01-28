"""
Global configuration settings for DataEval.
"""

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
    "get_seed",
    "set_seed",
    "GlobalConfig",
]

from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, field_validator

from dataeval.protocols import DeviceLike

### CONSTS ###

EPSILON = 1e-12


### GLOBAL CONFIG ###


class GlobalConfig(BaseModel):
    """
    Global configuration for DataEval runtime settings.

    This Pydantic model backs the global configuration state and provides
    validation on assignment. Users typically interact with the functional
    API (get_*, set_*, use_*) rather than this class directly.

    Attributes
    ----------
    device : torch.device or None, default None
        Default PyTorch device for computations.
    batch_size : int or None, default None
        Default batch size for data processing.
    max_processes : int or None, default None
        Maximum number of worker processes for parallel tasks.
    seed : int or None, default None
        Random seed for reproducibility.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    device: torch.device | None = None
    batch_size: int | None = None
    max_processes: int | None = None
    seed: int | None = None

    @field_validator("max_processes")
    @classmethod
    def validate_max_processes(cls, v: int | None) -> int | None:
        if v == 0:
            raise ValueError("max_processes cannot be zero; use None to default to CPU count.")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("batch_size must be greater than 0.")
        return v


# Global config instance
_config = GlobalConfig()


### CONTEXT MANAGER ###


class _ConfigContextManager:
    """Generic context manager for temporarily overriding configuration values."""

    def __init__(self, attr_name: str, value: Any) -> None:
        self._attr_name = attr_name
        self._value = value
        self._old: Any = None

    def __enter__(self) -> None:
        self._old = getattr(_config, self._attr_name)
        setattr(_config, self._attr_name, self._value)

    def __exit__(self, *args: tuple[Any, ...]) -> None:
        setattr(_config, self._attr_name, self._old)


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
    _config.device = None if device is None else _todevice(device)


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
    if override is not None:
        return _todevice(override)
    if _config.device is not None:
        return _config.device
    if hasattr(torch, "get_default_device"):
        return torch.get_default_device()
    return torch.device("cpu")


def use_device(device: DeviceLike) -> _ConfigContextManager:
    """
    Context manager to temporarily override the default device.

    Parameters
    ----------
    device : DeviceLike
        The device to use within the context.

    Returns
    -------
    _ConfigContextManager
        Context manager that restores the previous device on exit.

    Examples
    --------
    >>> with use_device("cuda:0"):
    ...     # Operations here use cuda:0
    ...     get_device()
    device(type='cuda', index=0)
    >>> # Original device is restored
    >>> get_device()
    device(type='cpu')
    """
    return _ConfigContextManager("device", None if device is None else _todevice(device))


def set_batch_size(batch_size: int | None) -> None:
    """
    Sets the default batch size to use when processing data.

    Parameters
    ----------
    batch_size : int or None
        The default batch size to use. None will unset the global batch size.

    Raises
    ------
    ValueError
        If the batch size is less than 1.
    """
    _config.batch_size = batch_size


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
        if override < 1:
            raise ValueError("Provided batch_size must be greater than 0.")
        return override

    if _config.batch_size is None:
        raise ValueError(
            "No batch_size provided. Either pass batch_size as a parameter to the call "
            "or set a global batch_size using dataeval.config.set_batch_size()."
        )
    return _config.batch_size


def use_batch_size(batch_size: int) -> _ConfigContextManager:
    """
    Context manager to temporarily override the default batch size.

    Parameters
    ----------
    batch_size : int
        The batch size to use within the context.

    Returns
    -------
    _ConfigContextManager
        Context manager that restores the previous batch size on exit.
    """
    return _ConfigContextManager("batch_size", batch_size)


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
    _config.max_processes = processes


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
    return _config.max_processes


def use_max_processes(processes: int) -> _ConfigContextManager:
    """
    Context manager to temporarily override the maximum number of processes.

    Parameters
    ----------
    processes : int
        The maximum number of processes to use within the context.

    Returns
    -------
    _ConfigContextManager
        Context manager that restores the previous value on exit.
    """
    return _ConfigContextManager("max_processes", processes)


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
    _config.seed = seed

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
    return _config.seed

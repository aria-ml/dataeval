from __future__ import annotations

__version__ = "0.0.0"

import os
from importlib.util import find_spec

_IS_TORCH_AVAILABLE = find_spec("torch") is not None
_IS_TORCHVISION_AVAILABLE = find_spec("torchvision") is not None
_IS_TENSORFLOW_AVAILABLE = find_spec("tensorflow") is not None and find_spec("tensorflow_probability") is not None

del find_spec

_ENV_DATAEVAL_MAX_PROCESSES = "DATAEVAL_MAX_PROCESSES"
_ENV_DATAEVAL_DEFAULT_DEVICE = "DATAEVAL_DEFAULT_DEVICE"

_max_processes_env = os.environ.get(_ENV_DATAEVAL_MAX_PROCESSES, "")
MAX_PROCESSES = int(_max_processes_env) if _max_processes_env.isdigit() else os.cpu_count()
"""
Sets the maximum number of processes used when performing calculations
that support multiprocessing.  Defaults to `os.cpu_count()`.

Also configurable through the environment variable `DATAEVAL_MAX_PROCESSES`
"""

DEFAULT_DEVICE: str | None = os.environ.get(_ENV_DATAEVAL_DEFAULT_DEVICE)
"""
Sets the default device to use for functionality that supports both CPU and
GPU based processing.  Set to `cuda` or `gpu` to enable usage of the GPU.

Also configurable through the environment variable `DATAEVAL_DEFAULT_DEVICE`
"""

del os


def use_gpu() -> bool:
    return bool(DEFAULT_DEVICE and any(DEFAULT_DEVICE.lower() == gpu for gpu in ("gpu", "cuda")))


def max_processes() -> int | None:
    return MAX_PROCESSES


from dataeval import detectors, metrics  # noqa: E402

__all__ = ["MAX_PROCESSES", "DEFAULT_DEVICE", "detectors", "metrics"]

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval import workflows

    __all__ += ["workflows"]

if _IS_TENSORFLOW_AVAILABLE or _IS_TORCH_AVAILABLE:  # pragma: no cover
    from dataeval import utils

    __all__ += ["utils"]

"""Global configuration settings for DataEval."""

__all__ = [
    "get_device",
    "set_device",
    "use_device",
    "get_batch_size",
    "resolve_batch_size",
    "set_batch_size",
    "use_batch_size",
    "get_max_processes",
    "set_max_processes",
    "use_max_processes",
    "get_load_threads",
    "set_load_threads",
    "use_load_threads",
    "get_seed",
    "set_seed",
    "use_seed",
    "GlobalConfig",
]

from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, field_validator

from dataeval._experimental import warn_experimental_once
from dataeval.protocols import DeviceLike

# GLOBAL CONFIG ###


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
    load_threads : int or None, default None
        Number of threads that load dataset items at once while a whole-dataset walk reads them.
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
    load_threads: int | None = None
    seed: int | None = None

    @field_validator("max_processes")
    @classmethod
    def validate_max_processes(cls, v: int | None) -> int | None:
        """Validate that max_processes is not zero."""
        if v == 0:
            raise ValueError("max_processes cannot be zero; use None to default to CPU count.")
        return v

    @field_validator("load_threads")
    @classmethod
    def validate_load_threads(cls, v: int | None) -> int | None:
        """Validate that load_threads is at least one."""
        if v is not None and v < 1:
            raise ValueError("load_threads must be at least 1; use None to load serially.")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int | None:
        """Validate that batch_size is greater than zero."""
        if v is not None and v < 1:
            raise ValueError("batch_size must be greater than 0.")
        return v


# Global config instance
_config = GlobalConfig()


# CONTEXT MANAGER ###


class _ConfigContextManager:
    """Generic context manager for temporarily overriding configuration values."""

    def __init__(self, attr_name: str, value: Any) -> None:
        self._attr_name = attr_name
        self._value = value
        self._old: Any = None

    def __enter__(self) -> None:
        self._old = getattr(_config, self._attr_name)
        setattr(_config, self._attr_name, self._value)

    def __exit__(self, *args: object) -> None:
        setattr(_config, self._attr_name, self._old)


# FUNCS ###


def _todevice(device: DeviceLike) -> torch.device:
    return torch.device(*device) if isinstance(device, tuple) else torch.device(device)


def set_device(device: DeviceLike | None) -> None:
    """
    Set the default device to use when executing against a PyTorch backend.

    Parameters
    ----------
    device : DeviceLike or None
        The default device to use. See documentation for more information.

    See Also
    --------
    :class:`torch.device`
    """
    _config.device = None if device is None else _todevice(device)


def get_device(override: DeviceLike | None = None) -> torch.device:
    """
    Return the PyTorch device to use.

    Parameters
    ----------
    override : DeviceLike or None, default None
        The user specified override if provided, otherwise returns the default device.

    Returns
    -------
    :class:`torch.device`
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
    Set the default batch size to use when processing data.

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
    Return the batch size to use.

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
            "or set a global batch_size using dataeval.config.set_batch_size().",
        )
    return _config.batch_size


def resolve_batch_size(*candidates: int | None) -> int:
    """
    Resolve a batch size from an ordered list of candidates.

    Returns the first non-None candidate (validated > 0). If every candidate is
    None, falls back to the global batch size via :func:`get_batch_size`, which
    raises if no global is set. Precedence is caller order: the earliest
    non-None candidate wins (outer/override before inner/default).

    Parameters
    ----------
    *candidates : int or None
        Candidate batch sizes in priority order (highest priority first).

    Returns
    -------
    int
        The resolved batch size.

    Raises
    ------
    ValueError
        If no candidate is non-None and no global batch size is set.
    ValueError
        If a non-None candidate is less than 1.
    """
    for candidate in candidates:
        if candidate is not None:
            return get_batch_size(candidate)
    return get_batch_size()


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
    Set the maximum number of worker processes to use when running tasks that support parallel processing.

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
    Return the maximum number of worker processes to use when running tasks that support parallel processing.

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


_LOAD_THREADS_DETAILS = (
    "It helps when each dataset read takes more than a few milliseconds and releases the GIL, and it can be slower "
    "than the serial loop on small images. The dataset must be safe to read from several threads at once."
)


def set_load_threads(threads: int | None) -> None:
    """
    Set how many threads load dataset items at once.

    DataEval uses them wherever it reads a whole dataset item by item: :class:`~dataeval.Embeddings`,
    :class:`~dataeval.Metadata`, :func:`~dataeval.core.image_stats` and so :class:`~dataeval.quality.Outliers` and
    :class:`~dataeval.quality.Duplicates`, the extractors in :mod:`dataeval.extractors` called on a dataset and so the
    drift and OOD detectors given one, :class:`~dataeval.performance.Feasibility` given a model, and
    :func:`~dataeval.data.unzip_dataset`, which reads ahead while it is iterated. So do
    :class:`~dataeval.data.Filter`, :class:`~dataeval.data.Relabel`, :class:`~dataeval.data.ClassBalance` and
    :class:`~dataeval.data.DetectionCrops`, which read every item once when a view is built.

    Parameters
    ----------
    threads : int or None
        The number of threads. None, the default, and 1 read one item at a time on the calling thread, as before.

    Raises
    ------
    ValueError
        If `threads` is below one.

    See Also
    --------
    set_max_processes : The separate setting for computing, not reading: worker processes, such as the ones
        :func:`~dataeval.core.image_stats` computes in while it reads in these threads, and scikit-learn's ``n_jobs``.

    Notes
    -----
    **Parallelism.** With more than one thread, ``dataset[i]`` is called from several threads at once and in no fixed
    order. A dataset that is not safe for that, such as one that holds an open file handle, can return wrong data, or
    raise. For a dataset that is safe to read from several threads, results and their order are unchanged. Threads help
    when each read takes more than a few milliseconds and releases the GIL, such as decoding a large image, and can be
    slower than the serial loop when reads are short. A read holds up to about four times the thread count of items
    ahead of the one in use. A random transform inside the dataset draws from shared random state, so seeded results
    can differ between runs. A transform that a :class:`~dataeval.data.View` applies runs in those threads too, so one
    you wrote must be safe for that. It is not tested with datasets that land their data on a GPU, on free-threaded
    builds of Python, or on Windows or macOS.

    **Serial reads.** A dataset already in memory, such as a list or an array, and a multi-object tracking dataset are
    read one item at a time on the calling thread. So is a :class:`~dataeval.data.SequenceFrames`, whose frames come
    off each video's decoder in order, and any view, crops or merge DataEval builds on one. Behind a wrapper that is not
    DataEval's, a :class:`~dataeval.data.SequenceFrames` still reads the right frames but can decode a video again for
    each read out of order, so leave threads off for that. :func:`~dataeval.core.video_stats`,
    :func:`~dataeval.core.frame_stats` and :func:`~dataeval.core.ego_stats` measure several videos at once instead, in
    threads that :func:`set_max_processes` sizes.

    Setting more than one thread raises :class:`~dataeval.exceptions.ExperimentalWarning` once per process.

    .. versionadded:: 1.3
    """
    if threads is not None and threads > 1:
        warn_experimental_once("load_threads", details=_LOAD_THREADS_DETAILS)
    _config.load_threads = threads


def get_load_threads() -> int | None:
    """
    Return how many threads load dataset items at once, or None when items are read one at a time.

    Returns
    -------
    int or None
        The number of threads set with :func:`set_load_threads`.
    """
    return _config.load_threads


def use_load_threads(threads: int) -> _ConfigContextManager:
    """
    Context manager to temporarily override the number of threads that load dataset items.

    Parameters
    ----------
    threads : int
        The number of threads to use within the context. See :func:`set_load_threads` for what to expect.

    Returns
    -------
    _ConfigContextManager
        Context manager that restores the previous value on exit.

    Notes
    -----
    The setting reaches only the reads made inside the block. :class:`~dataeval.Metadata` and
    :class:`~dataeval.Embeddings` read their dataset on first use, not when they are built, so run the pass inside the
    block, for example with ``Metadata(dataset).dataframe`` or ``Embeddings(dataset, ...).compute()``. One built inside
    and first used after the block reads one item at a time.
    """
    if threads > 1:
        warn_experimental_once("load_threads", details=_LOAD_THREADS_DETAILS)
    return _ConfigContextManager("load_threads", threads)


def set_seed(seed: int | None, all_generators: bool = False, deterministic: bool = False) -> None:
    """
    Set the seed for use by classes that allow for a random state or seed.

    Parameters
    ----------
    seed : int or None
        The seed to use. When None, clears the DataEval seed and resets
        all generators (NumPy, PyTorch) and deterministic settings
        regardless of other parameters.
    all_generators : bool, default False
        Whether to set the seed for all generators, including NumPy and PyTorch.
    deterministic : bool, default False
        Whether to force PyTorch to use deterministic algorithms. When True,
        calls ``torch.use_deterministic_algorithms(True)``, which ensures
        reproducible results but may reduce performance.
    """
    _config.seed = seed

    if all_generators or seed is None:
        np.random.seed(seed)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            torch.seed()
            torch.cuda.seed_all()

    if deterministic or seed is None:
        torch.use_deterministic_algorithms(seed is not None)


def get_seed() -> int | None:
    """
    Return the seed for random state or seed.

    Returns
    -------
    int or None
        The seed to use.
    """
    return _config.seed


def use_seed(seed: int | None) -> _ConfigContextManager:
    """
    Context manager to temporarily override the DataEval seed.

    Parameters
    ----------
    seed : int or None
        The seed to use within the context. None runs the block unseeded.

    Returns
    -------
    _ConfigContextManager
        Context manager that restores the previous seed on exit.

    See Also
    --------
    set_seed, get_seed

    Notes
    -----
    Only the value :func:`get_seed` returns is swapped, which is the seed DataEval
    passes to the estimators it calls. Unlike :func:`set_seed`, the global NumPy and
    PyTorch generators are left alone: their state cannot be restored on exit, so
    seeding them from a scope that promises to undo itself would be a promise this
    cannot keep. Seed them with :func:`set_seed` when a block needs that.

    Examples
    --------
    >>> set_seed(1234)
    >>> with use_seed(42):
    ...     get_seed()
    42
    >>> # Previous seed is restored
    >>> get_seed()
    1234
    """
    return _ConfigContextManager("seed", seed)

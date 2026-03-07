"""Utility to wrap multiprocessing/threading pools for easier debugging and profiling."""

import multiprocessing
import sys
from collections.abc import Callable, Iterable, Iterator
from os import cpu_count
from typing import Any, Literal, TypeVar

from typing_extensions import Self

_R = TypeVar("_R")
_T = TypeVar("_T")

# fork is fastest (no serialization) and safe on Linux.
# macOS defaults to spawn (fork unsafe with Objective-C runtime).
# Windows only supports spawn.
_DEFAULT_CONTEXT: Literal["fork", "spawn"] = "fork" if sys.platform == "linux" else "spawn"


class PoolWrapper:
    """
    Wrap pool executors to allow easy switching between multiprocessing and single-threaded execution.

    Defaults to 'fork' on Linux (fastest, no serialization overhead) and 'spawn' elsewhere.
    Also supports 'threads' for workloads where the GIL is released during computation.
    """

    def __init__(self, processes: int | None, context: Literal["fork", "spawn"] = _DEFAULT_CONTEXT) -> None:
        procs = 1 if processes is None else max(1, (cpu_count() or 1) + processes + 1) if processes < 0 else processes
        self._pool = multiprocessing.get_context(context).Pool(procs) if procs > 1 else None

    def imap_unordered(self, func: Callable[[_T], _R], iterable: Iterable[_T]) -> Iterator[_R]:
        """Apply `func` to each item in `iterable`, optionally using a pool."""
        return map(func, iterable) if self._pool is None else self._pool.imap_unordered(func, iterable)

    def __enter__(self, *args: Any, **kwargs: Any) -> Self:
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the runtime context and clean up the pool if it was created."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()

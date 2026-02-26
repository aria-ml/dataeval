"""Utility to wrap multiprocessing.Pool for easier debugging and profiling."""

from collections.abc import Callable, Iterable, Iterator
from multiprocessing import Pool
from os import cpu_count
from typing import Any, TypeVar

from typing_extensions import Self

_R = TypeVar("_R")
_T = TypeVar("_T")


class PoolWrapper:
    """
    Wrap `multiprocessing.Pool` to allow for easy switching between multiprocessing and single-threaded execution.

    This helps with debugging and profiling, as well as usage with Jupyter notebooks
    in VS Code, which does not support subprocess debugging.
    """

    def __init__(self, processes: int | None) -> None:
        procs = 1 if processes is None else max(1, (cpu_count() or 1) + processes + 1) if processes < 0 else processes
        self._pool = Pool(procs) if procs > 1 else None

    def imap_unordered(self, func: Callable[[_T], _R], iterable: Iterable[_T]) -> Iterator[_R]:
        """Apply `func` to each item in `iterable`, optionally using multiprocessing."""
        return map(func, iterable) if self._pool is None else self._pool.imap_unordered(func, iterable)

    def __enter__(self, *args: Any, **kwargs: Any) -> Self:
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the runtime context and clean up the pool if it was created."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()

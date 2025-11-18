from __future__ import annotations

__all__ = []

from collections.abc import Callable, Iterable, Iterator
from multiprocessing import Pool
from os import cpu_count
from typing import Any, TypeVar

_S = TypeVar("_S")
_T = TypeVar("_T")


class PoolWrapper:
    """
    Wraps `multiprocessing.Pool` to allow for easy switching between
    multiprocessing and single-threaded execution.

    This helps with debugging and profiling, as well as usage with Jupyter notebooks
    in VS Code, which does not support subprocess debugging.
    """

    def __init__(self, processes: int | None) -> None:
        procs = 1 if processes is None else max(1, (cpu_count() or 1) + processes + 1) if processes < 0 else processes
        self.pool = Pool(procs) if procs > 1 else None

    def imap_unordered(self, func: Callable[[_S], _T], iterable: Iterable[_S]) -> Iterator[_T]:
        return map(func, iterable) if self.pool is None else self.pool.imap_unordered(func, iterable)

    def __enter__(self, *args: Any, **kwargs: Any) -> PoolWrapper:
        return self

    def __exit__(self, *args: Any) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

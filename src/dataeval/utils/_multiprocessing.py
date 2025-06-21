from __future__ import annotations

__all__ = []

from collections.abc import Callable, Iterable, Iterator
from multiprocessing import Pool
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
        self.pool = Pool(processes) if processes is None or processes > 1 else None

    def imap(self, func: Callable[[_S], _T], iterable: Iterable[_S]) -> Iterator[_T]:
        return map(func, iterable) if self.pool is None else self.pool.imap(func, iterable)

    def __enter__(self, *args: Any, **kwargs: Any) -> PoolWrapper:
        return self

    def __exit__(self, *args: Any) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

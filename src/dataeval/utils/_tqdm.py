"""Fallback for tqdm dependency"""

from __future__ import annotations

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    from collections.abc import Iterable, Iterator
    from typing import Any, Generic, TypeVar

    T = TypeVar("T")

    class tqdm(Generic[T]):
        """
        A dummy class that mimics tqdm's interface to prevent errors
        when tqdm is not installed. It can be iterated over and its
        methods can be called, but they do nothing.
        """

        def __init__(self, iterable: Iterable[T], *args: Any, **kwargs: Any) -> None:
            self.iterable = iterable

        def __iter__(self) -> Iterator[T]:
            return iter(self.iterable)

        def __enter__(self) -> tqdm:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

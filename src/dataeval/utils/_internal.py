"""Generic runtime utilities: module import caching, metadata type coercion, and process pooling."""

__all__ = []

import logging
import multiprocessing
import sys
from collections.abc import Callable, Iterable, Iterator
from importlib import import_module
from multiprocessing.pool import ThreadPool
from os import cpu_count
from types import ModuleType
from typing import Any, Literal, Self, TypeVar, overload

import numpy as np

from dataeval._log import get_logger

_logger = get_logger(__name__)

EPSILON = 1e-12
MODULE_CACHE = {}


def try_import(module_name: str) -> ModuleType | None:
    if module_name in MODULE_CACHE:
        return MODULE_CACHE[module_name]

    try:
        module = import_module(module_name)
    except ImportError:  # pragma: no cover
        _logger.log(logging.INFO, f"Unable to import {module_name}.")
        module = None

    MODULE_CACHE[module_name] = module
    return module


_TYPE_MAP = {int: 0, float: 1, str: 2}


@overload
def simplify_type(data: list[Any]) -> list[int | None] | list[float | None] | list[str | None]: ...
@overload
def simplify_type(data: str) -> int | float | str: ...


def simplify_type(
    data: list[Any] | str,
) -> list[int | None] | list[float | None] | list[str | None] | int | float | str:
    """
    Simplify a value or a list of values to the simplest form possible.

    In preferred order of `int`, `float`, or `string`. A missing value in a list stays
    ``None``, and so does a ``NaN`` in a list that resolves to text.

    Parameters
    ----------
    data : list[Any] | str
        A list of values or a single value

    Returns
    -------
    list[int | float | str | None] | int | float | str
        The same values converted to the numerical type if possible
    """
    if not isinstance(data, list):
        try:
            value = float(data)
        except (TypeError, ValueError):
            value = None
        return str(data) if value is None else int(value) if value.is_integer() else value

    # A missing value is not a value to promote. Read as one it became the category "None"
    # (or "nan"), and the one string made the whole column text, numbers and all.
    converted = [None if value is None else simplify_type(value) for value in data]
    widest = max((_TYPE_MAP.get(type(value), 2) for value in converted if value is not None), default=0)
    cast = list(_TYPE_MAP)[widest]
    return [None if value is None or (cast is str and is_absent(value)) else cast(value) for value in converted]


def is_absent(value: Any) -> bool:
    """Whether a row recorded no value at all.

    Checked before any correction is consulted, which is what keeps "not recorded" and
    "a value the mapping does not name" two different answers. A catch-all that swallowed
    absence would collapse them, and the reserved missing code exists to hold them apart.
    """
    return value is None or (isinstance(value, float | np.floating) and bool(np.isnan(value)))


def value_kind(value: Any) -> str:
    """Whether a value reads as a number, as text, or is a boolean.

    The split every judgment about a mixed column turns on, in one place, so that the rule
    that sets a column aside and the report that describes it cannot disagree about which
    values are the problem. Read through :func:`simplify_type`, so a numeral is numeric
    whichever way it is spelled -- metadata that has been through JSON is all text.

    A boolean is its own kind. Read as a number it is ``0`` or ``1``, which is a faithful
    column only while every value is one: beside numbers ``True`` would become the ``1``
    they already hold.
    """
    if isinstance(value, bool | np.bool_):
        return "boolean"
    return "text" if isinstance(simplify_type(value), str) else "numeric"


def promotion_is_lossy(values: list[Any]) -> bool:
    """Whether these values are of more than one kind: numbers, text or booleans.

    :func:`simplify_type` gives a column one type by promoting every value to the widest one
    present, and where that widest type is text the promotion is not a widening but a loss:
    a value that reads as ``1.0`` becomes the *category* ``"1"``, so the column can no longer
    be binned, ordered or read as continuous, and every bias evaluator scores it as a
    category set. Nothing said so.

    The question is whether a value **reads** as a number, not whether it arrived as one.
    Metadata that has been through JSON is all text, so a column of counts is a column of
    numerals and has to keep working; ``["1", "2", "3"]`` is a numeric column that happens
    to be spelled in strings, and it resolves to one with nothing held back. What cannot be
    resolved is a column where only *some* values read as numbers -- ``[1.0, "N", 2.0]`` and
    ``["1", "2", "many"]`` are the same problem in two spellings, and neither reading is one
    this library can pick on the caller's behalf.

    A column of values none of which read as numbers is an ordinary category set and is
    left exactly as it is. Booleans beside either are the same problem: one column cannot
    hold ``True`` apart from ``1``, or from ``"True"``, without picking a reading for it.
    Missing values are no kind at all.
    """
    kinds = {value_kind(value) for value in values if not is_absent(value)}
    return len(kinds) > 1


R = TypeVar("R")
T = TypeVar("T")


# fork is fastest (no serialization) and safe on Linux.
# macOS defaults to spawn (fork unsafe with Objective-C runtime).
# Windows only supports spawn.
DEFAULT_CONTEXT: Literal["fork", "spawn"] = "fork" if sys.platform == "linux" else "spawn"


class PoolWrapper:
    """
    Wrap pool executors to allow easy switching between multiprocessing and single-threaded execution.

    Defaults to 'fork' on Linux (fastest, no serialization overhead) and 'spawn' elsewhere.
    Also supports 'threads' for workloads where the GIL is released during computation.
    """

    def __init__(self, processes: int | None, context: Literal["fork", "spawn", "threads"] = DEFAULT_CONTEXT) -> None:
        procs = 1 if processes is None else max(1, (cpu_count() or 1) + processes + 1) if processes < 0 else processes
        if procs <= 1:
            self._pool = None
        elif context == "threads":
            self._pool = ThreadPool(procs)
        else:
            self._pool = multiprocessing.get_context(context).Pool(procs)

    def imap_unordered(self, func: Callable[[T], R], iterable: Iterable[T]) -> Iterator[R]:
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

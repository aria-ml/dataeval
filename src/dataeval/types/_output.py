"""Output containers returned by evaluators and the metadata decorator that stamps them."""

__all__ = [
    "BaseCollectionMixin",
    "DataFrameOutput",
    "DictOutput",
    "MappingOutput",
    "Output",
    "SequenceOutput",
    "set_metadata",
]

import inspect
import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial, wraps
from typing import Any, Generic, ParamSpec, TypeVar, overload

import numpy as np
import polars as pl

from dataeval.types._execution import ExecutionMetadata, __version__

_T = TypeVar("_T", covariant=True)


class Output(Generic[_T]):
    """Base class for all evaluator output types."""

    _meta: ExecutionMetadata | None = None

    def data(self) -> _T:
        """Return the output data."""
        ...

    def meta(self) -> ExecutionMetadata:
        """
        Metadata about the execution of the function or method for the Output class.

        Returns
        -------
        ExecutionMetadata
        """
        return self._meta or ExecutionMetadata._empty()


class DataFrameOutput(Output[pl.DataFrame]):
    """An Output that wraps a Polars DataFrame and proxies its interface.

    Attribute access, indexing, and iteration are delegated to the underlying
    DataFrame so instances can be used directly in DataFrame contexts.
    :meth:`data` and :meth:`meta` remain available alongside all DataFrame
    methods and properties.

    Subclasses pass the required DataFrame as the first positional argument
    and may accept additional keyword arguments.

    .. warning:: **Return-type loss on DataFrame operations**

        Methods delegated via :meth:`__getattr__` (e.g. ``filter``,
        ``select``, ``sort``) return a plain :class:`polars.DataFrame`, *not*
        an instance of the subclass. Any subclass-specific attributes such as
        :attr:`~dataeval.quality.OutliersOutput.calculation_results` or :meth:`meta` will not
        be available on the result.

    .. note:: **Instance attribute names to avoid in subclasses**

        Because instance attributes shadow the proxy, do not use any of the
        following names for subclass ``__init__`` parameters or attributes:
        ``columns``, ``schema``, ``dtypes``, ``shape``, ``height``, ``width``.

    Parameters
    ----------
    data : pl.DataFrame
        The underlying DataFrame.
    """

    def __init__(self, data: pl.DataFrame) -> None:
        self._df = data

    def data(self) -> pl.DataFrame:
        """Return the output data as a polars DataFrame."""
        return self._df

    # --- DataFrame proxy ---
    # Special (dunder) methods are looked up on the type, not the instance,
    # so they bypass __getattr__ entirely and must be forwarded explicitly.

    def __repr__(self) -> str:
        """Return the repr of the underlying DataFrame."""
        return repr(self.data())

    def __str__(self) -> str:
        """Return the string representation of the underlying DataFrame."""
        return str(self.data())

    def __len__(self) -> int:
        """Return the number of rows in the underlying DataFrame."""
        return len(self.data())

    def __iter__(self) -> Iterator[pl.Series]:
        """Iterate over the columns of the underlying DataFrame."""
        return iter(self.data())

    def __contains__(self, item: str) -> bool:
        """Check if a column name exists in the underlying DataFrame."""
        return item in self.data()

    def __getitem__(self, item: Any) -> Any:
        """Index into the underlying DataFrame."""
        return self.data()[item]

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying DataFrame.

        .. note::
            Returns whatever Polars returns — typically a
            :class:`polars.DataFrame` — so subclass methods and metadata
            are not preserved on the result.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.data(), name)


class DictOutput(Output[dict[str, Any]]):
    """An Output that exposes its public instance attributes as a dictionary."""

    def data(self) -> dict[str, Any]:
        """
        Return the output data as a dictionary.

        Returns
        -------
        dict[str, Any]
        """
        return {k: v for k, v in self.__dict__.items() if k != "_meta"}

    @staticmethod
    def _format_value(v: Any) -> str:
        if isinstance(v, pl.DataFrame):
            return f"DataFrame(shape={v.shape})"
        if isinstance(v, np.ndarray):
            return f"ndarray(shape={v.shape}, dtype={v.dtype})"
        return repr(v)

    def __repr__(self) -> str:
        """Return a summary representation with formatted values."""
        items = ", ".join(f"{k}={self._format_value(v)}" for k, v in self.data().items())
        return f"{self.__class__.__name__}({items})"

    def __str__(self) -> str:
        """Return the string representation of the data dictionary."""
        return str(self.data())


class BaseCollectionMixin(Collection[Any]):
    """Mixin providing collection interface for Output subclasses."""

    __slots__ = ["_data"]

    def data(self) -> Any:
        """
        Return the output data as a collection.

        Returns
        -------
        Collection
        """
        return self._data

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a detailed string representation of the collection."""
        return f"{self.__class__.__name__}({repr(self._data)})"

    def __str__(self) -> str:
        """Return the string representation of the underlying data."""
        return str(self._data)


_TKey = TypeVar("_TKey", str, int, float, set)
_TValue = TypeVar("_TValue")


class MappingOutput(Mapping[_TKey, _TValue], BaseCollectionMixin, Output[Mapping[_TKey, _TValue]]):
    """An Output that wraps a mapping and proxies its interface."""

    def __init__(self, data: Mapping[_TKey, _TValue]) -> None:
        self._data = data

    def __getitem__(self, key: _TKey) -> _TValue:
        """Return the value for the given key."""
        return self._data[key]

    def __iter__(self) -> Iterator[_TKey]:
        """Iterate over the keys of the mapping."""
        return iter(self._data)


class SequenceOutput(Sequence[_TValue], BaseCollectionMixin, Output[Sequence[_TValue]]):
    """An Output that wraps a sequence and proxies its interface."""

    def __init__(self, data: Sequence[_TValue]) -> None:
        self._data = data

    @overload
    def __getitem__(self, index: int) -> _TValue: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[_TValue]: ...

    def __getitem__(self, index: int | slice) -> _TValue | Sequence[_TValue]:
        """Return the item or slice at the given index."""
        return self._data[index]

    def __iter__(self) -> Iterator[_TValue]:
        """Iterate over the items in the sequence."""
        return iter(self._data)


_P = ParamSpec("_P")
_R = TypeVar("_R", bound=Output)


def set_metadata(fn: Callable[_P, _R] | None = None, *, state: Sequence[str] | None = None) -> Callable[_P, _R]:  # noqa: C901
    """Stamp Output classes with runtime metadata."""
    if fn is None:
        return partial(set_metadata, state=state)  # type: ignore

    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        def fmt(v: Any) -> Any:
            if np.isscalar(v):
                return v
            if hasattr(v, "shape"):
                return f"{v.__class__.__name__}: shape={v.shape}"
            if hasattr(v, "__len__"):
                return f"{v.__class__.__name__}: len={len(v)}"
            return f"{v.__class__.__name__}"

        # Collect function metadata
        # set all params with defaults then update params with mapped arguments and explicit keyword args
        fn_params = inspect.signature(fn).parameters
        arguments = {k: None if v.default is inspect.Parameter.empty else v.default for k, v in fn_params.items()}
        arguments.update(zip(fn_params, args, strict=False))
        arguments.update(kwargs)
        arguments = {k: fmt(v) for k, v in arguments.items()}
        is_method = "self" in arguments
        state_attrs = {k: fmt(getattr(args[0], k)) for k in state or []} if is_method else {}
        module = args[0].__class__.__module__ if is_method else fn.__module__.removeprefix("src.")
        class_prefix = f".{args[0].__class__.__name__}." if is_method else "."
        name = f"{module}{class_prefix}{fn.__name__}"
        arguments = {k: v for k, v in arguments.items() if k != "self"}

        _logger = logging.getLogger(module)
        time = datetime.now(timezone.utc)
        _logger.log(logging.INFO, f">>> Executing '{name}': args={arguments} state={state} <<<")

        # EXECUTE FUNCTION #####
        result = fn(*args, **kwargs)
        ############################

        duration = (datetime.now(timezone.utc) - time).total_seconds()
        _logger.log(logging.INFO, f">>> Completed '{name}': args={arguments} state={state} duration={duration} <<<")

        # Update output with recorded metadata
        metadata = ExecutionMetadata(name, time, duration, arguments, state_attrs, __version__)
        object.__setattr__(result, "_meta", metadata)
        return result

    return wrapper

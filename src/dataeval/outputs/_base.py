from __future__ import annotations

__all__ = []

import inspect
import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial, wraps
from typing import Any, Generic, ParamSpec, TypeVar, overload

import numpy as np

from dataeval import __version__


@dataclass(frozen=True)
class ExecutionMetadata:
    """
    Metadata about the execution of the function or method for the Output class.

    Attributes
    ----------
    name: str
        Name of the function or method
    execution_time: datetime
        Time of execution
    execution_duration: float
        Duration of execution in seconds
    arguments: dict[str, Any]
        Arguments passed to the function or method
    state: dict[str, Any]
        State attributes of the executing class
    version: str
        Version of DataEval
    """

    name: str
    execution_time: datetime
    execution_duration: float
    arguments: dict[str, Any]
    state: dict[str, Any]
    version: str

    @classmethod
    def empty(cls) -> ExecutionMetadata:
        return ExecutionMetadata(
            name="",
            execution_time=datetime.min,
            execution_duration=0.0,
            arguments={},
            state={},
            version=__version__,
        )


T = TypeVar("T", covariant=True)


class GenericOutput(Generic[T]):
    _meta: ExecutionMetadata | None = None

    def data(self) -> T: ...
    def meta(self) -> ExecutionMetadata:
        """
        Metadata about the execution of the function or method for the Output class.

        Returns
        -------
        ExecutionMetadata
        """
        return self._meta or ExecutionMetadata.empty()


class Output(GenericOutput[dict[str, Any]]):
    def data(self) -> dict[str, Any]:
        """
        The output data as a dictionary.

        Returns
        -------
        dict[str, Any]
        """
        return {k: v for k, v in self.__dict__.items() if k != "_meta"}

    def __str__(self) -> str:
        return str(self.data())


class BaseCollectionMixin(Collection[Any]):
    __slots__ = ["_data"]

    def data(self) -> Any:
        """
        The output data as a collection.

        Returns
        -------
        Collection
        """
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._data)})"

    def __str__(self) -> str:
        return str(self._data)


TKey = TypeVar("TKey", str, int, float, set)
TValue = TypeVar("TValue")


class MappingOutput(Mapping[TKey, TValue], BaseCollectionMixin, GenericOutput[Mapping[TKey, TValue]]):
    def __init__(self, data: Mapping[TKey, TValue]) -> None:
        self._data = data

    def __getitem__(self, key: TKey) -> TValue:
        return self._data[key]

    def __iter__(self) -> Iterator[TKey]:
        return iter(self._data)


class SequenceOutput(Sequence[TValue], BaseCollectionMixin, GenericOutput[Sequence[TValue]]):
    def __init__(self, data: Sequence[TValue]) -> None:
        self._data = data

    @overload
    def __getitem__(self, index: int) -> TValue: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[TValue]: ...

    def __getitem__(self, index: int | slice) -> TValue | Sequence[TValue]:
        return self._data[index]

    def __iter__(self) -> Iterator[TValue]:
        return iter(self._data)


P = ParamSpec("P")
R = TypeVar("R", bound=GenericOutput)


def set_metadata(fn: Callable[P, R] | None = None, *, state: Sequence[str] | None = None) -> Callable[P, R]:
    """Decorator to stamp Output classes with runtime metadata"""

    if fn is None:
        return partial(set_metadata, state=state)  # type: ignore

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        def fmt(v: Any) -> Any:
            if np.isscalar(v):
                return v
            if hasattr(v, "shape"):
                return f"{v.__class__.__name__}: shape={getattr(v, 'shape')}"
            if hasattr(v, "__len__"):
                return f"{v.__class__.__name__}: len={len(v)}"
            return f"{v.__class__.__name__}"

        # Collect function metadata
        # set all params with defaults then update params with mapped arguments and explicit keyword args
        fn_params = inspect.signature(fn).parameters
        arguments = {k: None if v.default is inspect.Parameter.empty else v.default for k, v in fn_params.items()}
        arguments.update(zip(fn_params, args))
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

        ##### EXECUTE FUNCTION #####
        result = fn(*args, **kwargs)
        ############################

        duration = (datetime.now(timezone.utc) - time).total_seconds()
        _logger.log(logging.INFO, f">>> Completed '{name}': args={arguments} state={state} duration={duration} <<<")

        # Update output with recorded metadata
        metadata = ExecutionMetadata(name, time, duration, arguments, state_attrs, __version__)
        object.__setattr__(result, "_meta", metadata)
        return result

    return wrapper

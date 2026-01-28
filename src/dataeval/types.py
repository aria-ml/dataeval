"""Data types used in DataEval."""

__all__ = [
    "Array1D",
    "Array2D",
    "Array3D",
    "Array4D",
    "Array5D",
    "Array6D",
    "Array7D",
    "Array8D",
    "Array9D",
    "ArrayND",
    "SourceIndex",
    "ExecutionMetadata",
]


import inspect
import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial, wraps
from typing import Any, ClassVar, Generic, Literal, NamedTuple, ParamSpec, TypeAlias, TypeVar, overload

import numpy as np
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from dataeval import __version__
from dataeval._helpers import apply_config, get_overrides
from dataeval.protocols import Array, SequenceLike

DType = TypeVar("DType", covariant=True)


Array1D: TypeAlias = Array | SequenceLike[DType]
Array2D: TypeAlias = Array | SequenceLike[Array1D[DType]]
Array3D: TypeAlias = Array | SequenceLike[Array2D[DType]]
Array4D: TypeAlias = Array | SequenceLike[Array3D[DType]]
Array5D: TypeAlias = Array | SequenceLike[Array4D[DType]]
Array6D: TypeAlias = Array | SequenceLike[Array5D[DType]]
Array7D: TypeAlias = Array | SequenceLike[Array6D[DType]]
Array8D: TypeAlias = Array | SequenceLike[Array7D[DType]]
Array9D: TypeAlias = Array | SequenceLike[Array8D[DType]]
ArrayND: TypeAlias = (
    Array
    | Array1D[DType]
    | Array2D[DType]
    | Array3D[DType]
    | Array4D[DType]
    | Array5D[DType]
    | Array6D[DType]
    | Array7D[DType]
    | Array8D[DType]
    | Array9D[DType]
)

# Default values for ClusterConfigMixin
DEFAULT_CLUSTER_ALGORITHM: Literal["kmeans", "hdbscan"] = "hdbscan"
DEFAULT_CLUSTER_N_CLUSTERS: int | None = None


class EvaluatorConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class ClusterConfigMixin(BaseModel):
    cluster_algorithm: Literal["kmeans", "hdbscan"] = DEFAULT_CLUSTER_ALGORITHM
    n_clusters: int | None = DEFAULT_CLUSTER_N_CLUSTERS


class Evaluator:
    """Base class for all evaluators."""

    def __init__(self, kwargs: dict[str, Any], *, exclude: set[str] | None = None) -> None:
        config_cls = getattr(self, "Config", None)
        if config_cls is None:
            raise NotImplementedError("Evaluator subclasses must define a Config class.")
        base_config = kwargs.get("config") or config_cls()
        self._config = base_config.model_copy(update=get_overrides(kwargs, exclude))
        apply_config(self, self._config)


class SourceIndex(NamedTuple):
    """
    The indices of the source item, target and channel.

    Attributes
    ----------
    item: int
        Index of the source item
    target : int | None
        Index of the box/target within the source item.
        - None: References item-level data
        - int: References a specific target/detection within the item (0-indexed per item)
        For Object Detection datasets, this maps to the target_index in Metadata.
    channel : int | None
        Index of the channel of the source image (if applicable)
    """

    item: int
    target: int | None = None
    channel: int | None = None

    def __repr__(self) -> str:
        """Compact representation showing only non-None fields."""
        parts = [f"{self.item}"]
        if self.target is not None:
            parts.append(f"{self.target}")
        if self.target is None and self.channel is not None:
            parts.append("None")
        if self.channel is not None:
            parts.append(f"{self.channel}")
        return f"SourceIndex({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable string showing the full path."""
        parts = [str(self.item)]
        if self.target is not None:
            parts.append(str(self.target))
        if self.target is None and self.channel is not None:
            parts.append("-")
        if self.channel is not None:
            parts.append(str(self.channel))
        return "/".join(parts)

    @classmethod
    def from_string(cls, s: str) -> Self:
        """
        Construct a SourceIndex from a human-readable string.

        Parameters
        ----------
        s : str
            String in the format "item", "item/target", or "item/-/channel", "item/target/channel"
            Use "-" to represent None for the target field.

        Returns
        -------
        SourceIndex

        Examples
        --------
        >>> SourceIndex.from_string("0")
        SourceIndex(0)
        >>> SourceIndex.from_string("0/3")
        SourceIndex(0, 3)
        >>> SourceIndex.from_string("0/-/1")
        SourceIndex(0, None, 1)
        >>> SourceIndex.from_string("0/3/1")
        SourceIndex(0, 3, 1)
        """
        item, target, channel = None, None, None
        parts = s.split("/")

        if len(parts) > 0:
            item = int(parts[0])
        if len(parts) > 1:
            target = None if parts[1] == "-" else int(parts[1])
        if len(parts) > 2:
            channel = None if parts[2] == "-" else int(parts[2])
        if item is None or len(parts) > 3:
            raise ValueError(f"Invalid SourceIndex string format: {s}")
        return cls(item, target, channel)


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
    def empty(cls) -> Self:
        return cls(
            name="",
            execution_time=datetime.min,
            execution_duration=0.0,
            arguments={},
            state={},
            version=__version__,
        )


T = TypeVar("T", covariant=True)


class Output(Generic[T]):
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


class DictOutput(Output[dict[str, Any]]):
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


class MappingOutput(Mapping[TKey, TValue], BaseCollectionMixin, Output[Mapping[TKey, TValue]]):
    def __init__(self, data: Mapping[TKey, TValue]) -> None:
        self._data = data

    def __getitem__(self, key: TKey) -> TValue:
        return self._data[key]

    def __iter__(self) -> Iterator[TKey]:
        return iter(self._data)


class SequenceOutput(Sequence[TValue], BaseCollectionMixin, Output[Sequence[TValue]]):
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
R = TypeVar("R", bound=Output)


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

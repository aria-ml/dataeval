"""Data types used in DataEval."""

__all__ = [
    "Array1D",
    "Array2D",
    "Array3D",
    "Array4D",
    "Array5D",
    "ArrayND",
    "BaseCollectionMixin",
    "ClusterConfigMixin",
    "DataFrameOutput",
    "DictOutput",
    "Evaluator",
    "EvaluatorConfig",
    "ExecutionMetadata",
    "MappingOutput",
    "Output",
    "ReprMixin",
    "SequenceOutput",
    "SourceIndex",
    "StatsMap",
    "set_metadata",
]


import inspect
import logging
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial, wraps
from typing import Any, ClassVar, Generic, Literal, NamedTuple, ParamSpec, TypeAlias, TypeVar, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from dataeval import __version__
from dataeval._helpers import apply_config, get_overrides
from dataeval.protocols import Array, FeatureExtractor, SequenceLike

_DType = TypeVar("_DType", covariant=True)


Array1D: TypeAlias = Array | SequenceLike[_DType]
Array2D: TypeAlias = Array | SequenceLike[Array1D[_DType]]
Array3D: TypeAlias = Array | SequenceLike[Array2D[_DType]]
Array4D: TypeAlias = Array | SequenceLike[Array3D[_DType]]
Array5D: TypeAlias = Array | SequenceLike[Array4D[_DType]]
ArrayND: TypeAlias = Array | Array1D[_DType] | Array2D[_DType] | Array3D[_DType] | Array4D[_DType] | Array5D[_DType]

StatsMap: TypeAlias = Mapping[str, NDArray[Any]]
"""
A mapping of metric names to their corresponding numpy array values.
Each array should have the same length along the first dimension, representing the number of samples.
"""

# Default values for ClusterConfigMixin
_DEFAULT_CLUSTER_ALGORITHM: Literal["kmeans", "hdbscan"] = "hdbscan"
_DEFAULT_CLUSTER_N_CLUSTERS: int | None = None


class EvaluatorConfig(BaseModel):
    """Base configuration class for all evaluators."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ClusterConfigMixin(BaseModel):
    """Configuration mixin for evaluators that use clustering."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    extractor: FeatureExtractor | None = None
    batch_size: int | None = None
    cluster_algorithm: Literal["kmeans", "hdbscan"] = _DEFAULT_CLUSTER_ALGORITHM
    n_clusters: int | None = _DEFAULT_CLUSTER_N_CLUSTERS


class ReprMixin:
    """Mixin providing consistent ``__repr__`` via ``__init__`` signature introspection.

    Looks up each ``__init__`` parameter on ``self`` (trying both ``name`` and
    ``_name``).  Subclasses can override :meth:`_repr_extras` to append
    additional key-value pairs (e.g. ``fitted=True``).
    """

    def _repr_extras(self) -> dict[str, Any]:
        """Override to append extra state to ``__repr__``."""
        return {}

    def __repr__(self) -> str:  # noqa: C901
        """Return a string representation showing init parameters and extras."""
        sig = inspect.signature(self.__init__)  # type: ignore[misc]
        params: list[str] = []
        for name in sig.parameters:
            if name == "self":
                continue
            if hasattr(self, name):
                params.append(f"{name}={getattr(self, name)!r}")
            elif hasattr(self, f"_{name}"):
                params.append(f"{name}={getattr(self, f'_{name}')!r}")
        for k, v in self._repr_extras().items():
            params.append(f"{k}={v!r}")
        return f"{self.__class__.__name__}({', '.join(params)})"


class Evaluator:
    """Base class for all evaluators."""

    def __init__(self, kwargs: dict[str, Any] | None = None, *, exclude: set[str] | None = None) -> None:
        if kwargs is None:
            return
        config_cls = getattr(self, "Config", None)
        if config_cls is None:
            raise NotImplementedError("Evaluator subclasses must define a Config class.")
        base_config = kwargs.get("config") or config_cls()
        self._config = base_config.model_copy(update=get_overrides(kwargs, exclude))
        apply_config(self, self._config)

    def _repr_extras(self) -> dict[str, Any]:
        """Override to append extra state to ``__repr__``."""
        return {}

    def _repr(self, *, extras: bool = True) -> str:  # noqa: C901
        """Build repr string, optionally suppressing extras."""
        config = getattr(self, "_config", None)
        if config is not None and hasattr(config, "model_fields"):
            # Pydantic config (bias, performance, quality, scope)
            fields = config.model_fields
        elif config is not None and hasattr(config, "__dataclass_fields__"):
            # Dataclass config (drift, OOD)
            fields = config.__dataclass_fields__
        else:
            # Fallback: try self.config (drift/OOD store config without underscore)
            config = getattr(self, "config", None)
            if config is not None and hasattr(config, "__dataclass_fields__"):
                fields = config.__dataclass_fields__
            else:
                fields = {}
        params = [f"{k}={getattr(config, k)!r}" for k in fields]
        if extras:
            for k, v in self._repr_extras().items():
                params.append(f"{k}={v!r}")
        return f"{self.__class__.__name__}({', '.join(params)})"

    def __repr__(self) -> str:
        """Return a string representation showing the evaluator configuration."""
        return self._repr()


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

    def __repr__(self) -> str:
        """Return a detailed string representation of the execution metadata."""
        return (
            f"ExecutionMetadata(name={self.name!r}, "
            f"execution_time={self.execution_time.isoformat()}, "
            f"execution_duration={self.execution_duration:.4f}s, "
            f"version={self.version!r})"
        )

    def __str__(self) -> str:
        """Return a string representation showing the name and duration."""
        return f"{self.name} ({self.execution_duration:.4f}s)"

    @classmethod
    def _empty(cls) -> Self:
        return cls(
            name="",
            execution_time=datetime.min,
            execution_duration=0.0,
            arguments={},
            state={},
            version=__version__,
        )


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
        :attr:`~OutliersOutput.calculation_results` or :meth:`meta` will not
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

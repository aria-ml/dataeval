from __future__ import annotations

__all__ = []

import inspect
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import partial, wraps
from typing import Any, Callable, Iterator, TypeVar

import numpy as np
from typing_extensions import ParamSpec

from dataeval import __version__


class Output:
    _name: str
    _execution_time: datetime
    _execution_duration: float
    _arguments: dict[str, str]
    _state: dict[str, str]
    _version: str

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {str(self.dict())}"

    def dict(self) -> dict[str, Any]:
        """
        Output attributes as a dictionary.

        Returns
        -------
        dict[str, Any]
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def meta(self) -> dict[str, Any]:
        """
        Execution metadata as a dictionary.

        Returns
        -------
        dict[str, Any]
        """
        return {k.removeprefix("_"): v for k, v in self.__dict__.items() if k.startswith("_")}


TKey = TypeVar("TKey", str, int, float, set)
TValue = TypeVar("TValue")


class MappingOutput(Mapping[TKey, TValue], Output):
    __slots__ = ["_data"]

    def __init__(self, data: Mapping[TKey, TValue]):
        self._data = data

    def __getitem__(self, key: TKey) -> TValue:
        return self._data.__getitem__(key)

    def __iter__(self) -> Iterator[TKey]:
        return self._data.__iter__()

    def __len__(self) -> int:
        return self._data.__len__()

    def dict(self) -> dict[str, TValue]:
        return {str(k): v for k, v in self._data.items()}

    def __str__(self) -> str:
        return str(self.dict())


P = ParamSpec("P")
R = TypeVar("R", bound=Output)


def set_metadata(fn: Callable[P, R] | None = None, *, state: list[str] | None = None) -> Callable[P, R]:
    """Decorator to stamp Output classes with runtime metadata"""

    if fn is None:
        return partial(set_metadata, state=state)  # type: ignore

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        def fmt(v):
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
        metadata = {
            "_name": name,
            "_execution_time": time,
            "_execution_duration": duration,
            "_arguments": arguments,
            "_state": state_attrs,
            "_version": __version__,
        }
        for k, v in metadata.items():
            object.__setattr__(result, k, v)
        return result

    return wrapper

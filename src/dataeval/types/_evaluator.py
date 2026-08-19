"""Evaluator base class and shared ``__repr__`` behavior."""

__all__ = [
    "Evaluator",
    "ReprMixin",
]

import inspect
from typing import Any

from dataeval._helpers import apply_config, get_overrides


class ReprMixin:
    """Mixin providing consistent ``__repr__`` via ``__init__`` signature introspection.

    Looks up each ``__init__`` parameter on ``self`` (trying both ``name`` and
    ``_name``).  Subclasses can override :meth:`_repr_extras` to append
    additional key-value pairs (e.g. ``fitted=True``).
    """

    def _repr_extras(self) -> dict[str, Any]:
        """Override to append extra state to ``__repr__``."""
        return {}

    def _repr_overrides(self) -> dict[str, str]:
        """Override to replace an init parameter's rendered value with a custom string.

        Keys are ``__init__`` parameter names; values are inserted verbatim (no
        ``repr``) in place of the looked-up attribute (e.g. ``{"model": "ResNet"}``).
        """
        return {}

    def __repr__(self) -> str:  # noqa: C901
        """Return a string representation showing init parameters and extras."""
        sig = inspect.signature(self.__init__)  # type: ignore[misc]
        overrides = self._repr_overrides()
        params: list[str] = []
        for name in sig.parameters:
            if name == "self":
                continue
            if name in overrides:
                params.append(f"{name}={overrides[name]}")
            elif hasattr(self, name):
                params.append(f"{name}={getattr(self, name)!r}")
            elif hasattr(self, f"_{name}"):
                params.append(f"{name}={getattr(self, f'_{name}')!r}")
        for k, v in self._repr_extras().items():
            params.append(f"{k}={v!r}")
        return f"{self.__class__.__name__}({', '.join(params)})"


class Evaluator:
    """Base class for all evaluators."""

    @property
    def encoding_digest(self) -> str | None:
        """Fingerprint of the encoding this evaluator's metadata was read under, if it has one.

        Named in ``set_metadata(state=...)`` by the evaluators that read factors, so a
        result carries the encoding that produced it. Comparing two passes is only sound if
        each can say which cuts it was computed against — otherwise a score that moved is
        unattributable between *the override worked* and *the data changed*.

        None for an evaluator that has not run, and for a bare container that keeps no
        record.
        """
        return getattr(getattr(self, "metadata", None), "encoding_digest", None)

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

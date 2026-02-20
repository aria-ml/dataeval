"""Infrastructure for marking experimental and deprecated API items."""

__all__ = []

import functools
import importlib
import warnings
from typing import Any, TypeVar

from dataeval.exceptions import DeprecatedWarning, ExperimentalWarning

F = TypeVar("F")


def _make_warning_message(
    name: str,
    kind: str,
    *,
    since: str | None = None,
    removal: str | None = None,
    alternative: str | None = None,
    details: str | None = None,
) -> str:
    """Build a standardized warning message string."""
    if kind == "experimental":
        msg = f"'{name}' is experimental and may change or be removed in any future release without notice."
    else:
        msg = f"'{name}' is deprecated"
        if since:
            msg += f" since version {since}"
        msg += "."
        if removal:
            msg += f" It will be removed in version {removal}."
    if alternative:
        msg += f" Use '{alternative}' instead."
    if details:
        msg += f" {details}"
    return msg


def experimental(
    _target: F | None = None,
    *,
    alternative: str | None = None,
    details: str | None = None,
) -> F:
    """Mark a function or class as experimental.

    When applied to a function, warns on each call.
    When applied to a class, warns on instantiation.

    Can be used with or without arguments::

        @experimental
        def my_func(): ...


        @experimental(alternative="other_func")
        def my_func(): ...
    """

    def decorator(target: F) -> F:
        name = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))
        msg = _make_warning_message(name, "experimental", alternative=alternative, details=details)

        if isinstance(target, type):
            original_init = target.__init__

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(msg, ExperimentalWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            target.__init__ = new_init  # type: ignore[attr-defined]
            return target  # type: ignore[return-value]

        @functools.wraps(target)  # type: ignore[arg-type]
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, ExperimentalWarning, stacklevel=2)
            return target(*args, **kwargs)  # type: ignore[operator]

        return wrapper  # type: ignore[return-value]

    if _target is not None:
        return decorator(_target)
    return decorator  # type: ignore[return-value]


def deprecated(
    _target: F | None = None,
    *,
    since: str | None = None,
    removal: str | None = None,
    alternative: str | None = None,
    details: str | None = None,
) -> F:
    """Mark a function or class as deprecated.

    When applied to a function, warns on each call.
    When applied to a class, warns on instantiation.

    Can be used with or without arguments::

        @deprecated
        def old_func(): ...


        @deprecated(since="1.0", removal="2.0", alternative="new_func")
        def old_func(): ...
    """

    def decorator(target: F) -> F:
        name = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))
        msg = _make_warning_message(
            name,
            "deprecated",
            since=since,
            removal=removal,
            alternative=alternative,
            details=details,
        )

        if isinstance(target, type):
            original_init = target.__init__

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(msg, DeprecatedWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            target.__init__ = new_init  # type: ignore[attr-defined]
            return target  # type: ignore[return-value]

        @functools.wraps(target)  # type: ignore[arg-type]
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, DeprecatedWarning, stacklevel=2)
            return target(*args, **kwargs)  # type: ignore[operator]

        return wrapper  # type: ignore[return-value]

    if _target is not None:
        return decorator(_target)
    return decorator  # type: ignore[return-value]


def _warn_on_access(
    name: str,
    kind: str = "experimental",
    *,
    since: str | None = None,
    removal: str | None = None,
    alternative: str | None = None,
    details: str | None = None,
) -> None:
    """Emit a warning when a name is accessed from a module.

    Intended for use inside module-level ``__getattr__`` functions.
    Uses ``stacklevel=3`` because the call chain is:
    user code -> ``__getattr__`` -> ``_warn_on_access``.
    """
    category = ExperimentalWarning if kind == "experimental" else DeprecatedWarning
    msg = _make_warning_message(
        name,
        kind,
        since=since,
        removal=removal,
        alternative=alternative,
        details=details,
    )
    warnings.warn(msg, category, stacklevel=3)


def _lazy_import_with_warning(
    module_name: str,
    attr_name: str,
    qualified_name: str,
    kind: str = "experimental",
    **kwargs: Any,
) -> Any:
    """Import an attribute lazily and emit a warning.

    Helper for use in module-level ``__getattr__`` functions.

    Parameters
    ----------
    module_name : str
        Dotted module path to import from.
    attr_name : str
        Name of the attribute within the module.
    qualified_name : str
        Fully qualified name for the warning message (e.g. ``"dataeval.bias.Parity"``).
    kind : str
        Either ``"experimental"`` or ``"deprecated"``.
    **kwargs
        Additional keyword arguments passed to ``_warn_on_access``
        (e.g., ``since``, ``removal``, ``alternative``).
    """
    _warn_on_access(qualified_name, kind, **kwargs)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)

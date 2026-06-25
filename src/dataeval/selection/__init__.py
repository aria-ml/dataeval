"""Deprecated location. Selection classes moved to :mod:`dataeval.data`.

Importing from ``dataeval.selection`` is deprecated; import these classes from
:mod:`dataeval.data` instead.
"""

__all__ = []

import warnings
from typing import Any

_MOVED = ("ClassBalance", "ClassFilter", "Indices", "Limit", "Reverse", "Select", "Selection", "Shuffle")


def __getattr__(name: str) -> Any:
    if name in _MOVED:
        warnings.warn(
            f"dataeval.selection.{name} has moved to dataeval.data.{name}; importing it from "
            "dataeval.selection is deprecated and will be removed in v1.2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        import dataeval.data

        return getattr(dataeval.data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

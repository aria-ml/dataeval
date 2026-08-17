"""Deprecated home for the pre-``View`` selection API.

The dataset-view wrapper and its operations now live in :mod:`dataeval.data` as
:class:`~dataeval.data.View` and :class:`~dataeval.data.Operation`. This module keeps
the old names importable during deprecation:

- ``Select`` — a deprecated :class:`~dataeval.data.View` subclass (accepts ``selections=``).
- ``Selector`` / ``Selection`` — deprecated names for :class:`~dataeval.data.Operation`.
- the concrete operations (``ClassFilter``, ``Limit``, ...) forward to :mod:`dataeval.data`.
"""

__all__ = []

import warnings
from collections.abc import Sequence
from typing import Any, TypeVar

from dataeval.data._view import Operation, View
from dataeval.protocols import Dataset

_TDatum = TypeVar("_TDatum")

# Operations that simply moved from dataeval.selection to dataeval.data (name unchanged).
_MOVED = ("ClassBalance", "ClassFilter", "Indices", "Limit", "Reverse", "Shuffle")


class Select(View[_TDatum]):
    """Deprecated alias for :class:`dataeval.data.View`.

    Use ``dataeval.data.View(dataset, operations=...)`` instead.
    """

    def __init__(
        self,
        dataset: Dataset[_TDatum],
        selections: Operation | Sequence[Operation] | None = None,
    ) -> None:
        warnings.warn(
            "Select is deprecated and will be removed in v1.2.0; use "
            "dataeval.data.View(dataset, operations=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(dataset, selections)

    @property
    def selection_groups(self) -> list[list[Operation]]:
        """Deprecated alias for :attr:`dataeval.data.View.operation_groups`."""
        return self.operation_groups


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
    if name in ("Selector", "Selection"):
        warnings.warn(
            f"dataeval.selection.{name} is deprecated and will be removed in v1.2.0; subclass "
            "dataeval.data.Operation and implement apply(view) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Operation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

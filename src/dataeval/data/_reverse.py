__all__ = []

from typing import Any

from dataeval.data._view import Operation, View
from dataeval.utils.data import DatasetKind


class Reverse(Operation):
    """Select dataset indices in reverse order."""

    requires: DatasetKind | None = None

    def apply(self, view: View[Any]) -> None:
        view.selection.reverse()

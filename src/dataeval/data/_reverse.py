__all__ = []

from typing import Any

from dataeval.data._view import Operation, View


class Reverse(Operation):
    """Select dataset indices in reverse order."""

    def apply(self, view: View[Any]) -> None:
        view.selection.reverse()

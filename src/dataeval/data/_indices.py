__all__ = []

from collections.abc import Sequence
from typing import Any

from dataeval.data._view import Operation, View


class Indices(Operation):
    """
    Selects only the given indices from the dataset, or excludes them when `exclude` is set.

    Parameters
    ----------
    indices : Sequence[int]
        The specific indices to select or exclude.
    exclude : bool, default False
        If True, exclude `indices` from the dataset view, preserving the order of
        the remaining items. If False, select `indices` in the order given.
    """

    def __init__(self, indices: Sequence[int], exclude: bool = False) -> None:
        self.indices: Sequence[int] = indices
        self.exclude: bool = bool(exclude)

    def apply(self, view: View[Any]) -> None:
        if self.exclude:
            exclude = set(self.indices)
            view.selection = [index for index in view.selection if index not in exclude]
        else:
            current = set(view.selection)
            view.selection = [index for index in self.indices if index in current]

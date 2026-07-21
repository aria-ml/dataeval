__all__ = []

from collections.abc import Sequence
from typing import Any

from dataeval.data._view import Operation, View


class Indices(Operation):
    """
    Selects only the given indices from the dataset.

    Parameters
    ----------
    indices : Sequence[int]
        The specific indices to select.
    """

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def apply(self, view: View[Any]) -> None:
        current = set(view.selection)
        view.selection = [index for index in self.indices if index in current]

__all__ = []

from typing import Any

from dataeval.data._view import Operation, View


class Limit(Operation):
    """
    Cap the dataset to the first ``size`` items currently selected.

    Applied in place, in pipeline order: ``Limit`` truncates whatever selection
    precedes it, so ``[Shuffle(), Limit(100)]`` keeps a random 100 while
    ``[Limit(100), Shuffle()]`` shuffles the first 100. Chaining is allowed —
    ``[Limit(1000), Shuffle(), Limit(100)]`` keeps a random 100 of the first 1000.

    Parameters
    ----------
    size : int
        The maximum size of the dataset.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def apply(self, view: View[Any]) -> None:
        view.selection = view.selection[: self.size]

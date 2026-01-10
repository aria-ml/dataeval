__all__ = []

from typing import Any

from dataeval.selection._select import Select, Selection, SelectionStage


class Reverse(Selection[Any]):
    """
    Select dataset indices in reverse order.
    """

    stage = SelectionStage.ORDER

    def __call__(self, dataset: Select[Any]) -> None:
        dataset._selection.reverse()

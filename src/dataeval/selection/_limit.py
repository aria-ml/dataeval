__all__ = []

from typing import Any

from dataeval.selection._select import Select, Selection, SelectionStage


class Limit(Selection[Any]):
    """
    Limit the size of the dataset.

    Parameters
    ----------
    size : int
        The maximum size of the dataset.
    """

    stage = SelectionStage.STATE

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, dataset: Select[Any]) -> None:
        dataset._size_limit = self.size

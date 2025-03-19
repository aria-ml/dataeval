from __future__ import annotations

from dataeval.typing import TDatasetMetadata

__all__ = []

from typing import Any

from dataeval.utils.data._selection import Select, Selection, SelectionStage


class Reverse(Selection[Any]):
    """
    Reverse the selection order of the dataset.
    """

    stage = SelectionStage.ORDER

    def __call__(self, dataset: Select[Any, TDatasetMetadata]) -> None:
        dataset._selection.reverse()

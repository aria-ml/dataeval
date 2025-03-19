from __future__ import annotations

from dataeval.typing import TDatasetMetadata

__all__ = []

from typing import Any, Sequence

from dataeval.utils.data._selection import Select, Selection, SelectionStage


class Indices(Selection[Any]):
    """
    Selects specific indices from the dataset.

    Parameters
    ----------
    indices : Sequence[int]
        The indices to select from the dataset.
    """

    stage = SelectionStage.FILTER

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __call__(self, dataset: Select[Any, TDatasetMetadata]) -> None:
        dataset._selection = [index for index in self.indices if index in dataset._selection]

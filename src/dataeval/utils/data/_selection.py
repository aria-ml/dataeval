from __future__ import annotations

__all__ = []

from enum import IntEnum
from typing import Any, Generic, Iterator, Sequence, TypeVar

from dataeval.typing import AnnotatedDataset, DatasetMetadata

_TDatum = TypeVar("_TDatum")


class SelectionStage(IntEnum):
    STATE = 0
    FILTER = 1
    ORDER = 2


class Selection(Generic[_TDatum]):
    stage: SelectionStage

    def __call__(self, dataset: Select[_TDatum]) -> None: ...

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"


class Select(AnnotatedDataset[_TDatum]):
    """
    Wraps a dataset and applies selection criteria to it.

    Parameters
    ----------
    dataset : Dataset
        The dataset to wrap.
    selections : Selection or list[Selection], optional
        The selection criteria to apply to the dataset.

    Examples
    --------
    >>> from dataeval.utils.data.selections import ClassFilter, Limit

    >>> # Construct a sample dataset with size of 100 and class count of 10
    >>> # Elements at index `idx` are returned as tuples:
    >>> # - f"data_{idx}", one_hot_encoded(idx % class_count), {"id": idx}
    >>> dataset = SampleDataset(size=100, class_count=10)

    >>> # Apply a selection criteria to the dataset
    >>> selections = [Limit(size=5), ClassFilter(classes=[0, 2])]
    >>> selected_dataset = Select(dataset, selections=selections)

    >>> # Iterate over the selected dataset
    >>> for data, target, meta in selected_dataset:
    ...     print(f"({data}, {np.argmax(target)}, {meta})")
    (data_0, 0, {'id': 0})
    (data_2, 2, {'id': 2})
    (data_10, 0, {'id': 10})
    (data_12, 2, {'id': 12})
    (data_20, 0, {'id': 20})
    """

    _dataset: AnnotatedDataset[_TDatum]
    _selection: list[int]
    _selections: Sequence[Selection[_TDatum]]
    _size_limit: int

    def __init__(
        self,
        dataset: AnnotatedDataset[_TDatum],
        selections: Selection[_TDatum] | list[Selection[_TDatum]] | None = None,
    ) -> None:
        self._dataset = dataset
        self._size_limit = len(dataset)
        self._selection = list(range(self._size_limit))
        self._selections = self._sort_selections(selections)
        self.__dict__.update(dataset.__dict__)

        # Ensure metadata is populated correctly as DatasetMetadata TypedDict
        _metadata = getattr(dataset, "metadata", {})
        if "id" not in _metadata:
            _metadata["id"] = dataset.__class__.__name__
        self._metadata = DatasetMetadata(**_metadata)

        if self._selections:
            self._apply_selections()

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def __str__(self) -> str:
        nt = "\n    "
        title = f"{self.__class__.__name__} Dataset"
        sep = "-" * len(title)
        selections = f"Selections: [{', '.join([str(s) for s in self._sort_selections(self._selections)])}]"
        return f"{title}\n{sep}{nt}{selections}\n\n{self._dataset}"

    def _sort_selections(self, selections: Selection[_TDatum] | Sequence[Selection[_TDatum]] | None) -> list[Selection]:
        if not selections:
            return []

        selections = [selections] if isinstance(selections, Selection) else selections
        grouped: dict[int, list[Selection]] = {}
        for selection in selections:
            grouped.setdefault(selection.stage, []).append(selection)
        selection_list = [selection for category in sorted(grouped) for selection in grouped[category]]
        return selection_list

    def _apply_selections(self) -> None:
        for selection in self._selections:
            selection(self)
        self._selection = self._selection[: self._size_limit]

    def __getattr__(self, name: str, /) -> Any:
        selfattr = getattr(self._dataset, name, None)
        return selfattr if selfattr is not None else getattr(self._dataset, name)

    def __getitem__(self, index: int) -> _TDatum:
        return self._dataset[self._selection[index]]

    def __iter__(self) -> Iterator[_TDatum]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._selection)

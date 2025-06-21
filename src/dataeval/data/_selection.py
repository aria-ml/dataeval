from __future__ import annotations

__all__ = []

from collections.abc import Iterator, Sequence
from enum import IntEnum
from typing import Generic, TypeVar

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


class Subselection(Generic[_TDatum]):
    def __call__(self, original: _TDatum) -> _TDatum: ...


class Select(AnnotatedDataset[_TDatum]):
    """
    Dataset wrapper that applies selection criteria for filtering.

    Wraps an existing dataset and applies one or more selection filters to
    create a subset view without modifying the original dataset. Supports
    chaining multiple selection criteria for complex filtering operations.

    Parameters
    ----------
    dataset : AnnotatedDataset[_TDatum]
        Source dataset to wrap and filter. Must implement AnnotatedDataset
        interface with indexed access to data tuples.
    selections : Selection or Sequence[Selection] or None, default None
        Selection criteria to apply for filtering the dataset. When None,
        returns all items from the source dataset. Default None creates
        unfiltered view for consistent interface.

    Examples
    --------
    >>> from dataeval.data.selections import ClassFilter, Limit

    >>> # Construct a sample dataset with size of 100 and class count of 10
    >>> # Elements at index `idx` are returned as tuples:
    >>> # - f"data_{idx}", one_hot_encoded(idx % class_count), {"id": idx}
    >>> dataset = SampleDataset(size=100, class_count=10)

    >>> # Apply selection criteria to the dataset
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

    Notes
    -----
    Selection criteria are applied in the order provided, allowing for
    efficient sequential filtering. The wrapper maintains all metadata
    and interface compatibility with the original dataset.
    """

    _dataset: AnnotatedDataset[_TDatum]
    _selection: list[int]
    _selections: Sequence[Selection[_TDatum]]
    _size_limit: int
    _subselections: list[tuple[Subselection[_TDatum], set[int]]]

    def __init__(
        self,
        dataset: AnnotatedDataset[_TDatum],
        selections: Selection[_TDatum] | Sequence[Selection[_TDatum]] | None = None,
    ) -> None:
        self.__dict__.update(dataset.__dict__)
        self._dataset = dataset
        self._size_limit = len(dataset)
        self._selection = list(range(self._size_limit))
        self._selections = self._sort_selections(selections)
        self._subselections = []

        # Ensure metadata is populated correctly as DatasetMetadata TypedDict
        _metadata = getattr(dataset, "metadata", {})
        if "id" not in _metadata:
            _metadata["id"] = dataset.__class__.__name__
        self._metadata = DatasetMetadata(**_metadata)

        self._apply_selections()

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata information including identifier and configuration."""
        return self._metadata

    def __str__(self) -> str:
        nt = "\n    "
        title = f"{self.__class__.__name__} Dataset"
        sep = "-" * len(title)
        selections = f"Selections: [{', '.join([str(s) for s in self._selections])}]"
        return f"{title}\n{sep}{nt}{selections}{nt}Selected Size: {len(self)}\n\n{self._dataset}"

    def _sort_selections(
        self, selections: Selection[_TDatum] | Sequence[Selection[_TDatum]] | None
    ) -> list[Selection[_TDatum]]:
        if not selections:
            return []

        selections_list = [selections] if isinstance(selections, Selection) else list(selections)
        grouped: dict[int, list[Selection[_TDatum]]] = {}
        for selection in selections_list:
            grouped.setdefault(selection.stage, []).append(selection)
        return [selection for category in sorted(grouped) for selection in grouped[category]]

    def _apply_selections(self) -> None:
        for selection in self._selections:
            selection(self)
        self._selection = self._selection[: self._size_limit]

    def _apply_subselection(self, datum: _TDatum, index: int) -> _TDatum:
        for subselection, indices in self._subselections:
            datum = subselection(datum) if self._selection[index] in indices else datum
        return datum

    def __getitem__(self, index: int) -> _TDatum:
        return self._apply_subselection(self._dataset[self._selection[index]], index)

    def __iter__(self) -> Iterator[_TDatum]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._selection)

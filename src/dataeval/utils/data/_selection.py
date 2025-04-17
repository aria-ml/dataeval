from __future__ import annotations

__all__ = []

from enum import IntEnum
from typing import Any, Callable, Generic, Iterator, Sequence, Tuple, TypeVar, cast

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
    _target_transforms: dict[int, Callable[[Any], Any] | None]
    _metadata_transforms: dict[int, Callable[[dict[str, Any], Any], dict[str, Any]] | None]

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
        
        # Initialize transform dictionaries
        self._target_transforms = {}
        self._metadata_transforms = {}

        # Ensure metadata is populated correctly as DatasetMetadata TypedDict
        _metadata = getattr(dataset, "metadata", {})
        if "id" not in _metadata:
            _metadata["id"] = dataset.__class__.__name__
        self._metadata = DatasetMetadata(**_metadata)

        self._apply_selections()

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def __str__(self) -> str:
        nt = "\n    "
        title = f"{self.__class__.__name__} Dataset"
        sep = "-" * len(title)
        selections = f"Selections: [{', '.join([str(s) for s in self._selections])}]"
        return f"{title}\n{sep}{nt}{selections}{nt}Selected Size: {len(self)}\n\n{self._dataset}"

    def _sort_selections(self, selections: Selection[_TDatum] | Sequence[Selection[_TDatum]] | None)\
          -> list[Selection[_TDatum]]:
        if not selections:
            return []

        selections_list = [selections] if isinstance(selections, Selection) else list(selections)
        grouped: dict[int, list[Selection[_TDatum]]] = {}
        for selection in selections_list:
            grouped.setdefault(selection.stage, []).append(selection)
        selection_list = [selection for category in sorted(grouped) for selection in grouped[category]]
        return selection_list

    def _apply_selections(self) -> None:
        for selection in self._selections:
            selection(self)
        self._selection = self._selection[: self._size_limit]

    def __getitem__(self, index: int) -> _TDatum:
        # Get the original index
        orig_idx = self._selection[index]
        
        # Get the original item
        item = self._dataset[orig_idx]
        
        # Apply transformations if needed
        if orig_idx in self._target_transforms or orig_idx in self._metadata_transforms:
            # Cast item to a tuple of specific structure for type checking
            metadata = cast(Tuple[Any, Any, dict[str, Any]], item)
            image, target, metadata = cast(Tuple[Any, Any, dict[str, Any]], item)

            # Apply target transformations
            if orig_idx in self._target_transforms and self._target_transforms[orig_idx] is not None:
                transform_func = self._target_transforms[orig_idx]

                transformed_target = \
                    transform_func(target) if transform_func is not None else target
            else:
                transformed_target = target
            
            # Apply metadata transformations
            if metadata and orig_idx in self._metadata_transforms and self._metadata_transforms[orig_idx] is not None:
                transform_func = self._metadata_transforms[orig_idx]
                if transform_func is not None:  # Extra check for type checker
                    transformed_metadata = transform_func(metadata, transformed_target)
                    metadata = transformed_metadata
            
            # Recreate the item tuple with transformed components
            # and cast back to original type for return
            return cast(_TDatum, (image, transformed_target, metadata))
        
        return item

    def __iter__(self) -> Iterator[_TDatum]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._selection)
    
class _TargetWrapper:
    """
    Wrapper class for object detection or segmentation targets that preserves interface without 
    instantiating protocols. A function that transforms a target, e.g. to exclude some detections
    from the return value of dataset.__getitem__(idx)[1], will return a _TargetWrapper instance. 

    See _classfilter.py for an example of how it is used. 
    """
    boxes: Any | None = None
    labels: Any = None
    scores: Any | None = None
    mask: Any | None = None
    
    def __init__(self, boxes: Any | None = None, labels: Any = None, 
                 scores: Any | None = None, mask: Any | None = None) -> None:
        if boxes is not None:
            self.boxes = boxes
        if labels is not None:
            self.labels = labels
        if scores is not None:
            self.scores = scores
        if mask is not None:
            self.mask = mask

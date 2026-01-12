__all__ = []

from collections.abc import Iterable, Mapping, Sequence, Sized
from typing import Any, Generic, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Array, ObjectDetectionDatum, ObjectDetectionTarget, SegmentationDatum, SegmentationTarget
from dataeval.selection._select import Select, Selection, SelectionStage, Subselection
from dataeval.utils.arrays import as_numpy


class ClassFilter(Selection[Any]):
    """
    Select dataset indices based on class labels, keeping only those present in `classes`.

    Parameters
    ----------
    classes : Sequence[int]
        The sequence of classes to keep.
    filter_detections : bool, default True
        Whether to filter detections from targets for object detection and segmentation datasets.
    """

    stage = SelectionStage.FILTER

    def __init__(self, classes: Sequence[int], filter_detections: bool = True) -> None:
        self.classes = classes
        self.filter_detections = filter_detections

    def __call__(self, dataset: Select[Any]) -> None:
        if not self.classes:
            return

        selection = []
        subselection = set()
        for idx in dataset._selection:
            target = dataset._dataset[idx][1]
            if isinstance(target, Array):
                # Get the label for the image
                label = int(np.argmax(as_numpy(target)))
                # Check to see if the label is in the classes to keep
                if label in self.classes:
                    # Include the image index
                    selection.append(idx)
            elif isinstance(target, ObjectDetectionTarget | SegmentationTarget):
                # Get the set of labels from the target
                labels = set(target.labels if isinstance(target.labels, Iterable) else [target.labels])
                # Check to see if any labels are in the classes to filter for
                if labels.intersection(self.classes):
                    # Include the image index
                    selection.append(idx)
                    # If we are filtering out other labels and there are other labels, add a subselection filter
                    if self.filter_detections and labels.difference(self.classes):
                        subselection.add(idx)
            else:
                raise TypeError(f"ClassFilter does not support targets of type {type(target)}.")

        dataset._selection = selection
        dataset._subselections.append((ClassFilterSubSelection(self.classes), subselection))


_T = TypeVar("_T")
_TDatum = TypeVar("_TDatum", ObjectDetectionDatum, SegmentationDatum)
_TTarget = TypeVar("_TTarget", ObjectDetectionTarget, SegmentationTarget)


def _try_mask_object(obj: _T, mask: NDArray[np.bool_]) -> _T:
    if not isinstance(obj, str | bytes | bytearray) and isinstance(obj, Sequence | Array) and len(obj) == len(mask):
        return obj[mask] if isinstance(obj, Array) else cast(_T, [item for i, item in enumerate(obj) if mask[i]])
    return obj


class ClassFilterTarget(Generic[_TTarget]):
    def __init__(self, target: _TTarget, mask: NDArray[np.bool_]) -> None:
        self.__dict__.update(target.__dict__)
        self._length = len(target.labels) if isinstance(target.labels, Sized) else int(bool(target.labels))
        self._mask = mask
        self._target = target

    def __getattribute__(self, name: str) -> Any:
        if name in ("_length", "_mask", "_target") or name.startswith("__") and name.endswith("__"):
            return super().__getattribute__(name)

        attr = getattr(self._target, name)
        return _try_mask_object(attr, self._mask)


class ClassFilterSubSelection(Subselection[Any]):
    def __init__(self, classes: Sequence[int]) -> None:
        self.classes = classes

    def _filter(self, d: Mapping[str, Any], mask: NDArray[np.bool_]) -> dict[str, Any]:
        return {k: self._filter(v, mask) if isinstance(v, dict) else _try_mask_object(v, mask) for k, v in d.items()}

    def __call__(self, datum: _TDatum) -> _TDatum:
        # build a mask for any arrays
        image, target, metadata = datum

        mask = np.isin(as_numpy(target.labels), self.classes)
        filtered_metadata = self._filter(metadata, mask)

        # return a masked datum
        filtered_datum = image, ClassFilterTarget(target, mask), filtered_metadata
        return cast(_TDatum, filtered_datum)

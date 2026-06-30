__all__ = []

from collections.abc import Iterable, Sequence
from typing import Any, TypeVar, cast

import numpy as np

from dataeval.data._select import Select, Selection, SelectionStage, Subselection
from dataeval.protocols import Array, ObjectDetectionDatum, ObjectDetectionTarget, SegmentationDatum, SegmentationTarget
from dataeval.utils._internal import MaskedTarget, as_numpy, mask_metadata
from dataeval.utils._validate import DatasetKind


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
    requires: DatasetKind | None = "any_target"

    def __init__(self, classes: Sequence[int], filter_detections: bool = True) -> None:
        self.classes = classes
        self.filter_detections = filter_detections

    def __call__(self, dataset: Select[Any]) -> None:  # noqa: C901
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


_TDatum = TypeVar("_TDatum", ObjectDetectionDatum, SegmentationDatum)


class ClassFilterSubSelection(Subselection[Any]):
    def __init__(self, classes: Sequence[int]) -> None:
        self.classes = classes

    def __call__(self, datum: _TDatum) -> _TDatum:
        # build a mask for any arrays
        image, target, metadata = datum

        mask = np.isin(as_numpy(target.labels), self.classes)
        filtered_metadata = mask_metadata(metadata, mask)

        # return a masked datum
        filtered_datum = image, MaskedTarget(target, mask), filtered_metadata
        return cast(_TDatum, filtered_datum)

__all__ = []

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from dataeval.data._view import Operation, View
from dataeval.protocols import Array, ObjectDetectionTarget, SegmentationTarget
from dataeval.utils._array import argmax_label, as_numpy
from dataeval.utils._mask import MaskedTarget, mask_metadata
from dataeval.utils.data import DatasetKind


class ClassFilter(Operation):
    """
    Select dataset indices based on class labels, keeping only those present in `classes`.

    Filters images by class (cardinality) and, for object-detection and segmentation
    datasets, masks out the detections that belong to other classes (content). Reads
    each datum through preceding operations, so ``[Relabel(...), ClassFilter([0])]``
    filters on the relabeled vocabulary.

    Parameters
    ----------
    classes : Sequence[int]
        The sequence of classes to keep.
    filter_detections : bool, default True
        Whether to filter detections from targets for object detection and segmentation datasets.
    """

    requires: DatasetKind | None = "any_target"

    def __init__(self, classes: Sequence[int], filter_detections: bool = True) -> None:
        self.classes = classes
        self.filter_detections = filter_detections

    def apply(self, view: View[Any]) -> None:  # noqa: C901
        if not self.classes:
            return

        keep = set(self.classes)
        selection: list[int] = []
        mask_where: set[int] = set()
        for idx in view.selection:
            target = view.read(idx)[1]  # through preceding operations (e.g. Relabel)
            if isinstance(target, Array):
                if argmax_label(target) in keep:
                    selection.append(idx)
            elif isinstance(target, ObjectDetectionTarget | SegmentationTarget):
                labels = set(np.atleast_1d(as_numpy(target.labels)).tolist())
                if labels & keep:
                    selection.append(idx)
                    if self.filter_detections and (labels - keep):
                        mask_where.add(idx)
            else:
                raise TypeError(f"ClassFilter does not support targets of type {type(target)}.")

        view.selection = selection
        if mask_where:
            view.map(_mask_to_classes(self.classes), where=mask_where)


def _mask_to_classes(classes: Sequence[int]) -> Callable[[Any], Any]:
    keep = list(classes)

    def mask(datum: Any) -> Any:
        image, target, metadata = datum
        detection_mask = np.isin(as_numpy(target.labels), keep)
        return image, MaskedTarget(target, detection_mask), mask_metadata(metadata, detection_mask)

    return mask

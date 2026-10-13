"""The class-only filter :class:`~dataeval.data.Filter` replaced, kept until 2.0."""

__all__ = []

from collections.abc import Sequence

from dataeval._experimental import deprecated
from dataeval.data._filter import Filter
from dataeval.utils.data import DatasetKind


@deprecated(since="1.2", removal="2.0", alternative="Filter(classes=...)")
class ClassFilter(Filter):
    """
    Select dataset items by class label, keeping only those holding a label in `classes`.

    .. deprecated:: 1.2
        Use ``Filter(classes=...)``. ``filter_detections=False`` is ``drop="items"``.

    Parameters
    ----------
    classes : Sequence[int]
        The classes to keep. An empty sequence keeps everything.
    filter_detections : bool, default True
        Whether to also remove the detections of other classes from the items kept.
    """

    requires: DatasetKind | None = "any_target"

    def __init__(self, classes: Sequence[int], filter_detections: bool = True) -> None:
        self.filter_detections = filter_detections
        if len(classes):
            super().__init__(classes=classes, drop="both" if filter_detections else "items")
        else:
            # ClassFilter([]) has always kept everything, where Filter refuses an empty `classes`.
            self.where, self.classes, self.drop = None, classes, "both"

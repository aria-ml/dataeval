"""A dataset's datums without their pixels: everything structuring and filtering read, and nothing they do not."""

__all__ = []

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from dataeval.protocols import DatasetMetadata
from dataeval.types._task import detect_task

_FRAME_FIELDS = ("frame_index", "time_s", "pts")
"""What the tracking structurer reads from a frame besides its size (see ``types/_track.py``)."""


def _frame_stub(frame: Any) -> SimpleNamespace:
    """One video frame's number, timings and size, and none of its pixels.

    A field the frame does not declare is left undeclared rather than set to None, because the structurer tells the
    two apart the way it does for the frame itself.
    """
    stub = SimpleNamespace(**{name: getattr(frame, name) for name in _FRAME_FIELDS if hasattr(frame, name)})
    shape = getattr(getattr(frame, "pixels", None), "shape", None)
    if shape is not None:
        stub.pixels = SimpleNamespace(shape=tuple(shape))
    return stub


def skeleton(datum: Any) -> tuple[Any, Any, Any]:
    """Return `datum` without its pixels, keeping everything structuring reads.

    Structuring classification, detection and segmentation data never reads the image, so it becomes None. Structuring
    tracking data reads each frame's number, timings and size, so a video becomes a list of frame stubs — reading the
    stream to build them is the one decode a skeleton costs.

    Parameters
    ----------
    datum : tuple[Any, Any, Any]
        An ``(item, target, metadata)`` datum.

    Returns
    -------
    tuple[Any, Any, Any]
        The datum with its item replaced, and its target and metadata as they were.
    """
    item, target, metadata = datum
    if detect_task(target) == "MOT":
        return [_frame_stub(frame) for frame in item], target, metadata
    return None, target, metadata


class Reads:
    """Skeletons presented as a dataset, so a structurer can read them in place of the view they came from.

    Parameters
    ----------
    skeletons : Sequence[tuple[Any, Any, Any]]
        One skeleton per datum, in the order the dataset presents them.
    metadata : DatasetMetadata
        The dataset-level metadata of the dataset the skeletons came from.
    """

    def __init__(self, skeletons: Sequence[tuple[Any, Any, Any]], metadata: DatasetMetadata) -> None:
        self._skeletons = skeletons
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self._skeletons)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        return self._skeletons[index]

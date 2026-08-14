"""Reading detections off a target, for the tasks whose labels are instances."""

__all__ = []

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ObjectDetectionTarget, SingleFrameObjectTrackingTarget
from dataeval.utils._internal import as_numpy


class InstanceBuildingMixin:
    """Box/label extraction shared by instance-producing structurers.

    Used by object detection and multi-object tracking strategies. A
    :obj:`~dataeval.protocols.SingleFrameObjectTrackingTarget` is an
    :obj:`~dataeval.protocols.ObjectDetectionTarget` plus ``track_ids``, so both tasks
    read boxes and labels the same way and tracking adds one call on top.
    """

    @staticmethod
    def _instance_arrays(
        target: ObjectDetectionTarget | SingleFrameObjectTrackingTarget,
    ) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32]]:
        """Extract per-detection labels, boxes and scores from a detection target.

        Returns
        -------
        tuple
            ``(labels, boxes, scores)`` with one entry per detection. ``scores``
            keeps its original shape, which may be per-detection or
            per-detection-per-class.
        """
        labels = as_numpy(target.labels).reshape(-1).astype(np.intp)
        count = len(labels)
        boxes = (
            as_numpy(target.boxes).astype(np.float32).reshape(count, 4) if count else np.empty((0, 4), dtype=np.float32)
        )
        scores = as_numpy(target.scores).astype(np.float32) if count else np.empty(0, dtype=np.float32)
        return labels, boxes, scores

    @staticmethod
    def _track_ids(target: SingleFrameObjectTrackingTarget, count: int) -> NDArray[np.intp]:
        """Extract per-detection track ids from one frame's tracking target.

        Parameters
        ----------
        target : SingleFrameObjectTrackingTarget
            Frame target to read ``track_ids`` from.
        count : int
            Detections in this frame, as already established by :meth:`_instance_arrays`.

        Returns
        -------
        NDArray[np.intp]
            One track id per detection, ``-1`` where a detection belongs to no track.
        """
        if not count:
            return np.empty(0, dtype=np.intp)
        return as_numpy(target.track_ids).reshape(-1).astype(np.intp)

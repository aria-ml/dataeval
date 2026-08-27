"""Reading detections off a target, for the tasks whose labels are instances."""

__all__ = []

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ObjectDetectionTarget, SingleFrameObjectTrackingTarget
from dataeval.types._target import instance_arrays, track_ids_of


class InstanceBuildingMixin:
    """Box/label extraction shared by instance-producing structurers.

    Used by object detection and multi-object tracking strategies. A
    :obj:`~dataeval.protocols.SingleFrameObjectTrackingTarget` is an
    :obj:`~dataeval.protocols.ObjectDetectionTarget` plus ``track_ids``, so both tasks
    read boxes and labels the same way and tracking adds one call on top.

    The reading itself lives in ``dataeval.types._target``, because following a
    :class:`~dataeval.types.SourceIndex` back to a detection needs the same answers and is
    not done from inside this package. These two forward to it so that the structurers'
    call sites are unchanged.
    """

    @staticmethod
    def _instance_arrays(
        target: ObjectDetectionTarget | SingleFrameObjectTrackingTarget,
    ) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32]]:
        """One target's per-detection labels, boxes and scores."""
        return instance_arrays(target)

    @staticmethod
    def _track_ids(target: SingleFrameObjectTrackingTarget, count: int) -> NDArray[np.intp]:
        """One frame's per-detection track ids, ``-1`` where a detection has no track."""
        return track_ids_of(target, count)

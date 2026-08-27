"""Object detection over images: one image per item, many detections per image."""

__all__ = []

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval._metadata._structurers._block import RowBlock
from dataeval._metadata._structurers._data import StructuredData
from dataeval._metadata._structurers._dataset import DatasetStructurer
from dataeval._metadata._structurers._instances import InstanceBuildingMixin
from dataeval._metadata._structurers._ordering import running_index
from dataeval._metadata._structurers._propagation import PropagationMixin
from dataeval._metadata._structurers._reporting import log_items_without_targets
from dataeval._metadata._structurers._reserved import reserved_block_columns
from dataeval.protocols import AnnotatedDataset, DatumMetadata, ObjectDetectionTarget, ProgressCallback
from dataeval.types._task import TASK_PROFILES

_logger = get_logger(__name__)


class ODImageStructurer(InstanceBuildingMixin, PropagationMixin, DatasetStructurer):
    """Object detection over images: items are images, targets are instances."""

    profile = TASK_PROFILES["OD"]
    multi_target = True

    # Object detection called its target rows ``"target"`` through v1.1.0. It is the
    # only task that ever did, so it is the only one that translates the name. This
    # overrides rather than merges with the base map, so ``"image"`` is repeated here.
    legacy_level_aliases = MappingProxyType({"target": "instance", "image": "unit"})

    def build(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> StructuredData:
        raw: list[Mapping[str, Any]] = []
        labels: list[NDArray[Any]] = []
        boxes: list[NDArray[Any]] = []
        scores: list[NDArray[Any]] = []
        srcidx: list[int] = []

        count = len(dataset)
        # An image with no detections contributes no instance rows at all. That is far
        # more common here than an unlabeled item is for classification, so it is
        # tracked and reported for the same reason: without it, label-aware analysis
        # silently covers a subset of the dataset and the first sign is a row-count
        # mismatch from add_factors much later.
        undetected: list[int] = []
        for i in range(count):
            _, target, metadata = self._datum(dataset, i)
            raw.append(metadata)
            if not isinstance(target, ObjectDetectionTarget):
                raise TypeError(
                    f"Encountered unsupported target type {type(target).__name__} for task {self.task}.",
                )
            instance_labels, instance_boxes, instance_scores = self._instance_arrays(target)
            if len(instance_labels):
                labels.append(instance_labels)
                boxes.append(instance_boxes)
                scores.append(instance_scores)
                srcidx.extend([i] * len(instance_labels))
            else:
                undetected.append(i)
            if progress_callback:
                progress_callback(i + 1, total=count)

        unit_of_instance = np.asarray(srcidx, dtype=np.intp)
        class_labels = np.concatenate(labels).astype(np.intp) if labels else np.empty(0, dtype=np.intp)
        box_values = np.concatenate(boxes).astype(np.float32) if boxes else np.empty((0, 4), dtype=np.float32)
        score_values = np.concatenate(scores).astype(np.float32) if scores else np.empty(0, dtype=np.float32)
        instance_index = running_index(unit_of_instance)
        instances = len(unit_of_instance)

        instances_per_item = np.bincount(unit_of_instance, minlength=count).astype(int).tolist()
        instance_factors, dropped = self._merge_factors(
            raw,
            ignore_lists=False,
            targets_per_item=instances_per_item,
        )
        unit_factors, _ = self._merge_factors(raw, ignore_lists=True)
        # Anything the target-level merge produced that the item-level merge also
        # produced is item metadata replicated across instances; keep it at the ``unit``
        # level and let propagation replicate it instead of storing it twice.
        instance_factors = {name: values for name, values in instance_factors.items() if name not in unit_factors}

        unit_block = RowBlock(
            "unit",
            count,
            reserved_block_columns("unit", count, item_index=list(range(count))),
            {"unit": self._own_positions(count)},
        )
        # ``instance_index`` is the instance level's own key component; ``target_index`` is the
        # legacy public spelling of "index within the item at whatever level the labels
        # sit". For this task they are the same number, written from the same array so
        # they cannot drift; a task whose targets are not instances would fill them apart.
        instance_block = RowBlock(
            "instance",
            instances,
            reserved_block_columns(
                "instance",
                instances,
                item_index=unit_of_instance,
                target_index=instance_index,
                class_label=class_labels,
                score=score_values,
                box=box_values,
                instance_index=instance_index,
            ),
            {**self._inherit(unit_block.ancestor_pos, unit_of_instance), "instance": self._own_positions(instances)},
        )

        log_items_without_targets(undetected, "instance", count)
        _logger.info(
            "Object Detection dataset: %d images, %d classes, %d detections",
            count,
            len(np.unique(class_labels)),
            instances,
        )
        return StructuredData(
            [unit_block, instance_block],
            {"unit": unit_factors, "instance": instance_factors},
            dropped,
            raw,
            class_labels,
            unit_of_instance,
        )

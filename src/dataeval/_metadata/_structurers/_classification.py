"""Image classification: one image per item, one label on the image itself."""

__all__ = []

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval._metadata._structurers._block import RowBlock
from dataeval._metadata._structurers._data import StructuredData
from dataeval._metadata._structurers._dataset import DatasetStructurer
from dataeval._metadata._structurers._ordering import running_index
from dataeval._metadata._structurers._propagation import PropagationMixin
from dataeval._metadata._structurers._reporting import log_items_without_targets
from dataeval._metadata._structurers._reserved import reserved_block_columns
from dataeval.protocols import AnnotatedDataset, Array, DatumMetadata, ProgressCallback
from dataeval.types import FactorLevelSchema
from dataeval.utils._internal import as_numpy

_logger = get_logger(__name__)


class ICStructurer(PropagationMixin, DatasetStructurer):
    """Image classification: items are images, targets are the images themselves.

    The instance level is separate from the ``unit`` level even though a classification
    instance *is* the whole image, because the two answer different questions: an
    image row exists for every dataset item, an instance row only where there is a
    label to attach. Collapsing them would delete an unlabeled item — and all of its
    metadata — from the dataframe entirely.

    Sharing the level with object detection is what keeps one object one thing: a
    detection is an instance in an object detection dataset, and the same detection
    seen through :class:`~dataeval.data.DetectionCrops` is an instance here too.
    """

    task = "IC"
    levels = FactorLevelSchema.of("unit", "instance")
    item_level = "unit"
    label_level = "instance"
    unit_type = "image"

    def build(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> StructuredData:
        raw: list[Mapping[str, Any]] = []
        labels: list[int] = []
        scores: list[NDArray[Any]] = []
        srcidx: list[int] = []

        count = len(dataset)
        unlabeled: list[int] = []
        for i in range(count):
            _, target, metadata = self._datum(dataset, i)
            raw.append(metadata)
            if not isinstance(target, Array):
                raise TypeError(
                    f"Encountered unsupported target type {type(target).__name__} for task {self.task}.",
                )
            values = as_numpy(target)
            if len(values):
                labels.append(int(np.argmax(values)))
                scores.append(values)
                srcidx.append(i)
            else:
                unlabeled.append(i)
            if progress_callback:
                progress_callback(i + 1, total=count)

        unit_of_instance = np.asarray(srcidx, dtype=np.intp)
        class_labels = np.asarray(labels, dtype=np.intp)
        score_values = np.asarray(scores, dtype=np.float32) if scores else np.empty(0, dtype=np.float32)
        instance_index = running_index(unit_of_instance)
        instance_count = len(unit_of_instance)

        instances_per_item = np.bincount(unit_of_instance, minlength=count).astype(int).tolist()
        instance_factors, dropped = self._merge_factors(
            raw,
            ignore_lists=False,
            targets_per_item=instances_per_item,
        )
        unit_factors, _ = self._merge_factors(raw, ignore_lists=True)
        # Same rule as object detection: a name both merges produced is item metadata
        # replicated onto the target rows, so keep it once at the ``unit`` level and let
        # propagation do the replicating.
        instance_factors = {name: values for name, values in instance_factors.items() if name not in unit_factors}

        unit_block = RowBlock(
            "unit",
            count,
            reserved_block_columns("unit", count, item_index=list(range(count))),
            {"unit": self._own_positions(count)},
        )
        instance_block = RowBlock(
            "instance",
            instance_count,
            reserved_block_columns(
                "instance",
                instance_count,
                item_index=unit_of_instance,
                # One instance per image at most, so the index within the image is
                # always 0 — but derive it rather than assume, as object detection does.
                # ``instance_index`` is the instance level's own key column and is
                # written by every structurer that declares the level, so that caller
                # code reading rows_at(level)[f"{level}_index"] does not branch on task.
                target_index=instance_index,
                class_label=class_labels,
                score=score_values,
                instance_index=instance_index,
            ),
            {
                **self._inherit(unit_block.ancestor_pos, unit_of_instance),
                "instance": self._own_positions(instance_count),
            },
        )

        log_items_without_targets(unlabeled, "instance", count)
        _logger.info("%s dataset: %d items, %d classes", self.task, count, len(np.unique(class_labels)))
        return StructuredData(
            [unit_block, instance_block],
            {"unit": unit_factors, "instance": instance_factors},
            dropped,
            raw,
            class_labels,
            unit_of_instance,
        )

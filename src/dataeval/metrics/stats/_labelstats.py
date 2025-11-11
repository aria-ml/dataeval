from __future__ import annotations

__all__ = []

from typing import Any

from dataeval.core import label_stats
from dataeval.data._metadata import Metadata
from dataeval.outputs import LabelStatsOutput
from dataeval.protocols import AnnotatedDataset
from dataeval.types import set_metadata


@set_metadata
def labelstats(dataset: Metadata | AnnotatedDataset[Any]) -> LabelStatsOutput:
    """
    Calculates :term:`statistics<Statistics>` for data labels.

    This function computes counting metrics (e.g., total per class, total per image)
    on the labels.

    Parameters
    ----------
    dataset : Metadata or ImageClassificationDataset or ObjectDetect

    Returns
    -------
    LabelStatsOutput
        A dataclass containing the computed counting metrics for the labels.

    Examples
    --------
    Calculate basic :term:`statistics<Statistics>` on labels for a dataset.

    >>> from dataeval.data import Metadata
    >>> stats = labelstats(Metadata(dataset))
    >>> print(stats.to_table())
    Class Count: 5
    Label Count: 15
    Average # Labels per Image: 1.88
    --------------------------------------
      Label: Total Count - Image Count
      horse:      2      -      2
        cow:      4      -      3
      sheep:      2      -      2
        pig:      2      -      2
    chicken:      5      -      5
    """
    # Convert AnnotatedDataset to Metadata if needed
    metadata = Metadata(dataset) if isinstance(dataset, AnnotatedDataset) else dataset

    # Extract labels grouped by image from the metadata
    # Build a list of lists where each inner list contains the labels for one image
    labels: list[list[int]] = [[] for _ in range(metadata.item_count)]
    for class_label, item_index in zip(metadata.class_labels, metadata.item_indices):
        labels[item_index].append(int(class_label))

    # Create index2label mapping from class_names
    # Assumes class_names are ordered by class index
    index2label = dict(enumerate(metadata.class_names))

    # Call the core function
    stats_dict = label_stats(labels, index2label)

    # Wrap the result in LabelStatsOutput
    return LabelStatsOutput(
        label_counts_per_class=stats_dict["label_counts_per_class"],
        label_counts_per_image=stats_dict["label_counts_per_image"],
        image_counts_per_class=stats_dict["image_counts_per_class"],
        image_indices_per_class=stats_dict["image_indices_per_class"],
        image_count=stats_dict["image_count"],
        class_count=stats_dict["class_count"],
        label_count=stats_dict["label_count"],
        class_names=stats_dict["class_names"],
    )

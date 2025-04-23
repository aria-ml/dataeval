from __future__ import annotations

__all__ = []

from collections import Counter, defaultdict
from typing import Any, Mapping, TypeVar

from dataeval.data._metadata import Metadata
from dataeval.outputs import LabelStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import AnnotatedDataset

TValue = TypeVar("TValue")


def _sort_to_list(d: Mapping[int, TValue]) -> list[TValue]:
    return [t[1] for t in sorted(d.items())]


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
    dataset = Metadata(dataset) if isinstance(dataset, AnnotatedDataset) else dataset

    label_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    index_location = defaultdict(list[int])
    label_per_image: list[int] = []

    index2label = dict(enumerate(dataset.class_names))

    for i, target in enumerate(dataset.targets):
        group = target.labels.tolist()

        # Count occurrences of each label in all sublists
        label_counts.update(group)

        # Get the number of labels per image
        label_per_image.append(len(group))

        # Create a set of unique items in the current sublist
        unique_items: set[int] = set(group)

        # Update image counts and index locations
        image_counts.update(unique_items)
        for item in unique_items:
            index_location[item].append(i)

    return LabelStatsOutput(
        label_counts_per_class=_sort_to_list(label_counts),
        label_counts_per_image=label_per_image,
        image_counts_per_class=_sort_to_list(image_counts),
        image_indices_per_class=_sort_to_list(index_location),
        image_count=len(label_per_image),
        class_count=len(label_counts),
        label_count=sum(label_counts.values()),
        class_names=list(index2label.values()),
    )

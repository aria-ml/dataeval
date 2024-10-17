from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypeVar

from numpy.typing import ArrayLike

from dataeval._internal.interop import to_numpy
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class LabelStatsOutput(OutputMetadata):
    """
    Output class for :func:`labelstats` stats metric

    Attributes
    ----------
    label_counts_per_class : dict[str | int, int]
        Dictionary whose keys are the different label classes and
        values are total counts of each class
    label_counts_per_image : list[int]
        Number of labels per image
    image_counts_per_label : dict[str | int, int]
        Dictionary whose keys are the different label classes and
        values are total counts of each image the class is present in
    image_indices_per_label : dict[str | int, list]
        Dictionary whose keys are the different label classes and
        values are lists containing the images that have that label
    image_count : int
        Total number of images present
    class_count : int
        Total number of classes present
    label_count : int
        Total number of labels present
    """

    label_counts_per_class: dict[str | int, int]
    label_counts_per_image: list[int]
    image_counts_per_label: dict[str | int, int]
    image_indices_per_label: dict[str | int, list[int]]
    image_count: int
    class_count: int
    label_count: int


TKey = TypeVar("TKey", int, str)


def sort(d: Mapping[TKey, Any]) -> dict[TKey, Any]:
    """
    Sort mappings by key in increasing order
    """
    return dict(sorted(d.items(), key=lambda x: x[0]))


@set_metadata("dataeval.metrics")
def labelstats(
    labels: Iterable[ArrayLike],
) -> LabelStatsOutput:
    """
    Calculates statistics for data labels

    This function computes counting metrics (e.g., total per class, total per image)
    on the labels.

    Parameters
    ----------
    labels : ArrayLike, shape - [label] | [[label]] or (N,M) | (N,)
        Lists or numpy array of labels.
        A set of lists where each list contains all labels per image -
        (e.g. [[label1, label2], [label2], [label1, label3]] or [label1, label2, label1, label3]).
        If a numpy array, N is the number of images, M is the number of labels per image.

    Returns
    -------
    LabelStatsOutput
        A dictionary-like object containing the computed counting metrics for the labels.

    Examples
    --------
    Calculating the statistics on labels for a set of data

    >>> stats = labelstats(labels)
    >>> stats.label_counts_per_class
    {'chicken': 3, 'cow': 8, 'horse': 9, 'pig': 7, 'sheep': 7}
    >>> stats.label_counts_per_image
    [3, 2, 3, 4, 1, 5, 4, 4, 4, 4]
    >>> stats.image_counts_per_label
    {'chicken': 2, 'cow': 6, 'horse': 7, 'pig': 5, 'sheep': 7}
    >>> (stats.image_count, stats.class_count, stats.label_count)
    (10, 5, 34)
    """
    label_counts = Counter()
    image_counts = Counter()
    index_location = defaultdict(list[int])
    label_per_image: list[int] = []

    for i, group in enumerate(labels):
        # Count occurrences of each label in all sublists
        group = to_numpy(group)

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
        label_counts_per_class=sort(label_counts),
        label_counts_per_image=label_per_image,
        image_counts_per_label=sort(image_counts),
        image_indices_per_label=sort(index_location),
        image_count=len(label_per_image),
        class_count=len(label_counts),
        label_count=sum(label_counts.values()),
    )

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from dataeval._internal.flags import to_distinct
from dataeval._internal.output import OutputMetadata, populate_defaults, set_metadata


@dataclass(frozen=True)
class LabelStatsOutput(OutputMetadata):
    """
    Attributes
    ----------
    label_per_class : dict[str, dict[str, int]]
        Dictionary whose keys are the different label classes and
        values are total counts of each class
    image_per_label : dict[str, dict[str, int]]
        Dictionary whose keys are the different label classes and
        values are total counts of each image the class is present int
    image_index_per_label : dict[str, dict[str, list]]
        Dictionary whose keys are the different label classes and
        values are lists containing the images that have that label
    label_count : NDArray
        Number of labels per image
    total_label : int
        Total number of labels present
    total_class : int
        Total number of classes present
    """

    label_counts_per_class: dict[str, dict[str, int]]
    label_counts_per_image: NDArray
    image_counts_per_label: dict[str, dict[str, int]]
    image_index_per_label: dict[str, dict[str, list]]
    total_label_count: int
    total_class_count: int


def run_labelstats(
    labels: Iterable,
):

    label_counts = Counter()
    image_counts = Counter()
    index_location = defaultdict(list)
    label_per_image = []

    for i, group in enumerate(labels):
        # Count occurrences of each label in all sublists
        label_counts.update(group)

        # Get the number of labels per image
        label_per_image.append(len(group))

        # Create a set of unique items in the current sublist
        unique_items = set(group)

        # Update image counts and index locations
        image_counts.update(unique_items)
        for item in unique_items:
            index_location[item].append(i)

    output: dict[str, dict[str, int] | dict[str, list] | int | NDArray] = {}
    output['label_counts_per_class'] = label_counts
    output['label_counts_per_image'] = np.asarray(label_per_image)
    output['image_counts_per_label'] = image_counts
    output['image_index_per_label'] = index_location
    output['total_class_count'] = len(list(label_counts))
    output['total_label_count'] = sum([v for _, v in label_counts.items()])
    return output


@set_metadata("dataeval.metrics")
def labelstats(
    labels: Iterable,
) -> LabelStatsOutput:
    """
    Calculates statistics for data labels

    This function computes counting metrics (e.g., total per class, total per image)
    on the labels. It supports multiple types of counting metrics
    that can be selected using the `flags` argument.

    Parameters
    ----------
    labels : ArrayLike, shape - [label] | [[label]] or (N,M) | (N,)
        Lists or numpy array of labels.
        A set of lists where each list contains all labels per image -
        (e.g. [[label1, label2], [label2], [label1, label3]] or [label1, label2, label1, label3]).
        If a numpy array, N is the number of images, M is the number of labels per image.
    flags : LabelStat, default LabelStat.ALL
        Metric(s) to calculate for each label. The default flag ``LabelStat.ALL``
        computes all available label counting metrics on each label.

    Returns
    -------
    StatsLabelOutput
        A dictionary-like object containing the computed counting metrics for the labels. The keys correspond
        to the names of the metric (e.g., 'total_class', 'total_image'), and ...

    Examples
    --------
    Calculating the statistics on labels for a set of data

    >>> labelstats(labels)
    """
    stats = run_labelstats(labels)
    return LabelStatsOutput(**populate_defaults(stats, LabelStatsOutput))

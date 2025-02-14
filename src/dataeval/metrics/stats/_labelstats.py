from __future__ import annotations

__all__ = []

import contextlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypeVar

import numpy as np

from dataeval._output import Output, set_metadata
from dataeval.typing import ArrayLike
from dataeval.utils._array import as_numpy

with contextlib.suppress(ImportError):
    import pandas as pd


@dataclass(frozen=True)
class LabelStatsOutput(Output):
    """
    Output class for :func:`.labelstats` stats metric.

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

    def to_table(self) -> str:
        max_char = max(len(key) if isinstance(key, str) else key // 10 + 1 for key in self.label_counts_per_class)
        max_char = max(max_char, 5)
        max_label = max(list(self.label_counts_per_class.values()))
        max_img = max(list(self.image_counts_per_label.values()))
        max_num = int(np.ceil(np.log10(max(max_label, max_img))))
        max_num = max(max_num, 11)

        # Display basic counts
        table_str = f"Class Count: {self.class_count}\n"
        table_str += f"Label Count: {self.label_count}\n"
        table_str += f"Average # Labels per Image: {round(np.mean(self.label_counts_per_image), 2)}\n"
        table_str += "--------------------------------------\n"

        # Display counts per class
        table_str += f"{'Label':>{max_char}}: Total Count - Image Count\n"
        for cls in self.label_counts_per_class:
            table_str += f"{cls:>{max_char}}: {self.label_counts_per_class[cls]:^{max_num}} "
            table_str += f"- {self.image_counts_per_label[cls]:^{max_num}}\n"

        return table_str

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        class_list = []
        total_count = []
        image_count = []
        for cls in self.label_counts_per_class:
            class_list.append(cls)
            total_count.append(self.label_counts_per_class[cls])
            image_count.append(self.image_counts_per_label[cls])

        return pd.DataFrame(
            {
                "Label": class_list,
                "Total Count": total_count,
                "Image Count": image_count,
            }
        )


TKey = TypeVar("TKey", int, str)


def sort(d: Mapping[TKey, Any]) -> dict[TKey, Any]:
    """
    Sort mappings by key in increasing order
    """
    return dict(sorted(d.items(), key=lambda x: x[0]))


def _ensure_2d(labels: Iterable[ArrayLike]) -> Iterable[ArrayLike]:
    if isinstance(labels, np.ndarray):
        return labels[:, None]
    else:
        return [[lbl] for lbl in labels]  # type: ignore


def _get_list_depth(lst):
    if isinstance(lst, list) and lst:
        return 1 + max(_get_list_depth(item) for item in lst)
    return 0


def _check_labels_dimension(labels: Iterable[ArrayLike]) -> Iterable[ArrayLike]:
    # Check for nested lists beyond 2 levels

    if isinstance(labels, np.ndarray):
        if labels.ndim == 1:
            return _ensure_2d(labels)
        elif labels.ndim == 2:
            return labels
        else:
            raise ValueError("The label array must not have more than 2 dimensions.")
    elif isinstance(labels, list):
        depth = _get_list_depth(labels)
        if depth == 1:
            return _ensure_2d(labels)
        elif depth == 2:
            return labels
        else:
            raise ValueError("The label list must not be empty or have more than 2 levels of nesting.")
    else:
        raise TypeError("Labels must be either a NumPy array or a list.")


@set_metadata
def labelstats(
    labels: Iterable[ArrayLike],
) -> LabelStatsOutput:
    """
    Calculates :term:`statistics<Statistics>` for data labels.

    This function computes counting metrics (e.g., total per class, total per image)
    on the labels.

    Parameters
    ----------
    labels : ArrayLike, shape - [label] | [[label]] or (N,M) | (N,)
        Lists or :term:`NumPy` array of labels.
        A set of lists where each list contains all labels per image -
        (e.g. [[label1, label2], [label2], [label1, label3]] or [label1, label2, label1, label3]).
        If a numpy array, N is the number of images, M is the number of labels per image.

    Returns
    -------
    LabelStatsOutput
        A dictionary-like object containing the computed counting metrics for the labels.

    Examples
    --------
    Calculating the :term:`statistics<Statistics>` on labels for a set of data

    >>> stats = labelstats(labels)
    >>> stats.label_counts_per_class
    {'chicken': 12, 'cow': 5, 'horse': 4, 'pig': 7, 'sheep': 4}
    >>> stats.label_counts_per_image
    [3, 3, 5, 3, 2, 5, 5, 2, 2, 2]
    >>> stats.image_counts_per_label
    {'chicken': 8, 'cow': 4, 'horse': 4, 'pig': 7, 'sheep': 4}
    >>> (stats.image_count, stats.class_count, stats.label_count)
    (10, 5, 32)
    """
    label_counts = Counter()
    image_counts = Counter()
    index_location = defaultdict(list[int])
    label_per_image: list[int] = []

    labels_2d = _check_labels_dimension(labels)

    for i, group in enumerate(labels_2d):
        group = as_numpy(group)

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
        label_counts_per_class=sort(label_counts),
        label_counts_per_image=label_per_image,
        image_counts_per_label=sort(image_counts),
        image_indices_per_label=sort(index_location),
        image_count=len(label_per_image),
        class_count=len(label_counts),
        label_count=sum(label_counts.values()),
    )

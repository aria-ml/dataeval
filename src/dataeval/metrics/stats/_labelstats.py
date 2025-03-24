from __future__ import annotations

__all__ = []

import contextlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

import numpy as np

from dataeval._output import Output, set_metadata
from dataeval.typing import AnnotatedDataset, ArrayLike
from dataeval.utils._array import as_numpy
from dataeval.utils.data._metadata import Metadata

with contextlib.suppress(ImportError):
    import pandas as pd

TValue = TypeVar("TValue")


@dataclass(frozen=True)
class LabelStatsOutput(Output):
    """
    Output class for :func:`.labelstats` stats metric.

    Attributes
    ----------
    label_counts_per_class : dict[int, int]
        Dictionary whose keys are the different label classes and
        values are total counts of each class
    label_counts_per_image : list[int]
        Number of labels per image
    image_counts_per_class : dict[int, int]
        Dictionary whose keys are the different label classes and
        values are total counts of each image the class is present in
    image_indices_per_class : dict[int, list]
        Dictionary whose keys are the different label classes and
        values are lists containing the images that have that label
    image_count : int
        Total number of images present
    class_count : int
        Total number of classes present
    label_count : int
        Total number of labels present
    class_names : list[str]
    """

    label_counts_per_class: list[int]
    label_counts_per_image: list[int]
    image_counts_per_class: list[int]
    image_indices_per_class: list[list[int]]
    image_count: int
    class_count: int
    label_count: int
    class_names: list[str]

    def to_table(self) -> str:
        max_char = max(len(name) if isinstance(name, str) else name // 10 + 1 for name in self.class_names)
        max_char = max(max_char, 5)
        max_label = max(list(self.label_counts_per_class))
        max_img = max(list(self.image_counts_per_class))
        max_num = int(np.ceil(np.log10(max(max_label, max_img))))
        max_num = max(max_num, 11)

        # Display basic counts
        table_str = [f"Class Count: {self.class_count}"]
        table_str += [f"Label Count: {self.label_count}"]
        table_str += [f"Average # Labels per Image: {round(np.mean(self.label_counts_per_image), 2)}"]
        table_str += ["--------------------------------------"]

        # Display counts per class
        table_str += [f"{'Label':>{max_char}}: Total Count - Image Count"]
        for cls in range(len(self.class_names)):
            table_str += [
                f"{self.class_names[cls]:>{max_char}}: {self.label_counts_per_class[cls]:^{max_num}}"
                + " - "
                + f"{self.image_counts_per_class[cls]:^{max_num}}".rstrip()
            ]

        return "\n".join(table_str)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        total_count = []
        image_count = []
        for cls in range(len(self.class_names)):
            total_count.append(self.label_counts_per_class[cls])
            image_count.append(self.image_counts_per_class[cls])

        return pd.DataFrame(
            {
                "Label": self.class_names,
                "Total Count": total_count,
                "Image Count": image_count,
            }
        )


def _ensure_2d(labels: ArrayLike) -> ArrayLike:
    if isinstance(labels, np.ndarray):
        return labels[:, None]
    else:
        return [[lbl] for lbl in labels]  # type: ignore


def _get_list_depth(lst):
    if isinstance(lst, list) and lst:
        return 1 + max(_get_list_depth(item) for item in lst)
    return 0


def _check_labels_dimension(labels: ArrayLike) -> ArrayLike:
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


def _sort_to_list(d: Mapping[int, TValue]) -> list[TValue]:
    return [v for _, v in sorted(d.items())]


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

    >>> from dataeval.utils.data import Metadata
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
    labels = [target.labels.tolist() for target in dataset.targets]

    labels_2d = _check_labels_dimension(labels)

    for i, group in enumerate(labels_2d):
        group = as_numpy(group).tolist()

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

from __future__ import annotations

__all__ = []

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import TypedDict

from dataeval.types import Array1D, Array2D


class LabelStatsDict(TypedDict):
    """
    Type definition for label statistics output.

    Attributes
    ----------
    label_counts_per_class : Mapping[int, int]
        Dictionary mapping class labels to their total occurrence count
    label_counts_per_image : Sequence[int]
        List containing the number of labels in each image
    image_counts_per_class : Mapping[int, int]
        Dictionary mapping class labels to the number of images containing that class
    image_indices_per_class : Mapping[int, Sequence[int]]
        Dictionary mapping class labels to lists of image indices containing that class
    image_count : int
        Total number of images in the dataset
    class_count : int
        Total number of unique classes
    label_count : int
        Total number of labels across all images
    class_names : Sequence[str]
        List of human-readable class names, ordered by class index
    """

    label_counts_per_class: Mapping[int, int]
    label_counts_per_image: Sequence[int]
    image_counts_per_class: Mapping[int, int]
    image_indices_per_class: Mapping[int, Sequence[int]]
    image_count: int
    class_count: int
    label_count: int
    class_names: Sequence[str]


def label_stats(
    labels: Array1D[int] | Array2D[int],
    index2label: Mapping[int, str] | None = None,
) -> LabelStatsDict:
    """
    Calculates statistics for data labels.

    This function computes counting metrics (e.g., total per class, total per image)
    on the labels. This is a core computation function that operates on basic data
    structures without dependencies on complex domain objects.

    Parameters
    ----------
    labels : _1DArray[int] or _2DArray[int]
        A sequence of label sequences, where each inner sequence contains the integer
        labels for a single image. For image classification, each inner sequence
        typically contains a single label. For object detection, each inner sequence
        contains multiple labels (one per detected object). Empty sequences represent
        images with no labels/detections.
    index2label : Mapping[int, str] | None, optional
        A mapping from label integers to class names. If None, class names will be
        generated as string representations of the label integers.

    Returns
    -------
    dict
        A dictionary containing the computed counting metrics for the labels with keys:
        - label_counts_per_class: Mapping[int, int] - Total count of each class
        - label_counts_per_image: Sequence[int] - Number of labels per image
        - image_counts_per_class: Mapping[int, int] - How many images contain each label
        - image_indices_per_class: Mapping[int, Sequence[int]] - Which images contain each label
        - image_count: int - Total number of images
        - class_count: int - Total number of classes
        - label_count: int - Total number of labels
        - class_names: Sequence[str] - Human-readable class names

    Examples
    --------
    Calculate basic statistics on labels for object detection.

    >>> labels = [[0, 0, 1], [1, 2], [], [0, 1, 2, 3]]
    >>> index2label = {0: "horse", 1: "cow", 2: "sheep", 3: "pig"}
    >>> stats = label_stats(labels, index2label)
    >>> stats["label_counts_per_class"]
    {0: 3, 1: 3, 2: 2, 3: 1}
    >>> stats["label_counts_per_image"]
    [3, 2, 0, 4]
    >>> stats["class_names"]
    ['horse', 'cow', 'sheep', 'pig']

    Calculate basic statistics on labels for image classification.

    >>> labels = [[0], [1], [2], [0]]
    >>> index2label = {0: "cat", 1: "dog", 2: "bird"}
    >>> stats = label_stats(labels, index2label)
    >>> stats["label_counts_per_class"]
    {0: 2, 1: 1, 2: 1}
    >>> stats["label_counts_per_image"]
    [1, 1, 1, 1]
    >>> stats["class_names"]
    ['cat', 'dog', 'bird']
    """
    # Initialize counters
    label_counts: dict[int, int] = defaultdict(int)
    image_counts: dict[int, int] = defaultdict(int)
    image_indices_per_class: dict[int, list[int]] = defaultdict(list)
    label_counts_per_image: list[int] = []

    # Single pass through the data
    for img_idx, img_labels in enumerate(labels):
        # Count labels in this image
        label_counts_per_image.append(len(img_labels))

        # Track which classes appear in this image (for image_counts)
        classes_in_image = set()

        for label in img_labels if isinstance(img_labels, Iterable) else [img_labels]:
            # Count total occurrences of each label
            label_counts[label] += 1

            # Track which images contain each label (avoid duplicates)
            if label not in classes_in_image:
                classes_in_image.add(label)
                image_indices_per_class[label].append(img_idx)

    # Count images per class
    for label, indices in image_indices_per_class.items():
        image_counts[label] = len(indices)

    # Determine all unique classes and create class names
    unique_classes = sorted(label_counts.keys()) if label_counts else []
    if index2label is None:
        class_names = [str(cls) for cls in unique_classes]
    else:
        class_names = [index2label[cls] for cls in unique_classes]

    # Calculate total label count
    total_labels = sum(label_counts.values())

    return {
        "label_counts_per_class": dict(label_counts),
        "label_counts_per_image": label_counts_per_image,
        "image_counts_per_class": dict(image_counts),
        "image_indices_per_class": dict(image_indices_per_class),
        "image_count": len(labels),
        "class_count": len(unique_classes),
        "label_count": total_labels,
        "class_names": class_names,
    }

__all__ = []

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import TypedDict

_logger = logging.getLogger(__name__)


class LabelStatsResult(TypedDict):
    """
    Type definition for label statistics output.

    Attributes
    ----------
    label_counts_per_class : Mapping[int, int]
        Mapping of class labels to their total occurrence count.
    label_counts_per_image : Sequence[int]
        List containing the number of labels in each image
    image_counts_per_class : Mapping[int, int]
        Mapping of class labels to the number of images containing that class.
    image_indices_per_class : Mapping[int, Sequence[int]]
        Mapping of class labels to sequences of image indices containing that class.
    classes_per_image : Sequence[Sequence[int]]
        Sequence containing class labels for each image, indexed by image position.
        Images with no labels have empty sequences.
    image_count : int
        Total number of images in the dataset
    class_count : int
        Total number of unique classes
    label_count : int
        Total number of labels across all images
    index2label : Mapping[int, str]
        Direct mapping from class index to class name for O(1) lookups
    empty_image_indices : Sequence[int]
        Indices of images with no labels
    empty_image_count : int
        Total number of images with no labels
    """

    label_counts_per_class: Mapping[int, int]
    label_counts_per_image: Sequence[int]
    image_counts_per_class: Mapping[int, int]
    image_indices_per_class: Mapping[int, Sequence[int]]
    classes_per_image: Sequence[Sequence[int]]
    image_count: int
    class_count: int
    label_count: int
    index2label: Mapping[int, str]
    empty_image_indices: Sequence[int]
    empty_image_count: int


def label_stats(
    labels: Iterable[int] | Iterable[Iterable[int]],
    index2label: Mapping[int, str] | None = None,
) -> LabelStatsResult:
    """
    Calculates statistics for data labels.

    This function computes counting metrics (e.g., total per class, total per image)
    on the labels. This is a core computation function that operates on basic data
    structures without dependencies on complex domain objects.

    Parameters
    ----------
    labels : Iterable[int] | Iterable[Iterable[int]]
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
    LabelStatsResult
        A mapping containing the computed counting metrics for the labels with keys:

        - label_counts_per_class: Mapping[int, int] - Total count of each class
        - label_counts_per_image: Sequence[int] - Number of labels per image
        - image_counts_per_class: Mapping[int, int] - How many images contain each label
        - image_indices_per_class: Mapping[int, Sequence[int]] - Which images contain each label
        - image_count: int - Total number of images
        - class_count: int - Total number of classes
        - label_count: int - Total number of labels
        - index2label: Mapping[int, str] - Direct mapping from class index to class name
        - empty_image_indices: Sequence[int] - Indices of images with no labels
        - empty_image_count: int - Number of images with no labels

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
    >>> stats["empty_image_indices"]
    [2]
    >>> stats["empty_image_count"]
    1

    Calculate basic statistics on labels for image classification.

    >>> labels = [[0], [1], [2], [0]]
    >>> index2label = {0: "cat", 1: "dog", 2: "bird"}
    >>> stats = label_stats(labels, index2label)
    >>> stats["label_counts_per_class"]
    {0: 2, 1: 1, 2: 1}
    >>> stats["label_counts_per_image"]
    [1, 1, 1, 1]
    >>> stats["empty_image_count"]
    0
    """
    _logger.info("Starting label_stats calculation")

    # Initialize counters - separate empty image tracking from class statistics
    label_counts: dict[int, int] = defaultdict(int)
    image_counts: dict[int, int] = defaultdict(int)
    image_indices_per_class: dict[int, list[int]] = defaultdict(list)
    classes_per_image: list[list[int]] = []
    label_counts_per_image: list[int] = []
    empty_image_indices: list[int] = []

    # Single pass through the data
    img_idx = None
    for img_idx, img_labels in enumerate(labels):
        # Track which classes appear in this image (for image_counts)
        classes_in_image = set()
        classes_in_image_list: list[int] = []

        label_count = None
        labels = img_labels if isinstance(img_labels, Iterable) else [img_labels]
        for label_count, label in enumerate(labels):
            # Ensure label is always native int type
            label = int(label)

            # Count total occurrences of each label
            label_counts[label] += 1

            # Track which images contain each label (avoid duplicates)
            if label not in classes_in_image:
                classes_in_image.add(label)
                classes_in_image_list.append(label)
                image_indices_per_class[label].append(img_idx)

        # Store classes for this image (empty list for images with no labels)
        classes_per_image.append(classes_in_image_list)

        # Track empty images separately
        if label_count is None:
            empty_image_indices.append(img_idx)

        label_counts_per_image.append(0 if label_count is None else label_count + 1)

    # Count images per class
    for label, indices in image_indices_per_class.items():
        image_counts[label] = len(indices)

    # Determine all unique classes and create index2label mapping
    unique_classes = sorted(label_counts.keys()) if label_counts else []
    if index2label is None:
        result_index2label: dict[int, str] = {cls: str(cls) for cls in unique_classes}
    else:
        result_index2label = {cls: index2label[cls] for cls in unique_classes}

    # Calculate total label count
    total_labels = sum(label_counts.values())

    img_count = 0 if img_idx is None else img_idx + 1

    _logger.info(
        "Label stats calculation complete: %d images, %d classes, %d total labels, %d empty images",
        img_count,
        len(unique_classes),
        total_labels,
        len(empty_image_indices),
    )
    _logger.debug("Class distribution: %s", dict(label_counts))

    return {
        "label_counts_per_class": dict(label_counts),
        "label_counts_per_image": label_counts_per_image,
        "image_counts_per_class": dict(image_counts),
        "image_indices_per_class": dict(image_indices_per_class),
        "classes_per_image": classes_per_image,
        "image_count": img_count,
        "class_count": len(unique_classes),
        "label_count": total_labels,
        "index2label": result_index2label,
        "empty_image_indices": empty_image_indices,
        "empty_image_count": len(empty_image_indices),
    }

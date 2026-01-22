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
    class_labels: Iterable[int],
    item_indices: Iterable[int] | None = None,
    index2label: Mapping[int, str] | None = None,
    image_count: int | None = None,
) -> LabelStatsResult:
    """
    Calculates statistics for data labels.

    This function computes counting metrics (e.g., total per class, total per image)
    on the labels. This is a core computation function that operates on basic data
    structures without dependencies on complex domain objects.

    Parameters
    ----------
    class_labels : Iterable[int]
        A flat sequence of integer class labels. For image classification, this has
        one label per image. For object detection, this has one label per detection
        across all images.
    item_indices : Iterable[int] | None, optional
        A sequence mapping each label to its source image index. Must have the same
        length as class_labels. If None, a 1:1 mapping is assumed (one label per image).
    index2label : Mapping[int, str] | None, optional
        A mapping from label integers to class names. If None, class names will be
        generated as string representations of the label integers.
    image_count : int | None, optional
        Total number of images. Required when item_indices is provided to detect
        empty images. If None and item_indices is provided, inferred from max index + 1.

    Returns
    -------
    LabelStatsResult
        A mapping containing the computed counting metrics for the labels with keys:

        - label_counts_per_class: Mapping[int, int] - Total count of each class
        - label_counts_per_image: Sequence[int] - Number of labels per image
        - image_counts_per_class: Mapping[int, int] - How many images contain each label
        - image_indices_per_class: Mapping[int, Sequence[int]] - Which images contain each label
        - classes_per_image: Sequence[Sequence[int]] - Class labels for each image
        - image_count: int - Total number of images
        - class_count: int - Total number of classes
        - label_count: int - Total number of labels
        - index2label: Mapping[int, str] - Direct mapping from class index to class name
        - empty_image_indices: Sequence[int] - Indices of images with no labels
        - empty_image_count: int - Number of images with no labels

    Examples
    --------
    Calculate basic statistics on labels for object detection.

    >>> class_labels = [0, 0, 1, 1, 2, 0, 1, 2, 3]
    >>> item_indices = [0, 0, 0, 1, 1, 3, 3, 3, 3]  # image 2 is empty
    >>> index2label = {0: "horse", 1: "cow", 2: "sheep", 3: "pig"}
    >>> stats = label_stats(class_labels, item_indices, index2label, image_count=4)
    >>> stats["label_counts_per_class"]
    {0: 3, 1: 3, 2: 2, 3: 1}
    >>> stats["label_counts_per_image"]
    [3, 2, 0, 4]
    >>> stats["empty_image_indices"]
    [2]
    >>> stats["empty_image_count"]
    1

    Calculate basic statistics on labels for image classification (1:1 mapping).

    >>> class_labels = [0, 1, 2, 0]
    >>> index2label = {0: "cat", 1: "dog", 2: "bird"}
    >>> stats = label_stats(class_labels, index2label=index2label)
    >>> stats["label_counts_per_class"]
    {0: 2, 1: 1, 2: 1}
    >>> stats["label_counts_per_image"]
    [1, 1, 1, 1]
    >>> stats["empty_image_count"]
    0
    """
    _logger.info("Starting label_stats calculation")

    # Convert to lists for indexing if needed
    class_labels_list = list(class_labels)
    total_labels = len(class_labels_list)

    # Handle item_indices: if None, assume 1:1 mapping
    if item_indices is None:
        item_indices_list = list(range(total_labels))
        inferred_image_count = total_labels
    else:
        item_indices_list = list(item_indices)
        if len(item_indices_list) != total_labels:
            raise ValueError(
                f"item_indices length ({len(item_indices_list)}) must match class_labels length ({total_labels})"
            )
        inferred_image_count = max(item_indices_list) + 1 if item_indices_list else 0

    # Determine actual image count
    img_count = image_count if image_count is not None else inferred_image_count

    # Initialize data structures
    label_counts: dict[int, int] = defaultdict(int)
    image_indices_per_class: dict[int, list[int]] = defaultdict(list)
    classes_per_image: list[list[int]] = [[] for _ in range(img_count)]
    label_counts_per_image: list[int] = [0] * img_count
    classes_seen_per_image: list[set[int]] = [set() for _ in range(img_count)]

    # Single pass through the data
    for label, img_idx in zip(class_labels_list, item_indices_list):
        # Ensure label is always native int type
        label = int(label)
        img_idx = int(img_idx)

        # Count total occurrences of each label
        label_counts[label] += 1

        # Count labels per image
        label_counts_per_image[img_idx] += 1

        # Track which images contain each label (avoid duplicates)
        if label not in classes_seen_per_image[img_idx]:
            classes_seen_per_image[img_idx].add(label)
            classes_per_image[img_idx].append(label)
            image_indices_per_class[label].append(img_idx)

    # Count images per class
    image_counts: dict[int, int] = {label: len(indices) for label, indices in image_indices_per_class.items()}

    # Find empty images
    empty_image_indices = [i for i, count in enumerate(label_counts_per_image) if count == 0]

    # Determine all unique classes and create index2label mapping
    unique_classes = sorted(label_counts.keys()) if label_counts else []
    if index2label is None:
        result_index2label: dict[int, str] = {cls: str(cls) for cls in unique_classes}
    else:
        result_index2label = {cls: index2label[cls] for cls in unique_classes}

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

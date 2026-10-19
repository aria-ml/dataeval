__all__ = []

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.exceptions import ShapeMismatchError

_logger = get_logger(__name__)


def _calculate_class_proportions(
    label_counts: dict[int, int],
    all_classes: list[int],
    total_labels: int,
) -> dict[int, float]:
    """Calculate proportion of total labels for each class."""
    if total_labels > 0:
        return {cls: label_counts[cls] / total_labels for cls in all_classes}
    return dict.fromkeys(all_classes, 0.0)


def _calculate_imbalance_ratios(
    label_counts: dict[int, int],
    all_classes: list[int],
) -> dict[int, float]:
    """Calculate imbalance ratio per class (ratio of max class to each class)."""
    if not label_counts or not any(count > 0 for count in label_counts.values()):
        return dict.fromkeys(all_classes, 1.0)

    max_count = max(label_counts.values())
    return {cls: float(max_count / label_counts[cls]) if label_counts[cls] > 0 else float("inf") for cls in all_classes}


def _calculate_image_imbalance(
    class_image_counts: Mapping[int, int],
) -> dict[int, float]:
    """Calculate image-level presence imbalance ratio (IRLbl from Charte et al. 2019)."""
    counts = list(class_image_counts.values())
    max_count = int(np.max(counts)) if len(counts) > 0 else 0
    if max_count == 0:
        return dict.fromkeys(class_image_counts, 1.0)
    return {cls: float(max_count / cnt) if cnt > 0 else float("inf") for cls, cnt in class_image_counts.items()}


def _calculate_imbalance_mixing(  # noqa: C901
    classes_per_image: Sequence[Sequence[int]],
    image_imbalance: Mapping[int, float],
    image_indices_full: Mapping[int, Sequence[int]],
    all_classes: Sequence[int],
    image_count: int,
) -> tuple[float, dict[int, float]]:
    """
    Compute dataset-level and per-class imbalance mixing (SCUMBLE from Charte et al. 2019).

    For each image, SCUMBLE is 1 minus the ratio of geometric to arithmetic mean of
    image_imbalance_ratio_per_class over the classes present in that image.
    Single-label and unlabeled images score 0.0.

    imbalance_mixing is the average score over images with labels (0.0 if no images have labels).
    imbalance_mixing_per_class is the average score over images containing each class (NaN for
    classes that never appear).
    """
    if image_count == 0:
        return 0.0, dict.fromkeys(all_classes, float("nan"))

    image_scores = np.zeros(image_count, dtype=np.float64)
    labeled_indices: list[int] = []

    for img_idx, classes in enumerate(classes_per_image):
        if not classes:
            continue
        labeled_indices.append(img_idx)
        if len(classes) >= 2:
            ratios = np.array([image_imbalance[cls] for cls in classes], dtype=np.float64)
            arithmetic = float(np.mean(ratios))
            geometric = float(np.exp(np.mean(np.log(ratios))))
            if arithmetic > 0:
                image_scores[img_idx] = float(np.clip(1.0 - (geometric / arithmetic), 0.0, 1.0))

    imbalance_mixing = float(np.mean(image_scores[labeled_indices])) if labeled_indices else 0.0

    imbalance_mixing_per_class: dict[int, float] = {}
    for cls in all_classes:
        indices = image_indices_full.get(cls, [])
        if len(indices) > 0:
            imbalance_mixing_per_class[cls] = float(np.mean(image_scores[indices]))
        else:
            imbalance_mixing_per_class[cls] = float("nan")

    return imbalance_mixing, imbalance_mixing_per_class


def _calculate_multi_label_metrics(
    classes_per_image: Sequence[Sequence[int]],
    class_count: int,
) -> tuple[float, float, list[tuple[int, ...]], list[int], int]:
    """
    Calculate label cardinality, label density, and label set distributions.

    Only non-empty images take part: an image without labels is not a label set, and it
    would otherwise pull the cardinality down. Label sets are ranked by how many images
    share them (ties broken by the classes themselves, so the order is deterministic).
    """
    non_empty_classes = [c for c in classes_per_image if len(c) > 0]
    set_counts = Counter(map(tuple, non_empty_classes))
    distinct_label_sets = len(set_counts)

    def rank(item: tuple[tuple[int, ...], int]) -> tuple[int, tuple[int, ...]]:
        return -item[1], item[0]

    ranked = sorted(set_counts.items(), key=rank)
    label_sets = [label_set for label_set, _ in ranked]
    label_set_counts = [count for _, count in ranked]

    label_cardinality = float(sum(map(len, non_empty_classes)) / len(non_empty_classes)) if non_empty_classes else 0.0
    label_density = float(label_cardinality / class_count) if class_count > 0 else 0.0

    return label_cardinality, label_density, label_sets, label_set_counts, distinct_label_sets


def _integral(arr: NDArray[Any], name: str) -> NDArray[Any]:
    """Return ``arr`` as integers, converting whole-valued floats and rejecting anything else."""
    if arr.dtype.kind in "iu":
        return arr
    if arr.dtype.kind == "f" and np.all(np.isfinite(arr) & (arr == np.floor(arr))):
        return arr.astype(np.int64)
    raise TypeError(f"{name} must contain integers, got values of dtype {arr.dtype} that are not whole numbers")


def _as_int_list(values: Iterable[int], name: str) -> list[int]:
    """
    Convert an iterable of integers to a list of native Python ints.

    Going through NumPy means ndarray input is converted in C (``ndarray.tolist()``)
    instead of one ``int()`` call per element, and guarantees native ``int`` values so
    results are JSON-serializable and free of ``np.int64`` keys.

    Raises
    ------
    ShapeMismatchError
        If the values are not one-dimensional.
    TypeError
        If the values are not integers. Whole-valued floats (``1.0``) are accepted;
        bools, strings, and floats with a fractional part or NaN are rejected rather
        than silently truncated.
    """
    if not (isinstance(values, np.ndarray | Sequence) or hasattr(values, "__array__")):
        values = list(values)  # generators, sets, other one-shot iterables

    arr = np.asarray(values)
    if arr.size == 0:
        return []
    if arr.ndim != 1:
        raise ShapeMismatchError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return _integral(arr, name).tolist()


def _count_labels(
    class_labels: Sequence[int],
    item_indices: Sequence[int],
    image_count: int,
) -> tuple[dict[int, int], list[int], list[list[int]], dict[int, list[int]]]:
    """
    Count labels and track image membership in a single pass.

    Inputs are assumed to be pre-validated: native ints, equal lengths, and every
    index in ``range(image_count)``. A negative index would silently wrap around.

    Parameters
    ----------
    class_labels : Sequence[int]
        One class label per annotation.
    item_indices : Sequence[int]
        Source image index for each annotation. May be in any order and repeat.
    image_count : int
        Total number of images, including images with no annotations.

    Returns
    -------
    tuple
        - label_counts: total occurrences of each class present in the data
        - label_counts_per_image: number of annotations in each image
        - classes_per_image: sorted distinct classes in each image
        - image_indices_per_class: sorted indices of the images containing each class present
          in the data
    """
    label_counts: defaultdict[int, int] = defaultdict(int)
    image_indices_per_class: defaultdict[int, list[int]] = defaultdict(list)
    label_counts_per_image: list[int] = [0] * image_count
    classes_per_image: list[list[int]] = [[] for _ in range(image_count)]

    # A list scan beats a per-image set here: images typically hold few distinct
    # classes, and an empty list is far cheaper than an empty set at 100k+ images.
    # Single pass through the data
    for label, img_idx in zip(class_labels, item_indices, strict=True):
        label_counts[label] += 1
        label_counts_per_image[img_idx] += 1
        classes = classes_per_image[img_idx]
        if label not in classes:
            classes.append(label)
            image_indices_per_class[label].append(img_idx)

    for classes in classes_per_image:
        classes.sort()

    return (
        dict(label_counts),
        label_counts_per_image,
        classes_per_image,
        {label: sorted(indices) for label, indices in image_indices_per_class.items()},
    )


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
    class_proportions : Mapping[int, float]
        Proportion of total labels for each class
    imbalance_ratio_per_class : Mapping[int, float]
        Class-level imbalance ratio: ratio of largest class to each class
        (measures imbalance relative to majority class). Infinite for classes that never appear.
    image_imbalance_ratio_per_class : Mapping[int, float]
        Image-level imbalance ratio: largest image count across class divided by each
        class's image count. Infinite for classes that never appear.
    imbalance_mixing_per_class : Mapping[int, float]
        Measures the amount of mixing between rare and common classes.
        Average image-level of the images containing each class: how much that class is mixed
        with classes of very different frequency. NaN for classes that never appear.
    imbalance_mixing : float
        Average SCUMBLE over the images with labels: how much the labels within an image
        are a mix of common and rare classes. 0 when no image has labels.
    classes_per_image : Sequence[Sequence[int]]
        Sequence containing class labels for each image, indexed by image position.
        Images with no labels have empty sequences.
    label_cardinality : float
        Average number of distinct classes per non-empty image
    label_density : float
        Label cardinality divided by total class count
    label_sets : Sequence[tuple[int, ...]]
        Unique combinations of classes observed across images
    label_set_counts : Sequence[int]
        Number of images with each combination in ``label_sets``, in the same order.
    distinct_label_sets : int
        Total number of unique label combinations
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
    class_proportions: Mapping[int, float]
    imbalance_ratio_per_class: Mapping[int, float]
    image_imbalance_ratio_per_class: Mapping[int, float]
    imbalance_mixing_per_class: Mapping[int, float]
    imbalance_mixing: float
    classes_per_image: Sequence[Sequence[int]]
    label_cardinality: float
    label_density: float
    label_sets: Sequence[tuple[int, ...]]
    label_set_counts: Sequence[int]
    distinct_label_sets: int
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
    Compute statistics for data labels.

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
        A mapping from label integers to class names. If provided, this defines the
        complete set of classes, even if some classes have no instances in the data.
        Classes found in the data but missing from this mapping are still included, named
        by ``str(class)``. If None, only classes present in the data will be included.
    image_count : int | None, optional
        Total number of images. Required when detecting empty images matters. If None and
        item_indices is provided, inferred from ``max(item_indices) + 1``. If None and
        item_indices is None, inferred from number of labels.

    Returns
    -------
    LabelStatsResult
        A mapping containing the computed counting metrics for the labels with keys.
        All ``*_per_class`` mappings share the same keys: the sorted union of the
        classes in the data and the keys of ``index2label``.

        - label_counts_per_class: Mapping[int, int] - Total count of each class
        - label_counts_per_image: Sequence[int] - Number of labels per image
        - image_counts_per_class: Mapping[int, int] - How many images contain each label
        - image_indices_per_class: Mapping[int, Sequence[int]] - Which images contain each label
        - class_proportions: Mapping[int, float] - Proportion of total labels for each class
        - imbalance_ratio_per_class: Mapping[int, float] - Ratio of largest class to each class
        - image_imbalance_ratio_per_class: Mapping[int, float] - Image presence imbalance ratio (IRLbl)
        - imbalance_mixing_per_class: Mapping[int, float] - Average SCUMBLE for images containing each class
        - imbalance_mixing: float - Average SCUMBLE over images with labels
        - classes_per_image: Sequence[Sequence[int]] - Class labels for each image
        - label_cardinality: float - Average number of distinct classes per non-empty image
        - label_density: float - Label cardinality divided by total class count
        - label_sets: Sequence[tuple[int, ...]] - Unique combinations of classes observed across images
        - label_set_counts: Sequence[int] - Number of images with each combination in label_sets
        - distinct_label_sets: int - Total number of unique label combinations
        - image_count: int - Total number of images
        - class_count: int - Total number of classes
        - label_count: int - Total number of labels
        - index2label: Mapping[int, str] - Direct mapping from class index to class name
        - empty_image_indices: Sequence[int] - Indices of images with no labels
        - empty_image_count: int - Number of images with no labels

    Raises
    ------
    ShapeMismatchError
        If ``item_indices`` and ``class_labels`` differ in length, or either is not
        one-dimensional.
    ValueError
        If ``item_indices`` contains a negative value, or ``image_count`` is smaller
        than ``max(item_indices) + 1``.
    TypeError
        If ``class_labels`` or ``item_indices`` contain non-integer values. Whole-valued
        floats (``1.0``) are accepted; bools and fractional or NaN values are not.

    Examples
    --------
    Compute basic statistics on labels for object detection.

    >>> class_labels = [0, 0, 1, 1, 2, 0, 1, 2, 4, 4]  # classes 3 and 5 are empty
    >>> item_indices = [0, 0, 0, 1, 1, 3, 3, 3, 4, 4]  # image 2 is empty
    >>> index2label = {0: "horse", 1: "cow", 2: "sheep", 3: "pig", 4: "cat", 5: "dog"}
    >>> stats = label_stats(class_labels, item_indices, index2label, image_count=5)
    >>> stats["label_counts_per_class"]
    {0: 3, 1: 3, 2: 2, 3: 0, 4: 2, 5: 0}
    >>> stats["label_counts_per_image"]
    [3, 2, 0, 3, 2]
    >>> stats["classes_per_image"]
    [[0, 1], [1, 2], [], [0, 1, 2], [4]]
    >>> stats["image_indices_per_class"][1]
    [0, 1, 3]
    >>> stats["class_proportions"]
    {0: 0.3, 1: 0.3, 2: 0.2, 3: 0.0, 4: 0.2, 5: 0.0}
    >>> stats["imbalance_ratio_per_class"]
    {0: 1.0, 1: 1.0, 2: 1.5, 3: inf, 4: 1.5, 5: inf}
    >>> stats["empty_image_indices"]
    [2]
    >>> stats["empty_image_count"]
    1

    Compute basic statistics on labels for image classification (1:1 mapping).

    >>> class_labels = [0, 1, 2, 0]
    >>> index2label = {0: "cat", 1: "dog", 2: "bird"}
    >>> stats = label_stats(class_labels, index2label=index2label)
    >>> stats["label_counts_per_class"]
    {0: 2, 1: 1, 2: 1}
    >>> stats["label_counts_per_image"]
    [1, 1, 1, 1]
    >>> stats["class_proportions"]
    {0: 0.5, 1: 0.25, 2: 0.25}
    >>> stats["imbalance_ratio_per_class"]
    {0: 1.0, 1: 2.0, 2: 2.0}
    >>> stats["empty_image_count"]
    0

    Compare datasets against the same classes: every key in ``index2label`` appears
    in every per-class output, even when the dataset has no instances of it.

    >>> wanted = {0: "cat", 1: "dog", 2: "bird"}
    >>> stats = label_stats([0, 0], index2label=wanted)
    >>> stats["label_counts_per_class"]
    {0: 2, 1: 0, 2: 0}
    >>> stats["image_indices_per_class"]
    {0: [0, 1], 1: [], 2: []}
    """
    _logger.info("Starting label_stats calculation")

    # Convert to list with native int type for indexing if needed
    class_labels_list = _as_int_list(class_labels, "class_labels")
    total_labels = len(class_labels_list)

    # Handle item_indices: if None, assume 1:1 mapping
    if item_indices is None:
        item_indices_list = list(range(total_labels))
        inferred_image_count = total_labels
    else:
        item_indices_list = _as_int_list(item_indices, "item_indices")
        if len(item_indices_list) != total_labels:
            raise ShapeMismatchError(
                f"item_indices length ({len(item_indices_list)}) must match class_labels length ({total_labels})",
            )
        if item_indices_list and min(item_indices_list) < 0:
            raise ValueError(f"item_indices must be non-negative, got minimum {min(item_indices_list)}")
        inferred_image_count = max(item_indices_list) + 1 if item_indices_list else 0

    # Determine actual image count
    img_count = image_count if image_count is not None else inferred_image_count
    if img_count < inferred_image_count:
        raise ValueError(
            f"image_count ({image_count}) is smaller than required by item_indices "
            f"(max index + 1 = {inferred_image_count})",
        )

    # Count labels and track image membership
    label_counts, label_counts_per_image, classes_per_image, image_indices_per_class = _count_labels(
        class_labels_list, item_indices_list, img_count
    )

    # Find empty images
    empty_image_indices = [i for i, count in enumerate(label_counts_per_image) if count == 0]

    # Classes to report: everything in the data plus every requested class
    all_classes = sorted(set(label_counts).union(index2label or []))
    result_index2label = {cls: (index2label or {}).get(cls, str(cls)) for cls in all_classes}

    # Fill in missing classes (those with no instances)
    label_counts_full = {cls: label_counts.get(cls, 0) for cls in all_classes}
    image_indices_full = {cls: image_indices_per_class.get(cls, []) for cls in all_classes}
    image_counts_full = {cls: len(indices) for cls, indices in image_indices_full.items()}

    # Calculate derived metrics
    class_proportions = _calculate_class_proportions(label_counts_full, all_classes, total_labels)
    imbalance_ratio_per_class = _calculate_imbalance_ratios(label_counts_full, all_classes)
    image_imbalance = _calculate_image_imbalance(image_counts_full)
    imbalance_mixing, imbalance_mixing_per_class = _calculate_imbalance_mixing(
        classes_per_image, image_imbalance, image_indices_full, all_classes, img_count
    )
    label_cardinality, label_density, label_sets, label_set_counts, distinct_label_sets = (
        _calculate_multi_label_metrics(classes_per_image, len(all_classes))
    )

    _logger.info(
        "Label stats calculation complete: %d images, %d classes, %d total labels, %d empty images",
        img_count,
        len(all_classes),
        total_labels,
        len(empty_image_indices),
    )
    _logger.debug("Class distribution: %s", label_counts_full)

    return {
        "label_counts_per_class": label_counts_full,
        "label_counts_per_image": label_counts_per_image,
        "image_counts_per_class": image_counts_full,
        "image_indices_per_class": image_indices_full,
        "class_proportions": class_proportions,
        "imbalance_ratio_per_class": imbalance_ratio_per_class,
        "image_imbalance_ratio_per_class": image_imbalance,
        "imbalance_mixing_per_class": imbalance_mixing_per_class,
        "imbalance_mixing": imbalance_mixing,
        "classes_per_image": classes_per_image,
        "label_cardinality": label_cardinality,
        "label_density": label_density,
        "label_sets": label_sets,
        "label_set_counts": label_set_counts,
        "distinct_label_sets": distinct_label_sets,
        "image_count": img_count,
        "class_count": len(all_classes),
        "label_count": total_labels,
        "index2label": result_index2label,
        "empty_image_indices": empty_image_indices,
        "empty_image_count": len(empty_image_indices),
    }

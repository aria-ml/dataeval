"""Utility functions for dataset splitting and data and metadata manipulation."""

__all__ = ["split_dataset", "unzip_dataset", "TrainValSplit", "DatasetSplits"]

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target

from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    Dataset,
    MetadataLike,
    ObjectDetectionTarget,
)
from dataeval.utils._internal import EPSILON, as_numpy, unwrap_image
from dataeval.utils.preprocessing import BoundingBox

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainValSplit:
    """
    Dataclass containing train and validation indices.

    Attributes
    ----------
    train: NDArray[np.intp]
        Indices for the training set
    val: NDArray[np.intp]
        Indices for the validation set
    """

    train: NDArray[np.intp]
    val: NDArray[np.intp]


@dataclass(frozen=True)
class DatasetSplits:
    """
    Output class containing test indices and a list of TrainValSplits.

    Attributes
    ----------
    test: NDArray[np.intp]
        Indices for the test set
    folds: Sequence[TrainValSplit]
        List of train and validation split indices
    """

    test: NDArray[np.intp]
    folds: Sequence[TrainValSplit]


class _KFoldSplitter(Protocol):
    """Protocol covering sklearn KFold variant splitters."""

    def __init__(self, n_splits: int) -> None: ...
    def split(self, X: Any, y: Any, groups: Any) -> Iterator[tuple[NDArray[Any], NDArray[Any]]]: ...  # noqa: N803


_KFOLD_GROUP_STRATIFIED_MAP: dict[tuple[bool, bool], type[_KFoldSplitter]] = {
    (False, False): KFold,
    (False, True): StratifiedKFold,
    (True, False): GroupKFold,
    (True, True): StratifiedGroupKFold,
}


def _build_multilabel_matrix(
    class_labels: NDArray[np.intp],
    item_indices: NDArray[np.intp],
    n_images: int,
) -> NDArray[np.int8]:
    """Build a binary multi-label matrix from detection-level labels.

    Parameters
    ----------
    class_labels : NDArray[np.intp]
        Per-detection class labels
    item_indices : NDArray[np.intp]
        Per-detection source image indices
    n_images : int
        Total number of images

    Returns
    -------
    NDArray[np.int8]
        Binary matrix of shape (n_images, n_classes) where entry (i, j) = 1
        if image i contains at least one detection of class j
    """
    n_classes = int(class_labels.max()) + 1
    matrix = np.zeros((n_images, n_classes), dtype=np.int8)
    matrix[item_indices, class_labels] = 1
    return matrix


class _IterativeStratifiedKFold:
    """Multi-label stratified K-fold cross-validation.

    Implements the iterative stratification algorithm from
    Sechidis et al. "On the Stratification of Multi-Label Data" (ECML-PKDD 2011).

    Assigns samples to folds by processing labels from rarest to most common,
    greedily placing each sample in the fold that most needs that label.
    """

    def __init__(self, n_splits: int) -> None:
        self.n_splits = n_splits

    def split(
        self,
        X: Any,  # noqa: ARG002, N803
        y: NDArray[Any],
        groups: Any = None,  # noqa: ARG002
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        fold_assignment = self._assign_folds(np.asarray(y))
        for fold_idx in range(self.n_splits):
            test_mask = fold_assignment == fold_idx
            yield (
                np.where(~test_mask)[0].astype(np.intp),
                np.where(test_mask)[0].astype(np.intp),
            )

    def _assign_folds(self, y: NDArray[Any]) -> NDArray[np.intp]:  # noqa: C901
        n_samples = y.shape[0]
        assignment = np.full(n_samples, -1, dtype=np.intp)
        desired = np.ones(self.n_splits) / self.n_splits

        # Process labels from rarest to most common
        label_order = np.argsort(y.sum(axis=0))

        for label_idx in label_order:
            has_label = y[:, label_idx].astype(bool)
            candidates = np.where(has_label & (assignment == -1))[0]

            if len(candidates) == 0:
                continue

            # Current per-fold count of this label
            assigned_mask = has_label & (assignment != -1)
            fold_counts = np.zeros(self.n_splits, dtype=np.float64)
            for k in range(self.n_splits):
                fold_counts[k] = np.sum(assignment[assigned_mask] == k)

            desired_counts = desired * float(has_label.sum())

            for sample in candidates:
                need = desired_counts - fold_counts
                fold_sizes = np.array(
                    [np.sum(assignment == k) for k in range(self.n_splits)],
                    dtype=np.float64,
                )
                # Primary: most need for this label; secondary: smallest fold
                best = int(np.lexsort((fold_sizes, -need))[0])
                assignment[sample] = best
                fold_counts[best] += 1

        # Assign remaining samples (no labels) to smallest fold
        remaining = np.where(assignment == -1)[0]
        for sample in remaining:
            fold_sizes = np.array([np.sum(assignment == k) for k in range(self.n_splits)])
            assignment[sample] = int(fold_sizes.argmin())

        return assignment


def _multilabel_make_splits(
    index: NDArray[np.intp],
    multilabel: NDArray[np.int8],
    n_folds: int,
) -> list[TrainValSplit]:
    """Create multi-label stratified K-fold splits using iterative stratification."""
    splitter = _IterativeStratifiedKFold(n_folds)
    split_defs: list[TrainValSplit] = []
    for train_idx, eval_idx in splitter.split(index, multilabel):
        t = np.atleast_1d(train_idx).astype(np.intp)
        v = np.atleast_1d(eval_idx).astype(np.intp)
        split_defs.append(TrainValSplit(t, v))
    return split_defs


def _multilabel_find_best_split(
    multilabel: NDArray[np.int8],
    split_defs: list[TrainValSplit],
    split_frac: float,
) -> TrainValSplit:
    """Find the split whose per-label frequencies best match the overall distribution."""

    def label_freq_diff(split: TrainValSplit) -> float:
        overall = multilabel.mean(axis=0)
        train_freq = multilabel[split.train].mean(axis=0) if len(split.train) else np.zeros_like(overall)
        val_freq = multilabel[split.val].mean(axis=0) if len(split.val) else np.zeros_like(overall)
        return float(np.sum(np.abs(train_freq - overall)) + np.sum(np.abs(val_freq - overall)))

    def split_ratio(split: TrainValSplit) -> float:
        return len(split.val) / (len(split.val) + len(split.train))

    def split_diff(split: TrainValSplit) -> float:
        return abs(split_frac - split_ratio(split))

    def split_inv_diff(split: TrainValSplit) -> float:
        return abs(1 - split_frac - split_ratio(split))

    # Prefer label balance for multi-label; fall back to ratio for non-stratified
    return min(split_defs, key=label_freq_diff)


def _multilabel_single_split(
    index: NDArray[np.intp],
    multilabel: NDArray[np.int8],
    split_frac: float,
) -> TrainValSplit:
    """Single train/eval split for multi-label data using iterative stratification."""
    class_counts = multilabel.sum(axis=0)
    positive_counts = class_counts[class_counts > 0]
    max_folds = int(positive_counts.min()) if len(positive_counts) else len(index)

    divisor = split_frac if split_frac <= 2 / 3 else 1 - split_frac
    n_folds = min(max(round(1 / (divisor + EPSILON)), 2), max_folds)
    _logger.log(logging.DEBUG, f"multilabel n_folds={n_folds} clipped between[2, {max_folds}]")

    split_candidates = _multilabel_make_splits(index, multilabel, n_folds)
    return _multilabel_find_best_split(multilabel, split_candidates, split_frac)


def _calculate_validation_fraction(num_folds: int, test_frac: float, val_frac: float) -> float:  # noqa: C901
    """
    Compute possible validation fraction based on the number of folds and test fraction.

    Parameters
    ----------
    num_folds : int
        number of train and validation cross-validation folds to generate
    test_frac : float
        The fraction of the data to extract for testing before folds are created
    val_frac : float
        The validation split will contain (val_frac * 100)% of any data not already allocated to the test set.
        Only required if requesting a single [train, val] split.

    Returns
    -------
    float
        The updated validation fraction of the remaining data after the testing fraction is removed

    Raises
    ------
    ValueError
        When number of folds requested is less than 1
    ValueError
        When the test fraction is not within 0.0 and 1.0 inclusively
    ValueError
        When more than one fold and the validation fraction are both requested
    ValueError
        When number of folds equals one but the validation fraction is 0.0
    ValueError
        When the validation fraction is not within 0.0 and 1.0 inclusively
    """
    if num_folds < 1:
        raise ValueError(f"Number of folds must be greater than or equal to 1, got {num_folds}")
    if test_frac < 0.0 or test_frac > 1.0:
        raise ValueError(f"test_frac out of bounds. Must be between 0.0 and 1.0, got {test_frac}")

    # val base is a variable placeholder so val_frac can be ignored if num_folds != 1
    val_base: float = 1.0
    if num_folds == 1:
        if val_frac == 0.0:
            raise ValueError("If num_folds equals 1, must assign a value to val_frac")
        if val_frac < 0.0 or val_frac > 1.0:
            raise ValueError(f"val_frac out of bounds. Must be between 0.0 and 1.0, got {val_frac}")
        val_base = val_frac
    # num folds must be >1 in this case
    elif val_frac != 0.0:
        raise ValueError("Can only specify val_frac when num_folds equals 1")

    # This value is mathematically bound between 0-1 inclusive
    return val_base * (1.0 / num_folds) * (1.0 - test_frac)


def _validate_labels(labels: NDArray[np.intp], total_partitions: int) -> None:
    """
    Check to make sure there is more input data than the total number of partitions requested.

    Parameters
    ----------
    labels : np.ndarray of ints
        All class labels from the input dataset
    total_partitions : int
        Number of [train, val, test] splits requested

    Raises
    ------
    ValueError
        When more partitions are requested than number of labels.
    ValueError
        When the labels are considered continuous by Scikit-Learn. This does not necessarily
        mean that floats are not accepted as a label format. Rather, this implies that
        there are too many unique values in the set relative to its cardinality.
    """
    if len(labels) <= total_partitions:
        raise ValueError(
            "Total number of labels must be greater than the total number of partitions. "
            f"Got {len(labels)} labels and {total_partitions} total [train, val, test] partitions.",
        )

    if type_of_target(labels) == "continuous":
        raise ValueError("Detected continuous labels. Labels must be discrete for proper stratification")


def _validate_stratifiable(labels: NDArray[np.intp], num_partitions: int) -> None:
    """
    Check if the dataset can be stratified by class label over the given number of partitions.

    Parameters
    ----------
    labels : NDArray of ints
        All class labels of the input dataset
    num_partitions : int
        Total number of [train, val, test] splits requested

    Returns
    -------
    bool
        True if dataset can be stratified else False

    Raises
    ------
    ValueError
        If the dataset cannot be stratified due to the total number of [train, val, test]
        partitions exceeding the number of instances of the rarest class label.
    """
    # Get the minimum count of all labels
    lowest_label_count = np.unique(labels, return_counts=True)[1].min()
    if lowest_label_count < num_partitions:
        raise ValueError(
            f"Unable to stratify due to label frequency. The lowest label count ({lowest_label_count}) is fewer "
            f"than the total number of partitions ({num_partitions}) requested.",
        )


def _validate_groupable(groups: NDArray[np.intp], num_partitions: int) -> None:
    """
    Warn user if the number of unique group_ids is incompatible with a grouped partition.

    If this is the case, returns groups=None, which tells the partitioner not to
    group the input data.

    Parameters
    ----------
    groups : NDArray of ints
        The id of the group each sample at the corresponding index belongs to
    num_partitions : int
        Total number of train, val, and test splits requested

    Returns
    -------
    bool
        True if the dataset can be grouped by the given group ids else False

    Raises
    ------
    ValueError
        If there are is only one unique group.
    ValueError
        If there are fewer groups than the requested number of partitions plus one
    """
    num_unique_groups = len(np.unique(groups))
    # Cannot separate if only one group exists
    if num_unique_groups == 1:
        raise ValueError(f"Unique groups ({num_unique_groups}) must be greater than 1.")

    if num_unique_groups < num_partitions:
        raise ValueError(f"Unique groups ({num_unique_groups}) must be greater than num partitions ({num_partitions}).")


def _get_groups(metadata: MetadataLike, split_on: Sequence[str] | None) -> NDArray[np.intp] | None:
    """
    Return individual group numbers based on a subset of metadata defined by groupnames.

    Parameters
    ----------
    metadata : MetadataLike
        dictionary containing all metadata
    split_on : Sequence[str] or None
        which groups from the metadata dictionary to consider for dataset grouping

    Returns
    -------
    np.ndarray
        group identifiers from metadata
    """
    # get only the factors that are present in the metadata
    if split_on is None:
        return None

    split_set = set(split_on)
    indices = [i for i, name in enumerate(metadata.factor_names) if name in split_set]
    binned_features = metadata.factor_data[:, indices]
    return np.unique(binned_features, axis=0, return_inverse=True)[1]


def _make_splits(
    index: NDArray[np.intp],
    labels: NDArray[np.intp],
    n_folds: int,
    groups: NDArray[np.intp] | None,
    stratified: bool,
) -> list[TrainValSplit]:
    """
    Split data into n_folds partitions of training and validation data.

    Parameters
    ----------
    index : NDArray of ints
        index corresponding to each label
    labels : NDArray of ints
        classification labels
    n_folds : int
        number of [train, val] folds
    groups : NDArray of ints or None
        group index for grouped partitions. Grouped partitions are split such that no group id is
        present in both a training and validation split.
    stratified : bool
        If True, maintain dataset class balance within each [train, val] split

    Returns
    -------
    split_defs : list[TrainValSplit]
        List of TrainValSplits, which specify train index, validation index, and the ratio of
        validation to all data.
    """
    split_defs: list[TrainValSplit] = []
    n_labels = len(np.unique(labels))
    splitter = _KFOLD_GROUP_STRATIFIED_MAP[(groups is not None, stratified)](n_folds)
    _logger.log(logging.DEBUG, f"splitter={splitter.__class__.__name__}(n_splits={n_folds})")
    good = False
    attempts = 0
    while not good and attempts < 3:
        attempts += 1
        _logger.log(
            logging.DEBUG,
            f"attempt={attempts}: splitter.split("
            f"index=arr(len={len(index)}, unique={np.unique(index)}), "
            f"labels=arr(len={len(index)}, unique={np.unique(index)}), "
            + ("groups=None" if groups is None else f"groups=arr(len={len(groups)}, unique={np.unique(groups)}))"),
        )
        splits = splitter.split(index, labels, groups)
        split_defs.clear()
        for train_idx, eval_idx in splits:
            # test_ratio = len(eval_idx) / len(index)
            t = np.atleast_1d(train_idx).astype(np.intp)
            v = np.atleast_1d(eval_idx).astype(np.intp)
            good = good or (len(np.unique(labels[t])) == n_labels and len(np.unique(labels[v])) == n_labels)
            split_defs.append(TrainValSplit(t, v))
    if not good and attempts == 3:
        _logger.warning("Unable to create a good split definition, not all classes are represented in each split.")
    return split_defs


def _find_best_split(  # noqa: C901
    labels: NDArray[np.intp],
    split_defs: list[TrainValSplit],
    stratified: bool,
    split_frac: float,
) -> TrainValSplit:
    """
    Find the split that most closely satisfies a criterion determined by the arguments passed.

    If stratified is True, returns the split whose class balance most closely resembles the overall
    class balance. If false, returns the split with the size closest to the desired split_frac.

    Parameters
    ----------
    labels : np.ndarray
        Labels upon which splits are (optionally) stratified
    split_defs : list of TrainValSplits
        Specifies the train index, validation index
    stratified : bool
        If True, maintain dataset class balance within each [train, val] split
    split_frac : float
        Desired fraction of the dataset sequestered for evaluation

    Returns
    -------
    TrainValSplit
        Indices of data partitioned for training and evaluation
    """

    # Minimization functions and helpers
    def freq(arr: NDArray[Any], minlength: int = 0) -> NDArray[np.floating[Any]]:
        counts = np.bincount(arr, minlength=minlength)
        return counts / np.sum(counts)

    def weight(arr: NDArray, class_freq: NDArray) -> float:
        return float(np.sum(np.abs(freq(arr, len(class_freq)) - class_freq)))

    def class_freq_diff(split: TrainValSplit) -> float:
        class_freq = freq(labels)
        return weight(labels[split.train], class_freq) + weight(labels[split.val], class_freq)

    def split_ratio(split: TrainValSplit) -> float:
        return len(split.val) / (len(split.val) + len(split.train))

    def split_diff(split: TrainValSplit) -> float:
        return abs(split_frac - split_ratio(split))

    def split_inv_diff(split: TrainValSplit) -> float:
        return abs(1 - split_frac - split_ratio(split))

    # Selects minimization function based on inputs
    if stratified:
        key_func = class_freq_diff
    elif split_frac <= 2 / 3:
        key_func = split_diff
    else:
        key_func = split_inv_diff

    return min(split_defs, key=key_func)


def _single_split(
    index: NDArray[np.intp],
    labels: NDArray[np.intp],
    split_frac: float,
    groups: NDArray[np.intp] | None = None,
    stratified: bool = False,
) -> TrainValSplit:
    """
    Handle the special case where only 1 partition of the data is desired.

    Such as when generating the test holdout split. In this case, the desired fraction of the data to be
    partitioned into the test data must be specified, and a single [train, val] pair is returned.

    Parameters
    ----------
    index : NDArray of ints
        Input Dataset index corresponding to each label
    labels : NDArray of ints
        Labels upon which splits are (optionally) stratified
    split_frac : float
        Fraction of incoming data to be set aside for evaluation
    groups : NDArray of ints, Optional
        Group_ids (same shape as labels) for optional group partitioning
    stratified : bool, default False
        Generates stratified splits if true (recommended)

    Returns
    -------
    TrainValSplit
        Indices of data partitioned for training and evaluation
    """
    unique_groups = len(labels) if groups is None else len(np.unique(groups))
    max_folds = min(min(np.unique(labels, return_counts=True)[1]), unique_groups) if stratified else unique_groups

    divisor = split_frac if split_frac <= 2 / 3 else 1 - split_frac
    n_folds = min(max(round(1 / (divisor + EPSILON)), 2), max_folds)  # Clips value between 2 and max_folds
    _logger.log(logging.DEBUG, f"n_folds={n_folds} clipped between[2, {max_folds}]")

    split_candidates = _make_splits(index, labels, n_folds, groups, stratified)
    return _find_best_split(labels, split_candidates, stratified, split_frac)


def _split_od(  # noqa: C901
    multilabel: NDArray[np.int8],
    num_folds: int,
    stratify: bool,
    test_frac: float,
    val_frac: float,
    total_partitions: int,
    split_on: Sequence[str] | None,
) -> tuple[TrainValSplit, list[TrainValSplit]]:
    """Split an object detection (multi-label) dataset at the image level."""
    n_images = len(multilabel)
    index = np.arange(n_images)

    if n_images <= total_partitions:
        raise ValueError(
            f"Total number of images must be greater than the total number of partitions. "
            f"Got {n_images} images and {total_partitions} total [train, val, test] partitions.",
        )
    if stratify:
        class_counts = multilabel.sum(axis=0)
        min_count = int(class_counts[class_counts > 0].min()) if (class_counts > 0).any() else 0
        if min_count < total_partitions:
            raise ValueError(
                f"Unable to stratify: the rarest class has only {min_count} images, "
                f"fewer than the {total_partitions} partitions requested.",
            )
    if split_on is not None:
        _logger.warning("split_on is not supported for object detection datasets; ignoring.")

    # --- Test split ---
    if test_frac:
        tvs = (
            _multilabel_single_split(index, multilabel, test_frac)
            if stratify
            else _single_split(index, np.argmax(multilabel, axis=1).astype(np.intp), test_frac, None, False)
        )
    else:
        tvs = TrainValSplit(index, np.array([], dtype=np.intp))
    _logger.debug("OD test split: train=%d, test=%d", len(tvs.train), len(tvs.val))

    # --- Train/Val split ---
    tv_ml = multilabel[tvs.train]
    if stratify:
        if num_folds == 1:
            tv_splits = [_multilabel_single_split(tvs.train, tv_ml, val_frac)]
        else:
            tv_splits = _multilabel_make_splits(tvs.train, tv_ml, num_folds)
    else:
        tv_labels = np.argmax(tv_ml, axis=1).astype(np.intp)
        if num_folds == 1:
            tv_splits = [_single_split(tvs.train, tv_labels, val_frac, None, False)]
        else:
            tv_splits = _make_splits(tvs.train, tv_labels, num_folds, None, False)
    _logger.debug("OD train/val: %d fold(s)", len(tv_splits))

    return tvs, tv_splits


def _split_ic(
    metadata: MetadataLike,
    labels: NDArray[np.intp],
    num_folds: int,
    stratify: bool,
    split_on: Sequence[str] | None,
    test_frac: float,
    val_frac: float,
    total_partitions: int,
) -> tuple[TrainValSplit, list[TrainValSplit]]:
    """Split an image classification (single-label) dataset."""
    index = np.arange(len(labels))

    _validate_labels(labels, total_partitions)
    if stratify:
        _validate_stratifiable(labels, total_partitions)

    groups = _get_groups(metadata, split_on)
    if groups is not None:
        group_partitions = total_partitions + 1 if val_frac else total_partitions
        _validate_groupable(groups, group_partitions)

    # --- Test split ---
    if test_frac:
        tvs = _single_split(index, labels, test_frac, groups, stratify)
    else:
        tvs = TrainValSplit(index, np.array([], dtype=np.intp))
    _logger.debug("IC test split: train=%d, test=%d", len(tvs.train), len(tvs.val))

    # --- Train/Val split ---
    tv_labels = labels[tvs.train]
    tv_groups = groups[tvs.train] if groups is not None else None

    if num_folds == 1:
        tv_splits = [_single_split(tvs.train, tv_labels, val_frac, tv_groups, stratify)]
    else:
        tv_splits = _make_splits(tvs.train, tv_labels, num_folds, tv_groups, stratify)
    _logger.debug("IC train/val: %d fold(s)", len(tv_splits))

    return tvs, tv_splits


def split_dataset(
    dataset: AnnotatedDataset[Any] | MetadataLike,
    num_folds: int = 1,
    stratify: bool = False,
    split_on: Sequence[str] | None = None,
    test_frac: float = 0.0,
    val_frac: float = 0.0,
) -> DatasetSplits:
    """
    Dataset splitting function. Returns a dataclass containing a list of train and validation indices.

    Parameters
    ----------
    dataset : AnnotatedDataset or MetadataLike
        Dataset to split.
    num_folds : int, default 1
        Number of [train, val] folds. If equal to 1, val_frac must be greater than 0.0
    stratify : bool, default False
        If true, dataset is split such that the class distribution of the entire dataset is
        preserved within each [train, val] partition, which is generally recommended.
    split_on : list or None, default None
        Keys of the metadata dictionary upon which to group the dataset.
        A grouped partition is divided such that no group is present within both the training and
        validation set. Split_on groups should be selected to mitigate validation bias
    test_frac : float, default 0.0
        Fraction of data to be optionally held out for test set
    val_frac : float, default 0.0
        Fraction of training data to be set aside for validation in the case where a single
        [train, val] split is desired

    Returns
    -------
    split_defs : DatasetSplits
        Output class containing a list of indices of training
        and validation data for each fold and optional test indices

    Notes
    -----
    When specifying groups and/or stratification, ratios for test and validation splits can vary
    as the stratification and grouping take higher priority than the percentages
    """
    _logger.info(
        "split_dataset: num_folds=%d, stratify=%s, test_frac=%.2f, val_frac=%.2f",
        num_folds,
        stratify,
        test_frac,
        val_frac,
    )

    val_frac = _calculate_validation_fraction(num_folds, test_frac, val_frac)
    total_partitions = num_folds + 1 if test_frac else num_folds

    # Import Metadata at runtime to avoid circular import
    from dataeval._metadata import Metadata

    metadata = dataset if isinstance(dataset, MetadataLike) else Metadata(dataset)
    class_labels = metadata.class_labels

    # Detect OD datasets: more detections than images means multi-label
    item_indices: NDArray[np.intp] = getattr(metadata, "item_indices", np.arange(len(class_labels), dtype=np.intp))
    n_images: int = getattr(metadata, "item_count", len(class_labels))
    is_od = len(item_indices) > 0 and len(class_labels) > n_images

    n_classes = len(np.unique(class_labels))
    if is_od:
        _logger.info(
            "Detected Object Detection dataset: %d images, %d classes, %d detections",
            n_images,
            n_classes,
            len(class_labels),
        )
        multilabel = _build_multilabel_matrix(class_labels, item_indices, n_images)
        tvs, tv_splits = _split_od(multilabel, num_folds, stratify, test_frac, val_frac, total_partitions, split_on)
    else:
        _logger.info(
            "Detected Image Classification dataset: %d samples, %d classes",
            len(class_labels),
            n_classes,
        )
        tvs, tv_splits = _split_ic(
            metadata, class_labels, num_folds, stratify, split_on, test_frac, val_frac, total_partitions
        )

    folds: list[TrainValSplit] = [TrainValSplit(tvs.train[split.train], tvs.train[split.val]) for split in tv_splits]

    _logger.info(
        "Split complete: test=%d, %d fold(s) [train=%d, val=%d]",
        len(tvs.val),
        len(folds),
        len(folds[0].train) if folds else 0,
        len(folds[0].val) if folds else 0,
    )

    return DatasetSplits(tvs.val, folds)


class _SizedIterator:
    def __init__(self, iterator: Iterator[Any], length: int) -> None:
        self._iterator = iterator
        self._length = length

    def __iter__(self) -> Iterator[Any]:
        return self._iterator

    def __next__(self) -> Any:
        return next(self._iterator)

    def __len__(self) -> int:
        return self._length


def unzip_dataset(
    dataset: Dataset[Any] | Dataset[tuple[Any, Any, Any]],
    per_target: bool,
) -> tuple[Iterator[NDArray[Any]], Iterator[list[BoundingBox] | None] | None]:
    """
    Unzips a dataset into separate generators for images and targets.

    This preserves performance by only loading each item from the dataset once.

    Parameters
    ----------
    dataset : Dataset
        The dataset to unzip, which may contain images and optional targets.
    per_target : bool
        If True, extract bounding box targets from the dataset.

    Returns
    -------
    tuple[Iterator[NDArray[Any]], Iterator[list[BoundingBox] | None] | None]
        Two iterators, one for images and one for targets.
    """
    from itertools import tee

    def _generate_pairs() -> Iterator[tuple[NDArray[Any], list[BoundingBox] | None]]:
        for i in range(len(dataset)):
            d = dataset[i]
            image = np.asarray(unwrap_image(d))
            if per_target and isinstance(d, tuple) and isinstance(d[1], ObjectDetectionTarget):
                try:
                    boxes = d[1].boxes if isinstance(d[1].boxes, Array) else as_numpy(d[1].boxes)
                    target = [BoundingBox(box[0], box[1], box[2], box[3], image_shape=image.shape) for box in boxes]
                except (ValueError, IndexError) as err:
                    raise ValueError(f"Invalid bounding box format for image {i}: {d[1].boxes}") from err
            else:
                target = None
            yield image, target

    # Create two independent iterators from the generator
    iter1, iter2 = tee(_generate_pairs(), 2)

    # Extract images and targets separately
    images_iter = _SizedIterator((pair[0] for pair in iter1), len(dataset))
    targets_iter = (pair[1] for pair in iter2) if per_target else None

    return images_iter, targets_iter

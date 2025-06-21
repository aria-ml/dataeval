from __future__ import annotations

__all__ = []

import logging
import warnings
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target

from dataeval.config import EPSILON
from dataeval.data._metadata import Metadata
from dataeval.outputs._base import set_metadata
from dataeval.outputs._utils import SplitDatasetOutput, TrainValSplit
from dataeval.typing import AnnotatedDataset

_logger = logging.getLogger(__name__)


class KFoldSplitter(Protocol):
    """Protocol covering sklearn KFold variant splitters"""

    def __init__(self, n_splits: int) -> None: ...
    def split(self, X: Any, y: Any, groups: Any) -> Iterator[tuple[NDArray[Any], NDArray[Any]]]: ...


KFOLD_GROUP_STRATIFIED_MAP: dict[tuple[bool, bool], type[KFoldSplitter]] = {
    (False, False): KFold,
    (False, True): StratifiedKFold,
    (True, False): GroupKFold,
    (True, True): StratifiedGroupKFold,
}


def calculate_validation_fraction(num_folds: int, test_frac: float, val_frac: float) -> float:
    """
    Calculate possible validation fraction based on the number of folds and test fraction.

    Parameters
    ----------
    num_folds : int
        number of train and validation cross-validation folds to generate
    test_frac : float
        The fraction of the data to extract for testing before folds are created
    val_frac : float
        The validation split will contain (val_frac * 100)% of any data not already allocated to the test set.
        Only required if requesting a single [train, val] split.

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

    Returns
    -------
    float
        The updated validation fraction of the remaining data after the testing fraction is removed
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


def validate_labels(labels: NDArray[np.intp], total_partitions: int) -> None:
    """
    Check to make sure there is more input data than the total number of partitions requested

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
            f"Got {len(labels)} labels and {total_partitions} total [train, val, test] partitions."
        )

    if type_of_target(labels) == "continuous":
        raise ValueError("Detected continuous labels. Labels must be discrete for proper stratification")


def validate_stratifiable(labels: NDArray[np.intp], num_partitions: int) -> None:
    """
    Check if the dataset can be stratified by class label over the given number of partitions

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
            f"than the total number of partitions ({num_partitions}) requested."
        )


def validate_groupable(groups: NDArray[np.intp], num_partitions: int) -> None:
    """
    Warns user if the number of unique group_ids is incompatible with a grouped partition containing
    num_folds folds. If this is the case, returns groups=None, which tells the partitioner not to
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


def get_groups(metadata: Metadata, split_on: Sequence[str] | None) -> NDArray[np.intp] | None:
    """
    Returns individual group numbers based on a subset of metadata defined by groupnames

    Parameters
    ----------
    metadata : dict
        dictionary containing all metadata
    groupnames : list
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
    binned_features = metadata.binned_data[:, indices]
    return np.unique(binned_features, axis=0, return_inverse=True)[1]


def make_splits(
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
    splitter = KFOLD_GROUP_STRATIFIED_MAP[(groups is not None, stratified)](n_folds)
    _logger.log(logging.DEBUG, f"splitter={splitter.__class__.__name__}(n_splits={n_folds})")
    good = False
    attempts = 0
    while not good and attempts < 3:
        attempts += 1
        _logger.log(
            logging.DEBUG,
            f"attempt={attempts}: splitter.split("
            + f"index=arr(len={len(index)}, unique={np.unique(index)}), "
            + f"labels=arr(len={len(index)}, unique={np.unique(index)}), "
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
        warnings.warn("Unable to create a good split definition, not all classes are represented in each split.")
    return split_defs


def find_best_split(
    labels: NDArray[np.intp], split_defs: list[TrainValSplit], stratified: bool, split_frac: float
) -> TrainValSplit:
    """
    Finds the split that most closely satisfies a criterion determined by the arguments passed.
    If stratified is True, returns the split whose class balance most closely resembles the overall
    class balance. If false, returns the split with the size closest to the desired split_frac

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


def single_split(
    index: NDArray[np.intp],
    labels: NDArray[np.intp],
    split_frac: float,
    groups: NDArray[np.intp] | None = None,
    stratified: bool = False,
) -> TrainValSplit:
    """
    Handles the special case where only 1 partition of the data is desired (such as when
    generating the test holdout split). In this case, the desired fraction of the data to be
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

    unique_groups = 2 if groups is None else len(np.unique(groups))
    max_folds = min(min(np.unique(labels, return_counts=True)[1]), unique_groups) if stratified else unique_groups

    divisor = split_frac if split_frac <= 2 / 3 else 1 - split_frac
    n_folds = min(max(round(1 / (divisor + EPSILON)), 2), max_folds)  # Clips value between 2 and max_folds
    _logger.log(logging.DEBUG, f"n_folds={n_folds} clipped between[2, {max_folds}]")

    split_candidates = make_splits(index, labels, n_folds, groups, stratified)
    return find_best_split(labels, split_candidates, stratified, split_frac)


@set_metadata
def split_dataset(
    dataset: AnnotatedDataset[Any] | Metadata,
    num_folds: int = 1,
    stratify: bool = False,
    split_on: Sequence[str] | None = None,
    test_frac: float = 0.0,
    val_frac: float = 0.0,
) -> SplitDatasetOutput:
    """
    Dataset splitting function. Returns a dataclass containing a list of train and validation indices.

    Parameters
    ----------
    dataset : AnnotatedDataset or Metadata
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
    split_defs : SplitDatasetOutput
        Output class containing a list of indices of training
        and validation data for each fold and optional test indices

    Notes
    -----
    When specifying groups and/or stratification, ratios for test and validation splits can vary
    as the stratification and grouping take higher priority than the percentages
    """

    val_frac = calculate_validation_fraction(num_folds, test_frac, val_frac)
    total_partitions = num_folds + 1 if test_frac else num_folds

    metadata = dataset if isinstance(dataset, Metadata) else Metadata(dataset)
    labels = metadata.class_labels

    validate_labels(labels, total_partitions)
    if stratify:
        validate_stratifiable(labels, total_partitions)

    groups = get_groups(metadata, split_on)
    if groups is not None:
        # Accounts for a test set that is 100 % of the data
        group_partitions = total_partitions + 1 if val_frac else total_partitions
        validate_groupable(groups, group_partitions)

    index = np.arange(len(labels))

    if test_frac:
        tvs = single_split(index, labels, test_frac, groups, stratify)
    else:
        tvs = TrainValSplit(index, np.array([], dtype=np.intp))

    tv_labels = labels[tvs.train]
    tv_groups = groups[tvs.train] if groups is not None else None

    if num_folds == 1:
        tv_splits = [single_split(tvs.train, tv_labels, val_frac, tv_groups, stratify)]
    else:
        tv_splits = make_splits(tvs.train, tv_labels, num_folds, tv_groups, stratify)

    folds: list[TrainValSplit] = [TrainValSplit(tvs.train[split.train], tvs.train[split.val]) for split in tv_splits]

    return SplitDatasetOutput(tvs.val, folds)

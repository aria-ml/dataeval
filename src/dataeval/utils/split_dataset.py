from __future__ import annotations

from dataclasses import dataclass

from dataeval.output import Output, set_metadata

__all__ = ["split_dataset", "SplitDatasetOutput"]

import warnings
from typing import Any, Iterator, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target


class TrainValSplit(NamedTuple):
    """Tuple containing train and validation indices"""

    train: NDArray[np.int_]
    val: NDArray[np.int_]


@dataclass(frozen=True)
class SplitDatasetOutput(Output):
    """Output class containing test indices and a list of TrainValSplits"""

    test: NDArray[np.int_]
    folds: list[TrainValSplit]


class KFoldSplitter(Protocol):
    """Protocol covering sklearn KFold variant splitters"""

    def __init__(self, n_splits: int): ...
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


def _validate_labels(labels: NDArray[np.int_], total_partitions: int) -> None:
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


def is_stratifiable(labels: NDArray[np.int_], num_partitions: int) -> bool:
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

    Warns
    -----
    UserWarning
        Warns user if the dataset cannot be stratified due to the total number of [train, val, test]
        partitions exceeding the number of instances of the rarest class label.
    """

    # Get the minimum count of all labels
    lowest_label_count = np.unique(labels, return_counts=True)[1].min()
    if lowest_label_count < num_partitions:
        warnings.warn(
            f"Unable to stratify due to label frequency. The lowest label count ({lowest_label_count}) is fewer "
            f"than the total number of partitions ({num_partitions}) requested.",
            UserWarning,
        )
        return False
    return True


def is_groupable(group_ids: NDArray[np.int_], num_partitions: int) -> bool:
    """
    Warns user if the number of unique group_ids is incompatible with a grouped partition containing
    num_folds folds. If this is the case, returns groups=None, which tells the partitioner not to
    group the input data.

    Parameters
    ----------
    group_ids : NDArray of ints
        The id of the group each sample at the corresponding index belongs to
    num_partitions : int
        Total number of train, val, and test splits requested

    Returns
    -------
    bool
        True if the dataset can be grouped by the given group ids else False

    Warns
    -----
    UserWarning
        Warns if there are fewer groups than the requested number of partitions plus one
    """

    num_unique_groups = len(np.unique(group_ids))
    # Cannot separate if only one group exists
    if num_unique_groups == 1:
        return False

    if num_unique_groups < num_partitions:
        warnings.warn(
            f"Groups must be greater than num partitions. Got {num_unique_groups} and {num_partitions}. "
            "Reverting to ungrouped partitioning",
            UserWarning,
        )
        return False
    return True


def bin_kmeans(array: NDArray[Any]) -> NDArray[np.int_]:
    """
    Find bins of continuous data by iteratively applying k-means clustering, and keeping the
    clustering with the highest silhouette score.

    Parameters
    ----------
    array : NDArray
        continuous data to bin

    Returns
    -------
    NDArray[int]:
        bin numbers assigned by the kmeans best clusterer.
    """

    if array.ndim == 1:
        array = array.reshape([-1, 1])
        best_score = 0.60
    else:
        best_score = 0.50
    bin_index = np.zeros(len(array), dtype=np.int_)
    for k in range(2, 20):
        clusterer = KMeans(n_clusters=k)
        cluster_labels = clusterer.fit_predict(array)
        score = silhouette_score(array, cluster_labels, sample_size=25_000)
        if score > best_score:
            best_score = score
            bin_index = cluster_labels.astype(np.int_)
    return bin_index


def get_group_ids(metadata: dict[str, Any], group_names: list[str], num_samples: int) -> NDArray[np.int_]:
    """
    Returns individual group numbers based on a subset of metadata defined by groupnames

    Parameters
    ----------
    metadata : dict
        dictionary containing all metadata
    groupnames : list
        which groups from the metadata dictionary to consider for dataset grouping
    num_samples : int
        number of labels. Used to ensure agreement between input data/labels and metadata entries.

    Raises
    ------
    IndexError
        raised if an entry in the metadata dictionary doesn't have the same length as num_samples

    Returns
    -------
    np.ndarray
        group identifiers from metadata
    """
    features2group = {k: np.array(v) for k, v in metadata.items() if k in group_names}
    if not features2group:
        return np.zeros(num_samples, dtype=np.int_)
    for name, feature in features2group.items():
        if len(feature) != num_samples:
            raise ValueError(
                f"Feature length does not match number of labels. "
                f"Got {len(feature)} features and {num_samples} samples"
            )

        if type_of_target(feature) == "continuous":
            features2group[name] = bin_kmeans(feature)
    binned_features = np.stack(list(features2group.values()), axis=1)
    _, group_ids = np.unique(binned_features, axis=0, return_inverse=True)
    return group_ids


def make_splits(
    index: NDArray[np.int_],
    labels: NDArray[np.int_],
    n_folds: int,
    groups: NDArray[np.int_] | None,
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
    good = False
    attempts = 0
    while not good and attempts < 3:
        attempts += 1
        splits = splitter.split(index, labels, groups)
        split_defs.clear()
        for train_idx, eval_idx in splits:
            # test_ratio = len(eval_idx) / len(index)
            t = np.atleast_1d(train_idx).astype(np.int_)
            v = np.atleast_1d(eval_idx).astype(np.int_)
            good = good or (len(np.unique(labels[t])) == n_labels and len(np.unique(labels[v])) == n_labels)
            split_defs.append(TrainValSplit(t, v))
    if not good and attempts == 3:
        warnings.warn("Unable to create a good split definition, not all classes are represented in each split.")
    return split_defs


def find_best_split(
    labels: NDArray[np.int_], split_defs: list[TrainValSplit], stratified: bool, split_frac: float
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

    def weight(arr: NDArray, class_freq: NDArray) -> np.float64:
        return np.sum(np.abs(freq(arr, len(class_freq)) - class_freq))

    def class_freq_diff(split: TrainValSplit) -> np.float64:
        class_freq = freq(labels)
        return weight(labels[split.train], class_freq) + weight(labels[split.val], class_freq)

    def split_ratio(split: TrainValSplit) -> np.float64:
        return np.float64(len(split.val) / (len(split.val) + len(split.train)))

    def split_diff(split: TrainValSplit) -> np.float64:
        return abs(split_frac - split_ratio(split))

    def split_inv_diff(split: TrainValSplit) -> np.float64:
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
    index: NDArray[np.int_],
    labels: NDArray[np.int_],
    split_frac: float,
    groups: NDArray[np.int_] | None = None,
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

    _, label_counts = np.unique(labels, return_counts=True)
    max_folds = label_counts.min()
    min_folds = np.unique(groups).shape[0] if groups is not None else 2
    divisor = split_frac + 1e-06 if split_frac <= 2 / 3 else 1 - split_frac - 1e-06
    n_folds = round(min(max(1 / divisor, min_folds), max_folds))  # Clips value between min_folds and max_folds

    split_candidates = make_splits(index, labels, n_folds, groups, stratified)
    return find_best_split(labels, split_candidates, stratified, split_frac)


@set_metadata
def split_dataset(
    labels: list[int] | NDArray[np.int_],
    num_folds: int = 1,
    stratify: bool = False,
    split_on: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    test_frac: float = 0.0,
    val_frac: float = 0.0,
) -> SplitDatasetOutput:
    """
    Top level splitting function. Returns a dataclass containing a list of train and validation indices.
    Indices for a test holdout may also be optionally included

    Parameters
    ----------
    labels : list or NDArray of ints
        Classification Labels used to generate splits. Determines the size of the dataset
    num_folds : int, default 1
        Number of [train, val] folds. If equal to 1, val_frac must be greater than 0.0
    stratify : bool, default False
        If true, dataset is split such that the class distribution of the entire dataset is
        preserved within each [train, val] partition, which is generally recommended.
    split_on : list or None, default None
        Keys of the metadata dictionary upon which to group the dataset.
        A grouped partition is divided such that no group is present within both the training and
        validation set. Split_on groups should be selected to mitigate validation bias
    metadata : dict or None, default None
        Dict containing data for potential dataset grouping. See split_on above
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

    Raises
    ------
    TypeError
        Raised if split_on is passed, but metadata is None or empty

    Note
    ----
    When specifying groups and/or stratification, ratios for test and validation splits can vary
    as the stratification and grouping take higher priority than the percentages
    """

    val_frac = calculate_validation_fraction(num_folds, test_frac, val_frac)
    total_partitions = num_folds + 1 if test_frac else num_folds

    if isinstance(labels, list):
        labels = np.array(labels, dtype=np.int_)

    label_length: int = len(labels)

    _validate_labels(labels, total_partitions)
    stratify &= is_stratifiable(labels, total_partitions)
    groups = None
    if split_on:
        if metadata is None or metadata == {}:
            raise TypeError("If split_on is specified, metadata must also be provided, got None")
        possible_groups = get_group_ids(metadata, split_on, label_length)
        # Accounts for a test set that is 100 % of the data
        group_partitions = total_partitions + 1 if val_frac else total_partitions
        if is_groupable(possible_groups, group_partitions):
            groups = possible_groups

    test_indices: NDArray[np.int_]
    index = np.arange(label_length)

    tv_indices, test_indices = (
        single_split(index=index, labels=labels, split_frac=test_frac, groups=groups, stratified=stratify)
        if test_frac
        else (index, np.array([], dtype=np.int_))
    )

    tv_labels = labels[tv_indices]
    tv_groups = groups[tv_indices] if groups is not None else None

    if num_folds == 1:
        tv_splits = [single_split(tv_indices, tv_labels, val_frac, tv_groups, stratify)]
    else:
        tv_splits = make_splits(tv_indices, tv_labels, num_folds, tv_groups, stratify)

    folds: list[TrainValSplit] = [TrainValSplit(tv_indices[split.train], tv_indices[split.val]) for split in tv_splits]

    return SplitDatasetOutput(test_indices, folds)

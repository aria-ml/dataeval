from __future__ import annotations

__all__ = ["split_dataset"]

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target


def validate_test_val(num_folds: int, test_frac: float | None, val_frac: float | None) -> tuple[float, float]:
    """Check input fractions to ensure unambiguous splitting arguments are passed return calculated
    test and validation fractions.


    Parameters
    ----------
    num_folds : int
        number of [train, val] cross-validation folds to generate
    test_frac : float, optional
        If specified, also generate a test set containing (test_frac*100)% of the data
    val_frac  : float, optional
        Only specify if requesting a single [train, val] split. The validation split will
        contain (val_frac*100)% of any data not already allocated to the test set

    Raises
    ------
    UnboundLocalError
        Raised if more than one fold AND the fraction of data to be used for validation are
        both requested. In this case, val_frac is ambiguous, since the validation fraction must be
        by definition 1/num_folds
    ValueError
        Raised if num_folds is 1 (or left blank) AND val_frac is unspecified. When only 1 fold is
        requested, we need to know how much of the data should be allocated for validation.
    ValueError
        Raised if the total fraction of data used for evaluation (val + test) meets or exceeds 1.0

    Returns
    -------
    tuple[float, float]
        Tuple of the validated and calculated values as appropriate for test and validation fractions
    """
    if (num_folds > 1) and (val_frac is not None):
        raise ValueError("If specifying val_frac, num_folds must be None or 1")
    if (num_folds == 1) and (val_frac is None):
        raise ValueError("If num_folds is None or 1, must assign a value to val_frac")
    t_frac = 0.0 if test_frac is None else test_frac
    v_frac = 1.0 / num_folds * (1.0 - t_frac) if val_frac is None else val_frac * (1.0 - t_frac)
    if (t_frac + v_frac) >= 1.0:
        raise ValueError(f"val_frac + test_frac must be less that 1.0, currently {v_frac+t_frac}")
    return t_frac, v_frac


def check_labels(
    labels: list[int] | NDArray[np.int_], total_partitions: int
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Check to make sure there are more input data than the total number of partitions requested
    Also converts labels to a numpy array, if it isn't already

    Parameters
    ----------
    labels : list or np.ndarray
        all class labels from the input dataset
    total_partitions : int
        number of train-val splits requested (+1 if a test holdout is specified)

    Raises
    ------
    IndexError
        Raised if more partitions are requested than number of labels. This is exceedingly rare and
        usually means you've specified some argument incorrectly.
    ValueError
        Raised if the labels are considered continuous by Scikit-Learn. This does not necessarily
        mean that floats are not accepted as a label format. Rather, this exception implies that
        there are too many unique values in the set relative to it's cardinality.

    Returns
    -------
    index : np.ndarray
        Integer index generated based on the total number of labels
    labels : np.ndarray
        labels, converted to an ndarray if passed as a list.
    """
    if len(labels) <= total_partitions:
        raise IndexError(f"""
            Total number of labels must greater than the number of total partitions.
            Got {len(labels)} labels and {total_partitions} total train/val/test partitions.""")
    if isinstance(labels, list):
        labels = np.array(labels)
    if type_of_target(labels) == "continuous":
        raise ValueError("Detected continuous labels, labels must be discrete for proper stratification")
    index = np.arange(len(labels))
    return index, labels


def check_stratifiable(labels: NDArray[np.int_], total_partitions: int) -> bool:
    """
    Very basic check to see if dataset can be stratified by class label. This is not a
    comprehensive test, as factors such as grouping also affect the ability to stratify by label

    Parameters
    ----------
    labels : list or np.ndarray
        all class labels from the input dataset
    total_partitions : int
        number of train-val splits requested (+1 if a test holdout is specified)

    Warns
    -----
    UserWarning
        Warns user if the dataset cannot be stratified due to the number of total (train, val, test)
        partitions exceeding the number of instances of the rarest class label.

    Returns
    -------
    stratifiable : bool
        True if dataset can be stratified according to the criteria above.
    """

    stratifiable = True
    _, label_counts = np.unique(labels, return_counts=True)
    rarest_label_count = label_counts.min()
    if rarest_label_count < total_partitions:
        warnings.warn(f"""
            Unable to stratify due to label frequency. The rarest label occurs {rarest_label_count},
            which is fewer than the total number of partitions requested. Setting stratify flag to 
            false.""")
        stratifiable = False
    return stratifiable


def check_groups(group_ids: NDArray[np.int_], num_partitions: int) -> bool:
    """
    Warns user if the number of unique group_ids is incompatible with a grouped partition containing
    num_folds folds. If this is the case, returns groups=None, which tells the partitioner not to
    group the input data.

    Parameters
    ----------
    group_ids : np.ndarray
        Identifies the group to which a sample at the same index belongs.
    num_partitions: int
        How many total (train, val) folds will be generated (+1 if also specifying a test fold).

    Warns
    -----
    UserWarning
        Warns if there are fewer groups than the minimum required to successfully partition the data
        into num_partitions. The minimum is defined as the number of partitions requested plus one.

    Returns
    -------
    groupable : bool
        True if dataset can be grouped by the given group ids, given the criteria above.
    """

    groupable = True
    num_unique_groups = len(np.unique(group_ids))
    min_unique_groups = num_partitions + 1
    if num_unique_groups < min_unique_groups:
        warnings.warn(f"""
            {min_unique_groups} unique groups required for {num_partitions} partitions. 
            Found {num_unique_groups} instead. Reverting to ungrouped partitioning""")
        groupable = False
    else:
        groupable = True
    return groupable


def bin_kmeans(array: NDArray[Any]) -> NDArray[np.int_]:
    """
    Find bins of continuous data by iteratively applying k-means clustering, and keeping the
    clustering with the highest silhouette score.

    Parameters
    ----------
    array : np.ndarray
        continuous data to bin

    Returns
    -------
        np.ndarray[int]: bin numbers assigned by the kmeans best clusterer.
    """
    array = np.array(array)
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


def angle2xy(angles: NDArray[Any]) -> NDArray[Any]:
    """
    Converts angle measurements to xy coordinates on the unit circle. Needed for binning angle data.

    Parameters
    ----------
    angles : np.ndarray
        angle data in either radians or degrees

    Returns
    -------
    xy : np.ndarray
        Nx2 array of xy coordinates for each angle (can be radians or degrees)
    """
    is_radians = ((angles >= -np.pi) & (angles <= 2 * np.pi)).all()
    radians = angles if is_radians else np.pi / 180 * angles
    xy = np.stack([np.cos(radians), np.sin(radians)], axis=1)
    return xy


def get_group_ids(metadata: dict[str, Any], group_names: list[str], num_samples: int) -> NDArray[np.int_]:
    """Returns individual group numbers based on a subset of metadata defined by groupnames

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
    group_ids: np.ndarray
        group identifiers from metadata
    """
    features2group = {k: np.array(v) for k, v in metadata.items() if k in group_names}
    if not features2group:
        return np.zeros(num_samples, dtype=int)
    for name, feature in features2group.items():
        if len(feature) != num_samples:
            raise IndexError(f"""Feature length does not match number of labels. 
                             Got {len(feature)} features and {num_samples} samples""")
        if type_of_target(feature) == "continuous":
            if ("ANGLE" in name.upper()) or ("AZIMUTH" in name.upper()):
                feature = angle2xy(feature)
            features2group[name] = bin_kmeans(feature)
    binned_features = np.stack(list(features2group.values()), axis=1)
    _, group_ids = np.unique(binned_features, axis=0, return_inverse=True)
    return group_ids


def make_splits(
    index: NDArray[np.int_],
    labels: NDArray[np.int_],
    n_folds: int,
    groups: NDArray[np.int_] | None = None,
    stratified: bool = False,
) -> list[dict[str, NDArray[np.int_]]]:
    """Split data into n_folds partitions of training and validation data.

    Parameters
    ----------
    index : np.ndarray
        index corresponding to each label (see below)
    labels : np.ndarray
        classification labels
    n_folds : int
        number or train/val folds
    groups : np.ndarray, Optional
        group index for grouped partitions. Grouped partitions are split such that no group id is
        present in both a training and validation split.
    stratified : bool, default=False
        If True, maintain dataset class balance within each train/val split

    Returns
    -------
    split_defs : list[dict]
        list of dictionaries, which specifying train index, validation index, and the ratio of
        validation to all data.
    """
    split_defs = []
    index = index.reshape([-1, 1])
    if groups is not None:
        splitter = StratifiedGroupKFold(n_folds) if stratified else GroupKFold(n_folds)
        splits = splitter.split(index, labels, groups)
    else:
        splitter = StratifiedKFold(n_folds) if stratified else KFold(n_folds)
        splits = splitter.split(index, labels)
    for train_idx, eval_idx in splits:
        test_ratio = len(eval_idx) / index.shape[0]
        split_defs.append({"train": train_idx.astype(int), "eval": eval_idx.astype(int), "eval_frac": test_ratio})
    return split_defs


def find_best_split(
    labels: NDArray[np.int_], split_defs: list[dict[str, NDArray[np.int_]]], stratified: bool, eval_frac: float
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Finds the split that most closely satisfies a criterion determined by the arguments passed.
    If stratified is True, returns the split whose class balance most closely resembles the overall
    class balance. If false, returns the split with the size closest to the desired eval_frac

    Parameters
    ----------
    labels : np.ndarray
        Labels upon which splits are (optionally) stratified
    split_defs : list[dict]
        List of dictionaries, which specifying train index, validation index, and the ratio of
        validation to all data.
    stratified: bool
        If True, maintain dataset class balance within each train/val split
    eval_frac: float
        Desired fraction of the dataset sequestered for evaluation

    Returns
    -------
    train_index : np.ndarray
        indices of data partitioned for training
    eval_index : np.ndarray
        indices of data partitioned for evaluation
    """

    def class_freq_diff(split):
        train_labels = labels[split["train"]]
        _, train_counts = np.unique(train_labels, return_counts=True)
        train_freq = train_counts / train_counts.sum()
        return np.square(train_freq - class_freq).sum()

    if stratified:
        _, class_counts = np.unique(labels, return_counts=True)
        class_freq = class_counts / class_counts.sum()
        best_split = min(split_defs, key=class_freq_diff)
        return best_split["train"], best_split["eval"]
    elif eval_frac <= 2 / 3:
        best_split = min(split_defs, key=lambda x: abs(eval_frac - x["eval_frac"]))  # type: ignore
        return best_split["train"], best_split["eval"]
    else:
        best_split = min(split_defs, key=lambda x: abs(eval_frac - (1 - x["eval_frac"])))  # type: ignore
        return best_split["eval"], best_split["train"]


def single_split(
    index: NDArray[np.int_],
    labels: NDArray[np.int_],
    eval_frac: float,
    groups: NDArray[np.int_] | None = None,
    stratified: bool = False,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Handles the special case where only 1 partition of the data is desired (such as when
    generating the test holdout split). In this case, the desired fraction of the data to be
    partitioned into the test data must be specified, and a single [train, eval] pair are returned.

    Parameters
    ----------
    index : np.ndarray
        Input Dataset index corresponding to each label
    labels : np.ndarray
        Labels upon which splits are (optionally) stratified
    eval_frac : float
        Fraction of incoming data to be set aside for evaluation
    groups : np.ndarray, Optional
        Group_ids (same shape as labels) for optional group partitioning
    stratified : bool, default=False
        Generates stratified splits if true (recommended)

    Returns
    -------
    train_index : np.ndarray
        indices of data partitioned for training
    eval_index : np.ndarray
        indices of data partitioned for evaluation
    """
    if groups is not None:
        n_unique_groups = np.unique(groups).shape[0]
        _, label_counts = np.unique(labels, return_counts=True)
        n_folds = min(n_unique_groups, label_counts.min())
    elif eval_frac <= 2 / 3:
        n_folds = max(2, int(round(1 / (eval_frac + 1e-6))))
    else:
        n_folds = max(2, int(round(1 / (1 - eval_frac - 1e-6))))
    split_candidates = make_splits(index, labels, n_folds, groups, stratified)
    best_train, best_eval = find_best_split(labels, split_candidates, stratified, eval_frac)
    return best_train, best_eval


def split_dataset(
    labels: list[int] | NDArray[np.int_],
    num_folds: int = 1,
    stratify: bool = False,
    split_on: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    test_frac: float | None = None,
    val_frac: float | None = None,
) -> dict[str, dict[str, NDArray[np.int_]] | NDArray[np.int_]]:
    """Top level splitting function. Returns a dict with each key-value pair containing
    train and validation indices. Indices for a test holdout may also be optionally included

    Parameters
    ----------
    labels : Union[list, np.ndarray]
        Classification Labels used to generate splits. Determines the size of the dataset
    num_folds : int, optional
        Number of train/val folds. If None, returns a single train/val split, and val_frac must be
        specified. Defaults to None.
    stratify : bool, default=False
        If true, dataset is split such that the class distribution of the entire dataset is
        preserved within each train/val partition, which is generally recommended.
    split_on : list, optional
        Keys of the metadata dictionary which map to columns upon which to group the dataset.
        A grouped partition is divided such that no group is present within both the training and
        validation set. Split_on groups should be selected to mitigate validation bias. Defaults to
        None, in which groups will not be considered when partitioning the data.
    metadata : dict, optional
        Dict containing data for potential dataset grouping. See split_on above. Defaults to None.
    test_frac : float, optional
        Fraction of data to be optionally held out for test set. Defaults to None, in which no test
        set is created.
    val_frac : float, optional
        Fraction of training data to be set aside for validation in the case where a single
        train/val split is desired. Defaults to None.

    Raises
    ------
    UnboundLocalError
        Raised if split_on is passed, but metadata is left as None. This is because split_on
        defines the keys in which metadata dict must be indexed to determine the group index of the
        data

    Returns
    -------
    split_defs : dict
        dictionary of folds, each containing indices of training and validation data.
        ex.
        {
            "Fold_00":  {
                            "train": [1,2,3,5,6,7,9,10,11],
                            "val": [0, 4, 8, 12]
                        },
            "test": [13, 14, 15, 16]
        }
    """

    test_frac, val_frac = validate_test_val(num_folds, test_frac, val_frac)
    total_partitions = num_folds + 1 if test_frac else num_folds
    index, labels = check_labels(labels, total_partitions)
    stratify &= check_stratifiable(labels, total_partitions)
    if split_on:
        if metadata is None:
            raise UnboundLocalError("If split_on is specified, metadata must also be provided")
        groups = get_group_ids(metadata, split_on, len(labels))
        groupable = check_groups(groups, total_partitions)
        if not groupable:
            groups = None
    else:
        groups = None
    split_defs: dict[str, dict[str, NDArray[np.int_]] | NDArray[np.int_]] = {}
    if test_frac:
        tv_idx, test_idx = single_split(index, labels, test_frac, groups, stratify)
        tv_labels = labels[tv_idx]
        tv_groups = groups[tv_idx] if groups is not None else None
        split_defs["test"] = test_idx
    else:
        tv_idx = np.arange(len(labels)).reshape((-1, 1))
        tv_labels = labels
        tv_groups = groups
    if num_folds == 1:
        train_idx, val_idx = single_split(tv_idx, tv_labels, val_frac, tv_groups, stratify)
        split_defs["fold_0"] = {"train": tv_idx[train_idx].squeeze(), "val": tv_idx[val_idx].squeeze()}
    else:
        tv_splits = make_splits(tv_idx, tv_labels, num_folds, tv_groups, stratify)
        for i, split in enumerate(tv_splits):
            train_split = tv_idx[split["train"]]
            val_split = tv_idx[split["eval"]]
            split_defs[f"fold_{i}"] = {"train": train_split.squeeze(), "val": val_split.squeeze()}
    return split_defs

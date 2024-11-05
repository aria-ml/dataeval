from __future__ import annotations

import warnings
from typing import NewType, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.utils.multiclass import type_of_target
from tqdm import tqdm


def check_labels(labels: Union[list, np.ndarray], total_partitions: int):
    """Check to make sure there are more input data than the total number of partitions requested
    Also converts labels to a :term:`NumPy` array, if it isn't already

    Args:
        labels (Union[list, np.ndarray]): all class labels from the input dataset
        total_partitions (int): number of train-val splits requested + 1 if an additional test 
            holdout is specified

    Returns:
        np.ndarray: labels, possibly converted to an array if passed as a list. 
    """
    if len(labels) < total_partitions:
        raise IndexError(f"""
            Total number of labels must greater than or equal to number of total partitions.
            Got {len(labels)} labels and {total_partitions} total train/val/test partitions.""")
    if isinstance(labels, list):
        labels = np.array(labels)
    index = np.arange(len(labels))
    return index, labels


def check_groups(groups: np.ndarray, num_folds: int):
    """
    Warns user if the number of unique group_ids is incompatible with a grouped partition containing
    num_folds folds. If this is the case, returns groups=None, which tells the partitioner not to
    group the input data. 
    """
    num_unique_groups = len(np.unique(groups))
    min_unique_groups = num_folds + 1
    if num_unique_groups < min_unique_groups:
        warnings.warn(f"""
            {min_unique_groups} unique groups required for {num_folds} partitions. 
            Found {num_unique_groups} instead. Reverting to ungrouped partitioning""")
        return None
    else:
        return groups
    

def bin_kmeans(array: np.ndarray):
    """Find bins of continuous data by iteratively applying k-means clustering, and keeping
    the clustering with the highest silhouette score. 

    Args:
        array (np.ndarray): continuous data to bin

    Returns:
        np.ndarray[int]: bin numbers assigned by the kmeans best clusterer.
    """
    array = np.array(array)
    if array.ndim == 1:
        array = array.reshape([-1,1])
        best_score = 0.60
    else:
        best_score = 0.50
    bin_index = np.zeros(len(array))
    for k in range(2,20):
        clusterer = KMeans(n_clusters=k)
        cluster_labels = clusterer.fit_predict(array)
        score = silhouette_score(array, cluster_labels, sample_size=25_000)
        if score > best_score:
            best_score = score
            bin_index = cluster_labels
    return bin_index


def angle2xy(angles: np.ndarray):
    """converts angle measurements to xy coordinates on the unit circle. Needed for binning
    angle data.

    Args:
        angles (np.ndarray): angle data in either radians or degrees

    Returns:
        np.ndarray: Nx2 array of xy coordinates for each angle in angles
    """
    is_radians = ((angles>=-np.pi) & (angles<=2*np.pi)).all()
    radians = angles if is_radians else np.pi/180 * angles
    xy = np.stack([np.cos(radians), np.sin(radians)], axis=1)
    return xy


def get_group_ids(metadata: dict, groupnames: list, num_samples:int):
    """Returns individual group numbers based on a subset of metadata defined by groupnames

    Args:
        metadata (dict): dictionary containing all metadata
        groupnames (list): which groups from the metadata dictionary to consider for dataset grouping
        num_samples (int): number of labels. Used to ensure agreement between input data/labels and 
            metadata entries. 

    Returns:
        np.ndarray: group identifiers from metadata
    """
    features2group = {k:np.array(v) for k,v in metadata.items() if k in groupnames}
    if not features2group:
        return np.zeros(num_samples,dtype=int)
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
    index: np.ndarray,
    labels: np.ndarray, 
    n_folds: int,
    groups: Optional[np.ndarray]=None, 
    stratified: Optional[bool]=False
    ):
    """Split data into n_folds partitions of training and validation data. 

    Args:
        index (np.ndarray): index corresponding to each label (see below)
        labels (np.ndarray): classification labels
        n_folds (int): number or train/val folds 
        groups (Optional[np.ndarray]): group_ids for grouped partitions. Grouped partitions are
            split such that no group id is present in both a training and validation split.
        stratified (Optional[bool]): If True, maintain dataset class :term:`balance<Balance>` within each train/val
            split

    Returns:
        split_defs (list[dict]): list of dictionaries, which specifying train index, validation index,
            and the ratio of validation to all data.  
    """
    split_defs = []
    index = index.reshape([-1,1])
    if groups is not None:
        splitter = StratifiedGroupKFold(n_folds) if stratified else GroupKFold(n_folds)
        splits = splitter.split(index, labels, groups)
    else:
        splitter = StratifiedKFold(n_folds) if stratified else KFold(n_folds)
        splits = splitter.split(index, labels)
    for train_idx, eval_idx in splits:
        test_ratio = len(eval_idx) / index.shape[0]
        split_defs.append(
            {
                "train": train_idx.astype(int), 
                "eval": eval_idx.astype(int), 
                "eval_frac": test_ratio
            }
    )
    return split_defs


def single_split(
        index: np.ndarray,
        labels: np.ndarray, 
        eval_frac: float,
        groups: Optional[np.ndarray]=None, 
        stratified: Optional[bool]=False
    ):
    """Handles the special case where only 1 partition of the data is desired (such as when 
    generating the test holdout split). In this case, the desired fraction of the data to be 
    partitioned into the test data must be specified, and a single [train, eval] pair are returned.

    Args:
        index (np.ndarray): Input Dataset index corresponding to each label
        labels (np.ndarray): Labels upon which splits are (optionally) stratified 
        eval_frac (float): Fraction of incoming data to be set aside for evaluation
        groups (Optional[np.ndarray]): Group_ids (same shape as labels) for optional group partitioning
        stratified (Optional[bool]): Generates stratified splits if true (recommended)

    Returns:
        train_index: indices of data partitioned for training
        eval_index: indices of data partitioned for evaluation
    """
    if eval_frac <= 2/3:
        n_folds = int(round(1/eval_frac))
        split_candidates = make_splits(index, labels, n_folds, groups, stratified)
        best_split = min(split_candidates, key = lambda x: abs(eval_frac-x["eval_frac"]))
        return best_split["train"], best_split["eval"]
    else:
        n_folds = int(round(1/(1-eval_frac)))
        split_candidates = make_splits(index, labels, n_folds, groups, stratified)
        best_split = min(split_candidates, key = lambda x: abs(eval_frac-(1-x["eval_frac"])))
        return best_split["eval"], best_split["train"]


def split_dataset(
        labels: Union[list, np.ndarray], 
        num_folds: Optional[int]=None, 
        test_frac: Optional[float]=None,
        val_frac: Optional[float]=None,
        split_on: Optional[list]=None, 
        metadata: Optional[dict]=None,
        stratified: Optional[bool]=None
        ):
    """Top level splitting function. Returns a dict with each key-value pair containing 
    train and validation indices. Indices for a test holdout may also be optionally included 

    Args:
        labels (Union[list, np.ndarray]): classification labels used to generate splits. 
            Determines the size of the dataset 

        num_folds (Optional[int], optional): Number of train/val folds. If None, returns a 
            single train/val split, and val_frac must be specified. Defaults to None.

        test_frac (Optional[float], optional): Fraction of data to be optionally held out for
            test set. Defaults to None, in which no test set is created. 

        val_frac (Optional[float], optional): Fraction of training data to be set aside for 
            validation in the case where a single train/val split is desired. Defaults to None.

        split_on (Optional[list], optional): Keys of the metadata dictionary which map to columns upon
            which to group the dataset. A grouped partition is divided such that no group is present
            within both the training and validation set. Split_on groups should be selected to
            mitigate validation :term:`bian<Bias>`. Defaults to None, in which groups will not be considered when
            partitioning the data.

        metadata (Optional[dict], optional): metadict containing data for potential dataset grouping.
            See split_on above. Defaults to None.

        stratified (Optional[bool], optional): If true, dataset is split such that the class 
            distribution of the entire dataset is preserved within each train/val partition, which
            is generally recommended. Defaults to None.

    Returns:
       split_defs (dict): dictionary of folds, each containing indices of training and validation
       data.

       ex. 
       {
        "Fold_00": {"train": [1,2,3,5,6,7,9,10,11], "val": [0, 4, 8, 12]},
        "test": [13, 14, 15, 16]
       } 
    """
    if (not num_folds) or (num_folds==1):
        if val_frac is None:
            raise UnboundLocalError("If not specifying num_folds, must assign a value to val_frac")
        num_folds=1
    else:
        if val_frac is not None:
            raise ValueError("If specifying val_frac, num_folds must be None or 1")
    total_partitions = num_folds+1 if test_frac else num_folds
    index, labels = check_labels(labels, total_partitions)
    if split_on:
        if metadata is None:
            raise UnboundLocalError("If split_on is specified, metadata must also be provided")
        groups = get_group_ids(metadata, split_on, len(labels))
        groups = check_groups(groups, total_partitions)
    else:
        groups=None
    split_defs = dict()
    if test_frac:
        tv_idx, test_idx = single_split(index, labels, test_frac, groups, stratified)
        tv_labels = labels[tv_idx]
        tv_groups = groups[tv_idx] if groups is not None else None
        split_defs["test"] = test_idx
    else:
        tv_idx = np.arange(len(labels)).reshape((-1,1))
        tv_labels = labels
        tv_groups = groups
    if val_frac:
        train_idx, val_idx = single_split(tv_idx, tv_labels, val_frac, tv_groups, stratified)
        split_defs["fold_0"] = {"train": tv_idx[train_idx], "val": tv_idx[val_idx]}
    else:
        tv_splits = make_splits(tv_idx, tv_labels, num_folds, tv_groups, stratified)
        for i,split in enumerate(tv_splits):
            train_split = tv_idx[split["train"]]
            val_split = tv_idx[split["eval"]]
            split_defs[f"fold_{i}"] = {"train": train_split, "val": val_split}
    return split_defs





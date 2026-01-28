"""
Utility functions for dataset splitting and data and metadata manipulation.
"""

__all__ = ["split_dataset", "unzip_dataset", "TrainValSplit", "DatasetSplits", "flatten_metadata", "merge_metadata"]

import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target

from dataeval.config import EPSILON
from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    Dataset,
    Metadata,
    ObjectDetectionTarget,
)
from dataeval.utils.arrays import as_numpy
from dataeval.utils.preprocessing import BoundingBox

_logger = logging.getLogger(__name__)


_TYPE_MAP = {int: 0, float: 1, str: 2}


class DropReason(Enum):
    INCONSISTENT_KEY = "inconsistent_key"
    INCONSISTENT_SIZE = "inconsistent_size"
    NESTED_LIST = "nested_list"


@overload
def _simplify_type(data: list[str]) -> list[int] | list[float] | list[str]: ...
@overload
def _simplify_type(data: str) -> int | float | str: ...


def _simplify_type(data: list[str] | str) -> list[int] | list[float] | list[str] | int | float | str:
    """
    Simplifies a value or a list of values to the simplest form possible,
    in preferred order of `int`, `float`, or `string`.

    Parameters
    ----------
    data : list[str] | str
        A list of values or a single value

    Returns
    -------
    list[int | float | str] | int | float | str
        The same values converted to the numerical type if possible
    """
    if not isinstance(data, list):
        try:
            value = float(data)
        except (TypeError, ValueError):
            value = None
        return str(data) if value is None else int(value) if value.is_integer() else value

    converted = []
    max_type = 0
    for value in data:
        value = _simplify_type(value)
        max_type = max(max_type, _TYPE_MAP.get(type(value), 2))
        converted.append(value)
    for i in range(len(converted)):
        converted[i] = list(_TYPE_MAP)[max_type](converted[i])
    return converted


def _get_key_indices(keys: Iterable[tuple[str, ...]]) -> dict[tuple[str, ...], int]:
    """
    Finds indices to minimize unique tuple keys

    Parameters
    ----------
    keys : Iterable[tuple[str, ...]]
        Collection of unique expanded tuple keys

    Returns
    -------
    dict[tuple[str, ...], int]
        Mapping of tuple keys to starting index
    """
    indices = dict.fromkeys(keys, -1)
    ks = list(keys)
    while len(ks) > 0:
        seen: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        for k in ks:
            seen.setdefault(k[indices[k] :], []).append(k)
        ks.clear()
        for sk in seen.values():
            if len(sk) > 1:
                ks.extend(sk)
                for k in sk:
                    indices[k] -= 1
    return indices


def _sorted_drop_reasons(d: dict[str, set[DropReason]]) -> dict[str, list[str]]:
    return {k: sorted({vv.value for vv in v}) for k, v in sorted(d.items(), key=lambda item: item[1])}


def _flatten_dict_inner(
    d: Mapping[str, Any],
    dropped: dict[tuple[str, ...], set[DropReason]],
    parent_keys: tuple[str, ...],
    size: int | None = None,
    nested: bool = False,
) -> tuple[dict[tuple[str, ...], Any], int | None]:
    """
    Recursive internal function for flattening a dictionary.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary to flatten
    dropped: set[tuple[str, ...]]
        Reference to set of dropped keys from the dictionary
    parent_keys : tuple[str, ...]
        Parent keys to the current dictionary being flattened
    size : int or None, default None
        Tracking int for length of lists
    nested : bool, default False
        Tracking if inside a list

    Returns
    -------
    tuple[dict[tuple[str, ...], Any], int | None]
        - [0]: Dictionary of flattened values with the keys reformatted as a
               hierarchical tuple of strings
        - [1]: Size, if any, of the current list of values
    """
    items: dict[tuple[str, ...], Any] = {}
    for k, v in d.items():
        new_keys: tuple[str, ...] = parent_keys + (k,)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, dict):
            fd, size = _flatten_dict_inner(v, dropped, new_keys, size=size, nested=nested)
            items.update(fd)
        elif isinstance(v, list | tuple):
            if nested:
                dropped.setdefault(parent_keys + (k,), set()).add(DropReason.NESTED_LIST)
            elif size is not None and size != len(v):
                dropped.setdefault(parent_keys + (k,), set()).add(DropReason.INCONSISTENT_SIZE)
            else:
                size = len(v)
                if all(isinstance(i, dict) for i in v):
                    for sub_dict in v:
                        fd, size = _flatten_dict_inner(sub_dict, dropped, new_keys, size=size, nested=True)
                        for fk, fv in fd.items():
                            items.setdefault(fk, []).append(fv)
                else:
                    items[new_keys] = v
        else:
            items[new_keys] = v
    return items, size


@overload
def flatten_metadata(
    d: Mapping[str, Any],
    return_dropped: Literal[True],
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
) -> tuple[dict[str, Any], int, dict[str, list[str]]]: ...


@overload
def flatten_metadata(
    d: Mapping[str, Any],
    return_dropped: Literal[False] = False,
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
) -> tuple[dict[str, Any], int]: ...


def flatten_metadata(
    d: Mapping[str, Any],
    return_dropped: bool = False,
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
):
    """
    Flattens a nested metadata dictionary and converts values to numeric values when possible.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary to flatten
    return_dropped: bool, default False
        Option to return a dictionary of dropped keys and the reason(s) for dropping
    sep : str, default "_"
        String separator to use when concatenating key names
    ignore_lists : bool, default False
        Option to skip expanding lists within metadata
    fully_qualified : bool, default False
        Option to return dictionary keys fully qualified instead of reduced

    Returns
    -------
    dict[str, Any]
        Dictionary of flattened values with the keys reformatted as strings
    int
        Size of the values in the flattened dictionary
    dict[str, list[str]], Optional
        Dictionary containing dropped keys and reason(s) for dropping
    """
    dropped_inner: dict[tuple[str, ...], set[DropReason]] = {}
    expanded, size = _flatten_dict_inner(d, dropped=dropped_inner, parent_keys=(), nested=ignore_lists)

    output = {}
    for k, v in expanded.items():
        cv = _simplify_type(v)
        if isinstance(cv, list):
            if len(cv) == size:
                output[k] = cv
            else:
                dropped_inner.setdefault(k, set()).add(DropReason.INCONSISTENT_KEY)
        else:
            output[k] = cv if not size else [cv] * size

    if fully_qualified:
        output = {sep.join(k): v for k, v in output.items()}
    else:
        keys = _get_key_indices(output)
        output = {sep.join(k[keys[k] :]): v for k, v in output.items()}

    size = size if size is not None else 1
    dropped = {sep.join(k): v for k, v in dropped_inner.items()}

    if return_dropped:
        return output, size, _sorted_drop_reasons(dropped)
    if dropped:
        dropped_items = "\n".join([f"    {k}: {v}" for k, v in _sorted_drop_reasons(dropped).items()])
        _logger.warning(f"Metadata entries were dropped:\n{dropped_items}")
    return output, size


def _flatten_for_merge(
    metadatum: Mapping[str, Any],
    ignore_lists: bool,
    fully_qualified: bool,
    targets: int | None,
) -> tuple[dict[str, list[Any]] | dict[str, Any], int, dict[str, list[str]]]:
    flattened, image_repeats, dropped_inner = flatten_metadata(
        metadatum, return_dropped=True, ignore_lists=ignore_lists, fully_qualified=fully_qualified
    )
    if targets is not None:
        # check for mismatch in targets per image and force ignore_lists
        if not ignore_lists and targets != image_repeats:
            flattened, image_repeats, dropped_inner = flatten_metadata(
                metadatum, return_dropped=True, ignore_lists=True, fully_qualified=fully_qualified
            )
        if targets != image_repeats:
            flattened = {k: [v] * targets for k, v in flattened.items()}
        image_repeats = targets
    return flattened, image_repeats, dropped_inner


def _merge(
    dicts: list[Mapping[str, Any]],
    ignore_lists: bool,
    fully_qualified: bool,
    targets_per_image: Sequence[int] | None,
) -> tuple[dict[str, list[Any]], dict[str, set[DropReason]], NDArray[np.intp]]:
    merged: dict[str, list[Any]] = {}
    isect: set[str] = set()
    union: set[str] = set()
    image_repeats = np.zeros(len(dicts), dtype=np.intp)
    dropped: dict[str, set[DropReason]] = {}
    for i, d in enumerate(dicts):
        targets = None if targets_per_image is None else targets_per_image[i]
        if targets == 0:
            continue
        flattened, image_repeats[i], dropped_inner = _flatten_for_merge(d, ignore_lists, fully_qualified, targets)
        isect = isect.intersection(flattened.keys()) if isect else set(flattened.keys())
        union.update(flattened.keys())
        for k, v in dropped_inner.items():
            dropped.setdefault(k, set()).update({DropReason(vv) for vv in v})
        for k, v in flattened.items():
            merged.setdefault(k, []).extend(flattened[k]) if isinstance(v, list) else merged.setdefault(k, []).append(v)

    for k in union - isect:
        dropped.setdefault(k, set()).add(DropReason.INCONSISTENT_KEY)

    if image_repeats.sum() == image_repeats.size:
        image_indices = np.arange(image_repeats.size)
    else:
        image_ids = np.arange(image_repeats.size)
        image_data = np.concatenate(
            [np.repeat(image_ids[i], image_repeats[i]) for i in range(image_ids.size)], dtype=np.intp
        )
        _, image_unsorted = np.unique(image_data, return_inverse=True)
        image_indices = np.sort(image_unsorted)

    merged = {k: _simplify_type(v) for k, v in merged.items() if k in isect}
    return merged, dropped, image_indices


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[True],
    return_numpy: Literal[False] = False,
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> tuple[dict[str, list[Any]], dict[str, list[str]]]: ...


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[False] = False,
    return_numpy: Literal[False] = False,
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> dict[str, list[Any]]: ...


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[True],
    return_numpy: Literal[True],
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> tuple[dict[str, NDArray[Any]], dict[str, list[str]]]: ...


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[False] = False,
    return_numpy: Literal[True],
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> dict[str, NDArray[Any]]: ...


def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: bool = False,
    return_numpy: bool = False,
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
):
    """
    Merges a collection of metadata dictionaries into a single flattened
    dictionary of keys and values.

    Nested dictionaries are flattened, and lists are expanded. Nested lists are
    dropped as the expanding into multiple hierarchical trees is not supported.
    The function adds an internal "_image_index" key to the metadata dictionary
    used by the `Metadata` class.

    Parameters
    ----------
    metadata : Iterable[Mapping[str, Any]]
        Iterable collection of metadata dictionaries to flatten and merge
    return_dropped: bool, default False
        Option to return a dictionary of dropped keys and the reason(s) for dropping
    return_numpy : bool, default False
        Option to return results as lists or NumPy arrays
    ignore_lists : bool, default False
        Option to skip expanding lists within metadata
    fully_qualified : bool, default False
        Option to return dictionary keys full qualified instead of minimized
    targets_per_image : Sequence[int] or None, default None
        Number of targets for each image metadata entry
    image_index_key : str, default "_image_index"
        User provided metadata key which maps the metadata entry to the source image.

    Returns
    -------
    dict[str, list[Any]] | dict[str, NDArray[Any]]
        A single dictionary containing the flattened data as lists or NumPy arrays
    dict[str, list[str]], Optional
        Dictionary containing dropped keys and reason(s) for dropping

    Notes
    -----
    Nested lists of values and inconsistent keys are dropped in the merged
    metadata dictionary

    Example
    -------
    >>> list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4}], "source": "example"}]
    >>> reorganized_metadata, dropped_keys = merge_metadata(list_metadata, return_dropped=True)
    >>> reorganized_metadata
    {'common': [1, 1], 'a': [1, 2], 'b': [3, 4], 'source': ['example', 'example'], '_image_index': [0, 0]}
    >>> dropped_keys
    {'target_c': ['inconsistent_key']}
    """

    dicts: list[Mapping[str, Any]] = list(metadata)

    if targets_per_image is not None and len(dicts) != len(targets_per_image):
        raise ValueError("Number of targets per image must be equal to number of metadata entries.")

    merged, dropped, image_indices = _merge(dicts, ignore_lists, fully_qualified, targets_per_image)

    output: dict[str, Any] = {k: np.asarray(v) for k, v in merged.items()} if return_numpy else merged

    if image_index_key not in output:
        output[image_index_key] = image_indices if return_numpy else image_indices.tolist()

    if return_dropped:
        return output, _sorted_drop_reasons(dropped)

    if dropped:
        dropped_items = "\n".join([f"    {k}: {v}" for k, v in _sorted_drop_reasons(dropped).items()])
        _logger.warning(f"Metadata entries were dropped:\n{dropped_items}")

    return output


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
    metadata : Metadata
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
        _logger.warning("Unable to create a good split definition, not all classes are represented in each split.")
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


def split_dataset(
    dataset: AnnotatedDataset[Any] | Metadata,
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
    split_defs : DatasetSplits
        Output class containing a list of indices of training
        and validation data for each fold and optional test indices

    Notes
    -----
    When specifying groups and/or stratification, ratios for test and validation splits can vary
    as the stratification and grouping take higher priority than the percentages
    """

    val_frac = calculate_validation_fraction(num_folds, test_frac, val_frac)
    total_partitions = num_folds + 1 if test_frac else num_folds

    # Import Metadata at runtime to avoid circular import
    from dataeval._metadata import Metadata as _Metadata

    metadata = dataset if isinstance(dataset, Metadata) else _Metadata(dataset)
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

    return DatasetSplits(tvs.val, folds)


class SizedIterator:
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
    dataset: Dataset[Any] | Dataset[tuple[Any, Any, Any]], per_target: bool
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
            image = np.asarray(d[0] if isinstance(d, tuple) else d)
            if per_target and isinstance(d, tuple) and isinstance(d[1], ObjectDetectionTarget):
                try:
                    boxes = d[1].boxes if isinstance(d[1].boxes, Array) else as_numpy(d[1].boxes)
                    target = [BoundingBox(box[0], box[1], box[2], box[3], image_shape=image.shape) for box in boxes]
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid bounding box format for image {i}: {d[1].boxes}")
            else:
                target = None
            yield image, target

    # Create two independent iterators from the generator
    iter1, iter2 = tee(_generate_pairs(), 2)

    # Extract images and targets separately
    images_iter = SizedIterator((pair[0] for pair in iter1), len(dataset))
    targets_iter = (pair[1] for pair in iter2) if per_target else None

    return images_iter, targets_iter

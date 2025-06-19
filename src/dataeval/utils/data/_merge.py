"""
Utility functions that help organize raw metadata.
"""

from __future__ import annotations

__all__ = ["merge", "flatten"]

import warnings
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

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
def flatten(
    d: Mapping[str, Any],
    return_dropped: Literal[True],
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
) -> tuple[dict[str, Any], int, dict[str, list[str]]]: ...


@overload
def flatten(
    d: Mapping[str, Any],
    return_dropped: Literal[False] = False,
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
) -> tuple[dict[str, Any], int]: ...


def flatten(
    d: Mapping[str, Any],
    return_dropped: bool = False,
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
):
    """
    Flattens a dictionary and converts values to numeric values when possible.

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
        Dictionary of flattened values with the keys reformatted as a hierarchical tuple of strings
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
        warnings.warn(f"Metadata entries were dropped:\n{dropped_items}")
    return output, size


def _flatten_for_merge(
    metadatum: Mapping[str, Any],
    ignore_lists: bool,
    fully_qualified: bool,
    targets: int | None,
) -> tuple[dict[str, list[Any]] | dict[str, Any], int, dict[str, list[str]]]:
    flattened, image_repeats, dropped_inner = flatten(
        metadatum, return_dropped=True, ignore_lists=ignore_lists, fully_qualified=fully_qualified
    )
    if targets is not None:
        # check for mismatch in targets per image and force ignore_lists
        if not ignore_lists and targets != image_repeats:
            flattened, image_repeats, dropped_inner = flatten(
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
    image_repeats = np.zeros(len(dicts), dtype=np.int_)
    dropped: dict[str, set[DropReason]] = {}
    for i, d in enumerate(dicts):
        targets = None if targets_per_image is None else targets_per_image[i]
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
            [np.repeat(image_ids[i], image_repeats[i]) for i in range(image_ids.size)], dtype=np.int_
        )
        _, image_unsorted = np.unique(image_data, return_inverse=True)
        image_indices = np.sort(image_unsorted)

    merged = {k: _simplify_type(v) for k, v in merged.items() if k in isect}
    return merged, dropped, image_indices


@overload
def merge(
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
def merge(
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
def merge(
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
def merge(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[False] = False,
    return_numpy: Literal[True],
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    image_index_key: str = "_image_index",
) -> dict[str, NDArray[Any]]: ...


def merge(
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

    Note
    ----
    Nested lists of values and inconsistent keys are dropped in the merged
    metadata dictionary

    Example
    -------
    >>> list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4}], "source": "example"}]
    >>> reorganized_metadata, dropped_keys = merge(list_metadata, return_dropped=True)
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
        warnings.warn(f"Metadata entries were dropped:\n{dropped_items}")

    return output

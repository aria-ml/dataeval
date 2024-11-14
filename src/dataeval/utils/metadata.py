from __future__ import annotations

__all__ = ["merge_metadata"]

import warnings
from typing import Any, Iterable, Mapping, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


def _try_cast(v: Any, t: type[T]) -> T | None:
    """Casts a value to a type or returns None if unable"""
    try:
        return t(v)  # type: ignore
    except (TypeError, ValueError):
        return None


@overload
def _convert_type(data: list[str]) -> list[int] | list[float] | list[str]: ...
@overload
def _convert_type(data: str) -> int | float | str: ...


def _convert_type(data: list[str] | str) -> list[int] | list[float] | list[str] | int | float | str:
    """
    Converts a value or a list of values to the simplest form possible, in preferred order of `int`,
    `float`, or `string`.

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
        value = _try_cast(data, float)
        return str(data) if value is None else int(value) if value.is_integer() else value

    converted = []
    TYPE_MAP = {int: 0, float: 1, str: 2}
    max_type = 0
    for value in data:
        value = _convert_type(value)
        max_type = max(max_type, TYPE_MAP.get(type(value), 2))
        converted.append(value)
    for i in range(len(converted)):
        converted[i] = list(TYPE_MAP)[max_type](converted[i])
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
    indices = {k: -1 for k in keys}
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


def _flatten_dict_inner(
    d: Mapping[str, Any], parent_keys: tuple[str, ...], size: int | None = None, nested: bool = False
) -> tuple[dict[tuple[str, ...], Any], int | None]:
    """
    Recursive internal function for flattening a dictionary.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary to flatten
    parent_keys : tuple[str, ...]
        Parent keys to the current dictionary being flattened
    size : int or None, default None
        Tracking int for length of lists
    nested : bool, default False
        Tracking if inside a list

    Returns
    -------
    tuple[dict[tuple[str, ...], Any], int | None]
        - [0]: Dictionary of flattened values with the keys reformatted as a hierarchical tuple of strings
        - [1]: Size, if any, of the current list of values
    """
    items: dict[tuple[str, ...], Any] = {}
    for k, v in d.items():
        new_keys: tuple[str, ...] = parent_keys + (k,)
        if isinstance(v, dict):
            fd, size = _flatten_dict_inner(v, new_keys, size=size, nested=nested)
            items.update(fd)
        elif isinstance(v, (list, tuple)):
            if not nested and (size is None or size == len(v)):
                size = len(v)
                if all(isinstance(i, dict) for i in v):
                    for sub_dict in v:
                        fd, size = _flatten_dict_inner(sub_dict, new_keys, size=size, nested=True)
                        for fk, fv in fd.items():
                            items.setdefault(fk, []).append(fv)
                else:
                    items[new_keys] = v
            else:
                warnings.warn(f"Dropping nested list found in '{parent_keys + (k, )}'.")
        else:
            items[new_keys] = v
    return items, size


def _flatten_dict(d: Mapping[str, Any], sep: str, ignore_lists: bool, fully_qualified: bool) -> dict[str, Any]:
    """
    Flattens a dictionary and converts values to numeric values when possible.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary to flatten
    sep : str
        String separator to use when concatenating key names
    ignore_lists : bool
        Option to skip expanding lists within metadata
    fully_qualified : bool
        Option to return dictionary keys full qualified instead of minimized

    Returns
    -------
    dict[str, Any]
        A flattened dictionary
    """
    expanded, size = _flatten_dict_inner(d, parent_keys=(), nested=ignore_lists)

    output = {}
    if fully_qualified:
        expanded = {sep.join(k): v for k, v in expanded.items()}
    else:
        keys = _get_key_indices(expanded)
        expanded = {sep.join(k[keys[k] :]): v for k, v in expanded.items()}
    for k, v in expanded.items():
        cv = _convert_type(v)
        if isinstance(cv, list) and len(cv) == size:
            output[k] = cv
        elif not isinstance(cv, list):
            output[k] = cv if not size else [cv] * size
    return output


def _is_metadata_dict_of_dicts(metadata: Mapping) -> bool:
    """EXPERIMENTAL: Attempt to detect if metadata is a dict of dicts"""
    # single dict
    if len(metadata) < 2:
        return False

    # dict of non dicts
    keys = list(metadata)
    if not isinstance(metadata[keys[0]], Mapping):
        return False

    # dict of dicts with matching keys
    return set(metadata[keys[0]]) == set(metadata[keys[1]])


def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    as_numpy: bool = False,
) -> dict[str, list[Any]] | dict[str, NDArray[Any]]:
    """
    Merges a collection of metadata dictionaries into a single flattened dictionary of keys and values.

    Nested dictionaries are flattened, and lists are expanded. Nested lists are dropped as the
    expanding into multiple hierarchical trees is not supported.

    Parameters
    ----------
    metadata : Iterable[Mapping[str, Any]]
        Iterable collection of metadata dictionaries to flatten and merge
    ignore_lists : bool, default False
        Option to skip expanding lists within metadata
    fully_qualified : bool, default False
        Option to return dictionary keys full qualified instead of minimized
    as_numpy : bool, default False
        Option to return results as lists or NumPy arrays

    Returns
    -------
    dict[str, list[Any]] | dict[str, NDArray[Any]]
        A single dictionary containing the flattened data as lists or NumPy arrays

    Note
    ----
    Nested lists of values and inconsistent keys are dropped in the merged metadata dictionary

    Example
    -------
    >>> list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3}, {"a": 2, "b": 4}], "source": "example"}]
    >>> merge_metadata(list_metadata)
    {'common': [1, 1], 'a': [1, 2], 'b': [3, 4], 'source': ['example', 'example']}
    """
    merged: dict[str, list[Any]] = {}
    isect: set[str] = set()
    union: set[str] = set()
    keys: list[str] | None = None
    dicts: list[Mapping[str, Any]]

    # EXPERIMENTAL
    if isinstance(metadata, Mapping) and _is_metadata_dict_of_dicts(metadata):
        warnings.warn("Experimental processing for dict of dicts.")
        keys = [str(k) for k in metadata]
        dicts = list(metadata.values())
        ignore_lists = True
    else:
        dicts = list(metadata)

    for d in dicts:
        flattened = _flatten_dict(d, sep="_", ignore_lists=ignore_lists, fully_qualified=fully_qualified)
        isect = isect.intersection(flattened.keys()) if isect else set(flattened.keys())
        union = union.union(flattened.keys())
        for k, v in flattened.items():
            merged.setdefault(k, []).extend(flattened[k]) if isinstance(v, list) else merged.setdefault(k, []).append(v)

    if len(union) > len(isect):
        warnings.warn(f"Inconsistent metadata keys found. Dropping {union - isect} from metadata.")

    output: dict[str, Any] = {}

    if keys:
        output["keys"] = np.array(keys) if as_numpy else keys

    for k in (key for key in merged if key in isect):
        cv = _convert_type(merged[k])
        output[k] = np.array(cv) if as_numpy else cv

    return output

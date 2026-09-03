"""Metadata flattening and merging utilities."""

__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.utils._internal import promotion_is_lossy, simplify_type

_logger = get_logger(__name__)


def _get_key_indices(keys: Iterable[tuple[str, ...]]) -> dict[tuple[str, ...], int]:  # noqa: C901
    """
    Find indices to minimize unique tuple keys.

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


class DropReason(Enum):
    INCONSISTENT_KEY = "inconsistent_key"
    INCONSISTENT_SIZE = "inconsistent_size"
    MIXED_TYPES = "mixed_types"
    NESTED_LIST = "nested_list"


def sorted_drop_reasons(d: dict[str, set[DropReason]]) -> dict[str, list[str]]:
    return {k: sorted({vv.value for vv in v}) for k, v in sorted(d.items(), key=lambda item: item[1])}


def flatten_dict_inner(  # noqa: C901
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
            fd, size = flatten_dict_inner(v, dropped, new_keys, size=size, nested=nested)
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
                        fd, size = flatten_dict_inner(sub_dict, dropped, new_keys, size=size, nested=True)
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
    simplify: bool = True,
) -> tuple[dict[str, Any], int, dict[str, list[str]]]: ...


@overload
def flatten_metadata(
    d: Mapping[str, Any],
    return_dropped: Literal[False] = False,
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    simplify: bool = True,
) -> tuple[dict[str, Any], int]: ...


def flatten_metadata(  # noqa: C901
    d: Mapping[str, Any],
    return_dropped: bool = False,
    sep: str = "_",
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    simplify: bool = True,
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
    expanded, size = flatten_dict_inner(d, dropped=dropped_inner, parent_keys=(), nested=ignore_lists)

    output = {}
    for k, v in expanded.items():
        # ``simplify=False`` leaves the values as the dataset wrote them. `_merge` asks for
        # that because converting a numeral here is irreversible: once ``"1"`` has become
        # ``1`` nothing downstream can tell it from a value that arrived as a number, and
        # telling those apart is exactly what deciding whether a column mixes types needs.
        # It simplifies the whole column itself once every entry has contributed to it.
        cv = simplify_type(v) if simplify else v
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
        return output, size, sorted_drop_reasons(dropped)
    if dropped:
        dropped_items = "\n".join([f"    {k}: {v}" for k, v in sorted_drop_reasons(dropped).items()])
        _logger.warning(f"Metadata entries were dropped:\n{dropped_items}")
    return output, size


def _flatten_for_merge(
    metadatum: Mapping[str, Any],
    ignore_lists: bool,
    fully_qualified: bool,
    targets: int | None,
) -> tuple[dict[str, list[Any]] | dict[str, Any], int, dict[str, list[str]]]:
    flattened, image_repeats, dropped_inner = flatten_metadata(
        metadatum,
        return_dropped=True,
        ignore_lists=ignore_lists,
        fully_qualified=fully_qualified,
        simplify=False,
    )
    if targets is not None:
        # check for mismatch in targets per image and force ignore_lists
        if not ignore_lists and targets != image_repeats:
            flattened, image_repeats, dropped_inner = flatten_metadata(
                metadatum,
                return_dropped=True,
                ignore_lists=True,
                fully_qualified=fully_qualified,
                simplify=False,
            )
        if targets != image_repeats:
            flattened = {k: [v] * targets for k, v in flattened.items()}
        image_repeats = targets
    return flattened, image_repeats, dropped_inner


def _was_dropped(column: str, dropped: Mapping[str, set[DropReason]], sep: str = "_") -> bool:
    """Whether a merged column's key names a path some entry dropped.

    ``dropped`` is keyed by the full path a value was reached by and ``merged`` by the
    shortened name given to the column, so the two only meet at the tail: a column named
    ``y`` is the one dropped as ``objs_y``. Matching the whole trailing segment rather than
    a bare substring keeps ``y`` from answering for ``entropy``.
    """
    return bool(dropped.get(column)) or any(
        reasons and name.endswith(f"{sep}{column}") for name, reasons in dropped.items()
    )


def _simplify_present(values: list[Any]) -> list[Any]:
    """Simplify the values an entry recorded, leaving the ones it did not as missing.

    :func:`simplify_type` reads every element as a string, so a missing value reaches it as
    ``None`` and comes back as the *string* ``"None"`` -- which then makes the whole column
    a string column, since one string forces the widest type. Both are wrong for a column
    that is simply incomplete: the absence stops being an absence, and the numbers that were
    recorded stop being numbers.

    Only the partial path routes through here, so a column with no missing values reaches
    :func:`simplify_type` exactly as it always has.
    """
    if not any(value is None for value in values):
        return simplify_type(values)
    present = simplify_type([value for value in values if value is not None])
    filled = iter(present)
    return [None if value is None else next(filled) for value in values]


def _merge(  # noqa: C901
    dicts: list[Mapping[str, Any]],
    ignore_lists: bool,
    fully_qualified: bool,
    targets_per_image: Sequence[int] | None,
    keep_partial: bool = False,
) -> tuple[dict[str, list[Any]], dict[str, set[DropReason]], dict[str, list[Any]]]:
    merged: dict[str, list[Any]] = {}
    isect: set[str] = set()
    union: set[str] = set()
    dropped: dict[str, set[DropReason]] = {}
    rows = 0
    for i, d in enumerate(dicts):
        targets = None if targets_per_image is None else targets_per_image[i]
        if targets == 0:
            continue
        flattened, size, dropped_inner = _flatten_for_merge(d, ignore_lists, fully_qualified, targets)
        if not size:
            # An entry that flattens to no rows contributes no values, the same way one
            # holding no targets is skipped above. Its scalars have nothing to attach to:
            # appending them anyway advanced a column past `rows`, leaving the merged
            # columns describing different row counts.
            for k, v in dropped_inner.items():
                dropped.setdefault(k, set()).update({DropReason(vv) for vv in v})
            continue
        isect = isect.intersection(flattened.keys()) if isect else set(flattened.keys())
        union.update(flattened.keys())
        for k, v in dropped_inner.items():
            dropped.setdefault(k, set()).update({DropReason(vv) for vv in v})
        for k, v in flattened.items():
            column = merged.setdefault(k, [])
            if keep_partial and len(column) < rows:
                # First sight of a key some earlier entry did not carry: it owes one
                # missing value for every row already merged, or it would line up against
                # the wrong entries from here on.
                column.extend([None] * (rows - len(column)))
            column.extend(v) if isinstance(v, list) else column.append(v)
        rows += int(size)
        if keep_partial:
            for column in merged.values():
                column.extend([None] * (rows - len(column)))

    if not keep_partial:
        for k in union - isect:
            dropped.setdefault(k, set()).add(DropReason.INCONSISTENT_KEY)

    # An entry that declared none of a key contributes missing values for it rather than
    # costing every other entry the key -- but only for a key whose *only* problem is that
    # absence. One dropped for a reason of its own, a nested list among them, stays dropped:
    # keeping it would resurrect values that were never usable.
    # Under `keep_partial` a key absent from some entries is never marked at all -- the
    # `union - isect` pass above is skipped -- so every reason still standing here came from
    # `flatten_metadata`, which means the key was unusable *within* one entry. Keeping it
    # rebuilt a column the entry could not fill, and the targets that did record a value
    # lost it to the padding rather than keeping it.
    #
    # Matched by suffix because the two dicts are keyed in different namespaces: `merged`
    # carries the shortened name `flatten_metadata` chose for the column, `dropped` always
    # carries the full path it was reached by. On the all-or-nothing path the `union - isect`
    # pass hides the mismatch by re-dropping the short name; here nothing does, so `y` went
    # looking for itself under `objs_y` and did not find it.
    kept = {k for k in merged if not _was_dropped(k, dropped)} if keep_partial else isect
    # Checked here rather than per entry: one entry usually carries a single scalar for a
    # key, so the disagreement only becomes visible once every entry's value sits in one
    # column -- which is also the moment `simplify_type` would resolve it by promoting the
    # numbers to text.
    #
    # Set aside rather than discarded, and left exactly as the dataset wrote them. Python
    # holds a column of mixed values perfectly well; it is polars that needs one type, and
    # a column nobody has said how to read has no business being in the factor store yet.
    # Keeping the values is what lets a repair be applied to them later without re-reading
    # the dataset, and what lets the counts and distinct values be reported meanwhile.
    for k in sorted(kept):
        if promotion_is_lossy(merged[k]):
            dropped.setdefault(k, set()).add(DropReason.MIXED_TYPES)
    unusable = {k: list(merged[k]) for k in sorted(kept) if DropReason.MIXED_TYPES in dropped.get(k, set())}
    kept = {k for k in kept if k not in unusable}
    simplify = _simplify_present if keep_partial else simplify_type
    merged = {k: simplify(v) for k, v in merged.items() if k in kept}
    return merged, dropped, unusable


@overload
def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: Literal[True],
    return_numpy: Literal[False] = False,
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    keep_partial: bool = False,
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
    keep_partial: bool = False,
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
    keep_partial: bool = False,
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
    keep_partial: bool = False,
) -> dict[str, NDArray[Any]]: ...


def merge_metadata(
    metadata: Iterable[Mapping[str, Any]],
    *,
    return_dropped: bool = False,
    return_numpy: bool = False,
    ignore_lists: bool = False,
    fully_qualified: bool = False,
    targets_per_image: Sequence[int] | None = None,
    keep_partial: bool = False,
):
    """
    Merge a collection of metadata dictionaries into a single flattened dictionary.

    Nested dictionaries are flattened, and lists are expanded. Nested lists are
    dropped as the expanding into multiple hierarchical trees is not supported.

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
    keep_partial : bool, default False
        Keep a key only some entries declare, giving the entries that did not declare it a
        missing value. By default such a key is dropped for every entry, which is the
        conservative reading -- a factor present for half a dataset can mislead an analysis
        that does not know it is half absent -- but it discards the half that was recorded,
        which for a large dataset with one incomplete entry is the whole factor.

        This covers a key some *entries* do not declare. A key inconsistent among the
        targets *within* one entry is still dropped either way: by the time the mismatch is
        seen, the values have been flattened into a short list, and which targets it came
        from is no longer recoverable.

    Returns
    -------
    dict[str, list[Any]] | dict[str, NDArray[Any]]
        A single dictionary containing the flattened data as lists or NumPy arrays
    dict[str, list[str]], Optional
        Dictionary containing dropped keys and reason(s) for dropping

    Notes
    -----
    Nested lists of values and inconsistent keys are dropped in the merged
    metadata dictionary, unless ``keep_partial`` asks for the inconsistent ones to be
    kept with missing values. A nested list is dropped either way: it has no usable
    values to keep.

    Example
    -------
    >>> list_metadata = [{"common": 1, "target": [{"a": 1, "b": 3, "c": 5}, {"a": 2, "b": 4}], "source": "example"}]
    >>> reorganized_metadata, dropped_keys = merge_metadata(list_metadata, return_dropped=True)
    >>> reorganized_metadata
    {'common': [1, 1], 'a': [1, 2], 'b': [3, 4], 'source': ['example', 'example']}
    >>> dropped_keys
    {'target_c': ['inconsistent_key']}
    """
    dicts: list[Mapping[str, Any]] = list(metadata)

    if targets_per_image is not None and len(dicts) != len(targets_per_image):
        raise ValueError("Number of targets per image must be equal to number of metadata entries.")

    merged, dropped, _ = _merge(dicts, ignore_lists, fully_qualified, targets_per_image, keep_partial)

    output: dict[str, Any] = {k: np.asarray(v) for k, v in merged.items()} if return_numpy else merged

    if return_dropped:
        return output, sorted_drop_reasons(dropped)

    if dropped:
        dropped_items = "\n".join([f"    {k}: {v}" for k, v in sorted_drop_reasons(dropped).items()])
        _logger.warning(f"Metadata entries were dropped:\n{dropped_items}")

    return output

"""Units, labels, replicas and splits for the feasibility evaluator: what counts as one independent item."""

__all__ = []

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._metadata import Metadata
from dataeval.data._view import View
from dataeval.exceptions import ShapeMismatchError


@dataclass(frozen=True)
class Units:
    """
    The labeled items of a dataset and the physical unit each belongs to.

    ``labels``, ``item_indices`` and ``groups`` are aligned (one entry per labeled item). ``groups`` holds
    contiguous codes ``0..G-1``. ``population`` is the string every later module quotes verbatim.
    ``class_names`` is set only when the dataset declares ``index2label`` (ordered by class index; the declared
    indices may be sparse, so use ``n_classes``, not its length, for the class count). ``n_classes`` is the largest
    declared index + 1, else the largest observed label + 1. ``item_ids`` holds one datum id per dataset item
    (position = dataset index); a dataset with no ids gets positional ints, which are then the replica keys.
    ``grouped`` is whether some group holds more than one item.
    """

    labels: NDArray[np.intp]
    item_indices: NDArray[np.intp]
    groups: NDArray[np.intp]
    n_classes: int
    item_ids: NDArray[Any]
    n_excluded: int
    population: str
    class_names: list[str] | None
    grouped: bool


def _dataset_id(dataset: Any) -> str:
    try:
        return str(dataset.metadata["id"])
    except (AttributeError, KeyError, TypeError):
        return type(dataset).__name__


def _declared_indices(dataset: Any) -> list[int]:
    try:
        declared = dataset.metadata.get("index2label")
    except AttributeError:
        return []
    return sorted(int(k) for k in declared) if declared else []


def _declared_class_names(dataset: Any) -> list[str] | None:
    indices = _declared_indices(dataset)
    return [str(dataset.metadata["index2label"][k]) for k in indices] if indices else None


def _group_values(dataset: Any, md: Metadata, groups: ArrayLike | str | None) -> NDArray[Any]:
    """Raw group value per labeled item: its own item index, a metadata factor's raw value, or the given id."""
    items = np.asarray(md.item_indices, dtype=np.intp)
    if groups is None:
        return items
    if isinstance(groups, str):
        names = list(md.factor_names)
        if groups not in names:
            raise ValueError(
                f"groups='{groups}' names a metadata factor, but the dataset has {names or 'none'}. "
                "Pass a factor name or an array of group ids."
            )
        return np.asarray(md.factor_values)[:, names.index(groups)]  # raw values, aligned with labeled items
    arr = np.asarray(groups).reshape(-1)
    if len(arr) != len(dataset):
        raise ShapeMismatchError(f"Got {len(arr)} group ids for a dataset of {len(dataset)} items.")
    return arr[items]


def resolve_units(dataset: Any, *, groups: ArrayLike | str | None) -> Units:
    """
    Resolve labeled items and physical units; a derived view needs ``groups=`` to say what is independent.

    A ``View`` straight over a plain dataset defaults to its source indices (``selection``), so a repeated item
    groups with its copies; any other derived view must pass ``groups=``.
    """
    if groups is None and isinstance(dataset, View) and getattr(dataset.source, "source", None) is None:
        groups = np.asarray(dataset.selection)
    if groups is None and getattr(dataset, "source", None) is not None:
        raise ValueError(
            f"{type(dataset).__name__} is a derived view of another dataset, so its items are not independent "
            "(crops, frames and segments share a source). Pass groups= naming the physical unit of each item, "
            "as an array aligned with the dataset or a metadata factor name."
        )
    md = Metadata(dataset)
    if md.multi_target:
        raise ValueError(
            f"Feasibility phase 1 supports classification datasets only, but {_dataset_id(dataset)} yields "
            f"{md.unit_type} metadata with one row per detection or track, not per item."
        )
    items = np.asarray(md.item_indices, dtype=np.intp)
    n_excluded = len(dataset) - len(items)
    kind = "labeled items" if n_excluded else "items"
    labels = np.asarray(md.class_labels, dtype=np.intp)
    class_names = _declared_class_names(dataset)
    declared = _declared_indices(dataset)
    n_classes = max(declared) + 1 if declared else (int(labels.max()) + 1 if len(labels) else 0)
    unit_rows = md.rows_at("unit").sort("item_index")
    codes = np.unique(_group_values(dataset, md, groups), return_inverse=True)[1].reshape(-1).astype(np.intp)
    return Units(
        labels=labels,
        item_indices=items,
        groups=codes,
        n_classes=n_classes,
        item_ids=np.asarray(unit_rows["item_id"].to_list()),
        n_excluded=n_excluded,
        population=f"{kind} of {_dataset_id(dataset)}",
        class_names=class_names,
        grouped=bool(len(codes)) and int(np.bincount(codes).max()) > 1,
    )


def _datum_ids(units: Units) -> dict[Any, int]:
    ids = units.item_ids.tolist()
    if len(set(ids)) != len(ids):
        raise ValueError(
            "replicas= is keyed by datum id, but this dataset repeats ids (as derived views do), "
            "so a key would not name one item. Use a dataset with unique datum ids."
        )
    return {datum_id: i for i, datum_id in enumerate(ids)}


def _replica_labels(key: Any, values: ArrayLike, n_classes: int) -> NDArray[np.intp]:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0 or not np.all(arr == np.floor(arr)):
        raise ValueError(f"replicas[{key!r}] must be a non-empty array of integer class labels.")
    if arr.min() < 0 or arr.max() >= n_classes:
        raise ValueError(f"replicas[{key!r}] has labels outside [0, {n_classes}).")
    return arr.astype(np.intp)


def resolve_replicas(
    replicas: Mapping[int | str, ArrayLike],
    dataset: Any,  # noqa: ARG001 - kept for the contract; ids come from ``units``, never a re-read
    units: Units,
) -> dict[int, NDArray[np.intp]]:
    """Map datum-id keyed extra labels to ``{labeled-item position: labels}``; ids must be unique in the dataset."""
    if not replicas:
        return {}
    index_of_id = _datum_ids(units)
    position_of_item = {int(item): pos for pos, item in enumerate(units.item_indices)}
    out: dict[int, NDArray[np.intp]] = {}
    for key, values in replicas.items():
        if key not in index_of_id:
            raise ValueError(f"replicas key {key!r} is not a datum id of the dataset.")
        if index_of_id[key] not in position_of_item:
            raise ValueError(f"replicas key {key!r} names an unlabeled item, which is excluded from the population.")
        out[position_of_item[index_of_id[key]]] = _replica_labels(key, values, units.n_classes)
    return out


def split_units(units: Units, *, fit_fraction: float, seed: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Split labeled-item positions into fit and audit, whole physical groups to one side or the other."""
    n_groups = int(units.groups.max()) + 1 if len(units.groups) else 0
    n_fit = int(round(fit_fraction * n_groups))
    if not 0 < n_fit < n_groups:
        raise ValueError(
            f"fit_fraction={fit_fraction} over {n_groups} physical units leaves the fit or the audit side empty."
        )
    fit_groups = np.random.default_rng(seed).permutation(n_groups)[:n_fit]
    in_fit = np.isin(units.groups, fit_groups)
    return np.flatnonzero(in_fit).astype(np.intp), np.flatnonzero(~in_fit).astype(np.intp)


def audit_order(units: Units, audit: NDArray[np.intp], seed: int) -> NDArray[np.intp]:
    """Seeded order of the audit items: groups are permuted and expanded, so a prefix never splits a group."""
    audit = np.asarray(audit, dtype=np.intp)
    audit_groups = units.groups[audit]
    order = np.random.default_rng(seed).permutation(np.unique(audit_groups))
    return np.concatenate([audit[audit_groups == g] for g in order]) if len(audit) else audit

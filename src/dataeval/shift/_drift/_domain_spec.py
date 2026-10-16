"""Domain specs for DriftKnownDomains: which values of one metadata factor make up each known domain."""

__all__ = []

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dataeval._helpers import reject_filtered_metadata
from dataeval._metadata._corrections import _remapped, _within
from dataeval.exceptions import ShapeMismatchError
from dataeval.types import Remap
from dataeval.types._factors import _validate_range

_CATCH_ALL_NAME = "other"
_SCALARS = (str, int, float, bool)
_UNMATCHED = object()  # the catch-all answer when the spec has no None group


def _plain(value: Any) -> Any:
    """Unwrap a NumPy scalar: matching compares types, and ``np.int64(1)`` is not ``int``."""
    return value.item() if isinstance(value, np.generic) else value


def _range_name(over: tuple[Any, Any]) -> str:
    low, high = over
    return f"[{'-inf' if low is None else low}, {'inf' if high is None else high})"


def _item_name(item: Any) -> str:
    return _range_name(item) if isinstance(item, tuple) else str(item)


def _group_name(group: Any) -> str:
    if group is None:
        return _CATCH_ALL_NAME
    if isinstance(group, list):
        return "|".join(_item_name(item) for item in group)
    return _item_name(group)


def _normalize_item(item: Any, where: str) -> Any:
    item = _plain(item)
    if isinstance(item, tuple):
        item = tuple(_plain(bound) for bound in item)
        try:
            _validate_range(item, where)
        except ValueError as e:
            raise ValueError(f"{e} A tuple is a (low, high) range; use a list for a group of values.") from None
        return item
    if isinstance(item, _SCALARS):
        return item
    raise ValueError(f"{where} item {item!r} must be a value or a (low, high) range.")


def _normalize_group(group: Any, where: str) -> Any:
    if group is None:
        return None
    if isinstance(group, np.ndarray):
        group = group.tolist()
    if isinstance(group, list):
        if not group:
            raise ValueError(f"{where} is an empty list; give it at least one value or range.")
        return [_normalize_item(item, where) for item in group]
    return _normalize_item(group, where)


def _items(group: Any) -> list[Any]:
    if group is None:
        return []
    return group if isinstance(group, list) else [group]


def _overlap(a: tuple[Any, Any], b: tuple[Any, Any]) -> bool:
    def low(x: Any) -> float:
        return -np.inf if x is None else float(x)

    def high(x: Any) -> float:
        return np.inf if x is None else float(x)

    return low(a[0]) < high(b[1]) and low(b[0]) < high(a[1])


def _check(pairs: list[tuple[str, Any]]) -> None:  # noqa: C901
    """Refuse groups that would not split values into one domain each."""
    names = [name for name, _ in pairs]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"Domain names must be non-empty strings, got {names!r}.")
    duplicate = next((name for name in names if names.count(name) > 1), None)
    if duplicate is not None:
        raise ValueError(f"Two domain groups are named {duplicate!r}; name them with the dict form.")
    if sum(group is None for _, group in pairs) > 1:
        raise ValueError("Only one domain group can be the None catch-all.")

    exact = [(name, item) for name, group in pairs for item in _items(group) if not isinstance(item, tuple)]
    ranges = [(name, item) for name, group in pairs for item in _items(group) if isinstance(item, tuple)]
    seen: dict[Any, tuple[str, Any]] = {}
    for name, value in exact:
        if value in seen:
            other_name, other = seen[value]
            if other_name != name or type(other) is not type(value):
                raise ValueError(
                    f"{other!r} in {other_name!r} and {value!r} in {name!r} are equal in Python, so they "
                    f"cannot be told apart; keep one.",
                )
        else:
            seen[value] = (name, value)
    for name, value in exact:
        for range_owner, over in ranges:
            if range_owner != name and _within(value, over):
                raise ValueError(
                    f"{value!r} in {name!r} falls in the range {_range_name(over)} of {range_owner!r}; "
                    f"a value belongs to one domain.",
                )
    for i, (a_name, a) in enumerate(ranges):
        for b_name, b in ranges[i + 1 :]:
            if a_name != b_name and _overlap(a, b):
                raise ValueError(
                    f"The ranges {_range_name(a)} of {a_name!r} and {_range_name(b)} of {b_name!r} overlap; "
                    f"a value belongs to one domain.",
                )


def _encode(group: Any) -> Any:
    if isinstance(group, tuple):
        return {"range": list(group)}
    if isinstance(group, list):
        return [_encode(item) for item in group]
    return group


def _decode(group: Any) -> Any:
    if isinstance(group, Mapping):
        return tuple(group["range"])
    if isinstance(group, list):
        return [_decode(item) for item in group]
    return group


@dataclass(frozen=True)
class DomainSpec:
    """Which values of one factor make up each known domain.

    ``groups`` holds ``(name, group)`` pairs in domain order, where a group is a value, a
    ``(low, high)`` range, a list of those, or None for the catch-all. ``groups`` is None
    when every distinct value of the factor is its own domain.
    """

    factor: str
    groups: tuple[tuple[str, Any], ...] | None

    @classmethod
    def parse(cls, spec: Any) -> "DomainSpec":  # noqa: C901
        """Read a user's ``domains=`` argument, refusing one that does not split values cleanly."""
        if isinstance(spec, DomainSpec):
            return spec
        if isinstance(spec, str):
            if not spec:
                raise ValueError("domains= needs a factor name.")
            return cls(spec, None)
        if not isinstance(spec, Mapping) or len(spec) != 1:
            raise ValueError(f"domains= takes one factor name, or a mapping of one factor to its groups; got {spec!r}.")
        ((factor, groups),) = spec.items()
        if not isinstance(factor, str) or not factor:
            raise ValueError(f"domains= needs a factor name as its key, got {factor!r}.")
        if isinstance(groups, Mapping):
            pairs = [(name, _normalize_group(group, f"Domain {name!r}")) for name, group in groups.items()]
        elif isinstance(groups, list | np.ndarray):
            normalized = [_normalize_group(group, f"Domain group {i}") for i, group in enumerate(list(groups))]
            pairs = [(_group_name(group), group) for group in normalized]
        else:
            raise ValueError(
                f"domains={{{factor!r}: ...}} takes a dict of name to group, or a list of groups; got {groups!r}.",
            )
        if not pairs:
            raise ValueError(f"domains={{{factor!r}: ...}} names no groups.")
        _check(pairs)
        return cls(factor, tuple(pairs))

    @property
    def has_catch_all(self) -> bool:
        return self.groups is not None and any(group is None for _, group in self.groups)

    def canonical(self) -> str | dict[str, dict[str, Any]]:
        """Return the spec in its named form, as :meth:`parse` reads it back."""
        return self.factor if self.groups is None else {self.factor: dict(self.groups)}

    def domain_order(self, values: Sequence[Any]) -> tuple[str, ...]:
        """Return domain names in class order: the groups' order, or the sorted distinct values."""
        if self.groups is not None:
            return tuple(name for name, _ in self.groups)
        return tuple(str(value) for value in sorted({value for value in values if value is not None}))

    def assign(self, values: Sequence[Any]) -> list[str | None]:
        """Name the domain each value belongs to, or None where it matches none or is missing."""
        if self.groups is None:
            return [None if value is None else str(value) for value in values]
        mapping: dict[Any, Any] = {None: _UNMATCHED}
        for name, group in self.groups:
            if group is None:
                mapping[None] = name
            for item in _items(group):
                mapping[item] = name
        remap = Remap(self.factor, mapping)
        named = [None if value is None else _remapped(value, remap) for value in values]
        return [None if name is _UNMATCHED else name for name in named]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe form; ranges are tagged so they do not read back as lists of values."""
        groups = None if self.groups is None else [[name, _encode(group)] for name, group in self.groups]
        return {"factor": self.factor, "groups": groups}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "DomainSpec":
        """Rebuild a spec written by :meth:`to_json`, validating it again."""
        if data["groups"] is None:
            return cls.parse(data["factor"])
        return cls.parse({data["factor"]: {name: _decode(group) for name, group in data["groups"]}})


def item_values(metadata: Any, factor: str, n_samples: int, caller: str) -> list[Any]:
    """One raw value of ``factor`` per sample, in dataset-index order.

    Raises
    ------
    ValueError
        When the metadata is filtered, has no such factor, or records it below the item level.
    ShapeMismatchError
        When the item rows do not pair one to one with the samples.
    """
    reject_filtered_metadata(metadata, caller)
    info = metadata.factor_info.get(factor)
    if info is None:
        raise ValueError(
            f"{caller}: the metadata has no factor {factor!r}; it has {sorted(metadata.factor_names)}. "
            f"A factor mixing types is dropped as unusable, see Metadata.unusable.",
        )
    levels = metadata.levels
    if levels.index(info.level) > levels.index(metadata.item_level):
        raise ValueError(f"{caller}: domains are per image, but {factor!r} is recorded per {info.level}.")
    rows = metadata.rows_at(metadata.item_level).select("item_index", factor).sort("item_index")
    if not rows.height == metadata.item_count == n_samples or rows["item_index"].to_list() != list(range(n_samples)):
        raise ShapeMismatchError(
            f"{caller}: the metadata has {metadata.item_count} items in {rows.height} rows, but there are "
            f"{n_samples} samples; each sample needs exactly one item row.",
        )
    return rows[factor].to_list()

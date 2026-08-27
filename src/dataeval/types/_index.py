"""Index types locating data within a source dataset."""

__all__ = [
    "SourceIndex",
]

import warnings
from typing import Any, NamedTuple, cast, get_args

from typing_extensions import Self

from dataeval.types._factors import FactorLevel


# The fields live on a base class and `SourceIndex` subclasses it, rather than
# `SourceIndex` deriving from `NamedTuple` directly. A class deriving from `NamedTuple`
# may not define `__new__` ("Cannot overwrite NamedTuple attribute __new__") and may not
# call `super()` in its methods; an ordinary subclass of one may do both, and the
# `__new__` override below is what keeps the retired ``target=`` spelling constructing.
class _SourceIndexBase(NamedTuple):
    item: int
    key: int | None
    level: FactorLevel | None


# Checked on construction, which is the one place the v1.0 spelling of the third slot — a
# channel index — silently becomes something else. ``channel=`` is loud, but
# ``SourceIndex(0, 1, 2)`` is not, and an int in the level slot survives all the way to a
# ``TypeError`` from ``str.join`` or a value filed against a level nothing matches.
#
# This is narrower than validating against a dataset: *which* levels exist is a property of
# the task and is still checked where an address is used. That a level is one of the four
# names at all is a property of the type, and is checked here.
_LEVELS: frozenset[str] = frozenset(get_args(FactorLevel))


class SourceIndex(_SourceIndexBase):
    """
    An address into a dataset: which item, which level, and which row at that level.

    One address names exactly one row. It does not describe that row's parentage — a
    detection addressed as ``SourceIndex(3, 7, "instance")`` is unambiguous without naming
    the frame or the track it sits in, because the row itself carries those. That is what
    lets a single tuple address every level of a graph that is a diamond rather than a
    chain (see :doc:`/concepts/MetadataLevels`).

    Attributes
    ----------
    item : int
        Index of the source item — the thing the dataset yields. An image for an
        image-based task, a video for multi-object tracking.
    key : int or None, default None
        Which row at `level`, within `item`. Each level names its rows with a different
        column, and `key` holds that column's value:

        =============  ==================  ==================
        level          key column          unique within
        =============  ==================  ==================
        ``sequence``   — (`key` is None)   the dataset
        ``unit``       ``unit_index``      the item
        ``track``      ``track_id``        the item
        ``instance``   ``target_index``    the item
        =============  ==================  ==================

        The instance key is ``target_index``, **not** ``instance_index``:
        ``instance_index`` is dense within a frame and repeats across the frames of one
        sequence, so it does not name a row on video.
    level : FactorLevel or None, default None
        Which level `key` addresses. ``None`` is not "unknown" — it is the task-generic
        level, resolved by :meth:`resolve`: the item level when `key` is None, and the
        label level under an integer key. So ``SourceIndex(3)`` names image 3 on an
        image task and video 3 on a tracking one, and ``SourceIndex(3, 7)`` names
        detection 7 of item 3 on either.

    Notes
    -----
    **State a level only when ``None`` would resolve to a different one.** Two spellings of
    one address are not ``==`` and do not hash alike — ``SourceIndex(3, 7)`` and
    ``SourceIndex(3, 7, "instance")`` name the same detection but are different keys in a
    mapping. Since :class:`~dataeval.quality.Outliers` and
    :class:`~dataeval.quality.Duplicates` return addresses as dictionary keys and group
    members, producers emit the minimal spelling and a level is stated only where it has to
    be: for ``unit`` and ``track``, which have no unkeyed spelling.

    `level` is checked here only for being one of the four level names. *Which* levels
    exist is a property of the dataset rather than of the type, so an address naming a level
    the data does not have is caught where the address is used, against that metadata's own
    schema.

    Examples
    --------
    >>> SourceIndex(3)
    SourceIndex(3)
    >>> SourceIndex(3, 7)
    SourceIndex(3, 7)
    >>> SourceIndex(3, 12, "unit")
    SourceIndex(3, 12, 'unit')
    """

    __slots__ = ()

    def __new__(
        cls,
        item: int,
        key: int | None = None,
        level: FactorLevel | None = None,
        *,
        target: int | None = None,
    ) -> Self:
        """Construct an address, accepting `target` as the retired spelling of `key`.

        Raises
        ------
        TypeError
            When both `key` and `target` are given, which have no resolution between them.
        ValueError
            When `level` is not one of :data:`~dataeval.types.FactorLevel`'s levels. Which
            levels a *dataset* has is checked where the address is used; this catches only
            a value that is not a level at all, which is what the retired third positional
            argument — a channel index — becomes.

        Warns
        -----
        DeprecationWarning
            When `target` is given. It is removed in v1.3.0.
        """
        if target is not None:
            if key is not None:
                raise TypeError("pass one of key= or target=, not both; target= is the retired spelling of key=")
            warnings.warn(
                "SourceIndex(target=...) is the retired spelling of SourceIndex(key=...) and is removed in v1.3.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            key = target
        # `is not None` first so the common address, which states no level, pays one
        # identity check. Constructed once per detection, so the ordering is not incidental.
        if level is not None and level not in _LEVELS:
            raise ValueError(
                f"level={level!r} is not a level; the levels are {', '.join(get_args(FactorLevel))}. "
                f"The third argument was a channel index before v1.2 and is the level now.",
            )
        return super().__new__(cls, item, key, level)

    # Deliberately narrowing a method the typing spec marks final. `NamedTuple._replace`
    # routes through `_make` to `tuple.__new__`, so it skips `__new__` — and it is the
    # documented way to set a field after the fact, which for this type means the level
    # slot. Validating everywhere else and not here would leave the likeliest hole open.
    def _replace(self, **changes: Any) -> Self:  # type: ignore[misc]
        """Return a copy with `changes` applied, validated as a fresh construction."""
        return type(self)(**{**self._asdict(), **changes})

    @property
    def kind(self) -> "FactorLevel | None":
        """Which kind of row this addresses, canonically: two spellings of one row agree.

        ``None`` is an item's own row — what an unkeyed address names, whichever level that
        turns out to be, since an item has exactly one such row. Anything else is the level
        of a row *within* an item, with an unstated level resolved to the label level, which
        is ``instance`` on every task (see :meth:`resolve`).

        This is what to group addresses by. Reading :attr:`level` directly would put
        ``SourceIndex(3, 7)`` and ``SourceIndex(3, 7, "instance")`` in separate groups
        though they name one row, and would separate an unkeyed ``SourceIndex(3)`` from an
        explicit ``SourceIndex(3, None, "sequence")`` the same way.

        Examples
        --------
        >>> SourceIndex(3).kind is None
        True
        >>> SourceIndex(3, None, "sequence").kind is None
        True
        >>> SourceIndex(3, 7).kind == SourceIndex(3, 7, "instance").kind
        True
        >>> SourceIndex(3, 12, "unit").kind
        'unit'
        """
        if self.key is None:
            return None
        return self.level or "instance"

    @property
    def sort_key(self) -> tuple[int, int, str]:
        """A total order over addresses, safe where a stated level meets an unstated one.

        Ordering addresses as the tuples they are compares ``None`` against a level name
        and raises rather than ordering, which every ``sorted()`` over a mixed sequence
        would otherwise hit. This flattens the three fields into comparable parts and puts
        an item's own row ahead of the rows within it, as every other ordering here does.
        """
        return self.item, -1 if self.key is None else self.key, self.level or ""

    @property
    def target(self) -> int | None:
        """Retired alias for :attr:`key`, kept for the v1.2 cycle.

        .. deprecated:: 1.2
            Use :attr:`key` instead. Removed in v1.3.0.
        """
        warnings.warn(
            "SourceIndex.target is the retired spelling of SourceIndex.key and is removed in v1.3.0. "
            "The address now names one row at one level, and `key` is the level's own column — "
            "`target_index` for an instance, `unit_index` for a frame, `track_id` for a track.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.key

    def resolve(self, item_level: FactorLevel, label_level: FactorLevel) -> FactorLevel:
        """Name the level this address sits at, against a dataset's own levels.

        An unstated level is the task-generic one, which is the item level for an address
        with no key and the label level for one with a key. Resolution happens here rather
        than at construction because the answer depends on the dataset: the same
        ``SourceIndex(3)`` names an image on a classification set and a video on a tracking
        one, and a producer measuring a datum does not know which it is looking at.

        Parameters
        ----------
        item_level : FactorLevel
            The level of one dataset item, as :attr:`dataeval.Metadata.item_level` reports
            it — ``unit`` for image-based tasks, ``sequence`` for tracking.
        label_level : FactorLevel
            The level of one labelled thing, as :attr:`dataeval.Metadata.label_level`
            reports it. ``instance`` for every dataset task.

        Returns
        -------
        FactorLevel
            :attr:`level` when it is stated, and the resolved default when it is not.

        Examples
        --------
        >>> SourceIndex(3).resolve("sequence", "instance")
        'sequence'
        >>> SourceIndex(3, 7).resolve("sequence", "instance")
        'instance'
        >>> SourceIndex(3, 12, "unit").resolve("sequence", "instance")
        'unit'
        """
        if self.level is not None:
            return self.level
        return item_level if self.key is None else label_level

    def __repr__(self) -> str:
        """Compact representation, omitting trailing unstated fields."""
        parts = [f"{self.item}"]
        if self.key is not None or self.level is not None:
            parts.append(f"{self.key}")
        if self.level is not None:
            parts.append(f"{self.level!r}")
        return f"SourceIndex({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable string showing the full path.

        An unstated level is rendered by leaving it off rather than by a placeholder, so
        the two-part form is unchanged. ``-`` stands in for an unstated *key* under a
        stated level, which is the only place a slot has to be held open.
        """
        parts = [str(self.item)]
        if self.key is not None or self.level is not None:
            parts.append("-" if self.key is None else str(self.key))
        if self.level is not None:
            parts.append(self.level)
        return "/".join(parts)

    @classmethod
    def from_string(cls, s: str) -> Self:
        """
        Construct a SourceIndex from a human-readable string.

        Parameters
        ----------
        s : str
            String in the format ``"item"``, ``"item/key"`` or ``"item/key/level"``. Use
            ``"-"`` for an unstated key.

        Returns
        -------
        SourceIndex

        Raises
        ------
        ValueError
            When the string has no parts or more than three, or when its third part is not
            one of :data:`~dataeval.types.FactorLevel`'s levels. The level is checked here
            because this is the one entry point where the address arrives as untrusted
            text, and a typo would otherwise become a level nothing matches.

        Examples
        --------
        >>> SourceIndex.from_string("3")
        SourceIndex(3)
        >>> SourceIndex.from_string("3/7")
        SourceIndex(3, 7)
        >>> SourceIndex.from_string("3/12/unit")
        SourceIndex(3, 12, 'unit')
        >>> SourceIndex.from_string("3/-/sequence")
        SourceIndex(3, None, 'sequence')
        """
        parts = s.split("/")
        # `str.split` never returns an empty list, so emptiness shows up as an empty first
        # part — checked here rather than left to `int("")`, whose message names neither
        # this type nor the string that produced it.
        if not parts[0] or len(parts) > 3:
            raise ValueError(f"Invalid SourceIndex string format: {s}")
        item = int(parts[0])
        key = None if len(parts) < 2 or parts[1] == "-" else int(parts[1])
        level: FactorLevel | None = None
        if len(parts) > 2 and parts[2] != "-":
            levels: tuple[FactorLevel, ...] = get_args(FactorLevel)
            if parts[2] not in levels:
                raise ValueError(
                    f"Invalid SourceIndex string format: {s}; {parts[2]!r} is not a level. "
                    f"The levels are {', '.join(levels)}.",
                )
            level = cast("FactorLevel", parts[2])
        return cls(item, key, level)

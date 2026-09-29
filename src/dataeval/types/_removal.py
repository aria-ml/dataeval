"""Removal plans: the rows of a dataset that a policy, applied to an evaluator's result, removes."""

__all__ = ["RemovalPlan"]

import operator
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

from dataeval.types._index import SourceIndex

# Whole items first, then the rows inside them, which is the order a reader scans a dataset in.
_KIND_ORDER: tuple[str | None, ...] = (None, "instance", "unit", "track", "sequence")


def canonical_address(index: "int | SourceIndex") -> SourceIndex:
    """Return `index` as an address in its minimal spelling, which states a level only for a frame or a track.

    Two spellings of one row are not ``==``. ``SourceIndex.kind`` already groups them; this rebuilds each address
    from its kind, so a set of addresses holds each row once whichever producer spelled it.
    """
    if not isinstance(index, SourceIndex):
        try:
            return SourceIndex(operator.index(index))
        except TypeError:
            raise TypeError(f"an address is an int or a SourceIndex; got {type(index).__name__}") from None
    kind = index.kind
    if kind is None:
        return SourceIndex(index.item)
    return SourceIndex(index.item, index.key, None if kind == "instance" else kind)


@dataclass(frozen=True, init=False, repr=False)
class RemovalPlan:
    """
    The rows of one dataset that a policy removes, as :class:`~dataeval.types.SourceIndex` addresses.

    Returned by :meth:`~dataeval.quality.DuplicatesOutput.deduplicate` and
    :meth:`~dataeval.quality.OutliersOutput.prune`, and applied with :class:`~dataeval.data.Indices`:
    ``View(dataset, Indices(plan, exclude=True))`` is `dataset` without the rows the plan names.

    Parameters
    ----------
    discard : Iterable[int | SourceIndex], default ()
        The rows to remove. An int names a whole item and is stored as ``SourceIndex(i)``.

    Attributes
    ----------
    discard : frozenset[SourceIndex]
        The rows to remove, each in its minimal spelling: a level is stated only for a frame (``"unit"``) or a
        track, so two spellings of one row are stored once.

    Raises
    ------
    TypeError
        If an entry of `discard` is neither an int nor a SourceIndex.

    Notes
    -----
    **A plan belongs to the dataset it was computed on.** An address's ``item`` is a position in that dataset,
    and on a tracking dataset a detection's ``target_index`` counts the detections of its whole sequence. Both
    are renumbered once anything is removed, so apply a plan once, to the dataset whose result produced it.

    Plans combine with ``|``, which keeps every row either plan names. Combining is how you apply several plans to
    one dataset: pass the combined plan to one :class:`~dataeval.data.Indices` rather than stacking one ``Indices``
    per plan. A detection's ``target_index`` counts the detections its item holds when ``Indices`` reads it, so a
    second ``Indices`` would read numbers the first one had already shifted.

    Examples
    --------
    >>> from dataeval.types import RemovalPlan, SourceIndex
    >>> plan = RemovalPlan([3, SourceIndex(5, 2)]) | RemovalPlan([SourceIndex(5, 2, "instance")])
    >>> plan
    RemovalPlan(items=1, instances=1)
    >>> list(plan)
    [SourceIndex(3), SourceIndex(5, 2)]
    """

    discard: frozenset[SourceIndex]

    def __init__(self, discard: Iterable[int | SourceIndex] = ()) -> None:
        object.__setattr__(self, "discard", frozenset(canonical_address(index) for index in discard))

    def __or__(self, other: object) -> "RemovalPlan":
        """
        Return a plan holding every row this plan or `other` names.

        Parameters
        ----------
        other : RemovalPlan
            The plan to combine with.

        Returns
        -------
        RemovalPlan
            The union of the two plans.
        """
        if not isinstance(other, RemovalPlan):
            return NotImplemented
        return RemovalPlan(self.discard | other.discard)

    def __contains__(self, index: object) -> bool:
        """
        Return whether the plan names the row `index` addresses.

        Parameters
        ----------
        index : object
            An int or a :class:`~dataeval.types.SourceIndex`, in any spelling of the row.

        Returns
        -------
        bool
            True if the plan names that row. False if it does not, or if `index` is not an address.

        Notes
        -----
        The address is compared in its minimal spelling, as the plan stores it, so ``3`` finds ``SourceIndex(3)``
        and ``SourceIndex(5, 2, "instance")`` finds ``SourceIndex(5, 2)``. Only the row itself is looked up: a
        detection inside an item the plan names is not in the plan, although applying the plan removes it.

        Examples
        --------
        >>> from dataeval.types import RemovalPlan, SourceIndex
        >>> plan = RemovalPlan([3, SourceIndex(5, 2)])
        >>> 3 in plan
        True
        >>> SourceIndex(5, 2, "instance") in plan
        True
        >>> SourceIndex(3, 0) in plan
        False
        """
        try:
            address = canonical_address(cast("int | SourceIndex", index))
        except TypeError:
            return False
        return address in self.discard

    def __iter__(self) -> Iterator[SourceIndex]:
        """
        Iterate the addresses in :attr:`~dataeval.types.SourceIndex.sort_key` order.

        Returns
        -------
        Iterator[SourceIndex]
            Each item's own row first, then the rows inside it.
        """
        return iter(sorted(self.discard, key=lambda index: index.sort_key))

    def __len__(self) -> int:
        """
        Return how many distinct rows the plan removes.

        Returns
        -------
        int
            The number of addresses.
        """
        return len(self.discard)

    def __repr__(self) -> str:
        """
        Return the number of rows removed at each level.

        Returns
        -------
        str
            For example ``RemovalPlan(items=2, instances=5)``.
        """
        counts = Counter(index.kind for index in self.discard)
        parts = [f"{'items' if kind is None else f'{kind}s'}={counts[kind]}" for kind in _KIND_ORDER if counts[kind]]
        return f"RemovalPlan({', '.join(parts)})"

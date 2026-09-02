"""Positional foreign keys between levels, and the two shapes they take.

A level's rows relate to its parent level's by position: child row *i* inherits from
parent row ``positions[i]``, or from nothing when that is ``-1``. That is what makes
downward factor propagation a gather rather than a join, and it is what
``RowBlock.ancestor_pos`` has always recorded.

This module gives that relationship a type, so the *representation* can vary while the
meaning does not. Most edges a structurer builds are far more regular than a general
position array: a walk over a dataset emits children grouped by parent and ascending,
because that is the order it visits them in. Such an edge is fully described by where
each parent's run of children starts, which is one number per *parent* rather than one
per child, and turns broadcasting into a repeat instead of a gather.

:meth:`LinkIndex.of` picks the representation by inspecting the positions rather than
by asking the caller to declare it — a wrong declaration would be silent, while a
detected property cannot be.
"""

__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray


def to_series(name: str, values: Any) -> pl.Series:
    """Build a Series from a column's values, including an empty fixed-width array.

    An ``(n, k)`` array becomes a fixed-width ``Array`` column rather than a ``List``,
    which is what keeps a box column one contiguous buffer with no per-row offsets.
    Polars reads the width off the array's shape — except when there are no rows to read
    it from. On the supported polars floor ``pl.Series(name, np.empty((0, 4)))`` raises
    ``InvalidOperationError: cannot reshape empty array into shape (0, 4)`` instead of
    producing an empty ``Array(_, 4)``, and a dataset whose items carry no detections
    produces exactly that array.

    Stating the dtype for the no-rows case sidesteps the inference altogether. That
    works on every supported version, so this is one code path rather than a version
    check — the newer polars reaches the same dtype either way.

    Only the no-rows case is handled. A zero-*width* array is not constructible on the
    floor at all, with or without this, and no column here has one: a box is four wide
    and a score is at least one.

    Parameters
    ----------
    name : str
        Name for the resulting series.
    values : Any
        Column values, as an array or any sequence polars accepts. A 1-D object array is
        read through its Python values, which is where a nullable column gets a real dtype
        rather than ``Object``.

    Returns
    -------
    pl.Series
        The column, with a fixed-width ``Array`` dtype where the values are 2-D.
    """
    if isinstance(values, np.ndarray) and values.ndim == 1 and values.dtype == object:
        # An object array states no element type, so polars keeps it as an opaque
        # ``Object`` column — one that cannot be counted, binned or compared, and that
        # raises rather than answering ``n_unique``. The Python values do carry a type, so
        # they are what it infers from instead: an all-null column reaches ``Null`` and a
        # nullable bool reaches ``Boolean``, where the array form of either reaches
        # ``Object``. Homogeneous strings — much the commonest object array here — reach
        # ``String`` by both routes, so nothing that already worked changes.
        return pl.Series(name, values.tolist())
    if isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[0] == 0:
        # A 1-D empty of the same dtype is inferred without trouble, so it is the
        # cheapest way to name the inner type without a numpy-to-polars mapping here.
        inner = pl.Series(values.reshape(-1)).dtype
        return pl.Series(name, [], dtype=pl.Array(inner, values.shape[1]))
    return pl.Series(name, values)


def gather_nulling(name: str, values: Any, positions: NDArray[np.intp]) -> pl.Series:
    """Gather ``values`` at ``positions`` as a Series, nulling the negative positions.

    The one implementation of the clamp-then-null semantics that downward propagation
    has always had, shared by :meth:`GatherLink.broadcast` and by the structuring
    layer's own gather so the two cannot drift. A **negative position** means the row
    has no ancestor at that level and yields null.

    The missing rows are nulled in the *index* rather than in the gathered column: a null
    index gathers to null whatever the column holds, whereas ``Series.scatter`` is
    unimplemented for the nested dtypes — ``List``, ``Array``, ``Struct`` — that a
    box column or a list-valued factor lands in, and raises ``ComputeError`` there
    instead of nulling the row. Both forms are one vectorized operation.

    Parameters
    ----------
    name : str
        Name for the resulting series.
    values : array-like
        One value per row at the level being read from, in that level's row order.
    positions : NDArray[np.intp]
        One position into ``values`` per output row, negative where there is none.

    Returns
    -------
    pl.Series
        One value per entry of ``positions``, null at the negative ones.
    """
    series = to_series(name, values)
    if not len(positions):
        # No rows to read. Stated rather than gathered because polars 1.0.0 panics on an
        # empty gather into a fixed-width ``Array`` column — ``box`` is one — and a filter
        # that keeps nothing is an ordinary outcome rather than an edge case.
        return series.head(0).rename(name)
    missing = positions < 0
    if not missing.any():
        return series.gather(positions).rename(name)
    if missing.all():
        # Nothing to read, and the gather below would still have to build an index to
        # read nothing with. The dtype comes from the values so the column still types
        # the diagonal concat that a later flattening does.
        return pl.Series(name, [None] * len(positions), dtype=series.dtype)
    # Clamped before nulling so the gather stays in range: polars reads a negative index
    # as counting from the end, so this is not optional even though every clamped slot is
    # about to become null.
    index = pl.Series(np.where(missing, 0, positions)).scatter(np.flatnonzero(missing), None)
    return series.gather(index).rename(name)


def _lookup(table: NDArray[np.intp], positions: NDArray[np.intp], missing: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Read ``table`` at ``positions``, yielding the no-ancestor marker where ``missing``.

    The marker is carried through rather than looked up — ``table[-1]`` would otherwise
    silently read the last row — which is why the index is clamped before the read even
    though every clamped slot is about to be overwritten.

    An **empty** ``table`` is the level with no rows at all: a tracking dataset in which
    nothing is tracked has no ``track`` rows, so every row below one is already marked
    missing and there is nothing to read. Skipped rather than clamped, because clamping
    to 0 would index off the end of an empty array.
    """
    if table.size == 0:
        return np.full(positions.shape, -1, dtype=np.intp)
    return np.where(missing, -1, table[np.where(missing, 0, positions)])


def _require_agreement(left: NDArray[np.intp], right: NDArray[np.intp], context: str) -> None:
    """Refuse two routes that name different ancestors for the same row.

    Where a level graph offers several routes to one ancestor, the routes are two ways of
    asking the same question, so they may differ only in *whether* they have an answer —
    never in what it is. A detection reaches its sequence through its frame and, when a
    tracker linked it, through its track as well; those two sequences are the same
    sequence, or the dataset says a row is inside two different ancestors at once.

    Checked rather than assumed, and checked here rather than in a test, because a test
    pins it for the structurers shipped with the library and says nothing about a caller's
    own structurer or a hand-built store — which is where a violation will come from. The
    failure mode it guards is wrong numbers rather than an exception: without this, one
    route's answer silently wins and every rollup and every propagated factor is computed
    against a parentage the other route contradicts.

    Raises
    ------
    ValueError
        When some row has an ancestor on both routes and the two disagree.
    """
    both = (left >= 0) & (right >= 0)
    disagreeing = np.flatnonzero(both & (left != right))
    if not disagreeing.size:
        return
    first = int(disagreeing[0])
    raise ValueError(
        f"Two routes {context} disagree about {disagreeing.size} row(s): row {first} reaches "
        f"parent row {int(left[first])} along one route and {int(right[first])} along another. "
        "Routes to the same ancestor may differ in whether they have an answer, never in what "
        "it is, so this dataset's links place that row inside two different ancestors at once.",
    )


class LinkIndex(ABC):
    """A positional foreign key from one level's rows to its parent level's.

    Attributes
    ----------
    child_len : int
        Number of rows at the child level.
    parent_len : int
        Number of rows at the parent level.
    """

    child_len: int
    parent_len: int

    @staticmethod
    def of(positions: Any, parent_len: int) -> "LinkIndex":
        """Build the tightest representation the positions support.

        Parameters
        ----------
        positions : array-like of int
            One parent row position per child row, ``-1`` where the child has no
            ancestor at that level.
        parent_len : int
            Number of rows at the parent level.

        Returns
        -------
        LinkIndex
            A :class:`RunLengthLink` when the children are grouped by parent, ascending
            and all present; a :class:`GatherLink` otherwise.

        Raises
        ------
        ValueError
            When a position names a row the parent level does not have.

        Notes
        -----
        The check is a few O(n) passes, which is negligible against what it saves and,
        unlike a caller's promise, cannot be wrong.

        ``parent_len`` is validated rather than trusted, because the two representations
        would otherwise disagree about it: a run-length form derives its parent count
        from the runs it builds, so an out-of-range position would silently *extend* the
        parent level, while the general form would keep the declared length and only fail
        later, in a compose that no longer meets. Both mean the same thing only while the
        declaration and the positions agree.
        """
        values = np.ascontiguousarray(positions, dtype=np.intp)
        # Sorted and non-negative at the front implies non-negative throughout, so the
        # two conditions collapse into one scan plus one comparison.
        grouped = values.size == 0 or (values[0] >= 0 and bool(np.all(np.diff(values) >= 0)))
        # Sorted, so the last entry is already the largest; only the general form pays a max.
        highest = int(values[-1] if grouped else values.max()) if values.size else -1
        if highest >= parent_len:
            raise ValueError(
                f"Position {highest} names a row beyond the {parent_len} row(s) at the parent level; "
                "the positions and the parent level disagree about how many rows it has.",
            )
        if not grouped:
            return GatherLink(values, parent_len)
        offsets = np.concatenate(([0], np.cumsum(np.bincount(values, minlength=parent_len)))).astype(np.intp)
        return RunLengthLink(offsets, int(values.size))

    @abstractmethod
    def positions(self) -> NDArray[np.intp]:
        """One parent position per child row, ``-1`` where there is no ancestor.

        The general form, materialized on demand. Needed to compose an edge with
        another — a child's grandparent is its parent's parent, gathered — and to group
        child rows by parent.

        .. warning::
            Never hand this to :meth:`polars.Series.gather`. A negative position there
            means "count from the end", so a row with no ancestor would silently be
            given the last parent's value. :meth:`broadcast` is the only safe consumer.
        """

    @abstractmethod
    def counts(self) -> NDArray[np.intp]:
        """Count the child rows under each parent row, in parent row order."""

    @abstractmethod
    def broadcast(self, name: str, values: Any) -> pl.Series:
        """Spread one value per parent row across the child rows that inherit it.

        Parameters
        ----------
        name : str
            Name for the resulting series.
        values : array-like
            One value per row at the parent level, in that level's row order.

        Returns
        -------
        pl.Series
            One value per child row, null where the child has no ancestor.
        """

    def restrict(self, child_keep: NDArray[np.intp], parent_remap: NDArray[np.intp]) -> "LinkIndex":
        """Rebuild this edge over a surviving subset of both levels' rows.

        Representation-independent: the surviving positions are read through
        :meth:`positions` and handed back to :meth:`of`, which re-picks the tightest form
        for them — a filter can turn a general edge into a run-length one and does.

        Parameters
        ----------
        child_keep : NDArray[np.intp]
            Positions of the surviving child rows, ascending.
        parent_remap : NDArray[np.intp]
            For each old parent position, its new position, or ``-1`` when it did not
            survive. Length is the old ``parent_len``.

        Returns
        -------
        LinkIndex
            The edge between the surviving rows.
        """
        return LinkIndex.of(self._restricted_positions(child_keep, parent_remap), int((parent_remap >= 0).sum()))

    def compose(self, upward: "LinkIndex") -> "LinkIndex":
        """Follow this edge and then ``upward``, giving the link to the further level.

        A child's grandparent is its parent's parent. Composing rather than storing
        every ancestor pair means only the schema's own edges are held, and a derived
        link cannot go stale relative to the edges it came from.

        A child with no parent has no grandparent either, so its marker is carried
        through rather than looked up.

        Parameters
        ----------
        upward : LinkIndex
            The edge from this link's parent level to a level above it.

        Returns
        -------
        LinkIndex
            The edge from this link's child level to ``upward``'s parent level.

        Raises
        ------
        ValueError
            When the two edges do not meet — ``upward`` describes a different level than
            the one this link points at.
        """
        if self.parent_len != upward.child_len:
            raise ValueError(
                f"Cannot compose a link into {self.parent_len} parent row(s) with one out of "
                f"{upward.child_len}; the two edges do not meet at the same level.",
            )
        mine = self.positions()
        return LinkIndex.of(_lookup(upward.positions(), mine, mine < 0), upward.parent_len)

    @staticmethod
    def first_known(routes: "Sequence[LinkIndex]", context: str = "between two levels") -> "LinkIndex":
        """Combine several routes to the same level: the first route that knows wins.

        Only a diamond produces more than one route, and they agree wherever both are
        total. They differ exactly where one branch stops short — an untracked detection
        reaches its sequence through its frame but not through a track — so a row's
        ancestor is taken from the earliest route that records one.

        That agreement is **verified** by :func:`_require_agreement` rather than trusted.
        Verifying it costs the early exit this used to take when the first route was already
        total. The routes themselves were composed eagerly before and after, so nothing new
        is *built* — what is paid is a few vectorized passes per further route, and the
        ``np.where`` short-circuit that exit used to skip. That is charged once per level
        pair per store, since :meth:`~LevelStore.link` memoizes the result, and it buys the
        one invariant here whose violation is silent.

        Route order is :meth:`~dataeval.types.FactorLevelSchema.paths` order, which is
        canonical parent order at every step. For the tracking diamond that puts the
        ``unit`` branch first, reproducing the precedence the structurers have always
        had, where the ``unit`` branch was merged last and so overwrote the ``track``
        branch's markers.

        Parameters
        ----------
        routes : Sequence[LinkIndex]
            Composed links from one level to the same ancestor, in preference order.
        context : str, default "between two levels"
            Phrase naming the levels, used only in the disagreement message.

        Returns
        -------
        LinkIndex
            One link taking each row's position from the first route that has one.

        Raises
        ------
        ValueError
            When ``routes`` is empty, when the routes describe different levels, or when
            two routes name different ancestors for the same row.
        """
        if not routes:
            raise ValueError("Cannot combine an empty set of routes.")
        if len({(route.child_len, route.parent_len) for route in routes}) != 1:
            raise ValueError("Every route must run between the same two levels.")
        combined = routes[0].positions()
        for route in routes[1:]:
            positions = route.positions()
            _require_agreement(combined, positions, context)
            combined = np.where(combined < 0, positions, combined)
        return LinkIndex.of(combined, routes[0].parent_len)

    def _restricted_positions(self, child_keep: NDArray[np.intp], parent_remap: NDArray[np.intp]) -> NDArray[np.intp]:
        """Positions of the surviving children, renumbered onto the surviving parents.

        A child that had no ancestor still has none, so its ``-1`` is carried through
        rather than looked up — ``parent_remap[-1]`` would otherwise silently read the
        last parent's new position.

        Two invariants a filter has to preserve are checked here rather than left to a
        convention, because after a filter these are the properties most likely to break
        and their failure mode is wrong numbers rather than an exception:

        - Every position lands in range, or is the no-ancestor marker.
        - **A filter never manufactures a no-ancestor marker.** A child kept while the
          parent it does have was dropped would acquire one, and that marker claims the
          *data* says this row has no such ancestor — a statement about the dataset, not
          about the query. Keeping such a child is what the survivor rule forbids, so
          tripping this means the closure that chose ``child_keep`` is wrong.

        Raised rather than asserted: the cost is two vectorized passes against a filter
        that is already gathering over the same rows, and an invariant whose violation is
        silent should not be the thing that ``python -O`` removes.

        Raises
        ------
        RuntimeError
            When either invariant is broken.
        """
        kept = self.positions()[child_keep]
        restricted = _lookup(parent_remap, kept, kept < 0)
        survivors = int((parent_remap >= 0).sum())
        if bool(np.any(restricted >= survivors)):
            raise RuntimeError(
                f"Restricting this link produced a position outside the {survivors} surviving "
                "parent row(s); the parent remapping and the parent level disagree.",
            )
        if bool(np.any((kept >= 0) & (restricted < 0))):
            raise RuntimeError(
                "Restricting this link dropped a parent while keeping its child, which would "
                "fabricate a no-ancestor marker on a row whose ancestor the data does record. "
                "A row survives a filter only if every parent it has survives.",
            )
        return restricted


class RunLengthLink(LinkIndex):
    """An edge whose children are grouped by parent, ascending, with none missing.

    Stored as one offset per parent plus a terminator, so its size is proportional to
    the *parent* level rather than the child level — for a tracking dataset's
    ``instance -> unit`` edge that is one number per frame instead of one per
    detection. Broadcasting is :func:`numpy.repeat`, which never materializes an index
    array at all, and the child count per parent is a subtraction.
    """

    def __init__(self, offsets: NDArray[np.intp], child_len: int) -> None:
        self._offsets = offsets
        self.child_len = child_len
        self.parent_len = len(offsets) - 1

    def __repr__(self) -> str:
        return f"RunLengthLink(child_len={self.child_len}, parent_len={self.parent_len})"

    def positions(self) -> NDArray[np.intp]:
        return np.repeat(np.arange(self.parent_len, dtype=np.intp), self.counts())

    def counts(self) -> NDArray[np.intp]:
        return np.diff(self._offsets).astype(np.intp)

    def broadcast(self, name: str, values: Any) -> pl.Series:
        # A plain numeric column repeats without an index array at all, which is roughly
        # 2.5x the throughput of a gather at a million rows and is the case that matters
        # — factors are overwhelmingly 1-D. Everything else (strings, fixed-width boxes,
        # nested dtypes) takes the general path, which is correct for every dtype.
        #
        # A null-free numeric Series qualifies too, and has to be named explicitly: the
        # normalized store reads a level's column off its own frame and so hands this a
        # ``pl.Series`` rather than the array a structurer built, which would otherwise
        # take the general path for every propagated factor in the library. The
        # conversion is zero-copy exactly when the null check passes, and a column with
        # nulls is left alone because ``to_numpy`` would widen its dtype to hold them.
        #
        # Note ``repeat_by().explode()`` is *not* an option: it yields a null for a
        # parent with no children rather than nothing, inserting rows that do not exist.
        if isinstance(values, pl.Series) and values.dtype.is_numeric() and values.null_count() == 0:
            values = values.to_numpy()
        if isinstance(values, np.ndarray) and values.ndim == 1:
            return to_series(name, np.repeat(values, self.counts()))
        # Safe without clamping: a run-length edge has no negative positions by
        # construction, which is precisely what lets it be one.
        return to_series(name, values).gather(self.positions()).rename(name)


class GatherLink(LinkIndex):
    """An edge in the general form: one parent position per child row.

    The representation for edges a walk cannot emit in order — a track's detections are
    scattered across the frames it appears in — and for any edge where some child has no
    ancestor, which a run-length form has no way to express.
    """

    def __init__(self, positions: NDArray[np.intp], parent_len: int) -> None:
        self._positions = positions
        self.child_len = len(positions)
        self.parent_len = parent_len

    def __repr__(self) -> str:
        return f"GatherLink(child_len={self.child_len}, parent_len={self.parent_len})"

    def positions(self) -> NDArray[np.intp]:
        return self._positions

    def counts(self) -> NDArray[np.intp]:
        present = self._positions[self._positions >= 0]
        return np.bincount(present, minlength=self.parent_len).astype(np.intp)

    def broadcast(self, name: str, values: Any) -> pl.Series:
        # Clamp-then-null lives in one place, shared with the structuring layer's own
        # gather, so the two spellings of "a negative position is a null" cannot drift.
        return gather_nulling(name, values, self._positions)

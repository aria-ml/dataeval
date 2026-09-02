"""The normalized store: one frame per level, plus the edges between them.

Every fact is held once, at the granularity it was measured. A level's frame holds only
what is defined *at* that level; reading it from a descendant is a gather along the
schema's edges rather than a stored copy, and the flat dataframe is these frames plus
those gathers — which is why it is derivable from the store and not the other way round.

Two widths, deliberately named apart:

- :meth:`LevelStore.resolve` widens a level's rows with *every* column the store holds —
  ancestor values gathered down, everything else typed-null. It is what
  :attr:`~dataeval.Metadata.dataframe` and :meth:`~dataeval.Metadata.rows_at` return, and
  it costs one column per column in the store.
- :meth:`LevelStore.select` widens with *only* the named columns. It is what every
  array-shaped accessor reads through.

Routing a projection through ``resolve`` would rebuild the whole flat frame to read a
handful of columns off it, which is the cost the normalized store exists to remove. The
two are separate methods rather than a flag so that the expensive one cannot be reached by
accident from the cheap one's callers.

The store is immutable. Every writer returns a new store and the caller rebinds, so a
derived view — :meth:`~dataeval.Metadata.at`, and the filters that will follow it — shares
one safely instead of hand-copying the pieces it is made of.
"""

__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._metadata._links import LinkIndex
from dataeval._metadata._structurers import StructuredData
from dataeval.types import FactorLevel, FactorLevelSchema


# ``eq=False`` so the identity comparison inherited from ``object`` stands. A generated
# ``__eq__`` would compare the frames, and ``pl.DataFrame.__eq__`` answers with a frame
# rather than a bool, so comparing two distinct stores raises instead of answering; the
# generated ``__hash__`` that ``frozen=True`` would come with raises too, on the dicts.
@dataclass(frozen=True, eq=False)
class LevelStore:
    """Per-level frames and the links between them.

    Attributes
    ----------
    schema : FactorLevelSchema
        The level graph the frames and links are shaped by.
    frames : Mapping[str, pl.DataFrame]
        One frame per level that has rows, holding only that level's own columns, in
        row order coarsest level first.
    links : Mapping[tuple[str, str], LinkIndex]
        One entry per *schema edge* — ``(child, immediate parent)`` — and no more.
        Anything further up is composed on demand by :meth:`link` and memoized, so a
        link to a grandparent cannot disagree with the edges it is made of and storage
        stays proportional to the graph rather than to its transitive closure.
    propagating : Mapping[str, frozenset[str]]
        Which of each level's columns descendants inherit. Factors and their companion
        columns do; reserved columns do not, because every level produces its own —
        an instance row's ``item_index`` is computed for that row, not gathered from
        its unit.
    column_order : tuple[str, ...]
        Canonical left-to-right column order of the flat frame. Carried rather than
        derived: the flat frame lists every block's reserved columns before any level's
        factors, which walking the frames level by level does not reproduce.
    """

    schema: FactorLevelSchema
    frames: Mapping[FactorLevel, pl.DataFrame]
    links: Mapping[tuple[FactorLevel, FactorLevel], LinkIndex]
    propagating: Mapping[FactorLevel, frozenset[str]]
    column_order: tuple[str, ...]
    # Memoized transitive links. Excluded from equality and repr because it is a pure
    # function of ``links`` and ``schema`` — two stores that agree on those agree on
    # every composition, whether or not either has computed one yet. It is the single
    # piece of mutable state inside a frozen value, and it is why nothing may mutate
    # ``links`` in place: a derived link would outlive the edges it came from.
    _composed: dict[tuple[FactorLevel, FactorLevel, FactorLevel | None], LinkIndex] = field(
        default_factory=dict, compare=False, repr=False
    )

    # ------------------------------------------------------------------ building

    @classmethod
    def empty(cls, schema: FactorLevelSchema) -> "LevelStore":
        """Build a store with no rows, for an instance that has not been structured yet."""
        return cls(schema, {}, {}, {}, ())

    @classmethod
    def of(cls, schema: FactorLevelSchema, data: StructuredData) -> "LevelStore":
        """Build a store from what a structurer produced.

        The links are read through :meth:`RowBlock.positions_at` rather than indexed: a
        block may legitimately omit a parent from its ancestor map, which is the
        block-wide statement that no row here has such an ancestor, and the edge for
        that is a marker per row rather than a ``KeyError``.

        Only the schema's own edges are built. A block that records a *transitive*
        ancestor's positions as well is not consulted for them — :meth:`link` composes
        that from the edges, which issue 3.2 established reproduces them exactly.

        Every declared edge gets an entry, including one whose level or parent produced
        no block at all: that is a level with no rows, whose edge is simply empty on one
        side or marked absent on the other. Skipping it instead would leave :meth:`link`
        composing a route through an edge it has no entry for, which raises ``KeyError``
        from :meth:`_route` rather than the ``ValueError`` this class documents.
        """
        blocks = data.blocks_by_level
        return cls(
            schema=schema,
            frames=data.native_frames(),
            links={
                (level, parent): LinkIndex.of(
                    blocks[level].positions_at(parent) if level in blocks else np.empty(0, dtype=np.intp),
                    blocks[parent].size if parent in blocks else 0,
                )
                for level in schema
                for parent in schema.parents_of(level)
            },
            propagating={level: frozenset(names) for level, names in data.factors.items()},
            column_order=data.column_order,
        )

    # ------------------------------------------------------------------ reading

    @property
    def counts(self) -> Mapping[FactorLevel, int]:
        """Number of rows at each level, in row order."""
        return {level: frame.height for level, frame in self.frames.items()}

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column the store holds, in canonical order."""
        return self.column_order

    def height(self, level: FactorLevel) -> int:
        """Count the rows at ``level``, answering zero when the level has no frame."""
        frame = self.frames.get(level)
        return 0 if frame is None else frame.height

    def frame(self, level: FactorLevel) -> pl.DataFrame:
        """Read the level's own rows and columns, with nothing gathered onto them.

        Raises
        ------
        KeyError
            When the level has no frame, which for a structured store means it is not
            part of the schema.
        """
        return self.frames[level]

    def dtype_of(self, column: str) -> pl.DataType:
        """Dtype a column has wherever it is defined, or ``Null`` where it is nowhere.

        A column no level types is null on every row — ``box`` on a classification
        dataset — and ``Null`` is the dtype the flat frame has always given it.
        """
        for frame in self.frames.values():
            dtype = frame.schema.get(column)
            if dtype is not None:
                return dtype
        return pl.Null()

    def link(self, level: FactorLevel, ancestor: FactorLevel, via: FactorLevel | None = None) -> LinkIndex:
        """Positional link from ``level``'s rows up to ``ancestor``'s.

        Where the graph offers several routes — which only the tracking diamond does —
        each row takes its ancestor from the first route that records one. The routes are
        verified to agree wherever both know an answer; they differ only where one branch
        stops short, as it does for a detection no tracker linked, which reaches its
        sequence through its frame but not through a track. Route order is canonical
        parent order, which puts the ``unit`` branch first and so reproduces the
        precedence the structurers have always had.

        Parameters
        ----------
        level : str
            Level whose rows the link starts from.
        ancestor : str
            Level above it to reach.
        via : str or None, default None
            Restrict to routes passing through this level. None takes every route, which
            is the union of what they know and is total wherever any branch is. Naming a
            branch instead asks a narrower question — ``via="track"`` for an
            ``instance -> sequence`` link reaches only the detections a tracker linked —
            and is why the two are different rollups rather than two spellings of one.

        Returns
        -------
        LinkIndex
            One parent position per row at ``level``, ``-1`` where that row has no
            ancestor there by the selected route(s).

        Raises
        ------
        ValueError
            When ``ancestor`` does not sit above ``level``; when ``via`` is ``level``
            itself; when no route to ``ancestor`` passes through ``via``; or when two
            routes name different ancestors for the same row.
        """
        key = (level, ancestor, via)
        if via is None and (stored := self.links.get((level, ancestor))) is not None:
            return stored
        if (cached := self._composed.get(key)) is not None:
            return cached

        if not (paths := self.schema.paths(level, ancestor)):
            raise ValueError(
                f"{ancestor!r} is not above {level!r} in this dataset's level graph, so there is "
                f"no link between them. Levels above {level!r} are {list(self.schema.ancestors(level))}.",
            )
        routes = paths if via is None else self.schema.routes_through(level, ancestor, via)
        through = "" if via is None else f" through {via!r}"
        composed = LinkIndex.first_known(
            [self._route(level, path) for path in routes],
            context=f"from {level!r} to {ancestor!r}{through}",
        )
        self._composed[key] = composed
        return composed

    def _route(self, level: FactorLevel, path: Sequence[FactorLevel]) -> LinkIndex:
        """Compose the schema edges along one upward path into a single link.

        ``path`` comes from :meth:`~dataeval.types.FactorLevelSchema.paths` and always
        names at least one step, so there is always a first edge to start from.
        """
        link = self.links[(level, path[0])]
        current = path[0]
        for step in path[1:]:
            link = link.compose(self.links[(current, step)])
            current = step
        return link

    def partial_ancestry(self, level: FactorLevel, at: FactorLevel) -> bool:
        """Whether some row at ``at`` has no ancestor at ``level``.

        True only for the in-between case: ``level`` does reach ``at``, but not from every
        row. A detection no tracker linked is the instance of it — it has a frame and no
        track, so a per-track factor is null there while being present on its neighbours.
        Callers that need a total column have to exclude such a factor, which is a
        property of the links rather than of the values, so it is answered here.

        Returns
        -------
        bool
            True when at least one row at ``at`` records no ancestor position at
            ``level``. False when every row has one, when ``at`` has no rows at all, and
            when ``level`` does not sit above ``at`` — which is a different question,
            answered by :meth:`~dataeval.types.FactorLevelSchema.propagates_to`.
        """
        if not self.schema.is_ancestor(level, at) or at not in self.frames:
            return False
        return bool(np.any(self.link(at, level).positions() < 0))

    def source_of(self, level: FactorLevel, column: str) -> FactorLevel | None:
        """Which level supplies ``column`` for rows at ``level``, or None for nowhere.

        The level's own frame wins, so a native column is never shadowed by an
        ancestor's. Only :attr:`propagating` columns are looked for above: a reserved
        column is produced by each level for itself, so its absence from this level's
        frame means these rows genuinely have no value for it rather than that it should
        be gathered.
        """
        frame = self.frames.get(level)
        if frame is None:
            return None
        if column in frame.columns:
            return level
        for ancestor in self.schema.ancestors(level):
            above = self.frames.get(ancestor)
            if above is not None and column in self.propagating.get(ancestor, frozenset()) and column in above.columns:
                return ancestor
        return None

    def column(self, level: FactorLevel, name: str) -> pl.Series:
        """One value of ``name`` per row at ``level``, gathered from wherever it lives.

        Null throughout where no level supplies it, typed by :meth:`dtype_of` so that a
        frame built from these still concatenates against one that has real values.
        """
        source = self.source_of(level, name)
        if source is None:
            return pl.Series(name, [None] * self.height(level), dtype=self.dtype_of(name))
        if source == level:
            return self.frames[level][name]
        return self.link(level, source).broadcast(name, self.frames[source][name])

    def select(self, level: FactorLevel, columns: Sequence[str]) -> pl.DataFrame:
        """Read the named columns only, on ``level``'s rows.

        The narrow read. Each column is resolved on its own — taken natively, gathered
        from the one ancestor that defines it, or filled with typed nulls — so the cost
        is the columns asked for rather than every column the store holds.
        """
        return pl.DataFrame([self.column(level, name) for name in columns])

    def resolve(self, level: FactorLevel) -> pl.DataFrame:
        """Every column the store holds, on ``level``'s rows, in canonical order.

        The wide read, and exactly one horizontal slice of the flat frame: the level's
        own columns, every ancestor factor gathered down, and typed nulls for everything
        that belongs to another branch or to a level below.

        Raises
        ------
        RuntimeError
            When the level holds a column :attr:`column_order` does not name. Selecting
            the canonical order would drop it, and this is the method that defines what
            the flat frame contains — a column silently missing from it reads as a
            factor that was never added rather than as one that was lost.
        """
        frame = self.frames[level]
        ordered = set(self.column_order)
        if unordered := [name for name in frame.columns if name not in ordered]:
            raise RuntimeError(
                f"level {level!r} holds {unordered} but the store's column order does not name "
                f"them, so resolving it would drop them. Every writer must extend column_order.",
            )
        present = set(frame.columns)
        additions: list[pl.Series] = []
        for ancestor in self.schema.ancestors(level):
            above = self.frames.get(ancestor)
            if above is None:
                continue
            names = [
                name
                for name in above.columns
                if name not in present and name in self.propagating.get(ancestor, frozenset())
            ]
            if not names:
                continue
            link = self.link(level, ancestor)
            additions.extend(link.broadcast(name, above[name]) for name in names)
            present.update(names)
        additions.extend(
            pl.Series(name, [None] * frame.height, dtype=self.dtype_of(name))
            for name in self.column_order
            if name not in present
        )
        widened = frame.with_columns(additions) if additions else frame
        return widened.select(self.column_order)

    def flat(self) -> pl.DataFrame:
        """Rows at every level, coarsest first, with factors propagated down.

        The flat frame is the store plus the gathers, which this is the statement of.
        Nothing reads it that could read a level instead — it exists for
        :attr:`~dataeval.Metadata.dataframe`, whose contract is every level at once.
        """
        if not self.frames:
            return pl.DataFrame({name: [] for name in self.column_order})
        return pl.concat([self.resolve(level) for level in self.frames], how="vertical")

    # ------------------------------------------------------------------ writing

    def _replacing(self, frames: Mapping[FactorLevel, pl.DataFrame], **changes: Any) -> "LevelStore":
        """Rebuild with new frames, keeping the links, which no column write can affect."""
        return LevelStore(
            schema=self.schema,
            frames=frames,
            links=self.links,
            propagating=changes.get("propagating", self.propagating),
            column_order=changes.get("column_order", self.column_order),
            # Carried, not dropped: the compositions describe the edges, and the edges
            # are unchanged. A store whose *rows* change must not carry them, which is
            # why this helper is only for column writes.
            _composed=self._composed,
        )

    def with_column(self, level: FactorLevel, series: pl.Series, *, propagates: bool = True) -> "LevelStore":
        """Write one column at ``level``, replacing any column of that name anywhere.

        Cleared from every other level first: re-adding a factor at a new level must not
        leave the old level still holding its values, which would make
        :meth:`source_of` answer with whichever level it reached first.

        Parameters
        ----------
        level : str
            Level the values are defined at. A level the schema declares but that
            produced no frame at all is a no-op — including the clearing above, since
            there is no destination for the column to move to. A level with *no rows*
            is not that case: it has a frame of height zero and is written normally.
        series : pl.Series
            One value per row at ``level``, in that level's row order.
        propagates : bool, default True
            Whether descendant rows inherit the column. True for factors and their
            companion columns, which is everything written after structuring.
        """
        if level not in self.frames:
            return self
        name = series.name
        frames: dict[FactorLevel, pl.DataFrame] = {
            other: frame.drop(name) if other != level and name in frame.columns else frame
            for other, frame in self.frames.items()
        }
        frames[level] = frames[level].with_columns(series)
        propagating = {
            other: (names | {name}) if other == level and propagates else (names - {name})
            for other, names in self.propagating.items()
        }
        propagating.setdefault(level, frozenset({name}) if propagates else frozenset())
        order = self.column_order if name in self.column_order else (*self.column_order, name)
        return self._replacing(frames, propagating=propagating, column_order=order)

    def without_columns(self, names: Iterable[str]) -> "LevelStore":
        """Drop columns from wherever they are held.

        Names that no level holds are ignored, so a caller can offer both spellings of
        a companion column without first checking which one exists.
        """
        dropped = set(names)
        if not dropped:
            return self
        frames: dict[FactorLevel, pl.DataFrame] = {
            level: frame.drop(dropped & set(frame.columns)) for level, frame in self.frames.items()
        }
        propagating = {level: values - dropped for level, values in self.propagating.items()}
        order = tuple(name for name in self.column_order if name not in dropped)
        return self._replacing(frames, propagating=propagating, column_order=order)

    def positions_from(self, level: FactorLevel, ancestor: FactorLevel) -> NDArray[np.intp]:
        """Raw ancestor positions, for callers that compare them rather than gather with them.

        .. warning::
            Never hand these to :meth:`polars.Series.gather` — a negative position there
            counts from the end. :meth:`column` and :meth:`LinkIndex.broadcast` are the
            safe consumers.
        """
        return self.link(level, ancestor).positions()

    # ------------------------------------------------------------------ filtering

    def _inherited_survival(
        self, level: FactorLevel, keep: Mapping[FactorLevel, NDArray[np.intp]]
    ) -> NDArray[np.bool_]:
        """Which rows at ``level`` have every parent they actually have still surviving.

        A row with no ancestor at some parent level is *not* failing the test — it has
        nothing there to lose, which is the whole content of invariant I5. Reading a
        ``-1`` as a dropped parent would drop the untracked detections on every filter,
        and reading it through the parent's survival array would silently answer with the
        last parent's fate.
        """
        alive = np.ones(self.height(level), dtype=np.bool_)
        for parent in self.schema.parents_of(level):
            surviving = keep.get(parent)
            if surviving is None:
                continue
            lives = np.zeros(self.height(parent), dtype=np.bool_)
            lives[surviving] = True
            positions = self.link(level, parent).positions()
            # ``lives`` is empty exactly when the parent level has no rows, in which case
            # every position is already the marker and the gather would be out of range.
            inherited = lives[np.maximum(positions, 0)] if lives.size else np.zeros(len(positions), dtype=np.bool_)
            alive &= (positions < 0) | inherited
        return alive

    def _close_downward(self, seeded: Mapping[FactorLevel, NDArray[np.intp]]) -> dict[FactorLevel, NDArray[np.intp]]:
        """Fill in every level the seed did not name, by the parents-all-survive rule.

        Schema order is a topological order of the edges, so a level's parents are
        decided before it is and one pass suffices — no fixpoint. This is the half the
        two filters share; all that separates ``where`` from ``having`` is which levels
        arrive already seeded.
        """
        keep = dict(seeded)
        for level in self.schema:
            if level in keep or level not in self.frames:
                continue
            keep[level] = np.flatnonzero(self._inherited_survival(level, keep)).astype(np.intp)
        return keep

    def surviving_where(self, level: FactorLevel, mask: NDArray[np.bool_]) -> dict[FactorLevel, NDArray[np.intp]]:
        """Survivors of restricting ``level``'s rows to ``mask``, level by level.

        ``where`` does not filter upwards: a level's strict ancestors are seeded whole,
        so filtering frames leaves every sequence in place, and leaves every *track* in
        place too — ``track`` is a sibling of ``unit``, not a descendant, so a track whose
        every observation fell in a dropped frame still has a row. That is surprising
        rather than wrong, and reporting it belongs to the public filter rather than here.

        Parameters
        ----------
        level : str
            Level the predicate was evaluated at.
        mask : NDArray[np.bool_]
            One flag per row at ``level``, True to keep.

        Returns
        -------
        dict[str, NDArray[np.intp]]
            Surviving row positions at every level that has a frame, ascending.
        """
        seeded: dict[FactorLevel, NDArray[np.intp]] = {level: np.flatnonzero(mask).astype(np.intp)}
        for ancestor in self.schema.ancestors(level):
            if ancestor in self.frames:
                seeded[ancestor] = np.arange(self.height(ancestor), dtype=np.intp)
        return self._close_downward(seeded)

    def surviving_having(self, level: FactorLevel, mask: NDArray[np.bool_]) -> dict[FactorLevel, NDArray[np.intp]]:
        """Survivors of keeping the ancestors that have a matching row at ``level``.

        The seed travels *up* only: each strict ancestor keeps the rows some matching row
        at ``level`` points at, and everything else — including ``level`` itself — then
        falls out of the parents-all-survive rule. That is what makes ``having`` cut
        sideways across the diamond: a car detection in a frame that also holds a person
        keeps its frame but loses its track, and so is itself dropped.

        At a level with no ancestors there is nothing for the seed to travel to and this
        keeps everything. That is a filter that did nothing, which the public spelling
        rejects rather than passes on.

        Parameters
        ----------
        level : str
            Level the predicate was evaluated at.
        mask : NDArray[np.bool_]
            One flag per row at ``level``, True for a row that matches.

        Returns
        -------
        dict[str, NDArray[np.intp]]
            Surviving row positions at every level that has a frame, ascending.
        """
        seed = np.flatnonzero(mask).astype(np.intp)
        seeded: dict[FactorLevel, NDArray[np.intp]] = {}
        for ancestor in self.schema.ancestors(level):
            if ancestor not in self.frames:
                continue
            reached = self.link(level, ancestor).positions()[seed]
            seeded[ancestor] = np.unique(reached[reached >= 0]).astype(np.intp)
        return self._close_downward(seeded)

    def restrict(self, keep: Mapping[FactorLevel, NDArray[np.intp]]) -> "LevelStore":
        """Rebuild the store over the surviving rows, renumbering every edge onto them.

        The links are remapped rather than recomputed: a surviving child's parent is
        wherever its old parent landed, which :meth:`LinkIndex.restrict` resolves while
        checking that the closure did not keep a child whose parent it dropped. A filter
        may change an edge's representation — survivors of a run-length edge stay grouped,
        so it comes back as one — which is why the edges are rebuilt through
        :meth:`LinkIndex.of` rather than edited in place.

        Parameters
        ----------
        keep : Mapping[str, NDArray[np.intp]]
            Surviving row positions at each level, ascending. Must name every level that
            has a frame; a partial mapping would leave the unnamed levels whole while
            their edges were renumbered around them.

        Returns
        -------
        LevelStore
            A store over the surviving rows, holding the same columns.

        Raises
        ------
        ValueError
            When ``keep`` does not name every level that has a frame.
        """
        if missing := [level for level in self.frames if level not in keep]:
            raise ValueError(
                f"restrict needs survivors for every level that has a frame; {missing} are missing. "
                "A level left unnamed keeps all its rows while its edges are renumbered around it.",
            )
        remap: dict[FactorLevel, NDArray[np.intp]] = {}
        frames: dict[FactorLevel, pl.DataFrame] = {}
        for level in self.schema:
            survivors = keep.get(level, np.empty(0, dtype=np.intp))
            positions = np.full(self.height(level), -1, dtype=np.intp)
            positions[survivors] = np.arange(len(survivors), dtype=np.intp)
            remap[level] = positions
            if level in self.frames:
                # ``head(0)`` rather than an empty gather: polars 1.0.0 panics indexing a
                # frame that holds a fixed-width ``Array`` column with an empty index.
                frames[level] = self.frames[level][survivors] if survivors.size else self.frames[level].head(0)
        # ``_composed`` is deliberately not carried: it describes the *old* rows, and a
        # composition that outlived them would answer with positions into frames that no
        # longer exist. Column writes may carry it; a row change may never.
        return LevelStore(
            schema=self.schema,
            frames=frames,
            links={
                edge: link.restrict(keep.get(edge[0], np.empty(0, dtype=np.intp)), remap[edge[1]])
                for edge, link in self.links.items()
            },
            propagating=self.propagating,
            column_order=self.column_order,
        )

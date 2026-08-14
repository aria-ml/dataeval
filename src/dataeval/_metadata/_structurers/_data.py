"""The bundle a structurer hands back, and the two flat forms derivable from it."""

__all__ = []

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval._metadata._links import to_series
from dataeval._metadata._structurers._block import RowBlock
from dataeval._metadata._structurers._gather import column_values, gather_series, holds_only_nulls
from dataeval._metadata._structurers._layout import RowLayout
from dataeval.types import FactorLevel


@dataclass(frozen=True)
class StructuredData:
    """Everything a structurer extracts from a dataset, before any binning.

    Attributes
    ----------
    blocks : Sequence[RowBlock]
        Row blocks ordered coarsest level first.
    factors : Mapping[str, Mapping[str, Any]]
        Factor values keyed by the level they are defined at.
    dropped_factors : Mapping[str, Sequence[str]]
        Factors discarded during metadata merging, with reasons.
    raw : Sequence[Mapping[str, Any]]
        Untouched per-item metadata dictionaries.
    class_labels : NDArray[np.intp]
        One label per target-level row.
    item_indices : NDArray[np.intp]
        Source item index for each target-level row.
    """

    blocks: Sequence[RowBlock]
    factors: Mapping[FactorLevel, Mapping[str, Any]]
    dropped_factors: Mapping[str, Sequence[str]] = field(default_factory=dict)
    raw: Sequence[Mapping[str, Any]] = field(default_factory=list)
    class_labels: NDArray[np.intp] = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    item_indices: NDArray[np.intp] = field(default_factory=lambda: np.empty(0, dtype=np.intp))

    def __post_init__(self) -> None:
        """Validate the bundle before anything reads it."""
        self._reject_repeated_levels()
        self._reject_repeated_factor_names()

    def _reject_repeated_levels(self) -> None:
        """Reject a bundle carrying more than one block at the same level.

        One block per level is already assumed by everything that indexes the layout by
        level rather than by position: :attr:`RowLayout.counts` builds a dict keyed on
        level, so a second block silently overwrites the first's row count, and
        :meth:`RowLayout.partial_ancestry` returns on the first block matching the level
        and never looks at the rest. Both would answer confidently and wrongly.

        The assumption holds in every structurer today — each emits its levels once — so
        this states it rather than changes it. It is checked here because the levelled
        frames that :meth:`to_frame` builds are keyed by level outright, which turns a
        silent wrong answer into a missing block.
        """
        counts = Counter(block.level for block in self.blocks)
        if repeated := sorted(level for level, count in counts.items() if count > 1):
            raise ValueError(
                f"Level(s) {repeated} have more than one row block. A level's rows are one block, "
                "because the layout and the levelled frames are both keyed by level — a second "
                "block does not extend the first, it hides it. Concatenate the rows into one "
                "block per level before building the bundle.",
            )

    def _reject_repeated_factor_names(self) -> None:
        """Reject a factor name declared at more than one level.

        A factor becomes one dataframe column, and a column holds values for exactly
        one level: :meth:`RowLayout.expand` fills that level's rows and its
        descendants' and nulls everything else, so a second declaration of the same
        name does not merge with the first, it replaces it — and the losing level's
        rows are left holding nulls in a column that still counts as its factor.

        Checked here rather than left to a convention because the two existing
        structurers only avoid it by explicitly subtracting the overlap their two
        metadata merges produce. Nothing about that is visible to the next structurer,
        and the failure it prevents is silent: :meth:`Metadata._factor_level` resolves
        the name to one level and bins whatever that level's rows hold, which is a
        column of nulls. Qualify the names instead — ``frame_timestamp`` and
        ``instance_timestamp`` — the way ``add_factors`` does when a source index
        spans levels.
        """
        seen: dict[str, FactorLevel] = {}
        for level, factors in self.factors.items():
            for name in factors:
                if (first := seen.get(name)) is not None:
                    raise ValueError(
                        f"Factor {name!r} is declared at both the {first!r} and {level!r} levels. "
                        "A factor is one column and a column belongs to one level, so the second "
                        "declaration would null out the first level's rows. Give each level's "
                        f"values their own name, e.g. {f'{first}_{name}'!r} and {f'{level}_{name}'!r}.",
                    )
                seen[name] = level

    @property
    def layout(self) -> RowLayout:
        """Positional map for the rows this bundle describes."""
        return RowLayout.from_blocks(self.blocks)

    @property
    def column_order(self) -> tuple[str, ...]:
        """Canonical left-to-right column order of the flat frame.

        Reserved columns first, in the order the blocks introduce them, then factors in
        level order. Stated explicitly because :meth:`to_frame` builds one frame per
        block and lets a diagonal concat reconcile them: a block omits any column it has
        no values for, so the order the concat arrives at is first-*populated* order,
        which is not the same thing. Selecting this at the end pins the layout to the
        blocks' own declaration order regardless.

        Returns
        -------
        tuple[str, ...]
            Every column of the flat frame, in order, each appearing once.
        """
        order: dict[str, None] = {}
        for block in self.blocks:
            order.update(dict.fromkeys(block.columns))
        for factors in self.factors.values():
            order.update(dict.fromkeys(factors))
        return tuple(order)

    @property
    def blocks_by_level(self) -> Mapping[FactorLevel, RowBlock]:
        """Blocks keyed by level, which ``__post_init__`` guarantees is unique."""
        return {block.level: block for block in self.blocks}

    def native_frames(self) -> dict[FactorLevel, pl.DataFrame]:
        """One frame per level, holding only what is defined *at* that level.

        A level's reserved columns and the factors declared there — no ancestor values.
        This is the normalized form: each fact is stored once, at the granularity it was
        measured, and reading it from a descendant is a gather away rather than a stored
        copy. :meth:`to_frame` is this plus the gathers, which is why the flat frame is
        derivable from these and not the other way around.

        A reserved column a level has no values for is left out rather than carried as
        nulls: it is not native to that level, and a ``Null`` column carries no dtype for
        the concat in :meth:`to_frame` to reconcile.

        Returns
        -------
        dict[str, pl.DataFrame]
            Frame per level, in the bundle's block order.
        """
        frames: dict[FactorLevel, pl.DataFrame] = {}
        for block in self.blocks:
            columns = {
                name: to_series(name, values) for name, values in block.columns.items() if not holds_only_nulls(values)
            }
            own = self.factors.get(block.level, {})
            columns.update({name: to_series(name, values) for name, values in own.items()})
            frames[block.level] = pl.DataFrame(columns)
        return frames

    def _block_frame(self, block: RowBlock, native: pl.DataFrame) -> pl.DataFrame:
        """Widen one block's native frame with every ancestor factor that reaches it.

        A factor at a level this block records no ancestor position for is not readable
        here at all, so its column is omitted and the diagonal concat in :meth:`to_frame`
        fills it from the blocks that do type it. A ``Null`` column could not serve
        instead — polars raises rather than supertyping it against a real dtype.
        """
        gathered = [
            gather_series(name, values, positions)
            for level, factors in self.factors.items()
            if level != block.level and (positions := block.ancestor_pos.get(level)) is not None
            for name, values in factors.items()
        ]
        return native.with_columns(gathered) if gathered else native

    def to_frame(self, native: Mapping[FactorLevel, pl.DataFrame] | None = None) -> pl.DataFrame:
        """Flatten blocks and factors into a single dataframe.

        Builds one frame per block and concatenates them, rather than assembling
        full-height Python lists column by column. Each block's reserved columns are
        handed to polars as the arrays the structurers produced, and each factor is
        gathered onto the block's rows in one vectorized step, so no per-row Python
        object is created on the way in.

        Parameters
        ----------
        native : Mapping[str, pl.DataFrame] or None, default None
            Frames from a previous :meth:`native_frames` call, to widen rather than
            rebuild. A caller that keeps the normalized store — :class:`~dataeval.Metadata`
            does — has already paid for these, and building them twice per structuring
            costs one ``pl.Series`` per column per level for nothing. Omit to build them
            here.

        Returns
        -------
        pl.DataFrame
            Rows at every level, coarsest first, with factors propagated down to
            descendant levels and null elsewhere.
        """
        native = self.native_frames() if native is None else native
        frames = [self._block_frame(block, native[block.level]) for block in self.blocks]
        order = self.column_order
        if not frames:
            return pl.DataFrame({name: [] for name in order})
        frame = pl.concat(frames, how="diagonal")
        # A column no block typed is null everywhere — ``box`` on a classification
        # dataset, for instance. Every block dropped it, so the concat has never seen
        # it; it is restored here as Null, which is the dtype the flat builder inferred
        # for a column of nothing but None. Spelled as an explicit column of the frame's
        # own height rather than ``pl.lit(None)``, which *broadcasts*: against a frame the
        # concat left with no columns at all — every block empty, so every column dropped
        # — there is no height to broadcast to and the literal would add a phantom row.
        if absent := [name for name in order if name not in frame.columns]:
            frame = frame.with_columns([pl.Series(name, [None] * frame.height, dtype=pl.Null) for name in absent])
        return frame.select(order)

    def to_rows(self) -> dict[str, list[Any]]:
        """Flatten blocks and factors into a single column-oriented mapping.

        Returns
        -------
        dict[str, list[Any]]
            Mapping of column name to values across every row, with factors
            propagated down to descendant levels and nulled elsewhere.
        """
        # Reserved columns are block-local: each block supplies its own, and a
        # block that omits one carries null for it.
        rows: dict[str, list[Any]] = {name: [] for block in self.blocks for name in block.columns}
        for block in self.blocks:
            for name in rows:
                rows[name].extend(column_values(block.columns.get(name), block.size))

        # Factors are level-local and propagate downwards, which is exactly what
        # RowLayout.expand does — so the gather lives in one place. Assignment rather
        # than merge is safe because __post_init__ has already rejected a name declared
        # at two levels, so no two iterations write the same key.
        layout = self.layout
        for level, factors in self.factors.items():
            for name, values in factors.items():
                rows[name] = layout.expand(values, level)
        return rows

"""Factor info and level schema shared by the metadata structuring layer."""

__all__ = ["FactorLevel", "FactorLevelSchema", "FactorInfo"]

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

# Every level name the schema recognizes.
#
# A :obj:`~typing.Literal` rather than an enum because the string *is* the value:
# it lands directly in the dataframe's ``level`` column and is compared there. An
# enum-typed parameter would also reject the plain ``rows_at("unit")`` spelling
# that :class:`~dataeval.Metadata` is designed around, and ``enum.StrEnum`` is
# unavailable on the supported 3.10 floor.
FactorLevel: TypeAlias = Literal["sequence", "unit", "track", "instance"]

# The level vocabulary and its edges — the single place both are declared.
#
# Keys are every level, **in canonical order**: coarsest to finest. That order is
# load-bearing. It defines the row block ordering in the metadata dataframe and the
# notion of a "highest" level used when a factor's level has to be inferred, so
# iterate this mapping wherever ordered levels are needed. It must stay a
# topological order of the edges below: every level after its parents.
#
# Values are each level's parents, empty for a root. Factors propagate *downwards*
# along these edges (sequence -> unit -> instance) and never upwards; rows at a level
# above a factor's own level simply carry null values for it.
#
# This is a **directed acyclic graph, not a tree**: ``instance`` declares two parents.
# A per-frame detection sits inside both a frame and a track, and those are siblings
# under a sequence, so the graph is a diamond rather than a chain.
# :meth:`FactorLevelSchema.of` collapses edges through levels a task omits, so a schema
# that keeps only part of the graph still sees the right parents: an image-based task
# keeps neither ``sequence`` nor ``track``, and its ``instance`` level correctly reports
# ``unit`` as its only parent.
#
# Two consequences of the diamond, both load-bearing:
#
# - ``unit`` and ``track`` are **siblings**, so neither propagates to the other. A
#   per-frame factor is not readable from track rows and a per-track factor is not
#   readable from frame rows; :meth:`FactorLevelSchema.propagates_to` is what says so,
#   and :class:`~dataeval.Metadata` drops such a factor from factor analysis at that
#   view rather than inventing a value for it.
# - A row may be **missing** one parent while having the other. A detection that no
#   tracker linked has a frame but no track, so factors defined at ``track`` have no
#   value on it at all. That absence is carried positionally, as a negative parent
#   position — see ``RowBlock.ancestor_pos`` in the structuring layer.
#
# ``unit`` is one element of media: an image, an audio clip, a text document, a
# tabular row. For a task whose dataset item is an ordered run of them it is one
# frame or window of that run, which is why it is named for its role rather than
# for any medium. ``Structurer.unit_type`` carries the medium's own word for it.
#
# ``instance`` is one labelled thing inside an item: a detection for object detection
# or multi-object tracking, the image itself for whole-image classification. Every task
# shares it, so the same object keeps one level name whichever view produced it — a
# detection in an object detection dataset, the same detection seen through
# :class:`~dataeval.data.DetectionCrops`, and a per-frame detection in a tracking
# dataset are all instances.
#
# ``sequence`` is a video: one dataset item holding an ordered run of frames. It exists
# so that ``unit`` can mean "a frame" without also having to mean "a dataset item" —
# for multi-object tracking the item level is ``sequence`` and ``unit`` sits *between*
# the item level and the label level, which no image-based task needs.
#
# ``track`` is one tracked object across a sequence: the identity a tracker assigns, of
# which each instance is one observation. It is a level rather than a column so that
# metadata can be organized *by track* — a per-track factor is stored once per track and
# propagates down to every detection in it, and ``rows_at("track")`` reads it once per
# track instead of once per detection.
_FACTOR_LEVEL_HIERARCHY: Mapping[FactorLevel, tuple[FactorLevel, ...]] = MappingProxyType(
    {
        "sequence": (),
        "unit": ("sequence",),
        "track": ("sequence",),
        "instance": ("unit", "track"),
    },
)


def _as_parents(level: FactorLevel, value: Sequence[FactorLevel]) -> tuple[FactorLevel, ...]:
    """Normalize one level's declared parents to a tuple.

    A bare string is rejected rather than accepted as a single parent: ``str`` is
    itself a ``Sequence[str]``, so ``{"instance": "unit"}`` would silently become
    four single-character parents.
    """
    if isinstance(value, str):
        raise TypeError(
            f"Parents of {level!r} must be a sequence of level names, not the bare string "
            f"{value!r}; pass ({value!r},) for a single parent.",
        )
    return tuple(value)


def _closure(start: FactorLevel, parents: Mapping[FactorLevel, tuple[FactorLevel, ...]]) -> tuple[FactorLevel, ...]:
    """Every level reachable upwards from ``start``, nearest first.

    Breadth-first so that a diamond reports the levels nearest the start before the
    level where the branches meet, and reports each level once however many paths
    lead to it.
    """
    reached: list[FactorLevel] = []
    queue = deque(parents.get(start, ()))
    while queue:
        current = queue.popleft()
        if current in reached:
            continue
        reached.append(current)
        queue.extend(parents.get(current, ()))
    return tuple(reached)


def _relink(
    level: FactorLevel,
    keep: set[FactorLevel],
    hierarchy: Mapping[FactorLevel, tuple[FactorLevel, ...]],
) -> tuple[FactorLevel, ...]:
    """Nearest kept ancestors of ``level``, following every edge through dropped ones.

    A level a schema leaves out splices its edges rather than severing them: the walk
    continues up through it to whatever is kept above. Following *all* parents, not
    just the first, is what keeps both branches of a diamond when the level where they
    part is dropped.
    """
    nearest: list[FactorLevel] = []
    seen: set[FactorLevel] = set()
    queue = deque(hierarchy.get(level, ()))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        if current in keep:
            nearest.append(current)
        else:
            queue.extend(hierarchy.get(current, ()))
    return tuple(nearest)


def _validate_levels(levels: Sequence[FactorLevel]) -> None:
    """Reject unknown or repeated level names."""
    unknown = sorted(set(levels) - set(_FACTOR_LEVEL_HIERARCHY))
    if unknown:
        raise ValueError(f"Unknown level(s) {unknown}. Valid levels are {list(_FACTOR_LEVEL_HIERARCHY)}.")
    duplicated = sorted({level for level in levels if list(levels).count(level) > 1})
    if duplicated:
        raise ValueError(f"Level(s) {duplicated} appear more than once in the schema.")


def _validate_parents(levels: Sequence[FactorLevel], parents: Mapping[FactorLevel, tuple[FactorLevel, ...]]) -> None:
    """Reject parents that are unknown, absent from the schema, or repeated.

    An unknown parent would otherwise sit in the graph unnoticed until
    :meth:`FactorLevelSchema.ancestors` walked into it.
    """
    named = {parent for edges in parents.values() for parent in edges}
    unknown = sorted(named - set(_FACTOR_LEVEL_HIERARCHY))
    if unknown:
        raise ValueError(f"Unknown parent level(s) {unknown}. Valid levels are {list(_FACTOR_LEVEL_HIERARCHY)}.")
    dangling = sorted(named - set(levels))
    if dangling:
        raise ValueError(
            f"Parent level(s) {dangling} are not part of this schema. Every parent must also be "
            f"one of the schema's levels {list(levels)}; use FactorLevelSchema.of to re-link parents "
            "around omitted levels automatically.",
        )
    repeated = sorted({level for level, edges in parents.items() if len(set(edges)) != len(edges)})
    if repeated:
        raise ValueError(f"Level(s) {repeated} declare the same parent more than once.")


def _children_map(
    levels: Sequence[FactorLevel], parents: Mapping[FactorLevel, tuple[FactorLevel, ...]]
) -> dict[FactorLevel, list[FactorLevel]]:
    """Invert the parent edges, keyed by parent, in schema order."""
    children: dict[FactorLevel, list[FactorLevel]] = {}
    for level in levels:
        for parent in parents[level]:
            children.setdefault(parent, []).append(level)
    return children


def _validate_acyclic(levels: Sequence[FactorLevel], parents: Mapping[FactorLevel, tuple[FactorLevel, ...]]) -> None:
    """Reject a parent graph containing a cycle.

    Unreachable with a single parent per level, but a level may name several, and a
    cycle would make :meth:`FactorLevelSchema.ancestors` and every propagation walk
    non-terminating. Kahn's algorithm: peel off levels whose parents are all
    resolved; whatever will not peel is in, or below, a cycle.
    """
    indegree: dict[FactorLevel, int] = {level: len(parents[level]) for level in levels}
    children = _children_map(levels, parents)

    queue: deque[FactorLevel] = deque(level for level, degree in indegree.items() if degree == 0)
    resolved = 0
    while queue:
        for child in children.get(queue.popleft(), []):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
        resolved += 1

    if resolved != len(levels):
        cyclic = sorted(level for level in levels if indegree[level] > 0)
        raise ValueError(f"Level(s) {cyclic} form a cycle; the level graph must be acyclic.")


class FactorLevelSchema:
    """Ordered subset of the canonical levels with parent/child relationships.

    A schema describes the levels a particular task actually produces rows for, and
    how they are wired. Levels absent from the schema are elided from the hierarchy:
    an object detection image dataset uses ``("unit", "instance")``, so ``instance``
    reports ``unit`` as its only parent.

    Parameters
    ----------
    levels : Sequence[str]
        Levels in the schema, ordered coarsest to finest.
    parents : Mapping[str, Sequence[str]]
        Mapping of each level to its parents within this schema, empty for roots. A
        level may name more than one parent; a bare string is rejected, since ``str``
        is itself a sequence of strings.

    Raises
    ------
    ValueError
        When a level or parent is not one of the canonical levels, when a level
        is repeated, when a parent is not itself one of ``levels``, when a level
        names the same parent twice, or when the edges form a cycle.
    TypeError
        When a level's parents are given as a bare string.

    Example
    -------
    >>> schema = FactorLevelSchema.of("sequence", "unit", "track", "instance")
    >>> schema.parents_of("instance")
    ('unit', 'track')
    >>> schema.ancestors("instance")
    ('unit', 'track', 'sequence')

    Siblings do not propagate to each other, so a per-frame factor cannot be read from
    track rows:

    >>> schema.propagates_to("unit", "track")
    False
    >>> schema.propagates_to("unit", "instance")
    True

    Omitting a level splices the graph rather than severing it, so an image-based task
    still sees one parent:

    >>> FactorLevelSchema.of("unit", "instance").parents_of("instance")
    ('unit',)
    """

    def __init__(
        self,
        levels: Sequence[FactorLevel],
        parents: Mapping[FactorLevel, Sequence[FactorLevel]],
    ) -> None:
        _validate_levels(levels)
        resolved: dict[FactorLevel, tuple[FactorLevel, ...]] = {
            level: _as_parents(level, parents.get(level, ())) for level in levels
        }
        _validate_parents(levels, resolved)
        _validate_acyclic(levels, resolved)

        # Plain dicts, not MappingProxyType: a schema travels inside every Metadata
        # instance, and a mappingproxy cannot be pickled, which makes the whole
        # instance un-deep-copyable. Immutability is expressed through the read-only
        # properties below instead.
        self._levels: tuple[FactorLevel, ...] = tuple(levels)
        self._parents: dict[FactorLevel, tuple[FactorLevel, ...]] = resolved
        self._index: dict[FactorLevel, int] = {level: i for i, level in enumerate(self._levels)}

    @classmethod
    def of(cls, *levels: FactorLevel) -> "FactorLevelSchema":
        """Build a schema from canonical levels, re-linking parents around omitted levels.

        Parameters
        ----------
        *levels : str
            Canonical level names, in any order.

        Returns
        -------
        FactorLevelSchema
            Schema containing the requested levels in canonical order, with the
            canonical edges collapsed through whatever was left out.

        Raises
        ------
        ValueError
            When a level is unknown or supplied more than once.

        Notes
        -----
        Omitting a level splices the graph rather than severing it: every edge that
        ran *through* the omitted level is re-linked to the nearest kept level above
        it, following all of its parents. A schema that keeps only the ends of a
        diamond therefore still sees both branches meet.
        """
        # Validate the arguments as given, before de-duplicating: a repeated level is a
        # caller bug worth reporting, and a set would hide it from the duplicate check.
        _validate_levels(levels)
        requested = set(levels)

        # Iterating the canonical hierarchy, not `levels`, is what imposes canonical order on
        # the result regardless of the order the caller passed. Annotated because the
        # comprehension would otherwise widen Level's literals to str.
        included: tuple[FactorLevel, ...] = tuple(level for level in _FACTOR_LEVEL_HIERARCHY if level in requested)
        parents: dict[FactorLevel, tuple[FactorLevel, ...]] = {
            level: _relink(level, requested, _FACTOR_LEVEL_HIERARCHY) for level in included
        }
        return cls(included, parents)

    def __contains__(self, level: object) -> bool:
        return level in self._index

    def __iter__(self) -> Iterator[FactorLevel]:
        return iter(self._levels)

    def __len__(self) -> int:
        return len(self._levels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FactorLevelSchema):
            return NotImplemented
        return self._levels == other._levels and self._parents == other._parents

    def __hash__(self) -> int:
        return hash((self._levels, tuple(sorted(self._parents.items(), key=lambda kv: kv[0]))))

    def __repr__(self) -> str:
        return f"FactorLevelSchema({list(self._levels)})"

    @property
    def levels(self) -> tuple[FactorLevel, ...]:
        """Levels in this schema, ordered coarsest to finest."""
        return self._levels

    @property
    def parents(self) -> Mapping[FactorLevel, tuple[FactorLevel, ...]]:
        """Read-only view of each level's parents, empty for roots."""
        return MappingProxyType(self._parents)

    def parents_of(self, level: FactorLevel) -> tuple[FactorLevel, ...]:
        """Immediate parents of a level, empty when the level is a root."""
        return self._parents[self.validate(level)]

    def ancestors(self, level: FactorLevel) -> tuple[FactorLevel, ...]:
        """Ancestors of a level, nearest first.

        Parameters
        ----------
        level : str
            Level to walk upwards from.

        Returns
        -------
        tuple[str, ...]
            Every level above this one, breadth-first from its immediate parents
            outwards, each appearing once however many paths reach it.
        """
        return _closure(self.validate(level), self._parents)

    def paths(self, level: FactorLevel, ancestor: FactorLevel) -> tuple[tuple[FactorLevel, ...], ...]:
        """Every upward route from ``level`` to ``ancestor``, as sequences of edges.

        A chain has one route and needs none of this. A diamond has two, and they can
        disagree in exactly one way: a row may reach the meeting point along one branch
        and not the other — a detection no tracker linked has a frame, and through it a
        sequence, but no track. So "which route" is a real question with a real answer,
        and it belongs to the graph rather than to whichever structurer last wrote a
        dictionary literal.

        Parameters
        ----------
        level : str
            Level to walk upwards from.
        ancestor : str
            Level to reach.

        Returns
        -------
        tuple[tuple[str, ...], ...]
            One entry per route, each listing the levels stepped through *after*
            ``level``, ending at ``ancestor``. Empty when ``ancestor`` is not above
            ``level``. Routes are ordered by :meth:`parents_of` at each step, which is
            canonical order, so the first route is the one to prefer where several are
            total — see :meth:`~dataeval.Metadata` for how that preference is applied.

        Raises
        ------
        ValueError
            When either level is not part of this schema. A level this schema does not
            have is a different question from a level it has but cannot reach: the first
            is a caller error and raises, the second is a real answer and is the empty
            tuple.

        Examples
        --------
        >>> schema = FactorLevelSchema.of("sequence", "unit", "track", "instance")
        >>> schema.paths("instance", "unit")
        (('unit',),)
        >>> schema.paths("instance", "sequence")
        (('unit', 'sequence'), ('track', 'sequence'))
        """
        self.validate(level)
        self.validate(ancestor)
        if level == ancestor:
            return ()
        routes: list[tuple[FactorLevel, ...]] = []
        for parent in self._parents[level]:
            if parent == ancestor:
                routes.append((parent,))
                continue
            routes.extend((parent, *rest) for rest in self.paths(parent, ancestor))
        return tuple(routes)

    def descendants(self, level: FactorLevel) -> tuple[FactorLevel, ...]:
        """Levels that inherit from ``level``, in schema order."""
        self.validate(level)
        return tuple(other for other in self._levels if level in self.ancestors(other))

    def is_ancestor(self, candidate: FactorLevel, level: FactorLevel) -> bool:
        """Whether ``candidate`` sits above ``level`` by any path.

        Parameters
        ----------
        candidate : str
            Possible ancestor level.
        level : str
            Level whose ancestry is checked.

        Returns
        -------
        bool
            True when factors defined at ``candidate`` propagate down to ``level``.
            One path is enough: a level with two parents inherits from both.
        """
        return self.validate(candidate) in self.ancestors(level)

    def propagates_to(self, source: FactorLevel, target: FactorLevel) -> bool:
        """Whether values defined at ``source`` are visible on ``target`` rows."""
        return source == target or self.is_ancestor(source, target)

    def highest(self, levels: Iterable[FactorLevel]) -> FactorLevel:
        """Coarsest level among ``levels``, by schema order.

        Raises
        ------
        ValueError
            When ``levels`` is empty.

        Notes
        -----
        Schema order, not the parent graph: the graph is a partial order, so two
        levels on different branches are genuinely incomparable and "coarsest" would
        have no answer. Schema order is a topological order of the same edges, so an
        ancestor always wins over its descendants, and two incomparable levels resolve
        to whichever was declared first — deterministic, and a tie-break rather than a
        claim about the graph. Callers that need to know whether the choice mattered
        should ask :meth:`is_ancestor` directly.

        There is deliberately no ``lowest``. "Coarsest" has a safe tie-break because
        this is only ever asked of levels a value could have come from, and an
        ancestor is always a defensible answer. "Finest" has no such fallback: two
        sibling levels are incomparable, and picking the later-declared one would be
        a silent, arbitrary choice about *which* rows a caller gets. Anything that
        needs a finest level should name it.
        """
        candidates: list[FactorLevel] = [self.validate(level) for level in levels]
        if not candidates:
            raise ValueError("Cannot select the highest level from an empty collection.")
        return min(candidates, key=self._index.__getitem__)

    def validate(self, level: str) -> FactorLevel:
        """Return ``level`` when it belongs to this schema, otherwise raise.

        A schema contains only levels that exist. Retired spellings are translated by
        the caller before they get here, so that this stays a plain membership test
        and the deprecation warning points at the user's line.

        Parameters
        ----------
        level : str
            Level name to check. Typed as ``str`` rather than
            :data:`~dataeval.types.FactorLevel` on purpose: this is the runtime gate a
            name from outside the type system has to pass, and a signature that only
            accepted names already known to be valid could not be that gate.

        Returns
        -------
        str
            The validated level name.

        Raises
        ------
        ValueError
            When the level is not part of this schema.
        """
        for known in self._levels:
            if known == level:
                return known
        raise ValueError(f"Unknown level {level!r} for this dataset. Available levels are {list(self._levels)}.")


@dataclass
class FactorInfo:
    """Type information and provenance for a single metadata factor.

    Attributes
    ----------
    factor_type : {"categorical", "continuous", "discrete"}
        How the factor's values are treated during analysis.
    is_binned : bool, default False
        Whether a binned companion column was generated for this factor.
    is_digitized : bool, default False
        Whether a digitized companion column was generated for this factor.
    level : str, default "unit"
        Level the factor is defined at, drawn from the bound dataset's level
        schema (one of ``sequence``, ``unit``, ``track``, ``instance``). This is also the level the factor
        was binned at: its bin edges, its bin count and its continuous/discrete
        verdict all come from its values here, one per entity.
    aggregated_from : str or None, default None
        Level whose rows were rolled up to produce this factor, for one built by
        :meth:`~dataeval.Metadata.agg`, and None for one measured directly. The level
        only, not the operation: what distinguishes an aggregate from a measurement is
        that its values describe a *set* of finer rows, which is what a reader has to
        know before comparing it against a factor measured at ``level`` itself.
    """

    factor_type: Literal["categorical", "continuous", "discrete"]
    is_binned: bool = False
    is_digitized: bool = False
    level: FactorLevel = "unit"
    aggregated_from: FactorLevel | None = None

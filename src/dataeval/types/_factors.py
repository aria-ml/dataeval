"""Factor info and level schema shared by the metadata structuring layer."""

__all__ = [
    "AggregationRecord",
    "Aggregator",
    "BinSpec",
    "ClassAxis",
    "FactorInfo",
    "FactorLevel",
    "FactorLevelSchema",
    "LevelSpec",
    "ParseDateTime",
    "ParseValue",
    "Remap",
    "Rescale",
    "Unusable",
]

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, get_args

# Every level name the schema recognizes.
#
# A :obj:`~typing.Literal` rather than an enum because the string *is* the value:
# it lands directly in the dataframe's ``level`` column and is compared there. An
# enum-typed parameter would also reject the plain ``rows_at("unit")`` spelling
# that :class:`~dataeval.Metadata` is designed around, and ``enum.StrEnum`` is
# unavailable on the supported 3.10 floor.
FactorLevel: TypeAlias = Literal["sequence", "unit", "track", "instance"]

# Distinct values an ``Unusable`` repr shows per kind before abbreviating. The attribute
# keeps all of them; this is only how many are worth reading at a glance.
_SHOWN_VALUES = 8

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

    def routes_through(
        self, level: FactorLevel, ancestor: FactorLevel, via: FactorLevel
    ) -> tuple[tuple[FactorLevel, ...], ...]:
        """Routes from ``level`` to ``ancestor`` that step through ``via``.

        ``via`` names a level a route passes *through*, so ``level`` itself is not an
        answer — every route starts there — and saying so is clearer than the "no route
        passes through it" the general check would otherwise give. ``ancestor`` is an
        answer, trivially satisfied by every route, and is left alone rather than made a
        special case.

        A question about the graph, so it is answered by the graph. Both the store, which
        selects the links to compose, and :meth:`Aggregator.validate`, which has no store,
        ask it here — which is what lets a declaration be wrong about its route before any
        dataset is in hand.

        Returns
        -------
        tuple[tuple[str, ...], ...]
            The routes, in :meth:`paths` order.

        Raises
        ------
        ValueError
            When a level is not in this schema, when ``via`` is ``level``, or when no route
            passes through it.
        """
        self.validate(via)
        if via == level:
            raise ValueError(
                f"via={via!r} is the level being linked from, which every route starts at rather "
                f"than passes through. Name a level between {level!r} and {ancestor!r}, or omit "
                "via to take every route.",
            )
        paths = self.paths(level, ancestor)
        if selected := tuple(path for path in paths if via in path):
            return selected
        raise ValueError(
            f"No route from {level!r} to {ancestor!r} passes through {via!r}. The levels these "
            f"routes step through are {sorted({step for path in paths for step in path})}.",
        )

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


@dataclass(frozen=True)
class BinSpec:
    """How one factor's values were cut into codes.

    A bin edge is a claim about the world -- *below 0 °C is freezing*, *over 500 px is a
    large object* -- so where the cuts fell, and who chose them, is part of what a result
    computed from the factor means. Recording it is what lets a caller inspect the policy,
    apply the same one to data arriving later, and tell a cut they chose from one
    :class:`~dataeval.Metadata` derived on their behalf.

    Attributes
    ----------
    edges : tuple of float
        Cut points actually applied, ascending. The outer edges are ``-inf`` and ``+inf``
        wherever DataEval placed them itself, which is what makes a value outside the
        observed range land in an end bin rather than in a code of its own. An explicit
        edge list is recorded as given, infinite or not.
    provenance : {"derived", "accepted", "count", "edges"}
        Who chose what. ``"derived"``: DataEval chose both the count and the placement,
        and nobody has reviewed it. ``"accepted"``: derived placement that a person
        inspected and ratified. ``"count"``: the caller asked for a number of bins and
        DataEval chose where they fell. ``"edges"``: the caller said where to cut.
    method : {"uniform_width", "uniform_count", "clusters"} or None
        How the placement was chosen, or None when the caller supplied the edges outright.

    Notes
    -----
    Codes run from 0 to ``len(edges)`` inclusive, with :attr:`missing_code` above them.
    Code 0 means *below the first edge* and ``len(edges)`` means *at or above the last*;
    both are unreachable when the corresponding outer edge is infinite, so an
    infinitely-bounded spec leaves them empty rather than absent. The span is fixed by the
    edges rather than by which codes a particular sample happened to fill, so the same spec
    assigns the same code to the same value in any dataset -- which is what makes a
    recorded encoding reusable.
    """

    edges: tuple[float, ...]
    provenance: Literal["derived", "accepted", "count", "edges"]
    method: Literal["uniform_width", "uniform_count", "clusters"] | None = None

    @property
    def missing_code(self) -> int:
        """Code standing for a missing value, above every code the edges can produce.

        A missing value is not a small value, a large value, or a value between two edges,
        so it cannot share a bin with observed data without distorting whatever reads the
        result. Its position is an artifact of needing one -- it sits at the top because
        the codes have to go somewhere, not because missing is large.
        """
        return len(self.edges) + 1


@dataclass(frozen=True)
class LevelSpec:
    """Which value each code stands for, in code order.

    Recorded for the same reason as :class:`BinSpec`, and answering a second question
    besides: a code is only stable if something remembers what it meant. Reading the map
    back off the current data -- which is what ``np.unique`` does -- renumbers the factor
    whenever the set of observed values changes.

    Attributes
    ----------
    levels : tuple
        Value per code: code ``i`` stands for ``levels[i]``. Sorted when the factor is
        first structured, so the codes match what sorting the observed values would give.
        **Append-only afterwards**: a value not already present takes the next code and
        goes at the end, out of sort order, so that codes already assigned keep their
        meaning. Anything displaying levels sorts by the value rather than by the code.
    provenance : {"derived", "accepted", "declared"}
        Who chose the vocabulary. ``"derived"``: read off the values during structuring,
        unreviewed. ``"accepted"``: derived and then ratified by a person. ``"declared"``:
        supplied by the caller, so the codes were fixed before any data was seen.
    """

    levels: tuple[Any, ...]
    provenance: Literal["derived", "accepted", "declared"]

    @property
    def missing_code(self) -> int:
        """Code standing for a missing value, above every code the vocabulary describes.

        Derived rather than stored, exactly as :attr:`BinSpec.missing_code` is, so the code
        space is fixed by the vocabulary rather than by which codes a particular sample
        happened to fill. Two datasets recorded against one spec therefore share an
        alphabet whether or not either has a missing value in it, and a filter that removes
        the last such value does not change the factor's cardinality.

        A missing value is not one of the values, so it does not become a level: a
        vocabulary is what a code *means*, and "not recorded" means the same thing for every
        factor rather than something about this one.
        """
        return len(self.levels)


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
    encoding : BinSpec or LevelSpec or None, default None
        The map from values to codes: a :class:`BinSpec` where the factor was binned, a
        :class:`LevelSpec` where it was digitized. One field rather than two, because a
        factor has exactly one map and ``is_binned``/``is_digitized`` already say which
        kind it is. None only for a factor that reached neither path.

        Where the other fields record *that* a factor was encoded, this records *how* --
        which is what a reader needs to say what a code means, and what a later dataset
        needs to be given the same codes for the same values.
    missing : int, default 0
        Rows that recorded no value, and so hold the encoding's reserved missing code
        rather than one standing for a value.

        Worth reading before a bias result is: no evaluator treats that code differently
        from any other, so unrecorded rows are scored as a group of their own, sitting in
        every contingency table beside the groups the factor actually names. That is the
        honest default -- dropping them would quietly change the population a result
        describes -- but it is only honest if the number is available, and it is the one
        thing neither the vocabulary nor the edges say.
    """

    factor_type: Literal["categorical", "continuous", "discrete"]
    is_binned: bool = False
    is_digitized: bool = False
    level: FactorLevel = "unit"
    aggregated_from: FactorLevel | None = None
    encoding: BinSpec | LevelSpec | None = None
    missing: int = 0


@dataclass(frozen=True)
class Remap:
    """Replace named values, whole ranges of them, or everything left unnamed.

    The correction for a column whose values do not agree about what they are: a compass
    recorded sometimes in degrees and sometimes as a bearing, a sentinel band standing for
    a bad reading, a category set that wants collapsing. The mapping *is* the correction —
    what is declared and what is recorded are one object — so it can be reviewed in a diff
    and reapplied to the next dataset without being re-decided.

    Attributes
    ----------
    factor : str
        Factor the mapping applies to.
    mapping : Mapping
        What each value becomes. A key is one of three things:

        - **a value**, matched exactly as the dataset wrote it;
        - **a ``(low, high)`` range**, half-open ``[low, high)``, matching any number in
          it — which is how a sentinel band is retired to one value, and the thing
          :class:`Rescale` cannot express, since it transforms values rather than
          replacing them. ``None`` at either end is unbounded;
        - **``None``**, the catch-all for every value no other key matched. It is what
          lets a recorded mapping survive a second dataset, which will bring values the
          first never held.

        A value matched by nothing, where there is no catch-all, is **left as it was** —
        so a partial mapping is a partial mapping, and a column it leaves mixed simply
        stays unusable and says so.

        A row that recorded *nothing* is never matched, catch-all included. Absence is not
        a value, and it keeps the reserved missing code that says a reading was not taken
        rather than becoming one that says the mapping did not name it.
    provenance : {"declared"}, default "declared"
        Always ``"declared"``: a repair is somebody's decision, because DataEval never
        guesses at a column whose values disagree. Recorded so the descriptor says so
        rather than leaving a reader to infer it.
    """

    factor: str
    mapping: Mapping[Any, Any]
    provenance: Literal["declared"] = "declared"

    def __post_init__(self) -> None:
        """Normalize the mapping and refuse the shapes no dataset is needed to reject."""
        if not self.factor:
            raise ValueError("A remap needs a factor name, e.g. Remap('direction', {'N': 0}).")
        if not self.mapping:
            raise ValueError(
                f"Remap({self.factor!r}, {{}}) names nothing to replace. Give it at least one "
                f"value, range or a None catch-all.",
            )
        for key in self.mapping:
            if isinstance(key, tuple):
                _validate_range(key, f"Remap({self.factor!r})")
        object.__setattr__(self, "mapping", MappingProxyType(dict(self.mapping)))

    def __hash__(self) -> int:
        """Hash the declaration, reading the mapping as the pairs it holds.

        ``frozen=True`` generates a hash over every field and one of them is a mapping, so
        a record whose whole purpose is being stored and compared between runs could be
        compared but never put in a set -- and would say so only at the call.
        """
        return hash((self.factor, tuple(sorted(self.mapping.items(), key=repr)), self.provenance))


@dataclass(frozen=True)
class Rescale:
    """Apply ``value * multiply + add`` to the values in a range.

    The correction for a column that is readable but in the wrong units: a run of altitudes
    in feet among metres, a sensor whose readings carry a constant offset, a depth field
    that switched to millimetres partway through a collection.

    One affine form rather than four operations, because ``multiply`` covers multiply and
    divide, ``add`` covers add and subtract, and multiplying before adding is the order
    every unit conversion is already written in.

    Attributes
    ----------
    factor : str
        Factor the adjustment applies to.
    over : tuple
        Half-open ``[low, high)`` range of values it applies to, matching the convention
        binning already uses. ``None`` at either end is unbounded, so ``(None, None)`` is
        every value and ``(1000, None)`` is everything from 1000 up. A value outside the
        range is left exactly as it was.
    multiply : float, default 1.0
        Factor applied first. Use its reciprocal to divide.
    add : float, default 0.0
        Offset applied after multiplying. Use a negative number to subtract.
    provenance : {"declared"}, default "declared"
        Always ``"declared"``, for the reason :class:`Remap` gives.
    """

    factor: str
    over: tuple[float | None, float | None] = (None, None)
    multiply: float = 1.0
    add: float = 0.0
    provenance: Literal["declared"] = "declared"

    def __post_init__(self) -> None:
        """Refuse the shapes no dataset is needed to reject."""
        if not self.factor:
            raise ValueError("A rescale needs a factor name, e.g. Rescale('altitude', multiply=0.3048).")
        _validate_range(self.over, f"Rescale({self.factor!r})")
        if self.multiply == 0:
            raise ValueError(
                f"Rescale({self.factor!r}, multiply=0) gives every value in range the same "
                f"answer, which discards the readings rather than adjusting them. Use "
                f"Remap({self.factor!r}, {{{self.over!r}: {self.add!r}}}) to say that outright.",
            )


def _validate_range(over: Any, context: str) -> None:
    """Check a half-open range, wherever one is written.

    Raises
    ------
    ValueError
        When the range is not a pair, or its bounds are the wrong way round.
    """
    if not isinstance(over, tuple) or len(over) != 2:
        raise ValueError(f"{context} range {over!r} must be a (low, high) pair; None at either end is unbounded.")
    low, high = over
    if low is not None and high is not None and low > high:
        raise ValueError(f"{context} range {over!r} runs backwards: its low bound is above its high bound.")


# Periods a timestamp can be read as, coarsest first, in two families.
#
# A bare name is an **absolute** period and runs once: "2020-08" happened, and no later
# month is it. An ``x_of_y`` name is a **recurring position** and comes round again: every
# collection has a 14:00. The distinction is the whole reason both are here -- a dataset
# split by time separates perfectly on any absolute period, which is a restatement of the
# split rather than a finding, while the recurring ones stay comparable across the split.
#
# A closed vocabulary rather than a free-form pattern because each one has to name a bucket
# every reader agrees on: a period nobody can spell twice is not one a second collection
# could be grouped by.
DateTimeGranularity: TypeAlias = Literal[
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "month_of_year",
    "day_of_week",
    "hour_of_day",
]

# Spelled once, read by the record that validates against it and the reader that applies it.
# Taken off the alias rather than repeated beside it, so the annotation a type checker reads
# and the tuple the constructor checks against cannot drift apart.
DATETIME_GRANULARITIES: tuple[str, ...] = get_args(DateTimeGranularity)


# Units a number can count since the Unix epoch in. Declared rather than guessed, because
# the same integer is a plausible reading in every one of them -- 1_700_000_000 is a moment
# in 2023 read as seconds and one in 1970 read as milliseconds, and nothing about the column
# says which was meant. Seconds is the default because it is what this record *emits* when
# no period is asked for, so a reading round-trips through its own output.
EpochUnit: TypeAlias = Literal["s", "ms", "us", "ns"]

EPOCH_UNITS: tuple[str, ...] = get_args(EpochUnit)

# What one of each unit is worth in seconds.
EPOCH_SECONDS: Mapping[str, float] = MappingProxyType({"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9})


def _validated_drops(factor: str, drop: Sequence[str]) -> tuple[str, ...]:
    """Read a parse's drops as the tuple of substrings they have to be.

    Raises
    ------
    ValueError
        When the drops are a bare string, or any of them is not a non-empty one.
    """
    if isinstance(drop, str):
        raise ValueError(
            f"ParseValue({factor!r}, drop={drop!r}) reads as a sequence of substrings, and a bare string "
            f"is ambiguous: {drop!r} could be one substring or {len(drop)} characters. Write the "
            f"list you mean, e.g. drop=[{drop!r}].",
        )
    drops = tuple(drop)
    if any(not isinstance(item, str) or not item for item in drops):
        raise ValueError(
            f"ParseValue({factor!r}) drops {list(drops)!r}: every entry must be a non-empty string. An "
            f"empty one occurs everywhere and would mean nothing.",
        )
    return drops


def _validate_separator(factor: str, decimal: str, drops: tuple[str, ...]) -> None:
    """Check a parse's decimal separator against the drops it is read alongside.

    Raises
    ------
    ValueError
        When the separator is not one character, is itself dropped, or the rule as a whole
        would leave the values exactly as they are already read.
    """
    if len(decimal) != 1:
        raise ValueError(
            f"ParseValue({factor!r}, decimal={decimal!r}) must name a single character, e.g. decimal=','.",
        )
    if decimal in drops:
        raise ValueError(
            f"ParseValue({factor!r}) drops {decimal!r} and also reads it as the decimal separator. Drop "
            f"it or read it, not both.",
        )
    if not drops and decimal == ".":
        raise ValueError(
            f"ParseValue({factor!r}) drops nothing and reads '.' as the separator, which is how the "
            f"values are already read. Name what to remove, e.g. drop=[','].",
        )


@dataclass(frozen=True)
class ParseValue:
    """Read text as a value by removing what is not part of it.

    The correction for a column whose numbers are wearing decoration: a thousands separator,
    a unit written into the cell, a degree sign, a decimal comma. The values *are* numbers —
    nothing about them is in doubt — but no reading of the text finds them until the
    decoration is named, so the column is held back for mixing numbers with text.

    Declared as data, never as a function, for the same reason
    :class:`Remap` is: a correction is committed alongside code, read back months later and
    reapplied to the next collection. A rule that removes ``","`` says what it does in a
    diff, and does the same thing on a dataset this one never saw --- which is the whole
    difference between this and a mapping that would have to name every value in advance.

    This does not decide what the cleaned text *becomes*. The values go back through the
    same reading every column gets, so text that now spells a number is stored as one and
    text that still does not is left as text --- and a column its leftovers keep mixed stays
    unusable and says so, exactly as :class:`Remap` leaves one.

    Attributes
    ----------
    factor : str
        Factor the rule applies to.
    drop : Sequence[str]
        Substrings removed from every value, in the order given. Each is removed wherever it
        occurs, so ``["kg"]`` reads ``"12kg"`` and ``"12 kg"`` alike once ``" "`` is dropped
        too. Substrings rather than a set of characters, so a rule that removes ``"kg"``
        cannot also eat the ``k`` of a value it was never meant to touch.
    decimal : str, default "."
        The character this column separates a fraction with, swapped for ``"."`` after the
        drops. One character, and never one that ``drop`` removes --- a rule that deletes the
        separator and then reads it is two decisions that contradict each other.
    provenance : {"declared"}, default "declared"
        Always ``"declared"``, for the reason :class:`Remap` gives.

    Raises
    ------
    ValueError
        When the factor is unnamed, when ``drop`` is given as a bare string rather than a
        sequence of them, when the rule as written would change nothing, or when ``decimal``
        is not a single character or is itself dropped.

    See Also
    --------
    ParseDateTime : Read text as a timestamp, which needs a calendar rather than a cleanup.
    Remap : Replace named values outright, where the vocabulary is small and closed.

    Examples
    --------
    A thousands separator, and a unit written into the cell:

    >>> ParseValue("count", drop=[","])
    ParseValue(factor='count', drop=(',',), decimal='.', provenance='declared')

    >>> ParseValue("weight", drop=[" ", "kg"])
    ParseValue(factor='weight', drop=(' ', 'kg'), decimal='.', provenance='declared')

    A column recorded in a locale that separates fractions with a comma:

    >>> ParseValue("span", decimal=",")
    ParseValue(factor='span', drop=(), decimal=',', provenance='declared')
    """

    factor: str
    drop: Sequence[str] = ()
    decimal: str = "."
    provenance: Literal["declared"] = "declared"

    def __post_init__(self) -> None:
        """Normalize the drops and refuse the shapes no dataset is needed to reject."""
        if not self.factor:
            raise ValueError("A parse needs a factor name, e.g. ParseValue('count', drop=[',']).")
        drop = _validated_drops(self.factor, self.drop)
        _validate_separator(self.factor, self.decimal, drop)
        object.__setattr__(self, "drop", drop)


@dataclass(frozen=True)
class ParseDateTime:
    """Read text as a timestamp, and optionally as the period it falls in.

    The correction for a column of timestamps, which is held back for a reason no cleanup
    fixes: nearly every row holds a different value, so the column names its rows rather
    than grouping them, and being text it has no order to be cut along. Reading it as a
    time gives it both --- an order, and a period each row belongs to.

    A calendar is what separates this from :class:`ParseValue`. Which characters to remove says
    nothing about whether ``03/04`` is March or April, when a week begins, or which rows
    share a quarter; those are questions only a format and a granularity answer.

    What the column becomes depends on ``every``, because the answers are different kinds
    of thing:

    - **An absolute period** --- ``"year"``, ``"quarter"``, ``"month"``, ``"week"``,
      ``"day"``, ``"hour"`` --- labels each row with the period it falls in, ``"2020-08"``
      for a month. A closed, readable vocabulary that groups rows, which is what the column
      was missing.
    - **A recurring position** --- ``"month_of_year"``, ``"day_of_week"``, ``"hour_of_day"``
      --- labels it with where in the cycle it sits, ``14`` for a frame flown at 14:20. Read
      back as the number it is, and cut into bins like any other ordered reading.
    - **``None``** keeps the instant itself, in seconds since the Unix epoch. Naive
      timestamps are read as UTC, so the same declaration gives the same number on any
      machine.

    Which family to reach for is decided by what the timestamp is being compared *across*.
    A dataset split by time --- a reference campaign and a later one --- separates perfectly
    on any absolute period, because the period *is* the split, and a factor that drifts by
    construction restates the question rather than answering it. The recurring positions
    survive that split intact: every campaign has a 14:00, so "the later campaign started
    flying earlier in the day" is a finding about collection conditions rather than about
    the calendar.

    A timestamp is not always text. This reads three spellings of one, so a column keeps its
    meaning however the dataset recorded it:

    - **Text** is read under ``format``, or as ISO 8601 where none is given.
    - **A number** is read as a count since the Unix epoch, in the unit ``epoch`` names.
      Which unit is a declaration rather than a guess: ``1_700_000_000`` is a moment in
      2023 read as seconds and one in 1970 read as milliseconds, and nothing about the
      column says which was meant.
    - **A** :class:`~datetime.datetime` **or** :class:`~datetime.date` is already a moment
      and is used as it stands. A bare date is read as midnight.

    Anything else is left exactly as it was.

    Attributes
    ----------
    factor : str
        Factor the reading applies to.
    format : str or None, default None
        How the text is spelled, as a :meth:`~datetime.datetime.strptime` pattern ---
        ``"%d/%m/%Y %H:%M"`` for a column no standard describes. ``None`` reads ISO 8601,
        which is what a timestamp that has been through JSON almost always is. Read only
        for values that are text.
    every : str or None, default None
        Period each row is labelled by, from the two families above. ``None`` keeps the
        instant. Weeks and weekdays are ISO, so a week belongs to the year holding its
        Thursday --- ``"2020-W35"`` --- and Monday is ``1`` through Sunday ``7``.
    epoch : {"s", "ms", "us", "ns"}, default "s"
        Unit a *numeric* value counts the epoch in. Seconds by default, which is what this
        record emits when ``every`` is ``None`` --- so a column it has already read comes
        back through it unchanged in meaning. Read only for values that are numbers, and a
        boolean is never one of them.
    provenance : {"declared"}, default "declared"
        Always ``"declared"``, for the reason :class:`Remap` gives.

    Raises
    ------
    ValueError
        When the factor is unnamed, or ``every`` or ``epoch`` is not one of the values
        named.

    See Also
    --------
    ParseValue : Read text as a value by removing what is not part of it.
    Remap : Replace named values outright, where the vocabulary is small and closed.

    Notes
    -----
    A value the format does not read is left exactly as it was, so a partial reading is a
    partial reading: a column its leftovers keep mixed stays unusable and says so, rather
    than being quietly completed by a rule nobody wrote.

    The label is handed back for the same reading every column gets, so one that spells a
    number is stored as one: ``every="year"`` gives the number 2020 and is cut into bins,
    while ``every="month"`` gives the category ``"2020-08"``. Both group the rows; they
    differ in whether the grouping carries an order.

    Examples
    --------
    Group a campaign's frames by the month they were flown:

    >>> ParseDateTime("date_time", every="month")
    ParseDateTime(factor='date_time', format=None, every='month', epoch='s', provenance='declared')

    Compare time of day across campaigns a year apart, which a month could not:

    >>> ParseDateTime("date_time", every="hour_of_day")
    ParseDateTime(factor='date_time', format=None, every='hour_of_day', epoch='s', provenance='declared')

    Keep the instant, to be cut into bins like any other ordered reading:

    >>> ParseDateTime("date_time")
    ParseDateTime(factor='date_time', format=None, every=None, epoch='s', provenance='declared')

    A column no standard describes:

    >>> ParseDateTime("logged", format="%d/%m/%Y %H:%M", every="day")
    ParseDateTime(factor='logged', format='%d/%m/%Y %H:%M', every='day', epoch='s', provenance='declared')

    A column of milliseconds since the epoch, as JavaScript and many logs record them:

    >>> ParseDateTime("logged_ms", epoch="ms", every="day")
    ParseDateTime(factor='logged_ms', format=None, every='day', epoch='ms', provenance='declared')
    """

    factor: str
    format: str | None = None
    every: DateTimeGranularity | None = None
    epoch: EpochUnit = "s"
    provenance: Literal["declared"] = "declared"

    def __post_init__(self) -> None:
        """Refuse the shapes no dataset is needed to reject."""
        if not self.factor:
            raise ValueError("A datetime reading needs a factor name, e.g. ParseDateTime('date_time').")
        if self.every is not None and self.every not in DATETIME_GRANULARITIES:
            raise ValueError(
                f"ParseDateTime({self.factor!r}, every={self.every!r}) is not a period this reads. "
                f"Use one of {', '.join(DATETIME_GRANULARITIES)}, or None to keep the instant.",
            )
        if self.epoch not in EPOCH_UNITS:
            raise ValueError(
                f"ParseDateTime({self.factor!r}, epoch={self.epoch!r}) is not a unit this counts in. "
                f"Use one of {', '.join(EPOCH_UNITS)}.",
            )


@dataclass(frozen=True)
class Unusable:
    """A factor the walk could not read, and what it would take to read it.

    A column whose values disagree about their type is not a factor: it has no single type
    the store could give it, and reading it one way rather than another is a decision only
    the caller can make. Rather than guess, the walk sets the values aside and describes
    them here, so the decision can be made from what is actually in the column.

    A column whose values agree perfectly but hold a different one on nearly every row is
    not a factor either, for the opposite reason: it names its rows rather than grouping
    them. That one is described here too, because the decision it needs is the same shape
    --- how to read the values --- even though nothing about them was in doubt.

    Nothing further happens to a factor nobody repairs. It is absent from
    :attr:`~dataeval.Metadata.factor_names`, from :attr:`~dataeval.Metadata.factor_data`
    and from every evaluator, exactly as an unreadable column already was -- there is no
    gate and no error, and a caller who does not care never has to look.

    Attributes
    ----------
    reasons : tuple[str, ...]
        Why the factor could not be read, as recorded in
        :attr:`~dataeval.Metadata.dropped_factors`. More than one where more than one
        applies.
    level : str or None
        Level the factor would be defined at, where that is known. ``None`` for a factor
        dropped before any level could be settled on.
    repairable : bool
        Whether :meth:`~dataeval.Metadata.repair` can make this a factor. True for a column
        whose values are kept: one set aside for mixing numbers with text, and one dropped
        for naming its rows, which a :class:`ParseDateTime` can give a vocabulary to. False
        where the values are gone or no reading of them would produce a column -- a
        vector-valued statistic has no single-column form however it is read.
    counts : Mapping[str, int]
        Rows that read as ``"numeric"`` and rows that read as ``"text"``. A numeral is
        numeric whichever way it is spelled, so a column that has been through JSON is
        described by what its values *mean* rather than by how they were written.
    distinct : Mapping[str, tuple[Any, ...]]
        The distinct values behind those counts, in the spelling the dataset used, so that
        a repair can be written against what is actually there. Sorted within each kind.
        Every one of them unless ``sampled`` says otherwise.
    sampled : bool, default False
        Whether ``distinct`` holds a sample rather than the whole set. True only for a
        column dropped for naming its rows, where the values are near-unique by definition:
        the set is the size of the column, no mapping could name it, and a handful of
        examples is what a reading is actually chosen from. False everywhere else, where
        the values are the thing a repair has to cover and are reported in full.

    Notes
    -----
    The ``repr`` abbreviates long value lists whether or not they were sampled. Where
    ``sampled`` is False the attribute itself holds all of them, which is what makes it
    possible to write a mapping that covers the column; a caller that must not guess should
    read the flag rather than the length.
    """

    reasons: tuple[str, ...]
    level: FactorLevel | None = None
    repairable: bool = False
    counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    distinct: Mapping[str, tuple[Any, ...]] = field(default_factory=lambda: MappingProxyType({}))
    sampled: bool = False

    def __hash__(self) -> int:
        """Hash the report, reading its two mappings as the pairs they hold.

        ``frozen=True`` generates a hash over every field and two of them are mappings, so a
        value describing a factor could be compared but never put in a set or used as a dict
        key -- and would say so only at the call. The same reason :class:`Remap` and
        :class:`Aggregator` spell theirs out.
        """
        return hash((
            self.reasons,
            self.level,
            self.repairable,
            tuple(sorted(self.counts.items())),
            tuple(sorted(self.distinct.items(), key=repr)),
            self.sampled,
        ))

    def __repr__(self) -> str:
        """Abbreviate the value lists, which exist to be complete rather than to be read."""
        shown = {
            kind: values
            if len(values) <= _SHOWN_VALUES
            else (*values[:_SHOWN_VALUES], f"... +{len(values) - _SHOWN_VALUES} more")
            for kind, values in self.distinct.items()
        }
        sampled = ", sampled=True" if self.sampled else ""
        return (
            f"Unusable(reasons={self.reasons!r}, level={self.level!r}, "
            f"repairable={self.repairable!r}, counts={dict(self.counts)!r}, distinct={shown!r}{sampled})"
        )


@dataclass(frozen=True)
class Aggregator:
    """A named reduction, declared from one level to a level above it.

    A reduction on its own is not a well-formed statement. "Mean brightness" is ambiguous
    until it says mean over *what*: over the frames of a video, over the detections of a
    frame, or over the detections of a track are three different numbers from one column,
    and the middle one is the fan-out that :meth:`~dataeval.Metadata.agg` refuses without
    ``unique_by``. Carrying the reduction and its level pair in one value is what makes the
    statement checkable, storable and comparable between runs.

    Everything here except the factor set is decidable against a
    :class:`FactorLevelSchema` alone, with no dataset, which is what makes this a
    *declaration* rather than a call. A calculator can declare how its output rolls up
    beside the calculator, and be wrong about it at import time rather than at analysis
    time.

    Attributes
    ----------
    how : str
        Name of the reduction, from the reduction registry -- ``"mean"``, ``"count"``,
        ``"mode"`` and so on. The name carries a contract the raw expression cannot: which
        value types it applies to, and what it answers for a destination with nothing
        beneath it.
    source : str or None
        Level whose rows are rolled up, or None to infer it per factor from the level the
        factor is defined at. Inference is what :meth:`~dataeval.Metadata.aggregate`'s
        keyword form does; naming it is what makes this value complete on its own.
    target : str
        Level receiving one value per row. Must sit strictly above ``source``.
    factors : tuple of str, default ()
        Factors to roll up. Empty means every factor at ``source`` the reduction's value
        type admits, resolved against the dataset -- so an unresolved aggregator with an
        empty set names a *rule*, and its resolved form names the factors that rule chose.
    unique_by : str or None, default None
        Count each entity at this level once within a group. Required by a reduction over
        a column defined above ``source``, which repeats across the fan-out.
    via : str or None, default None
        Roll up along routes through this level rather than along every route. Only a
        diamond offers a choice: ``via="track"`` for an instance-to-sequence roll-up
        reaches only the detections a tracker linked, which is a different question from
        the default and not a different spelling of it.
    order_by : str or None, default None
        Column a temporal reduction reads the rows in the order of. None infers it from the
        source level, preferring a wall-clock time over a presentation timestamp over a
        position. A positional reduction ignores it; a temporal one at a level that has no
        ordering is refused rather than run against row order, which is not time.
    options : Mapping[str, Any], default empty
        Reduction-specific parameters, such as ``longest_run``'s ``tolerance``. Declared per
        reduction, so an option a reduction does not take raises rather than sitting there
        inert -- which is why ``tolerance`` is not a field of its own. Reads back as a
        mapping whatever was passed: the field is normalized on construction, so a caller
        holding an aggregator never has to test it for ``None``.
    min_coverage : float, default 1.0
        Share of the rows beneath a destination that must carry a value for the destination
        to get an answer rather than a null. The default is the all-or-nothing rule the
        structurers used to apply to a whole factor, at the granularity of one destination
        row -- where it can be relaxed. ``0.0`` summarizes whatever is there. Ignored by a
        reduction that is about missingness rather than distorted by it, such as ``count``.
    suffix : str or None, default None
        Override for the output name's suffix. None derives it from ``how`` and ``via``.
    provenance : {"declared", "derived"}, default "declared"
        Who chose ``source`` and ``factors``. ``"declared"``: the caller named them.
        ``"derived"``: :meth:`~dataeval.Metadata.aggregate` inferred them from a dataset,
        which makes this value a *fit* -- reusable, and reapplied rather than re-derived
        against a second dataset, exactly as a :class:`BinSpec`'s edges are.

    See Also
    --------
    :meth:`~dataeval.Metadata.aggregate` : Roll factors up by name
    :meth:`~dataeval.Metadata.agg` : The expression-level form beneath it
    """

    how: str
    source: FactorLevel | None
    target: FactorLevel
    factors: tuple[str, ...] = ()
    unique_by: FactorLevel | None = None
    via: FactorLevel | None = None
    order_by: str | None = None
    # A factory, not a shared instance: a mappingproxy default is refused by 3.11's
    # dataclasses, and the field is normalized into a per-instance mapping anyway.
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    min_coverage: float = 1.0
    suffix: str | None = None
    provenance: Literal["declared", "derived"] = "declared"

    def __post_init__(self) -> None:
        """Normalize the factor set and refuse the shapes no schema is needed to reject."""
        validate_coverage(self.min_coverage)
        if isinstance(self.factors, str):
            # ``str`` is itself a sequence of str, so a bare name would silently become one
            # single-character factor per letter -- the same trap ``_as_parents`` guards.
            raise TypeError(
                f"factors must be a sequence of factor names, not the bare string {self.factors!r}; "
                f"pass ({self.factors!r},) for a single factor.",
            )
        object.__setattr__(self, "factors", tuple(self.factors))
        # ``or {}`` because the signature documents ``options`` as optional, and passing the
        # ``None`` it documents raised ``'NoneType' object is not iterable`` from inside here.
        object.__setattr__(self, "options", MappingProxyType(dict(self.options or {})))
        if not self.how:
            raise ValueError("An aggregator needs a reduction name, e.g. Aggregator('mean', 'unit', 'sequence').")
        if self.source is not None and self.source == self.target:
            raise ValueError(
                f"An aggregator rolls rows up into a level above them, so source and target cannot "
                f"both be {self.target!r}.",
            )

    def __hash__(self) -> int:
        """Hash the declaration, reading ``options`` as the sorted pairs it holds.

        ``frozen=True`` generates a hash over every field, and one of the fields is a
        mapping -- so a value type whose whole point is being storable and comparable
        between runs could be compared but never put in a set or used as a dict key, and
        said so only at the call. Read as pairs it hashes like the rest of the declaration.
        """
        return hash((
            self.how,
            self.source,
            self.target,
            self.factors,
            self.unique_by,
            self.via,
            self.order_by,
            tuple(sorted(self.options.items())),
            self.min_coverage,
            self.suffix,
            self.provenance,
        ))

    @property
    def is_resolved(self) -> bool:
        """Whether this names a concrete roll-up rather than a rule for finding one."""
        return self.source is not None and bool(self.factors)

    @property
    def rolls_from(self) -> FactorLevel:
        """Source level of a resolved aggregator.

        Separate from :attr:`source` so that a resolved aggregator can be *used* without
        every call site re-proving that inference has happened. An unresolved one raises
        rather than answering, since there is no honest answer to give.
        """
        if self.source is None:
            raise ValueError(
                "This aggregator's source level has not been resolved yet, so it does not have one. "
                "Resolve it against a dataset, or name the level it rolls up from.",
            )
        return self.source

    @property
    def rolls_by(self) -> str:
        """Ordering column of a resolved temporal aggregator.

        The counterpart of :attr:`rolls_from`: a resolved temporal roll-up has one, an
        unresolved or positional one does not, and asking is how a caller stops re-proving
        it at every use.
        """
        if self.order_by is None:
            raise ValueError(
                "This aggregator names no ordering column. Only a temporal reduction has one, and "
                "only once it has been resolved against a dataset.",
            )
        return self.order_by

    def name_for(self, factor: str) -> str:
        """Output name for one rolled-up factor.

        The name is the only durable record of the operation --
        :class:`FactorInfo` records the level a factor was rolled up *from* and
        deliberately not what was done to it -- so the route has to appear in it wherever
        it is not the default. Two roll-ups of one factor to one level by different routes
        are different questions, and a name that could not tell them apart would leave them
        distinguished only by a uniqueness suffix.
        """
        if self.suffix is not None:
            return f"{factor}{self.suffix}"
        return f"{factor}_{self.how}" if self.via is None else f"{factor}_{self.how}_via_{self.via}"

    def validate(self, schema: FactorLevelSchema) -> None:
        """Check the level pair against a schema, with no dataset.

        ``how`` is the one field this cannot check: the reduction registry lives in the
        metadata layer, which imports this module rather than the other way round, so the
        name is checked where the roll-up is resolved. Everything else — the level triple,
        ``unique_by``, and the route ``via`` names — is decided here, with no dataset.

        Raises
        ------
        ValueError
            When a level is not in the schema, when ``target`` does not sit above
            ``source``, when ``unique_by`` is neither ``source`` nor above it, or when no
            route from ``source`` to ``target`` passes through ``via``.
        """
        schema.validate(self.target)
        if self.source is None:
            # Without a source there is no route to check ``via`` against either: which
            # branch a factor takes upward depends on the level it is measured at.
            return
        schema.validate(self.source)
        if not schema.is_ancestor(self.target, self.source):
            raise ValueError(
                f"An aggregator rolls rows up into a level above them, but {self.target!r} does not "
                f"sit above {self.source!r} in this level graph. Levels above {self.source!r} are "
                f"{list(schema.ancestors(self.source))}.",
            )
        if (
            self.unique_by is not None
            and self.unique_by != self.source
            and not schema.is_ancestor(self.unique_by, self.source)
        ):
            raise ValueError(
                f"unique_by={self.unique_by!r} must be {self.source!r} itself or one of the levels "
                f"above it {list(schema.ancestors(self.source))}; it names the entity each row is "
                "counted once for.",
            )
        if self.via is not None:
            schema.routes_through(self.source, self.target, self.via)


def validate_coverage(min_coverage: float) -> None:
    """Refuse a coverage threshold that is not a share.

    Shared by :class:`Aggregator` and by :meth:`~dataeval.Metadata.agg`, which reach the
    same engine: a threshold above 1 nulls every destination and then reports a coverage of
    1.0 alongside, which reads as the data being at fault rather than the argument.

    Raises
    ------
    ValueError
        When the threshold is outside ``[0, 1]``, ``NaN`` included -- a NaN comparison is
        false in both directions, so it would silently disable the threshold instead.
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(
            f"min_coverage is a share of the rows beneath a destination, so it lies in [0, 1]; got {min_coverage}.",
        )


@dataclass(frozen=True)
class AggregationRecord:
    """What one roll-up reached, and how much of it was recorded.

    A rolled-up column cannot explain its own nulls. A destination is null because it had
    no rows beneath it, or because too few of the rows it did have carried a value, or --
    if the roll-up was routed through a branch -- because it was never reached at all.
    Those are three different statements about a dataset and they look identical in the
    column, so the counts are kept beside it rather than left to be inferred.

    Attributes
    ----------
    source : str
        Level whose rows were rolled up.
    target : str
        Level that received one value per row.
    how : str or None
        Reduction's name, or None where the roll-up was written as an expression.
    via : str or None
        Branch the roll-up was routed through, or None for every route.
    outputs : tuple of str
        Factor names produced, after any renaming for collision. ``coverage`` and
        ``uncovered`` are aligned with this.
    took_part : int
        Source rows that had an ancestor at ``target`` and were summarized.
    no_ancestor : int
        Source rows excluded for having none. Zero for every complete route; for a routed
        one, exactly the rows that branch does not reach.
    childless : int
        Destination rows with nothing beneath them.
    coverage : tuple of float
        Per output, the lowest share of recorded values any destination with rows beneath
        it saw. This is the number that says which ``min_coverage`` would have been
        answerable.
    uncovered : tuple of int
        Per output, destinations nulled for falling below the threshold.
    gaps : int
        Steps in the ordering key larger than the tightest step within the same destination,
        summed over all of them; 0 for a roll-up that did not read its rows as an ordered
        series. It counts unevenness whatever caused it — a filter that removed rows, a
        key-frame selection, or a source that sampled unevenly — because from the ordering
        key those are the same observation, and a reduction that assumes even spacing is
        equally wrong in all three.
    """

    source: FactorLevel
    target: FactorLevel
    how: str | None
    via: FactorLevel | None
    outputs: tuple[str, ...]
    took_part: int
    no_ancestor: int
    childless: int
    coverage: tuple[float, ...]
    uncovered: tuple[int, ...]
    gaps: int = 0

    def coverage_of(self, output: str) -> float:
        """Lowest coverage recorded for one of this roll-up's outputs.

        Raises
        ------
        ValueError
            When this roll-up produced no such output.
        """
        if output not in self.outputs:
            raise ValueError(f"{output!r} is not one of this roll-up's outputs {list(self.outputs)}.")
        return self.coverage[self.outputs.index(output)]


@dataclass(frozen=True)
class ClassAxis:
    """What a class-conditional result conditioned on.

    Every evaluator that groups rows by class reads one variable to do it, and until
    :meth:`~dataeval.Metadata.classed_by` existed that variable was always the dataset's
    own labels. It no longer is, so a result has to be able to say which variable it was
    -- otherwise two runs that answered different questions are indistinguishable
    afterwards, and a reader comparing them attributes a moved score to the data.

    Carried on the output of every class-conditional evaluator and returned by
    :attr:`~dataeval.Metadata.class_axis_info`, so the same record can be asserted on
    before a run and read back after one.

    Attributes
    ----------
    name : str
        What the axis is called: ``"class_label"`` for the dataset's own labels, the
        factor's name for a pivot, or several joined by ``" x "`` for a composite.
    source : {"ground_truth", "derived"}
        ``"ground_truth"`` for the dataset's labels; ``"derived"`` for an axis a caller
        defined out of factors. The field a gate asserts on: an evaluator whose meaning
        depends on the labels being the dataset's own -- :class:`~dataeval.scope.Representation`
        resolves them against an ontology -- refuses a ``"derived"`` axis rather than
        reporting against concepts the names cannot match.
    level : str or None
        Level the axis is defined at, or None where the container has no level schema.
        Read together with ``rows_per_group_entity``: an axis defined above the rows being
        counted is replicated onto them, and this is what says so.
    groups : int
        Number of distinct groups the axis takes, i.e. how many classes the result reports.
    rows_per_group_entity : float or None
        Rows counted per entity at the axis's own level -- 1.0 where the axis was read at
        its own level, and above 1.0 where it fanned out onto descendant rows. A frame-level
        axis read from detection rows weights each frame by how many detections it holds,
        and this is the number that says by how much. None where the container cannot say.
    vocabulary : {"declared", "observed"}
        Whether the group names come from a vocabulary the caller declared -- through
        ``factor_levels=`` or a recorded ``encoding=`` -- or were read off the values that
        happened to be present. ``"observed"`` names are dataset-relative: a set missing
        ``fog`` yields a different alphabet, so two results are only comparable by name
        under ``"declared"``. Always ``"observed"`` for a composite axis, whose combinations
        are read off the data even where each component was declared.
    """

    name: str
    source: Literal["ground_truth", "derived"]
    level: FactorLevel | None = None
    groups: int = 0
    rows_per_group_entity: float | None = None
    vocabulary: Literal["declared", "observed"] = "observed"

"""In-memory ontology model with optional RDF/OWL/JSON-LD construction.

DataEval operates on a small, strongly-typed, dependency-free in-memory
representation of an ontology (:class:`Ontology` / :class:`OntologyConcept`).
File parsing is intentionally *not* part of the library: the
:meth:`Ontology.from_rdf` / :meth:`Ontology.from_rdflib` constructors accept
already-in-memory content and lazily import :mod:`rdflib` (an optional
dependency, installable via ``dataeval[ontology]``).

Concepts are typically typed ``owl:Class`` (or ``skos:Concept``), but their
hierarchy and labels come from ``rdfs:`` (``rdfs:subClassOf``, ``rdfs:label``)
and ``skos:`` (``skos:prefLabel``, ``skos:altLabel``, ``skos:definition``) —
OWL reuses these rather than defining its own.
"""

__all__ = ["Ontology"]

import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from dataeval.exceptions import OntologyCycleError, OntologyError
from dataeval.types import OntologyConcept

if TYPE_CHECKING:
    import rdflib

_logger = logging.getLogger(__name__)


class Ontology:
    """
    An immutable, in-memory directed acyclic graph of :class:`OntologyConcept`.

    The graph is built from a collection of concepts linked by their ``parents``
    (is-a edges). A concept may have more than one parent, so the graph is a DAG
    rather than a tree. Concepts asserted to denote the same class — via
    ``equivalent_to``, or by naming each other as parents — are collapsed into a
    single *canonical* concept, and the group's remaining ids become **aliases**
    that resolve to it; a concept naming *itself* is the degenerate case and is
    simply dropped. Cycles carrying neither signal are rejected. Parent ids
    referencing concepts not present in the collection are kept as *external*
    references — they participate in ancestor/LCA queries but are not themselves
    concepts.

    Once built, the graph is queryable for ancestors, descendants, siblings,
    lowest common ancestors, depth, and rooted subtrees, and resolves class
    names to concepts via :meth:`find`.

    Parameters
    ----------
    concepts : Iterable[OntologyConcept]
        Concepts comprising the ontology. Ids must be unique.

    Raises
    ------
    OntologyError
        If two concepts share an id.
    OntologyCycleError
        If the is-a graph contains a cycle.

    See Also
    --------
    Ontology.from_rdf : Build from in-memory RDF/OWL/JSON-LD content.
    Ontology.from_hierarchy : Build from a plain nested dict / list (no rdflib).
    """

    def __init__(self, concepts: Iterable[OntologyConcept]) -> None:
        collected: dict[str, OntologyConcept] = {}
        for concept in concepts:
            if concept.id in collected:
                raise OntologyError(f"Duplicate concept id: {concept.id!r}")
            # A concept is trivially its own superclass (RDFS/OWL entail it, and
            # reasoners materialize `X rdfs:subClassOf X`). The edge carries no
            # information, so drop it rather than reporting a self-loop as a cycle.
            if concept.id in concept.parents:
                _logger.debug("Dropping self-referential parent on concept %r", concept.id)
                concept = concept.model_copy(update={"parents": tuple(p for p in concept.parents if p != concept.id)})
            # Likewise `X owl:equivalentClass X`: reflexive, entailed, and empty
            # of information, so it is not an equivalence group of one.
            if concept.id in concept.equivalent_to:
                _logger.debug("Dropping self-referential equivalence on concept %r", concept.id)
                concept = concept.model_copy(
                    update={"equivalent_to": tuple(e for e in concept.equivalent_to if e != concept.id)}
                )
            collected[concept.id] = concept

        # alias id -> canonical id, one entry per non-canonical equivalence member
        self._alias: dict[str, str] = {}
        self._concepts: dict[str, OntologyConcept] = self._merge_equivalents(collected)

        # concept id -> insertion position, so subtrees can be ordered like the
        # ontology they came from without rescanning every concept
        self._position: dict[str, int] = {cid: i for i, cid in enumerate(self._concepts)}

        # children map keyed by parent id (external parents are valid keys)
        self._children: dict[str, list[str]] = {}
        # case-insensitive index over preferred label + synonyms (+ exact id)
        self._label_index: dict[str, list[str]] = {}
        self._build_indexes()
        self._check_acyclic()

    def _merge_equivalents(self, collected: dict[str, OntologyConcept]) -> dict[str, OntologyConcept]:
        """Collapse each equivalence group to one canonical concept, recording aliases."""
        aliases_of: dict[str, list[str]] = {}
        for group in _equivalence_groups(collected):
            # An undefined id has no label, parents, or definition to survive as,
            # so only defined members are eligible. Every group has one: unions
            # only ever originate from a defined concept's own assertions, so an
            # all-undefined group cannot form.
            defined = sorted(member for member in group if member in collected)
            # Among those, prefer a member carrying a real label. `from_rdflib`
            # falls back to `label = str(subject)` for an unlabelled class, and
            # electing that would demote the group's human label to a synonym and
            # surface a raw IRI wherever `concept.label` is displayed. Smallest
            # id still breaks the tie, so the choice stays order-independent.
            labelled = [member for member in defined if collected[member].label != member]
            canonical_id = (labelled or defined)[0]
            alias_ids = sorted(member for member in group if member != canonical_id)
            aliases_of[canonical_id] = alias_ids
            for alias in alias_ids:
                self._alias[alias] = canonical_id
        if not self._alias:
            return collected
        return {
            cid: _absorb(concept, aliases_of.get(cid, []), collected, self._alias)
            for cid, concept in collected.items()
            if cid not in self._alias
        }

    def _build_indexes(self) -> None:
        for concept in self._concepts.values():
            for parent in concept.parents:
                self._children.setdefault(parent, []).append(concept.id)
            for name in (concept.label, *concept.synonyms):
                self._label_index.setdefault(name.casefold(), []).append(concept.id)

    def _check_acyclic(self) -> None:
        # External parents are not concepts, so they don't count toward indegree.
        parents_of = {cid: concept.parents for cid, concept in self._concepts.items()}
        stuck = _unresolved_by_kahn(parents_of)
        if stuck:
            raise OntologyCycleError(f"Ontology contains a cycle (is-a edges): {_cycle_trace(parents_of, stuck)}")

    # --- mapping-like access ---

    def __repr__(self) -> str:
        """Return a concise structural summary of the ontology."""
        return (
            f"{type(self).__name__}({len(self._concepts)} concepts, "
            f"{len(self.roots)} roots, {len(self.leaves)} leaves, "
            f"{len(self.external_ids)} external)"
        )

    def __len__(self) -> int:
        """Return the number of defined concepts."""
        return len(self._concepts)

    def __iter__(self) -> Iterator[OntologyConcept]:
        """Iterate over defined concepts."""
        return iter(self._concepts.values())

    def __contains__(self, concept_id: str) -> bool:
        """Return whether ``concept_id`` is a defined concept or an alias of one."""
        return concept_id in self._concepts or concept_id in self._alias

    def __getitem__(self, concept_id: str) -> OntologyConcept:
        """Return the concept for ``concept_id`` (raises ``KeyError`` if absent)."""
        return self._concepts[self.canonical(concept_id)]

    def concept(self, concept_id: str) -> OntologyConcept:
        """Return the concept for ``concept_id`` (raises ``KeyError`` if absent)."""
        return self._concepts[self.canonical(concept_id)]

    def canonical(self, concept_id: str) -> str:
        """
        Resolve any known id to the id of the concept that defines it.

        Concepts asserted to denote the same class are collapsed at build time
        into one *canonical* concept; the group's other ids become **aliases**.
        Passing a canonical id returns it unchanged, passing an alias returns
        its canonical.

        An alias differs from an *external reference*: an alias names a class
        this ontology defines (under another id), while an external reference
        names one it does not define at all. Aliases never appear in
        :attr:`external_ids`.

        Parameters
        ----------
        concept_id : str
            A canonical concept id or an alias of one.

        Returns
        -------
        str
            The canonical concept id.

        Raises
        ------
        KeyError
            If ``concept_id`` is neither a defined concept nor an alias.
        """
        if concept_id in self._concepts:
            return concept_id
        canonical_id = self._alias.get(concept_id)
        if canonical_id is None:
            raise KeyError(concept_id)
        return canonical_id

    def aliases(self, concept_id: str) -> tuple[str, ...]:
        """
        Return the ids absorbed into ``concept_id``'s concept, id-sorted.

        Accepts a canonical id or any of its aliases; both return the same
        tuple. Empty when the concept was not part of an equivalence group.
        Raises ``KeyError`` if ``concept_id`` is not a known id.
        """
        return self.concept(concept_id).equivalent_to

    @property
    def ids(self) -> tuple[str, ...]:
        """Ids of all defined concepts."""
        return tuple(self._concepts)

    @property
    def roots(self) -> tuple[str, ...]:
        """Ids of defined concepts that declare no parents."""
        return tuple(c.id for c in self._concepts.values() if not c.parents)

    @property
    def leaves(self) -> tuple[str, ...]:
        """Ids of defined concepts that have no children (most specific concepts)."""
        return tuple(c.id for c in self._concepts.values() if c.id not in self._children)

    @property
    def external_ids(self) -> tuple[str, ...]:
        """
        Ids referenced as parents but not present as defined concepts.

        These are *external references*: the ontology references them (e.g. it
        was distributed as a subset) but does not define them, so they have no
        label, definition, or further ancestors. Their presence means the is-a
        hierarchy is truncated at those points.
        """
        return tuple(sorted(pid for pid in self._children if pid not in self._concepts))

    @property
    def label_collisions(self) -> dict[str, tuple[str, ...]]:
        """
        Case-folded names that resolve to more than one concept.

        Each entry maps a normalized name (a preferred label or synonym shared
        across concepts) to the distinct concept ids :meth:`find` would return for
        it — the artifact-side source of reconciliation *ambiguity*. Empty when
        every name resolves uniquely. Unlike :meth:`find`, exact-id matches are not
        considered, since an id is unique by construction.
        """
        collisions: dict[str, tuple[str, ...]] = {}
        for name, ids in self._label_index.items():
            unique = tuple(dict.fromkeys(ids))
            if len(unique) > 1:
                collisions[name] = unique
        return collisions

    # --- queries ---

    def find(self, name: str) -> tuple[str, ...]:
        """
        Resolve a human-readable name (or exact id) to matching concept ids.

        Matching is case-insensitive over each concept's preferred label and
        synonyms. An exact id match is also returned, resolved through any
        alias, so only canonical ids come back.

        Parameters
        ----------
        name : str
            Class name or concept id to resolve.

        Returns
        -------
        tuple[str, ...]
            Matching concept ids. Empty if unmatched; length > 1 if ambiguous.
        """
        ids = list(self._label_index.get(name.casefold(), ()))
        if name in self:
            canonical_id = self.canonical(name)
            if canonical_id not in ids:
                ids.append(canonical_id)
        return tuple(dict.fromkeys(ids))

    def _parents(self, concept_id: str) -> tuple[str, ...]:
        concept = self._concepts.get(concept_id)
        return concept.parents if concept is not None else ()

    def _children_of(self, concept_id: str) -> list[str]:
        return self._children.get(concept_id, [])

    def _ancestors(self, concept_id: str) -> list[str]:
        return _traverse(concept_id, self._parents)

    def ancestors(self, concept_id: str) -> tuple[str, ...]:
        """
        Return all ancestor ids of a concept, nearest-first (breadth-first).

        Ancestors are the concept's transitive *superclasses* (broader concepts).
        May include external reference ids. Raises ``KeyError`` if ``concept_id``
        is not a defined concept.
        """
        concept_id = self.canonical(concept_id)
        return tuple(self._ancestors(concept_id))

    def children(self, concept_id: str) -> tuple[str, ...]:
        """
        Return the ids of the direct subclasses (children) of ``concept_id``.

        Children are the defined concepts that declare ``concept_id`` among their
        ``parents``; order follows concept insertion order. Unlike
        :meth:`descendants` this is the immediate, non-transitive layer. Raises
        ``KeyError`` if ``concept_id`` is not a defined concept.
        """
        concept_id = self.canonical(concept_id)
        return tuple(self._children.get(concept_id, ()))

    def descendants(self, concept_id: str) -> tuple[str, ...]:
        """
        Return all descendant concept ids of ``concept_id``, nearest-first.

        Descendants are the concept's transitive *subclasses* (narrower concepts).
        Raises ``KeyError`` if ``concept_id`` is not a defined concept.
        """
        concept_id = self.canonical(concept_id)
        return tuple(_traverse(concept_id, self._children_of))

    def is_a(self, a: str, b: str) -> bool:
        """Return whether concept ``a`` is a subclass of ``b``.

        Equivalently, whether ``b`` is ``a`` itself or one of its ancestors
        (superclasses). Subsumption is *reflexive* — every concept is its own
        subclass, as RDFS/OWL entail — so ``is_a(x, x)`` is true whether or not
        the ontology spells the trivial edge out. For the strict form ("below
        ``b``, not ``b``"), test ``a != b and onto.is_a(a, b)``.

        Note this makes ``is_a`` slightly wider than :meth:`ancestors`, which
        stays proper: ``is_a(a, b)`` is ``b == a or b in ancestors(a)``.

        Equivalent concepts resolve to the same canonical id, so ``is_a`` is
        symmetric across an equivalence group — which falls out of reflexivity
        rather than needing a rule of its own. Either argument may be given as
        an alias.

        Raises ``KeyError`` if ``a`` is not a defined concept; ``b`` may be any
        id, including an external reference.
        """
        a = self.canonical(a)
        # `b` resolves through the alias map but not through `canonical`: it may
        # legitimately be an external reference, which must not start raising.
        b = self._alias.get(b, b)
        return b == a or b in self._ancestors(a)

    def lowest_common_ancestors(self, a: str, b: str) -> tuple[str, ...]:
        """
        Return all lowest common ancestors of ``a`` and ``b``, id-sorted.

        A *common ancestor* is an id in both concepts' ancestor sets; a concept
        counts as an ancestor of itself, so the LCA of a concept and its
        descendant is the concept itself. A common ancestor is *lowest* when
        none of its own descendants is also a common ancestor. On a tree this is
        always a single id, but on a DAG two concepts may meet at several
        mutually incomparable points, so the result may hold more than one. May
        include an external reference id (the meeting point can lie outside the
        defined concepts). Returns an empty tuple when the two share no ancestor.

        Raises ``KeyError`` if ``a`` or ``b`` is not a defined concept.
        """
        a = self.canonical(a)
        b = self.canonical(b)
        common = {a, *self._ancestors(a)} & {b, *self._ancestors(b)}
        if not common:
            return ()
        # An id is *lowest* unless it is a (proper) ancestor of another common
        # id — i.e. unless it appears in some common id's ancestor set.
        higher = set().union(*(self._ancestors(cid) for cid in common))
        return tuple(sorted(common - higher))

    def lowest_common_ancestor(self, a: str, b: str) -> str | None:
        """
        Return a single lowest common ancestor of ``a`` and ``b``, or ``None``.

        A deterministic projection of :meth:`lowest_common_ancestors`: on a tree
        the LCA is unique; on a DAG with several incomparable lowest common
        ancestors this returns the deepest (the id with the most ancestors), ties
        broken by id. Use :meth:`lowest_common_ancestors` to get the full set.
        Returns ``None`` when the two share no ancestor; may return an external
        reference id.

        Raises ``KeyError`` if ``a`` or ``b`` is not a defined concept.
        """
        candidates = self.lowest_common_ancestors(a, b)
        if not candidates:
            return None
        # candidates is id-sorted, so max() breaks ancestor-count ties by smallest id
        return max(candidates, key=lambda cid: len(self._ancestors(cid)))

    def siblings(self, concept_id: str) -> tuple[str, ...]:
        """
        Return defined concepts sharing at least one parent with ``concept_id``.

        Excludes the concept itself. Siblings under an *external* (undefined)
        parent are included, so this works on subset ontologies. Raises
        ``KeyError`` if ``concept_id`` is not a defined concept.
        """
        concept_id = self.canonical(concept_id)
        ordered: list[str] = []
        seen: set[str] = {concept_id}
        for parent in self._concepts[concept_id].parents:
            for child in self._children.get(parent, ()):
                if child not in seen:
                    seen.add(child)
                    ordered.append(child)
        return tuple(ordered)

    def depth_of(self, concept_id: str) -> int:
        """
        Return the length of the longest is-a path from a root to ``concept_id``.

        A concept with no parents has depth 0; a concept whose only parent is an
        external reference has depth 1. Raises ``KeyError`` if ``concept_id``
        is not a defined concept.
        """
        concept_id = self.canonical(concept_id)
        memo: dict[str, int] = {}

        def depth(cid: str) -> int:
            if cid in memo:
                return memo[cid]
            parents = self._parents(cid)
            memo[cid] = 1 + max((depth(p) for p in parents), default=-1)
            return memo[cid]

        return depth(concept_id)

    def subtree_ids(self, concept_id: str) -> frozenset[str]:
        """
        Return ``concept_id`` together with all its descendant ids (its subtree).

        A lightweight id-set form of :meth:`subtree`, for membership and
        disjointedness tests that do not need a full sub-ontology. Raises
        ``KeyError`` if ``concept_id`` is not a defined concept.
        """
        concept_id = self.canonical(concept_id)
        return frozenset((concept_id, *self.descendants(concept_id)))

    def subtree(self, concept_id: str) -> "Ontology":
        """
        Return a new :class:`Ontology` rooted at ``concept_id``.

        Contains the concept and all its descendants; parent links pointing
        outside the subtree are pruned so ``concept_id`` becomes a root. Raises
        ``KeyError`` if ``concept_id`` is not a defined concept.
        """
        node_ids = self.subtree_ids(concept_id)
        concepts = []
        # Order by this ontology's own concept order, not the id *set*: set order
        # varies with the hash seed, which would make the subtree's `ids`,
        # `roots`, `leaves`, and iteration differ between runs. Sorting the
        # subtree keeps this O(k log k) rather than rescanning all N concepts.
        for nid in sorted(node_ids, key=self._position.__getitem__):
            concept = self._concepts[nid]
            pruned = tuple(p for p in concept.parents if p in node_ids)
            concepts.append(concept.model_copy(update={"parents": pruned}))
        return Ontology(concepts)

    # --- construction from RDF (optional rdflib dependency) ---

    @classmethod
    def from_rdflib(cls, graph: "rdflib.Graph") -> Self:
        """
        Build an :class:`Ontology` from an in-memory :class:`rdflib.Graph`.

        Concepts are collected from subjects typed ``owl:Class`` / ``rdfs:Class`` /
        ``skos:Concept`` and from any subject of ``rdfs:subClassOf`` /
        ``skos:broader`` / ``skos:broaderTransitive`` / ``owl:equivalentClass``.
        For each: ``label`` is ``skos:prefLabel`` (falling back to ``rdfs:label``),
        ``synonyms`` are ``skos:altLabel`` (plus a differing ``rdfs:label``),
        ``parents`` are the IRI objects of ``rdfs:subClassOf`` / ``skos:broader``,
        ``equivalent_to`` are the IRI objects of ``owl:equivalentClass``, and
        ``definition`` is ``skos:definition``. Blank-node superclasses (e.g.
        ``owl:Restriction``) are ignored.

        ``skos:broaderTransitive`` is read only as a *fallback*, for a concept
        that declares no direct parent — it is the transitive property, so a
        materialized vocabulary asserts every ancestor with it, and taking those
        as parents would record closure edges as direct is-a edges.

        ``skos:exactMatch`` is deliberately *not* read: it is a cross-scheme
        mapping property, and cross-vocabulary equivalence belongs to
        :class:`~dataeval.types.Correspondence`, not to this graph.

        Parameters
        ----------
        graph : rdflib.Graph
            Parsed RDF graph.

        Returns
        -------
        Ontology
        """
        from rdflib import URIRef
        from rdflib.namespace import OWL, RDF, RDFS, SKOS

        subjects: set[URIRef] = set()
        for rdf_class in (OWL.Class, RDFS.Class, SKOS.Concept):
            subjects.update(s for s in graph.subjects(RDF.type, rdf_class) if isinstance(s, URIRef))
        for predicate in (RDFS.subClassOf, SKOS.broader, SKOS.broaderTransitive, OWL.equivalentClass):
            subjects.update(s for s in graph.subjects(predicate, None) if isinstance(s, URIRef))

        # Sorted, not set order: set iteration varies with the hash seed, which
        # would make `ids` — and so the integer class indices :class:`.Relabel`
        # derives from them — differ between processes for the same document.
        concepts = [_concept_from_graph(graph, subject) for subject in sorted(subjects)]
        _logger.debug("Built ontology with %d concepts from rdflib graph", len(concepts))
        return cls(concepts)

    @classmethod
    def from_rdf(cls, source: str | bytes, *, format: str | None = None) -> Self:  # noqa: A002
        """
        Build an :class:`Ontology` from in-memory RDF content.

        Parses already-in-memory serialized RDF (OWL/RDF-XML, Turtle, N-Triples,
        JSON-LD, ...) via :mod:`rdflib`. This does **not** read files; callers
        should load file contents themselves and pass the text/bytes.

        Parameters
        ----------
        source : str or bytes
            Serialized RDF content.
        format : str or None, optional
            rdflib format hint, e.g. ``"xml"``, ``"turtle"``, ``"json-ld"``,
            ``"nt"``. If ``None``, rdflib attempts to guess.

        Returns
        -------
        Ontology

        Raises
        ------
        ImportError
            If :mod:`rdflib` is not installed. Install via ``dataeval[ontology]``.
        """
        try:
            import rdflib
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "Ontology.from_rdf requires the optional 'rdflib' dependency. "
                "Install it with: pip install 'dataeval[ontology]'"
            ) from exc

        graph = rdflib.Graph()
        graph.parse(data=source, format=format)
        return cls.from_rdflib(graph)

    @classmethod
    def from_hierarchy(cls, data: "Mapping[str, Any] | Sequence[Any]") -> Self:
        """
        Build an :class:`Ontology` from a plain, hand-authored hierarchy.

        A dependency-free constructor for the common case where you don't have
        an RDF/OWL file. Labels double as concept ids (no IRIs, synonyms, or
        definitions). Accepts:

        - a flat list of labels: ``["car", "dog"]``
        - a one-level mapping: ``{"car": ["sedan", "SUV"], "dog": None}``
        - an arbitrarily nested mapping:
          ``{"vehicle": {"car": {"sedan": None}}}``

        Mapping values may be ``None`` (leaf), a list of labels (children), or a
        nested mapping. A label appearing under more than one parent yields a DAG.

        Parameters
        ----------
        data : Mapping or Sequence
            The hierarchy specification.

        Returns
        -------
        Ontology

        Raises
        ------
        OntologyError
            If a label is not a string or a node has an unexpected type.
        OntologyCycleError
            If the hierarchy contains a cycle.
        """
        return cls(_build_hierarchy(data))


def _traverse(start: str, neighbors: Callable[[str], Iterable[str]]) -> list[str]:
    """Breadth-first traversal from ``start`` over ``neighbors``, nearest-first.

    Excludes ``start`` itself and de-duplicates, preserving discovery order.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    queue = deque(neighbors(start))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        queue.extend(neighbors(current))
    return ordered


def _unresolved_by_kahn(parents_of: "Mapping[str, Sequence[str]]") -> set[str]:  # noqa: C901
    """Return the ids Kahn's algorithm cannot peel off: a cycle and its downstream.

    ``parents_of`` maps each node to its parent ids. Parents absent from the
    mapping are *external references*, and self-edges are trivial, so neither
    counts toward indegree. An empty result means the graph is acyclic.
    """
    indegree = {nid: sum(1 for p in ps if p != nid and p in parents_of) for nid, ps in parents_of.items()}
    children: dict[str, list[str]] = {}
    for nid, parents in parents_of.items():
        for parent in parents:
            if parent != nid and parent in parents_of:
                children.setdefault(parent, []).append(nid)
    queue = deque(nid for nid, degree in indegree.items() if degree == 0)
    removed = 0
    while queue:
        for child in children.get(queue.popleft(), ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
        removed += 1
    if removed == len(indegree):
        return set()
    return {nid for nid, degree in indegree.items() if degree > 0}


def _cycle_trace(parents_of: "Mapping[str, Sequence[str]]", stuck: set[str]) -> str:
    """Render one concrete cycle within the unresolved set as ``'a' -> 'b' -> 'a'``.

    The unresolved set holds the cycle *and* everything downstream of it, so
    reporting an arbitrary member points at nodes whose own edges are fine.
    Every unresolved node has an unresolved non-self parent, so walking
    parent-ward must revisit a node; that repeat delimits the real cycle.
    """
    path: list[str] = []
    position: dict[str, int] = {}
    current = min(stuck)  # deterministic entry point
    while current not in position:
        position[current] = len(path)
        path.append(current)
        current = next(p for p in parents_of[current] if p != current and p in stuck)
    cycle = path[position[current] :]
    return " -> ".join(repr(nid) for nid in (*cycle, cycle[0]))


def _equivalence_groups(collected: dict[str, OntologyConcept]) -> list[set[str]]:  # noqa: C901
    """Union-find over explicit equivalences and direct mutual subsumption.

    Two edge sources, both licensed: an explicit ``equivalent_to`` assertion,
    and a mutual pair (each side naming the other as a parent), which is how a
    reasoner materializes ``owl:equivalentClass``. Grouping is transitive by
    construction, which is the correct entailment. Longer cycles with neither
    signal are *not* grouped — they reach the acyclic check and raise.

    Mutual subsumption is detected between *groups*, iterated to a fixpoint,
    because merging can entail further equivalence: given ``A ≡ B``, ``A ⊑ C``
    and ``C ⊑ B``, the group ``{A, B}`` and ``C`` subsume each other even though
    no two concepts formed a mutual pair in the input. Stopping after one pass
    would leave that edge behind and manufacture a fresh cycle out of valid OWL.
    """
    parent: dict[str, str] = {}

    def root(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]  # path compression
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = root(a), root(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for cid, concept in collected.items():
        root(cid)
        for eid in concept.equivalent_to:
            union(cid, eid)

    # Group-level parent sets, recomputed after every round of merges.
    def supersets() -> dict[str, set[str]]:
        supers: dict[str, set[str]] = {}
        for cid, concept in collected.items():
            group = root(cid)
            for pid in concept.parents:
                parent_group = root(pid)
                if parent_group != group:
                    supers.setdefault(group, set()).add(parent_group)
        return supers

    merged = True
    while merged:
        merged = False
        supers = supersets()
        # Collect the whole round before unioning: a union invalidates `supers`.
        mutual = [(g, pg) for g, pgs in supers.items() for pg in pgs if g in supers.get(pg, ())]
        for group, parent_group in mutual:
            if root(group) != root(parent_group):
                union(group, parent_group)
                merged = True

    groups: dict[str, set[str]] = {}
    for node in list(parent):
        groups.setdefault(root(node), set()).add(node)
    return [members for members in groups.values() if len(members) > 1]


def _absorb(
    canonical: OntologyConcept,
    alias_ids: list[str],
    collected: dict[str, OntologyConcept],
    alias_map: dict[str, str],
) -> OntologyConcept:
    """Fold an equivalence group's members into its canonical concept.

    Also used for concepts in no group at all (empty ``alias_ids``), but then it
    rewrites *only* ``parents``: a non-member pointing at an alias id has to be
    redirected or the edge is lost from the children map. Everything else is left
    verbatim, so an equivalence between two concepts cannot change a third.
    """
    if not alias_ids:
        redirected = tuple(
            dict.fromkeys(r for pid in canonical.parents if (r := alias_map.get(pid, pid)) != canonical.id)
        )
        return canonical if redirected == canonical.parents else canonical.model_copy(update={"parents": redirected})

    members = [canonical, *(collected[alias] for alias in alias_ids if alias in collected)]
    synonyms: list[str] = []
    parents: list[str] = []
    definition = canonical.definition
    for member in members:
        synonyms.extend(name for name in (member.label, *member.synonyms) if name != canonical.label)
        parents.extend(r for pid in member.parents if (r := alias_map.get(pid, pid)) != canonical.id)
        if definition is None:
            definition = member.definition
    return canonical.model_copy(
        update={
            "synonyms": tuple(dict.fromkeys(synonyms)),
            "parents": tuple(dict.fromkeys(parents)),
            "definition": definition,
            "equivalent_to": tuple(alias_ids),
        }
    )


def _build_hierarchy(data: "Mapping[str, Any] | Sequence[Any]") -> list[OntologyConcept]:  # noqa: C901
    order: list[str] = []
    parents: dict[str, list[str]] = {}
    seen: set[str] = set()

    def add(name: str, parent: str | None) -> None:
        if name not in seen:
            seen.add(name)
            order.append(name)
            parents[name] = []
        if parent is not None and parent not in parents[name]:
            parents[name].append(parent)

    def walk(node: Any, parent: str | None) -> None:  # noqa: C901
        if isinstance(node, str):
            add(node, parent)
        elif isinstance(node, Mapping):
            for label, children in node.items():
                walk(label, parent)
                walk(children, label)
        elif isinstance(node, Sequence) and not isinstance(node, bytes):
            for item in node:
                walk(item, parent)
        elif node is not None:
            raise OntologyError(
                f"Unexpected hierarchy node {node!r} ({type(node).__name__}); expected mapping, list, str, or None."
            )

    walk(data, None)
    _reject_hierarchy_cycles(parents)
    return [OntologyConcept(id=name, label=name, parents=tuple(parents[name])) for name in order]


def _reject_hierarchy_cycles(parents: dict[str, list[str]]) -> None:
    """Reject any cycle in a hand-authored hierarchy.

    Stricter than :meth:`Ontology._check_acyclic`, deliberately. A nested
    mapping has no syntax for equivalence, so a node reachable from itself is
    always a typo — including the two-node ``a -> b -> a`` case, which from RDF
    would be read as a materialized ``owl:equivalentClass`` and merged. A node
    listed directly under *itself* stays legal: annotation schemas commonly list
    a category among its own choices, and that edge is dropped as trivial.
    """
    stuck = _unresolved_by_kahn(parents)
    if stuck:
        raise OntologyCycleError(f"Hierarchy contains a cycle: {_cycle_trace(parents, stuck)}")


def _concept_from_graph(graph: "rdflib.Graph", subject: "rdflib.URIRef") -> OntologyConcept:  # noqa: C901
    from rdflib import URIRef
    from rdflib.namespace import OWL, RDFS, SKOS

    def first_literal(*predicates: URIRef) -> str | None:
        for predicate in predicates:
            for obj in graph.objects(subject, predicate):
                return str(obj)
        return None

    rdfs_label = first_literal(RDFS.label)
    label = first_literal(SKOS.prefLabel) or rdfs_label or str(subject)

    synonyms: list[str] = []
    if rdfs_label is not None and rdfs_label != label:
        synonyms.append(rdfs_label)
    synonyms.extend(str(obj) for obj in graph.objects(subject, SKOS.altLabel))

    parents: list[str] = []
    for predicate in (RDFS.subClassOf, SKOS.broader):
        parents.extend(str(obj) for obj in graph.objects(subject, predicate) if isinstance(obj, URIRef))
    if not parents:
        # Fallback only. `skos:broaderTransitive` is the *transitive* property,
        # so a materialized vocabulary states every ancestor with it, not just
        # the direct one. Preferring the direct predicates keeps closure edges
        # from being recorded as parents, which would corrupt children() and
        # report a concept's own parent among its siblings.
        parents.extend(str(obj) for obj in graph.objects(subject, SKOS.broaderTransitive) if isinstance(obj, URIRef))

    equivalents = [str(obj) for obj in graph.objects(subject, OWL.equivalentClass) if isinstance(obj, URIRef)]

    return OntologyConcept(
        id=str(subject),
        label=label,
        synonyms=tuple(dict.fromkeys(synonyms)),
        parents=tuple(dict.fromkeys(parents)),
        equivalent_to=tuple(dict.fromkeys(equivalents)),
        definition=first_literal(SKOS.definition),
    )

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
    rather than a tree; cycles are rejected. Parent ids referencing concepts not
    present in the collection are kept as *external* references — they
    participate in ancestor/LCA queries but are not themselves concepts.

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
        self._concepts: dict[str, OntologyConcept] = {}
        for concept in concepts:
            if concept.id in self._concepts:
                raise OntologyError(f"Duplicate concept id: {concept.id!r}")
            self._concepts[concept.id] = concept

        # children map keyed by parent id (external parents are valid keys)
        self._children: dict[str, list[str]] = {}
        # case-insensitive index over preferred label + synonyms (+ exact id)
        self._label_index: dict[str, list[str]] = {}
        self._build_indexes()
        self._check_acyclic()

    def _build_indexes(self) -> None:
        for concept in self._concepts.values():
            for parent in concept.parents:
                self._children.setdefault(parent, []).append(concept.id)
            for name in (concept.label, *concept.synonyms):
                self._label_index.setdefault(name.casefold(), []).append(concept.id)

    def _check_acyclic(self) -> None:
        # Kahn's algorithm: peel off concepts whose (defined) parents are all
        # resolved; any left over are part of, or downstream of, a cycle.
        # External parents are not concepts, so they don't count toward indegree.
        indegree = {cid: sum(p in self._concepts for p in c.parents) for cid, c in self._concepts.items()}
        queue = deque(cid for cid, deg in indegree.items() if deg == 0)
        removed = 0
        while queue:
            for child in self._children.get(queue.popleft(), ()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
            removed += 1
        if removed != len(self._concepts):
            stuck = next(cid for cid, deg in indegree.items() if deg > 0)
            raise OntologyCycleError(f"Ontology contains a cycle involving {stuck!r}")

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
        """Return whether ``concept_id`` is a defined concept."""
        return concept_id in self._concepts

    def __getitem__(self, concept_id: str) -> OntologyConcept:
        """Return the concept for ``concept_id`` (raises ``KeyError`` if absent)."""
        return self._concepts[concept_id]

    def concept(self, concept_id: str) -> OntologyConcept:
        """Return the concept for ``concept_id`` (raises ``KeyError`` if absent)."""
        return self._concepts[concept_id]

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

    # --- queries ---

    def find(self, name: str) -> tuple[str, ...]:
        """
        Resolve a human-readable name (or exact id) to matching concept ids.

        Matching is case-insensitive over each concept's preferred label and
        synonyms. An exact id match is also returned.

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
        if name in self._concepts and name not in ids:
            ids.append(name)
        return tuple(dict.fromkeys(ids))

    def _require(self, concept_id: str) -> None:
        """Raise ``KeyError`` if ``concept_id`` is not a defined concept."""
        if concept_id not in self._concepts:
            raise KeyError(concept_id)

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
        self._require(concept_id)
        return tuple(self._ancestors(concept_id))

    def children(self, concept_id: str) -> tuple[str, ...]:
        """
        Return the ids of the direct subclasses (children) of ``concept_id``.

        Children are the defined concepts that declare ``concept_id`` among their
        ``parents``; order follows concept insertion order. Unlike
        :meth:`descendants` this is the immediate, non-transitive layer. Raises
        ``KeyError`` if ``concept_id`` is not a defined concept.
        """
        self._require(concept_id)
        return tuple(self._children.get(concept_id, ()))

    def descendants(self, concept_id: str) -> tuple[str, ...]:
        """
        Return all descendant concept ids of ``concept_id``, nearest-first.

        Descendants are the concept's transitive *subclasses* (narrower concepts).
        Raises ``KeyError`` if ``concept_id`` is not a defined concept.
        """
        self._require(concept_id)
        return tuple(_traverse(concept_id, self._children_of))

    def is_a(self, a: str, b: str) -> bool:
        """Return whether concept ``a`` is a (transitive) subclass of ``b``.

        Equivalently, whether ``b`` is an ancestor (superclass) of ``a``.
        Raises ``KeyError`` if ``a`` is not a defined concept; ``b`` may be any
        id, including an external reference.
        """
        self._require(a)
        return b in self._ancestors(a)

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
        self._require(a)
        self._require(b)
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
        self._require(concept_id)
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
        self._require(concept_id)
        memo: dict[str, int] = {}

        def depth(cid: str) -> int:
            if cid in memo:
                return memo[cid]
            parents = self._parents(cid)
            memo[cid] = 1 + max((depth(p) for p in parents), default=-1)
            return memo[cid]

        return depth(concept_id)

    def subtree(self, concept_id: str) -> "Ontology":
        """
        Return a new :class:`Ontology` rooted at ``concept_id``.

        Contains the concept and all its descendants; parent links pointing
        outside the subtree are pruned so ``concept_id`` becomes a root. Raises
        ``KeyError`` if ``concept_id`` is not a defined concept.
        """
        self._require(concept_id)
        node_ids = {concept_id, *self.descendants(concept_id)}
        concepts = []
        for nid in node_ids:
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
        ``skos:broader``. For each: ``label`` is ``skos:prefLabel`` (falling back
        to ``rdfs:label``), ``synonyms`` are ``skos:altLabel`` (plus a differing
        ``rdfs:label``), ``parents`` are the IRI objects of ``rdfs:subClassOf`` /
        ``skos:broader``, and ``definition`` is ``skos:definition``. Blank-node
        superclasses (e.g. ``owl:Restriction``) are ignored.

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
        for predicate in (RDFS.subClassOf, SKOS.broader):
            subjects.update(s for s in graph.subjects(predicate, None) if isinstance(s, URIRef))

        concepts = [_concept_from_graph(graph, subject) for subject in subjects]
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
    return [OntologyConcept(id=name, label=name, parents=tuple(parents[name])) for name in order]


def _concept_from_graph(graph: "rdflib.Graph", subject: "rdflib.URIRef") -> OntologyConcept:  # noqa: C901
    from rdflib import URIRef
    from rdflib.namespace import RDFS, SKOS

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

    return OntologyConcept(
        id=str(subject),
        label=label,
        synonyms=tuple(dict.fromkeys(synonyms)),
        parents=tuple(dict.fromkeys(parents)),
        definition=first_literal(SKOS.definition),
    )

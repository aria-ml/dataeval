"""Reconcile dataset class labels against an :class:`Ontology` and recover hierarchy."""

__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypedDict

from dataeval._ontology import Ontology

Relation = Literal["ancestor", "descendant", "sibling", "unrelated"]


class LabelReconciliationResult(TypedDict):
    """
    Result of reconciling class labels against an ontology.

    Attributes
    ----------
    matched : Mapping[str, str]
        Class name to the single concept id it resolved to.
    unmatched : Sequence[str]
        Class names that resolved to no concept.
    ambiguous : Mapping[str, Sequence[str]]
        Class names that resolved to more than one candidate concept id.
    ancestor_paths : Mapping[str, Sequence[str]]
        For each matched class name, its ancestor concept ids nearest-first
        (the is-a path toward the root, possibly ending at an external id).
    external_ancestors : Mapping[str, Sequence[str]]
        For each matched class name whose is-a path is truncated, the subset of
        its ancestor ids that are not defined in the ontology (external
        references). Present (non-empty) only for incomplete/subset ontologies; signals
        that the hierarchy above that label is unresolved rather than rooted.
    induced_edges : Sequence[tuple[str, str]]
        ``(parent_name, child_name)`` is-a edges of the sub-hierarchy induced by
        the matched classes — the transitive reduction restricted to those
        classes (intermediate concepts collapsed).
    relations : Mapping[tuple[str, str], str]
        For each ordered pair of distinct matched class names ``(a, b)``, the
        relation of ``a`` to ``b``: ``"ancestor"`` (a is a superclass of b),
        ``"descendant"`` (a is a subclass of b), ``"sibling"`` (shared ancestor
        but neither subsumes the other), or ``"unrelated"``.
    """

    matched: Mapping[str, str]
    unmatched: Sequence[str]
    ambiguous: Mapping[str, Sequence[str]]
    ancestor_paths: Mapping[str, Sequence[str]]
    external_ancestors: Mapping[str, Sequence[str]]
    induced_edges: Sequence[tuple[str, str]]
    relations: Mapping[tuple[str, str], Relation]


def _resolve(class_names: Iterable[str], ontology: Ontology) -> tuple[dict[str, str], list[str], dict[str, list[str]]]:
    matched: dict[str, str] = {}
    unmatched: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for name in dict.fromkeys(class_names):
        ids = ontology.find(name)
        if len(ids) == 0:
            unmatched.append(name)
        elif len(ids) == 1:
            matched[name] = ids[0]
        else:
            ambiguous[name] = list(ids)
    return matched, unmatched, ambiguous


def _induced_edges(matched: dict[str, str], ancestors: dict[str, set[str]]) -> list[tuple[str, str]]:
    # work in id-space, mapping each id back to its first matched name
    id_to_name: dict[str, str] = {}
    for name, cid in matched.items():
        id_to_name.setdefault(cid, name)
    ids = list(id_to_name)

    edges: list[tuple[str, str]] = []
    for child in ids:
        matched_ancestors = [a for a in ids if a != child and a in ancestors[child]]
        for parent in matched_ancestors:
            # keep the edge only if no other matched ancestor sits strictly between
            # child and parent (i.e. is a descendant of parent)
            between = any(other != parent and parent in ancestors[other] for other in matched_ancestors)
            if not between:
                edges.append((id_to_name[parent], id_to_name[child]))
    return edges


def _relation(a_id: str, b_id: str, ancestors: dict[str, set[str]]) -> Relation:
    # ancestors[x] is the set of x plus all its ancestors
    if a_id != b_id and b_id in ancestors[a_id]:
        return "descendant"
    if a_id != b_id and a_id in ancestors[b_id]:
        return "ancestor"
    if ancestors[a_id] & ancestors[b_id]:
        return "sibling"
    return "unrelated"


def _relations(matched: dict[str, str], ancestors: dict[str, set[str]]) -> dict[tuple[str, str], Relation]:
    names = list(matched)
    return {(a, b): _relation(matched[a], matched[b], ancestors) for a in names for b in names if a != b}


def label_reconciliation(class_names: Iterable[str], ontology: Ontology) -> LabelReconciliationResult:
    """
    Reconcile class labels against an ontology and recover their hierarchy.

    Reconciliation matches each class name to the ontology's concepts (by
    preferred label, synonyms, or exact id) and reports whether the label set
    conforms (every name matched unambiguously), alongside the recovered
    hierarchy of the matched classes.

    Parameters
    ----------
    class_names : Iterable[str]
        Dataset class names to reconcile, e.g. ``index2label.values()``.
    ontology : Ontology
        Ontology to reconcile against.

    Returns
    -------
    LabelReconciliationResult
        Match report (matched / unmatched / ambiguous), ancestor paths, the
        induced sub-hierarchy, and pairwise relations among matched classes.

    Notes
    -----
    Ambiguous names (resolving to more than one concept) are excluded from the
    hierarchy outputs; resolve them upstream (e.g. by passing concept ids).
    """
    matched, unmatched, ambiguous = _resolve(class_names, ontology)
    # Each matched concept's is-a path, computed once and reused below.
    ancestor_paths = {name: list(ontology.ancestors(cid)) for name, cid in matched.items()}
    external_ancestors = {
        name: ext for name, path in ancestor_paths.items() if (ext := [aid for aid in path if aid not in ontology])
    }
    # Ancestor set (self included) per matched id, derived from the paths above
    # so hierarchy reasoning needs no further graph traversal.
    ancestor_sets = {cid: {cid, *ancestor_paths[name]} for name, cid in matched.items()}
    return LabelReconciliationResult(
        matched=matched,
        unmatched=unmatched,
        ambiguous=ambiguous,
        ancestor_paths=ancestor_paths,
        external_ancestors=external_ancestors,
        induced_edges=_induced_edges(matched, ancestor_sets),
        relations=_relations(matched, ancestor_sets),
    )

"""Report structural and naming facts about an :class:`Ontology` artifact.

Where :func:`dataeval.core.label_reconciliation` validates a *dataset* against an
ontology, :func:`ontology_validation` validates the *ontology itself* — surfacing
the structural and naming facts (dangling ancestors, redundant edges, label
collisions, depth/breadth profile) needed to judge its quality. It deliberately
reports the *ingredients* and not a verdict: which findings constitute a failure,
and at what severity, is policy left to a downstream evaluator (the same way
``label_reconciliation`` reports ``matched``/``unmatched``/``ambiguous`` without
calling ``conforms``).
"""

__all__ = []

import re
from collections.abc import Mapping, Sequence
from typing import TypedDict

from dataeval._ontology import Ontology


class OntologyValidationResult(TypedDict):
    """
    Structural and naming facts about an :class:`Ontology` artifact.

    Reports the *ingredients* for judging ontology quality, not a pass/fail
    verdict. An empty finding collection is the raw "clean" signal, but whether a
    given finding (e.g. a dangling external ancestor) actually constitutes a
    failure is contextual policy for a downstream evaluator to decide.

    Attributes
    ----------
    roots : Sequence[str]
        Concept ids with no parents.
    leaves : Sequence[str]
        Concept ids with no children (most specific concepts).
    isolated : Sequence[str]
        Concept ids with neither parents nor children — disconnected singletons.
    external_ancestors : Mapping[str, Sequence[str]]
        For each concept whose is-a path is truncated, the subset of its ancestor
        ids that are not defined in the ontology (external references / "floating"
        ancestors). Empty unless the ontology is a subset of a fuller one.
    redundant_edges : Sequence[tuple[str, str]]
        ``(parent, child)`` direct is-a edges that are also implied transitively
        (``parent`` is reachable from ``child`` by another, longer path), so the
        direct edge is a non-reduced restatement of an existing subsumption.
    ancestor_siblings : Sequence[tuple[str, str]]
        ``(concept, ancestor)`` pairs where a concept shares a parent with one of
        its own ancestors — a contradictory placement (the concept is-a the
        ancestor yet is declared alongside it).
    unary_parents : Sequence[str]
        Concept ids with exactly one child — a single-child link adds depth
        without discriminating, a candidate redundant intermediate.
    label_collisions : Mapping[str, Sequence[str]]
        Normalized (case-folded) name to the more-than-one concept ids it resolves
        to over preferred labels and synonyms — the artifact-side cause of
        reconciliation ``ambiguous`` results.
    nonconforming_labels : Mapping[str, str]
        Concept id to its label, for labels that do not fully match
        ``label_pattern``. Always empty when ``label_pattern`` is ``None``.
    depth : Mapping[str, int]
        Concept id to the length of its longest is-a path from a root. The raw
        material for depth and depth-imbalance analysis.
    fan_out : Mapping[str, int]
        Concept id to its number of direct children. The raw material for breadth
        and over-broad-parent analysis.
    parent_count : Mapping[str, int]
        Concept id to its number of declared parents (multiple-inheritance load).
    """

    roots: Sequence[str]
    leaves: Sequence[str]
    isolated: Sequence[str]
    external_ancestors: Mapping[str, Sequence[str]]
    redundant_edges: Sequence[tuple[str, str]]
    ancestor_siblings: Sequence[tuple[str, str]]
    unary_parents: Sequence[str]
    label_collisions: Mapping[str, Sequence[str]]
    nonconforming_labels: Mapping[str, str]
    depth: Mapping[str, int]
    fan_out: Mapping[str, int]
    parent_count: Mapping[str, int]


def _external_ancestors(ontology: Ontology, ancestors: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Per concept, the ancestor ids not defined in the ontology (truncation points)."""
    result: dict[str, list[str]] = {}
    for cid, ancestor_ids in ancestors.items():
        external = [aid for aid in ancestor_ids if aid not in ontology]
        if external:
            result[cid] = external
    return result


def _is_implied(ancestors: Mapping[str, Sequence[str]], parent: str, siblings: Sequence[str]) -> bool:
    """Whether ``parent`` is a transitive ancestor of another (defined) of ``siblings``."""
    return any(other != parent and other in ancestors and parent in ancestors[other] for other in siblings)


def _redundant_edges(ontology: Ontology, ancestors: Mapping[str, Sequence[str]]) -> list[tuple[str, str]]:
    """Direct parent edges also implied by a longer path through another parent."""
    edges: list[tuple[str, str]] = []
    for concept in ontology:
        parents = concept.parents
        if len(parents) >= 2:
            edges.extend((parent, concept.id) for parent in parents if _is_implied(ancestors, parent, parents))
    return edges


def _ancestor_siblings(ontology: Ontology, ancestors: Mapping[str, Sequence[str]]) -> list[tuple[str, str]]:
    """Pairs where a concept shares a parent with one of its own ancestors."""
    pairs: list[tuple[str, str]] = []
    for concept in ontology:
        ancestor_set = set(ancestors[concept.id])
        pairs.extend((concept.id, sib) for sib in ontology.siblings(concept.id) if sib in ancestor_set)
    return pairs


def _nonconforming_labels(ontology: Ontology, label_pattern: str | None) -> dict[str, str]:
    """Concept id to label, for labels that do not fully match ``label_pattern``."""
    if label_pattern is None:
        return {}
    pattern = re.compile(label_pattern)
    return {c.id: c.label for c in ontology if not pattern.fullmatch(c.label)}


def ontology_validation(ontology: Ontology, *, label_pattern: str | None = None) -> OntologyValidationResult:
    """
    Validate an ontology artifact and report its structural and naming facts.

    Inspects the ontology's own graph and labels — independent of any dataset —
    and reports the connectivity, redundancy/contradiction, naming, and shape
    facts from which ontology quality can be judged. The result is verdict-free:
    it provides the ingredients (e.g. dangling ancestors, redundant edges, label
    collisions, per-concept depth and fan-out) for a downstream evaluator to turn
    into a pass/fail determination under its own policy and thresholds.

    Construction-time invariants (no duplicate ids, acyclic is-a graph) are
    already guaranteed by :class:`~dataeval.Ontology`, so a built ontology cannot
    violate them; this function reports the *legal-but-smelly* structure they do
    not preclude.

    Parameters
    ----------
    ontology : Ontology
        The ontology artifact to validate.
    label_pattern : str or None, optional
        A regular expression a concept label must fully match (via
        :func:`re.fullmatch`) to be considered well-formed; labels that do not are
        reported in ``nonconforming_labels``. The naming convention is policy, so
        the check is opt-in: when ``None`` (default) it is skipped and
        ``nonconforming_labels`` is empty. Pass e.g. ``r"[a-z0-9]+(_[a-z0-9]+)*"``
        to lint for ``lowercase_snake_case``.

    Returns
    -------
    OntologyValidationResult
        Connectivity (``roots``/``leaves``/``isolated``/``external_ancestors``),
        redundancy/contradiction (``redundant_edges``/``ancestor_siblings``/
        ``unary_parents``), naming (``label_collisions``/``nonconforming_labels``),
        and the per-concept shape profile (``depth``/``fan_out``/``parent_count``).

    See Also
    --------
    dataeval.core.label_reconciliation : Validate a dataset's labels *against* an
        ontology (rather than the ontology artifact itself).

    Notes
    -----
    Every finding identifies concepts by id. The findings are facts, not failures:
    a non-empty ``external_ancestors`` is expected for a deliberately distributed
    subset, for instance, and is only a defect for an ontology meant to be
    complete.
    """
    # Compute each concept's ancestors once and reuse across the helpers below,
    # rather than re-walking the is-a graph in each (cf. label_reconciliation).
    ancestors = {concept.id: ontology.ancestors(concept.id) for concept in ontology}
    roots = list(ontology.roots)
    leaves = list(ontology.leaves)
    leaf_set = set(leaves)
    fan_out = {c.id: len(ontology.children(c.id)) for c in ontology}

    return OntologyValidationResult(
        roots=roots,
        leaves=leaves,
        isolated=[cid for cid in roots if cid in leaf_set],
        external_ancestors=_external_ancestors(ontology, ancestors),
        redundant_edges=_redundant_edges(ontology, ancestors),
        ancestor_siblings=_ancestor_siblings(ontology, ancestors),
        unary_parents=[cid for cid, children in fan_out.items() if children == 1],
        label_collisions=ontology.label_collisions,
        nonconforming_labels=_nonconforming_labels(ontology, label_pattern),
        depth={c.id: ontology.depth_of(c.id) for c in ontology},
        fan_out=fan_out,
        parent_count={c.id: len(c.parents) for c in ontology},
    )

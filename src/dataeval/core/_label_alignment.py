"""Align a source label vocabulary against a target :class:`Ontology`.

Ontology alignment relates two label vocabularies by typed *correspondences*
(see :class:`dataeval.types.Correspondence`) so heterogeneous annotations can be
compared, scored across granularity, or combined. :func:`label_alignment` is the general
operation; label reconciliation (:func:`dataeval.core.label_reconciliation`) is
its restriction to a structureless source, exact matching, and equivalence alone.
"""

__all__ = []

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, TypedDict

from dataeval._ontology import Ontology
from dataeval.protocols import Matcher
from dataeval.types import AlignmentRelation, Correspondence, OntologyConcept

Mergeability = Literal["lossless", "lossy", "partial"]

# Relations that license carrying a source label over to the target vocabulary:
# equivalence (rename) and narrower (safe coarsening up the hierarchy).
_CARRYABLE: frozenset[AlignmentRelation] = frozenset({"equivalent", "narrower"})


class LabelAlignmentResult(TypedDict):
    """
    Result of aligning a source vocabulary against a target ontology.

    Attributes
    ----------
    correspondences : Sequence[Correspondence]
        Accepted correspondences, all relations: ``equivalent`` (exact or
        matcher), ``narrower`` (structurally entailed coarsening), and
        ``broader`` (granularity-mismatch diagnostics). A custom matcher may also
        contribute ``related``.
    unaligned_source : Sequence[str]
        Source concept ids with no *carryable* correspondence (not in ``class_remap``).
        Read open-world: out-of-vocabulary with respect to the target, not invalid.
    unaligned_target : Sequence[str]
        Target concept ids never referenced by any correspondence — the part of
        the reference vocabulary the source does not cover.
    class_remap : Mapping[str, str]
        ``source id -> target id`` for the correspondences that license a rewrite
        (``equivalent`` and ``narrower`` only): the safe carry-over of source
        labels into the target vocabulary. ``broader``/``related`` are excluded.
    mergeability : {"lossless", "lossy", "partial"}
        How completely the source is expressible in the target. ``"lossless"`` —
        every source concept is carryable and ``class_remap`` is injective (no two
        sources collapse to one target). ``"lossy"`` — every source is carryable
        but ``class_remap`` collapses two or more sources to a single target
        (specificity lost). ``"partial"`` — at least one source concept is
        unaligned (or only ``broader``/``related``) and cannot be carried over.
        Generalizes reconciliation *conformance* (conformant = lossless, all
        ``equivalent``).
    """

    correspondences: Sequence[Correspondence]
    unaligned_source: Sequence[str]
    unaligned_target: Sequence[str]
    class_remap: Mapping[str, str]
    mergeability: Mergeability


def _as_ontology(source: Ontology | Iterable[str]) -> Ontology:
    if isinstance(source, Ontology):
        return source
    if isinstance(source, str):
        raise TypeError("source must be an Ontology or a sequence of class names, not a single string")
    return Ontology.from_hierarchy(list(source))


def _candidates(concept: OntologyConcept, target: Ontology) -> tuple[str, ...]:
    """Target concept ids a source concept resolves to (label, synonyms, or id)."""
    found: list[str] = []
    for name in (concept.label, *concept.synonyms, concept.id):
        found.extend(target.find(name))
    return tuple(dict.fromkeys(found))


def _exact_anchor(source: Ontology, target: Ontology) -> list[Correspondence]:
    """Exact terminological matching: unique name hits become equivalences."""
    anchors: list[Correspondence] = []
    for concept in source:
        candidates = _candidates(concept, target)
        if len(candidates) == 1:
            anchors.append(
                Correspondence(source=concept.id, target=candidates[0], relation="equivalent", matcher="exact")
            )
    return anchors


def _accept_best(candidates: list[Correspondence]) -> Correspondence | None:
    """Accept the single highest-confidence proposal, or None if it is ambiguous."""
    best = max(c.confidence for c in candidates)
    top = [c for c in candidates if c.confidence == best]
    # Precision over recall: accept only when the best score points at one target.
    return top[0] if len({c.target for c in top}) == 1 else None


def _collect_proposals(
    source: Ontology,
    target: Ontology,
    matchers: Iterable[Matcher],
    anchored: set[str],
    threshold: float,
) -> dict[str, list[Correspondence]]:
    """Gather above-threshold matcher proposals for unanchored sources, by source."""
    proposals: dict[str, list[Correspondence]] = {}
    for matcher in matchers:
        for proposal in matcher(source, target):
            if proposal.source in anchored or proposal.confidence < threshold:
                continue
            proposals.setdefault(proposal.source, []).append(proposal)
    return proposals


def _apply_matchers(
    source: Ontology,
    target: Ontology,
    matchers: Iterable[Matcher],
    anchored: set[str],
    threshold: float,
) -> list[Correspondence]:
    """Run additional matchers on unanchored sources; accept the best per source."""
    proposals = _collect_proposals(source, target, matchers, anchored, threshold)
    return [best for candidates in proposals.values() if (best := _accept_best(candidates))]


def _propagate(
    source: Ontology,
    anchored: set[str],
    anchor_map: dict[str, str],
    anchor_conf: dict[str, float],
) -> list[Correspondence]:
    """Coarsen each unanchored source to its nearest equivalent-anchored ancestor."""
    narrower: list[Correspondence] = []
    for concept in source:
        if concept.id in anchored:
            continue
        for ancestor in source.ancestors(concept.id):
            if ancestor in anchor_map:
                narrower.append(
                    Correspondence(
                        source=concept.id,
                        target=anchor_map[ancestor],
                        relation="narrower",
                        confidence=anchor_conf[ancestor],
                        matcher="structural",
                    )
                )
                break
    return narrower


def _broader_diagnostics(
    equivalences: list[Correspondence],
    target: Ontology,
    class_remap: Mapping[str, str],
) -> list[Correspondence]:
    """Flag target subtrees finer than an equivalent source (granularity mismatch)."""
    reached = set(class_remap.values())
    diagnostics: list[Correspondence] = []
    for anchor in equivalences:
        if anchor.target not in target:
            continue
        diagnostics.extend(
            Correspondence(
                source=anchor.source,
                target=child,
                relation="broader",
                confidence=anchor.confidence,
                matcher="structural",
            )
            for child in target.children(anchor.target)
            if target.subtree_ids(child).isdisjoint(reached)
        )
    return diagnostics


def _mergeability(class_remap: Mapping[str, str], unaligned_source: Iterable[str]) -> Mergeability:
    if unaligned_source:
        return "partial"
    if len(set(class_remap.values())) < len(class_remap):
        return "lossy"
    return "lossless"


def label_alignment(
    source: Ontology | Iterable[str],
    target: Ontology,
    *,
    matchers: Iterable[Matcher] = (),
    threshold: float = 0.0,
) -> LabelAlignmentResult:
    """
    Align a source label vocabulary against a target ontology.

    Establishes typed :class:`~dataeval.types.Correspondence` objects from the
    source's concepts to the target's, in three passes: exact terminological
    anchoring (unique label/synonym/id matches become ``equivalent``), any
    additional ``matchers`` for the concepts left unanchored, then structural
    propagation that coarsens each still-unanchored source up to its nearest
    equivalent-anchored ancestor (``narrower``). ``broader`` correspondences are
    added as diagnostics where an equivalent source spans several finer target
    concepts. The result reports the safe label rewrite (``class_remap``), the
    open-world unaligned concepts on each side, and a ``mergeability`` summary.

    Parameters
    ----------
    source : Ontology or Iterable[str]
        The vocabulary to map *from*. A bare sequence of class names is treated
        as a structureless ontology (via :meth:`Ontology.from_hierarchy`); in
        that case ``label_alignment`` reduces to label reconciliation plus structural
        inference against the target.
    target : Ontology
        The reference vocabulary to map *to*.
    matchers : Iterable[Matcher], optional
        Additional element-level matchers (e.g. a fuzzy string-similarity matcher)
        consulted for source concepts the exact pass did not anchor. Each must
        implement the :class:`~dataeval.protocols.Matcher` protocol. Exact
        anchoring is always performed first.
    threshold : float, optional
        Minimum confidence for accepting a matcher's proposal, in ``[0, 1]``.
        Defaults to ``0.0`` (accept any proposal a matcher emits).

    Returns
    -------
    LabelAlignmentResult
        Correspondences, unaligned concepts on each side, the carry-over
        ``class_remap``, and the ``mergeability`` of the source into the target.

    See Also
    --------
    dataeval.core.label_reconciliation : The restriction of alignment to a
        single structureless source and exact equivalence.

    Notes
    -----
    Acceptance favors precision over recall: a source concept with no unique
    exact match and no above-threshold, unambiguous matcher proposal is left
    unanchored (and only coarsened if it has an anchored ancestor), rather than
    committing a likely-wrong correspondence.
    """
    src = _as_ontology(source)

    anchors = _exact_anchor(src, target)
    anchored = {c.source for c in anchors}
    anchors += _apply_matchers(src, target, matchers, anchored, threshold)
    anchored = {c.source for c in anchors}

    equivalences = [c for c in anchors if c.relation == "equivalent"]
    anchor_map = {c.source: c.target for c in equivalences}
    anchor_conf = {c.source: c.confidence for c in equivalences}
    narrower = _propagate(src, anchored, anchor_map, anchor_conf)

    class_remap = {c.source: c.target for c in (*anchors, *narrower) if c.relation in _CARRYABLE}
    broader = _broader_diagnostics(equivalences, target, class_remap)

    correspondences = [*anchors, *narrower, *broader]
    referenced = {c.target for c in correspondences}
    unaligned_source = tuple(c.id for c in src if c.id not in class_remap)
    unaligned_target = tuple(tid for tid in target.ids if tid not in referenced)

    return LabelAlignmentResult(
        correspondences=correspondences,
        unaligned_source=unaligned_source,
        unaligned_target=unaligned_target,
        class_remap=class_remap,
        mergeability=_mergeability(class_remap, unaligned_source),
    )

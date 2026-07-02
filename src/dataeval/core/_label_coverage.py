"""Project a dataset's label mass onto an :class:`Ontology` and report coverage.

Where :func:`label_reconciliation` checks that a dataset's class names *resolve*
to ontology concepts, :func:`label_coverage` goes a step further and reports *how
the dataset's label mass is distributed across the ontology's structure*: which
concepts and leaf species are populated, how breadth (siblings filled) and depth
(resolution reached) of coverage vary across the graph, and the empirical
distribution over leaves.

It is deliberately observation-only. It reports the dataset's *observed* coverage
and never an expectation or a recommendation: judging whether that coverage is
*sufficient* — and what to collect to improve it — requires an expected
distribution, which is policy left to a downstream evaluator (the same way
:func:`ontology_validation` reports structural facts without rendering a verdict).
A uniform prior, deficits, and a collection worklist all belong to that evaluator,
not here.
"""

__all__ = []

import logging
from collections.abc import Mapping, Sequence
from typing import TypedDict

from dataeval._ontology import Ontology
from dataeval.core._label_reconciliation import _resolve

_logger = logging.getLogger(__name__)


class LabelCoverageResult(TypedDict):
    """
    Observed distribution of a dataset's label mass over an :class:`Ontology`.

    Every per-concept mapping is keyed over *all* defined concepts, so an
    unlabeled concept appears with a zero/empty entry rather than being absent —
    that visibility of the unpopulated parts of the ontology is the whole point.
    All fields are observations; none assumes an expected distribution.

    Attributes
    ----------
    matched : Mapping[str, str]
        Dataset class name to the single concept id it resolved to. Distinct names
        may resolve to the same id (synonyms); their counts are summed downstream.
    unmatched : Mapping[str, int]
        Class name to its count, for names that resolved to no concept — label mass
        the ontology does not cover (a missing concept, or a junk label).
    ambiguous : Mapping[str, Sequence[str]]
        Class name to the more-than-one candidate concept ids it resolved to. Their
        counts are *not* attributed to any concept; resolve them upstream (e.g. by
        passing concept ids) to fold them into the coverage tallies.
    direct_count : Mapping[str, int]
        Concept id to the label mass landing *exactly* on it (0 when unlabeled).
        Labels may land on internal concepts, not only leaves.
    subtree_count : Mapping[str, int]
        Concept id to the mass on it plus all its descendants (its subtree). On a
        DAG a multi-parent concept contributes to every ancestor's subtree but is
        counted once per ancestor.
    covered_leaves : Mapping[str, tuple[int, int]]
        Concept id to ``(covered, total)`` leaf species in its subtree, where a leaf
        is *covered* if it has any direct mass. The breadth-of-coverage signal at a
        glance: ``(0, n)`` is a wholly dark branch.
    covered_children : Mapping[str, tuple[int, int]]
        Concept id to ``(covered, total)`` *direct* children whose subtree holds any
        mass — sibling fill under each parent. Leaves report ``(0, 0)``.
    coverage_by_depth : Mapping[int, tuple[int, int]]
        Is-a depth to ``(covered, total)`` concepts at that depth, where *covered*
        means the concept's subtree holds any mass. The depth profile of coverage.
    leaf_coverage : float
        Fraction of the ontology's leaf species with any direct mass — a single
        observed coverage scalar (no prior). ``0.0`` when the ontology has no leaves.
    leaf_distribution : Mapping[str, float]
        Leaf concept id to its share of total leaf-attributed mass (the entries sum
        to 1 over leaves, or are all ``0.0`` when no leaf is labeled). The empirical
        class distribution at the finest granularity, for an evaluator to compare
        against an expected one.
    """

    matched: Mapping[str, str]
    unmatched: Mapping[str, int]
    ambiguous: Mapping[str, Sequence[str]]
    direct_count: Mapping[str, int]
    subtree_count: Mapping[str, int]
    covered_leaves: Mapping[str, tuple[int, int]]
    covered_children: Mapping[str, tuple[int, int]]
    coverage_by_depth: Mapping[int, tuple[int, int]]
    leaf_coverage: float
    leaf_distribution: Mapping[str, float]


def _direct_counts(label_counts: Mapping[str, int], matched: Mapping[str, str], ids: Sequence[str]) -> dict[str, int]:
    """Mass landing exactly on each concept, summing synonyms; 0 for unlabeled concepts."""
    direct = dict.fromkeys(ids, 0)
    for name, cid in matched.items():
        direct[cid] += int(label_counts[name])
    return direct


def _subtree_counts(
    direct: Mapping[str, int], ancestors: Mapping[str, Sequence[str]], ontology: Ontology
) -> dict[str, int]:
    """Mass on each concept plus its descendants, propagated up de-duplicated ancestors."""
    subtree = dict(direct)
    for cid, mass in direct.items():
        if mass:
            # ancestors are de-duplicated, so a multi-parent concept adds its mass
            # to each distinct ancestor exactly once; external ancestors are not concepts.
            for ancestor in ancestors[cid]:
                if ancestor in ontology:
                    subtree[ancestor] += mass
    return subtree


def _covered_leaves(
    direct: Mapping[str, int], leaves: Sequence[str], ancestors: Mapping[str, Sequence[str]], ontology: Ontology
) -> dict[str, tuple[int, int]]:
    """Per concept, ``(covered, total)`` leaf species in its subtree (covered = has direct mass)."""
    tally = {cid: [0, 0] for cid in ancestors}
    for leaf in leaves:
        is_covered = direct[leaf] > 0
        for cid in (leaf, *(a for a in ancestors[leaf] if a in ontology)):
            tally[cid][1] += 1
            if is_covered:
                tally[cid][0] += 1
    return {cid: (counts[0], counts[1]) for cid, counts in tally.items()}


def label_coverage(label_counts: Mapping[str, int], ontology: Ontology) -> LabelCoverageResult:
    """
    Report how a dataset's label mass is distributed over an ontology.

    Resolves each dataset class name against the ontology (by preferred label,
    synonym, or exact id), attributes its count to the matched concept, and reports
    the resulting coverage of the ontology's structure — direct and subtree mass per
    concept, leaf and sibling coverage, the depth profile, and the empirical leaf
    distribution. The result is observation-only: it describes what the dataset
    *does* cover, leaving any notion of an expected distribution, sufficiency
    threshold, or collection recommendation to a downstream evaluator.

    Parameters
    ----------
    label_counts : Mapping[str, int]
        Dataset class name to its label count (e.g. ``label_stats(...)`` counts
        mapped through ``index2label``). Counts are instance counts; for object
        detection that is detections-per-class, for classification images-per-class.
    ontology : Ontology
        Ontology whose concepts define the space coverage is measured against.

    Returns
    -------
    LabelCoverageResult
        Resolution facts (``matched`` / ``unmatched`` / ``ambiguous``), per-concept
        mass (``direct_count`` / ``subtree_count``), breadth and depth coverage
        (``covered_leaves`` / ``covered_children`` / ``coverage_by_depth`` /
        ``leaf_coverage``), and the empirical ``leaf_distribution``.

    See Also
    --------
    dataeval.core.label_reconciliation : Resolve labels against an ontology and
        recover their hierarchy (the matching this builds on).
    dataeval.core.ontology_validation : Report structural facts about the ontology
        artifact itself, independent of any dataset.

    Notes
    -----
    Ambiguous names (resolving to more than one concept) carry mass that cannot be
    attributed to a single concept, so they are reported but excluded from the
    coverage tallies. Names resolving to no concept are reported in ``unmatched``
    with their counts, since that mass signals concepts the ontology is missing.

    Examples
    --------
    >>> from dataeval import Ontology
    >>> ontology = Ontology.from_hierarchy({"animal": {"mammal": ["cat", "dog"], "bird": ["owl", "hawk"]}})
    >>> counts = {"cat": 8, "dog": 2, "owl": 1}  # hawk never collected
    >>> result = label_coverage(counts, ontology)
    >>> result["leaf_coverage"]  # 3 of 4 leaf species have any examples
    0.75
    >>> result["covered_leaves"]["bird"]  # one of two bird species populated
    (1, 2)
    >>> result["coverage_by_depth"]  # (covered, total) per is-a depth
    {0: (1, 1), 1: (2, 2), 2: (3, 4)}
    >>> result["subtree_count"]["mammal"]  # cat + dog mass rolls up to mammal
    10
    >>> result["leaf_distribution"]["cat"]  # 8 of 11 leaf-attributed labels
    0.7272727272727273

    A class name absent from the ontology is reported as unmatched mass:

    >>> label_coverage({"cat": 5, "unicorn": 3}, ontology)["unmatched"]
    {'unicorn': 3}
    """
    _logger.info("Starting label_coverage over ontology with %d concepts", len(ontology))

    # Each concept's ancestors, computed once and reused across the helpers below
    # rather than re-walking the (un-memoized) is-a graph in each (cf. ontology_validation).
    ids = ontology.ids
    leaves = ontology.leaves
    ancestors = {cid: ontology.ancestors(cid) for cid in ids}

    # Resolve names against the ontology, reusing reconciliation's matcher so
    # matched/unmatched/ambiguous mean exactly what they do there.
    matched, unmatched_names, ambiguous = _resolve(label_counts, ontology)
    unmatched = {name: int(label_counts[name]) for name in unmatched_names}

    direct = _direct_counts(label_counts, matched, ids)
    subtree = _subtree_counts(direct, ancestors, ontology)
    covered_leaves = _covered_leaves(direct, leaves, ancestors, ontology)

    covered_children: dict[str, tuple[int, int]] = {}
    for cid in ids:
        children = ontology.children(cid)
        covered_children[cid] = (sum(subtree[ch] > 0 for ch in children), len(children))

    by_depth: dict[int, list[int]] = {}
    for cid in ids:
        slot = by_depth.setdefault(ontology.depth_of(cid), [0, 0])
        slot[1] += 1
        if subtree[cid] > 0:
            slot[0] += 1
    coverage_by_depth = {depth: (counts[0], counts[1]) for depth, counts in sorted(by_depth.items())}

    leaf_mass = sum(direct[leaf] for leaf in leaves)
    leaf_distribution = {leaf: (direct[leaf] / leaf_mass if leaf_mass else 0.0) for leaf in leaves}
    leaf_coverage = (sum(direct[leaf] > 0 for leaf in leaves) / len(leaves)) if leaves else 0.0

    _logger.info(
        "label_coverage complete: %d matched, %d unmatched, %d ambiguous, leaf_coverage=%.4f",
        len(matched),
        len(unmatched),
        len(ambiguous),
        leaf_coverage,
    )

    return LabelCoverageResult(
        matched=matched,
        unmatched=unmatched,
        ambiguous={name: list(ids) for name, ids in ambiguous.items()},
        direct_count=direct,
        subtree_count=subtree,
        covered_leaves=covered_leaves,
        covered_children=covered_children,
        coverage_by_depth=coverage_by_depth,
        leaf_coverage=leaf_coverage,
        leaf_distribution=leaf_distribution,
    )

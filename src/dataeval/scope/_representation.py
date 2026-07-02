"""Evaluate how well a dataset represents an :class:`Ontology`'s label space.

Where :func:`dataeval.core.label_coverage` reports the *observed* distribution of a
dataset's label mass over an ontology (facts, no prior), :class:`Representation` is
the evaluator on top: it overlays an *expected* distribution and turns the gap into
a prioritized collection worklist — which concepts to acquire, which to augment, by
how much — plus a check of any explicit representation assertions. The expected
distribution is policy and lives here, not in ``core``: by default every leaf
species is expected to be evenly represented, but a caller who knows a class is
genuinely rare can assert a minimum share for it instead.
"""

__all__ = ["Representation", "RepresentationOutput"]

import logging
from collections.abc import Mapping
from typing import Any

import polars as pl

from dataeval import Metadata
from dataeval._ontology import Ontology
from dataeval.core._label_coverage import LabelCoverageResult, label_coverage
from dataeval.core._label_stats import label_stats
from dataeval.protocols import AnnotatedDataset
from dataeval.types import DataFrameOutput, Evaluator, EvaluatorConfig, set_metadata

_logger = logging.getLogger(__name__)

_WORKLIST_SCHEMA: dict[str, pl.DataType] = {
    "concept": pl.String(),
    "label": pl.String(),
    "parent": pl.String(),
    "action": pl.String(),
    "count": pl.Int64(),
    "target": pl.Int64(),
    "deficit": pl.Int64(),
}
_VIOLATIONS_SCHEMA: dict[str, pl.DataType] = {
    "concept": pl.String(),
    "label": pl.String(),
    "floor": pl.Float64(),
    "actual": pl.Float64(),
    "shortfall": pl.Int64(),
}
_DARK_BRANCHES_SCHEMA: dict[str, pl.DataType] = {
    "concept": pl.String(),
    "label": pl.String(),
    "leaves": pl.Int64(),
}


class RepresentationOutput(DataFrameOutput):
    """
    A dataset's collection worklist against an :class:`Ontology`.

    The wrapped DataFrame is the worklist itself — one row per leaf species short of
    its expected share, sorted by ``deficit`` (largest first) — with columns
    ``concept``, ``label``, ``parent``, ``action`` (``"acquire"`` for unrepresented
    species, ``"augment"`` for under-represented ones), ``count``, ``target``, and
    ``deficit``. The summary scalars and supporting frames hang off it as attributes.

    Attributes
    ----------
    leaf_coverage : float
        Fraction of the ontology's leaf species with any examples (carried through
        from :func:`~dataeval.core.label_coverage`; observation, not policy).
    total_deficit : int
        Sum of all positive deficits — an estimate of how many labels the dataset is
        short of its expected distribution. The single budgeting number.
    violations : polars.DataFrame
        One row per asserted class (from ``expected``) whose observed share falls
        below its floor: ``concept``, ``label``, ``floor``, ``actual``, ``shortfall``.
        Empty when no assertions were made or all held.
    dark_branches : polars.DataFrame
        Maximal wholly-unpopulated internal branches, largest first: ``concept``,
        ``label``, ``leaves`` (leaf species under that branch). The branch-level
        headline above the per-species worklist.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        leaf_coverage: float,
        total_deficit: int,
        violations: pl.DataFrame,
        dark_branches: pl.DataFrame,
    ) -> None:
        super().__init__(data)
        self.leaf_coverage = leaf_coverage
        self.total_deficit = total_deficit
        self.violations = violations
        self.dark_branches = dark_branches


class Representation(Evaluator):
    """
    Evaluate a dataset's coverage of an ontology and prioritize what to collect.

    Resolves the dataset's class labels against the ontology, compares the observed
    distribution to an expected one, and returns a :class:`RepresentationOutput`
    worklist of the leaf species to acquire or augment. The default expectation is a
    uniform distribution over leaf species; pass ``expected`` to assert a minimum
    share (a fraction of the whole dataset) for specific classes you know to be rare,
    which both right-sizes their target and is validated as an assertion.

    Parameters
    ----------
    ontology : Ontology
        Ontology whose leaf species define the label space to cover.
    expected : Mapping[str, float] or None, default None
        Class name to its minimum expected share of the dataset (a fraction in
        ``[0, 1]``). Named classes use this floor as their target in place of the
        uniform share, and are validated in :attr:`RepresentationOutput.violations`;
        unnamed classes keep the uniform target. ``None`` means a uniform expectation
        for every leaf.
    config : Representation.Config or None, default None
        Optional configuration object; parameters passed directly to ``__init__``
        override its values.

    See Also
    --------
    dataeval.core.label_coverage : The observation-only coverage facts this builds on.
    dataeval.core.label_reconciliation : Resolve labels against an ontology.

    Notes
    -----
    Targets are rounded to the nearest whole label. A class named in ``expected``
    that does not resolve to exactly one concept is ignored (resolve it upstream).

    Examples
    --------
    >>> from dataeval import Ontology
    >>> from dataeval.scope import Representation
    >>> ontology = Ontology.from_hierarchy({"animal": {"mammal": ["cat", "dog"], "bird": ["owl"]}})
    >>> result = Representation(ontology).evaluate(dataset)
    >>> result.columns
    ['concept', 'label', 'parent', 'action', 'count', 'target', 'deficit']

    Assert that a known-rare class need only make up 5% of the dataset:

    >>> result = Representation(ontology, expected={"owl": 0.05}).evaluate(dataset)
    >>> result.violations.columns
    ['concept', 'label', 'floor', 'actual', 'shortfall']
    """

    class Config(EvaluatorConfig):
        """
        Configuration for the :class:`Representation` evaluator.

        Attributes
        ----------
        expected : Mapping[str, float] or None, default None
            Class name to its minimum expected share of the dataset. ``None`` means a
            uniform expectation across leaf species.
        """

        expected: Mapping[str, float] | None = None

    # Set by apply_config from Config.
    expected: Mapping[str, float] | None
    config: Config

    def __init__(
        self,
        ontology: Ontology,
        *,
        expected: Mapping[str, float] | None = None,
        config: Config | None = None,
    ) -> None:
        # ontology is a fixed reference, not a tunable config value, so it is kept off
        # the config the way Prioritize keeps its reference set off the config.
        super().__init__(locals(), exclude={"ontology"})
        self._ontology = ontology

    def _label(self, concept_id: str) -> str:
        """Human-readable label for a concept id (the id itself if external)."""
        return self._ontology[concept_id].label if concept_id in self._ontology else concept_id

    def _expected_by_concept(self) -> dict[str, float]:
        """Resolve ``expected`` class names to the single concept id each names."""
        resolved: dict[str, float] = {}
        for name, floor in (self.expected or {}).items():
            ids = self._ontology.find(name)
            if len(ids) == 1:
                resolved[ids[0]] = floor
            else:
                _logger.warning("expected name %r resolved to %d concepts; ignoring", name, len(ids))
        return resolved

    def _worklist(self, direct: Mapping[str, int], floors: Mapping[str, float], total: int) -> list[dict[str, Any]]:
        """Return every leaf short of its (asserted-or-uniform) target, largest deficit first."""
        leaves = self._ontology.leaves
        uniform = 1 / len(leaves) if leaves else 0.0
        rows: list[dict[str, Any]] = []
        for leaf in leaves:
            count = direct[leaf]
            target = round(floors.get(leaf, uniform) * total)
            if (deficit := target - count) > 0:
                rows.append({
                    "concept": leaf,
                    "label": self._label(leaf),
                    "parent": ", ".join(self._label(p) for p in self._ontology[leaf].parents),
                    "action": "acquire" if count == 0 else "augment",
                    "count": count,
                    "target": target,
                    "deficit": deficit,
                })
        return sorted(rows, key=lambda row: (-row["deficit"], row["concept"]))

    def _violations(self, direct: Mapping[str, int], floors: Mapping[str, float], total: int) -> list[dict[str, Any]]:
        """Return asserted classes whose observed share is below their floor."""
        rows: list[dict[str, Any]] = []
        for cid, floor in floors.items():
            count = direct.get(cid, 0)
            actual = count / total if total else 0.0
            if actual < floor:
                rows.append({
                    "concept": cid,
                    "label": self._label(cid),
                    "floor": floor,
                    "actual": actual,
                    "shortfall": round(floor * total) - count,
                })
        return rows

    def _dark_branches(self, coverage: LabelCoverageResult) -> list[dict[str, Any]]:
        """Return maximal internal subtrees with no mass anywhere under them, largest first."""
        subtree = coverage["subtree_count"]
        covered_leaves = coverage["covered_leaves"]
        covered_children = coverage["covered_children"]
        rows: list[dict[str, Any]] = []
        for cid in self._ontology.ids:
            # A branch is maximal when no parent is also dark: an empty parent's
            # subtree contains this one, so it (not this node) is the topmost dark root.
            # Direct parents suffice — a dark ancestor implies a dark parent on its path.
            is_internal = covered_children[cid][1] > 0
            no_dark_parent = not any(p in subtree and subtree[p] == 0 for p in self._ontology[cid].parents)
            if subtree[cid] == 0 and is_internal and no_dark_parent:
                rows.append({"concept": cid, "label": self._label(cid), "leaves": covered_leaves[cid][1]})
        return sorted(rows, key=lambda row: (-row["leaves"], row["concept"]))

    def _build_output(self, coverage: LabelCoverageResult, total: int) -> RepresentationOutput:
        direct = coverage["direct_count"]
        floors = self._expected_by_concept()
        worklist = self._worklist(direct, floors, total)
        return RepresentationOutput(
            pl.DataFrame(worklist, schema=_WORKLIST_SCHEMA),
            leaf_coverage=coverage["leaf_coverage"],
            total_deficit=sum(row["deficit"] for row in worklist),
            violations=pl.DataFrame(self._violations(direct, floors, total), schema=_VIOLATIONS_SCHEMA),
            dark_branches=pl.DataFrame(self._dark_branches(coverage), schema=_DARK_BRANCHES_SCHEMA),
        )

    @set_metadata
    def evaluate(self, data: AnnotatedDataset[Any] | Metadata) -> RepresentationOutput:
        """
        Evaluate a dataset's coverage of the ontology.

        Parameters
        ----------
        data : AnnotatedDataset or Metadata
            The dataset (or its :class:`~dataeval.Metadata`) to evaluate. Class labels
            and the ``index2label`` mapping are read from it; raw label counts are
            derived via :func:`~dataeval.core.label_stats`.

        Returns
        -------
        RepresentationOutput
            The collection worklist (``acquire`` / ``augment`` rows) with
            ``leaf_coverage``, ``total_deficit``, ``violations``, and ``dark_branches``.

        Examples
        --------
        >>> ontology = Ontology.from_hierarchy({
        ...     "vehicle": {"land": ["car", "bike"], "water": ["boat"], "air": ["plane"]}
        ... })
        >>> evaluator = Representation(ontology)
        >>> result = evaluator.evaluate(dataset)
        >>> result.data()
        shape: (2, 7)
        ┌─────────┬───────┬────────┬─────────┬───────┬────────┬─────────┐
        │ concept ┆ label ┆ parent ┆ action  ┆ count ┆ target ┆ deficit │
        │ ---     ┆ ---   ┆ ---    ┆ ---     ┆ ---   ┆ ---    ┆ ---     │
        │ str     ┆ str   ┆ str    ┆ str     ┆ i64   ┆ i64    ┆ i64     │
        ╞═════════╪═══════╪════════╪═════════╪═══════╪════════╪═════════╡
        │ bike    ┆ bike  ┆ land   ┆ acquire ┆ 0     ┆ 23     ┆ 23      │
        │ boat    ┆ boat  ┆ water  ┆ augment ┆ 22    ┆ 23     ┆ 1       │
        └─────────┴───────┴────────┴─────────┴───────┴────────┴─────────┘
        >>> result.total_deficit
        24
        >>> result.leaf_coverage
        0.75
        """
        metadata = data if isinstance(data, Metadata) else Metadata(data)
        stats = label_stats(metadata.class_labels, index2label=metadata.index2label)
        index2label = stats["index2label"]
        label_counts = {index2label[idx]: count for idx, count in stats["label_counts_per_class"].items()}
        coverage = label_coverage(label_counts, self._ontology)
        return self._build_output(coverage, total=sum(label_counts.values()))

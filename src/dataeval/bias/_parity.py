__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import polars as pl
from pydantic import Field

from dataeval import Metadata
from dataeval._experimental import experimental
from dataeval._helpers import (
    axis_record,
    factor_code_names,
    factors_excluding,
    independence_groups,
    is_metadata_like,
    lexical_categorical,
    resolve_label_axis,
)
from dataeval._log import get_logger
from dataeval.core._parity import parity
from dataeval.protocols import AnnotatedDataset, MetadataLike
from dataeval.types import ClassAxis, DictOutput, Evaluator, EvaluatorConfig, set_metadata
from dataeval.types._config import OpenUnitInterval, UnitInterval

_logger = get_logger(__name__)

DEFAULT_PARITY_SCORE_THRESHOLD = 0.3
DEFAULT_PARITY_P_VALUE_THRESHOLD = 0.05
DEFAULT_PARITY_LIFT_THRESHOLD = 1.25

FACTOR_STATUS = pl.Enum(["associated", "weak", "inconclusive", "not_detected", "untestable"])
CELL_STATUS = pl.Enum(["underrepresented", "overrepresented", "too_thin", "no_finding", "untestable"])

_STATE = [
    "score_threshold",
    "p_value_threshold",
    "lift_threshold",
    "assume_independent",
    "group_source",
    "test",
    "label",
    "class_axis",
    "class_axis_source",
    "class_axis_level",
    "encoding_digest",
]


@experimental
@dataclass(frozen=True, repr=False)
class ParityOutput(DictOutput):
    """
    Output class for the :class:`.Parity` :term:`bias<Bias>` evaluator.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Two frames: one row per factor, and one row per (factor, level, class) cell.

    Attributes
    ----------
    factors : pl.DataFrame
        One row per factor, in the metadata's factor order, with columns:

        - factor_name: str - Name of the metadata factor
        - status: Enum - ``associated``, ``weak``, ``inconclusive``, ``not_detected`` or
          ``untestable`` (see :class:`.Parity`)
        - score: float - Bias-corrected Cramér's V, NaN when untestable
        - p_value: float - Raw p-value of the homogeneity test
        - p_value_adjusted: float - Benjamini-Hochberg adjusted over the testable factors
        - is_associated: bool - ``score >= score_threshold`` and
          ``p_value_adjusted <= p_value_threshold``; False when untestable
        - has_insufficient_data: bool - The table is too thin to trust the reference
          distribution: Cochran's criterion on the expected counts (divided by the design
          effect), or fewer than 4 design degrees of freedom. A screen, not a guarantee
        - design_effect: float - Mean generalized design effect; 1 for independent rows
        - effective_groups: float - Kish effective number of groups behind the thinnest
          level; the row count for independent rows
        - df_num, df_den: float - Degrees of freedom of the reference distribution;
          ``df_den`` is infinite for independent rows
    cells : pl.DataFrame
        One row per (factor, observed level, observed class), observed zeros included,
        sorted by ``factor_name``, then ``status`` in the order below, then ``deficit``
        descending (a convenience order, not a priority). Columns:

        - factor_name, level, class_name: str - The cell, decoded to names where the
          metadata records them and to stringified codes otherwise
        - status: Enum - ``underrepresented``, ``overrepresented``, ``too_thin``,
          ``no_finding`` or ``untestable`` (see :class:`.Parity`)
        - is_actionable: bool - ``status`` is ``underrepresented`` or ``overrepresented``
        - observed: int - Rows in the cell
        - expected: float - Rows expected under homogeneity
        - deficit: float - ``expected - observed``
        - lift: float - ``observed / expected``
        - p_value: float - Exact two-sided p-value, NaN when untestable
        - p_value_adjusted: float - Benjamini-Hochberg adjusted over the testable cells
        - is_deviant: bool - Testable and ``p_value_adjusted <= p_value_threshold``,
          whatever the lift or sparsity
        - is_sparse: bool - Too little expected data, or too few effective groups, to rely
          on the cell
    class_axis : ClassAxis or None
        What this result conditioned on: the dataset's own labels, or an axis a caller
        defined through :meth:`~dataeval.Metadata.classed_by` or ``label=``. Read
        ``class_axis.source`` to tell the two apart without parsing a name, and
        ``class_axis.rows_per_group_entity`` to see whether a coarse axis was replicated
        onto finer rows. None only on an output built by hand rather than by an
        evaluator, which is the one case where nothing resolved an axis.

        .. versionadded:: 1.2
    test : str or None
        ``"pearson"`` for independent rows, ``"rao_scott_f"`` for grouped rows.

        .. versionadded:: 1.2
    group_source : str or None
        Where the independent sampling unit came from: ``"level"`` (the metadata's item
        level), ``"item"`` (declared ``item_indices``), ``"rows"`` (the recorded structure
        has one row per item) or ``"declared"`` (``assume_independent=True``).

        .. versionadded:: 1.2
    group_level : str or None
        The metadata level the rows were grouped by, for ``group_source="level"``.

        .. versionadded:: 1.2
    n_rows : int or None
        Number of rows tested.

        .. versionadded:: 1.2
    n_groups : int or None
        Number of independent groups; ``n_rows`` for independent rows.

        .. versionadded:: 1.2
    """

    factors: pl.DataFrame
    cells: pl.DataFrame
    class_axis: ClassAxis | None = None
    test: str | None = None
    group_source: str | None = None
    group_level: str | None = None
    n_rows: int | None = None
    n_groups: int | None = None


@experimental
class Parity(Evaluator):
    """
    Test whether metadata factors are associated with the class labels, factor by factor and cell by cell.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    For each factor, tests **label-proportion homogeneity**: the class distribution of
    rows is the same at every factor level. Groups (images, sequences) are independent of
    each other, and rows inside a group may depend on each other in any way. Each factor
    gets a bias-corrected Cramér's V as its effect size, and each (level, class) cell an
    exact test of whether it holds more or fewer rows than homogeneity predicts. A
    shortcut shows up as an over-represented cell, a collection gap as an
    under-represented one.

    Parameters
    ----------
    score_threshold : float, default 0.3
        Cramér's V at or above which a significant factor is ``associated`` rather than
        ``weak``.
    p_value_threshold : float, default 0.05
        Largest Benjamini-Hochberg **adjusted** p-value that counts as significant, for
        factors and cells alike.
    lift_threshold : float, default 1.25
        Smallest multiplicative deviation that makes a significant cell actionable: lift at
        or below ``1/lift_threshold`` or at or above ``lift_threshold``. Must be at least
        1; 1 disables the band. An interpretable practical-significance default, not a
        statistically derived optimum.
    label : str or Sequence[str] or None, default None
        Factor(s) to condition on instead of the class labels. None reads
        :attr:`~dataeval.Metadata.class_labels`, which requires the metadata be viewed
        at its label level; naming a factor is how the same question is asked at a
        coarser view, where there is no single class label per row. Several names are
        combined into one composite axis.
    assume_independent : bool, default False
        Declare each row its own independent item. Needed only for a custom container that
        records neither :class:`~dataeval.Metadata` levels nor ``item_indices``; the
        recorded structure always wins over it.

    Attributes
    ----------
    metadata : MetadataLike
        Preprocessed metadata
    score_threshold : float
        Cramér's V threshold for ``associated``
    p_value_threshold : float
        Adjusted p-value threshold for statistical significance
    lift_threshold : float
        Lift band for actionable cells
    label : str or Sequence[str] or None
        Factor(s) conditioned on instead of the class labels
    assume_independent : bool
        Whether the caller declared each row independent
    group_source : str or None
        Where the last call's independent sampling unit came from; None before a call
    test : str or None
        The factor test the last call ran, ``"pearson"`` or ``"rao_scott_f"``; None
        before a call

    Notes
    -----
    **The independent sampling unit (D2).** Parity must know which rows are independent.
    A :class:`~dataeval.Metadata` groups by its ``item_indices`` at any view: detections
    by image, frames by sequence (``group_source="level"``). A container with
    many-to-one ``item_indices`` groups by them (``"item"``). One row per item makes the
    rows the unit (``"rows"``). A container recording neither raises ``ValueError``
    unless ``assume_independent=True`` (``"declared"``), because row-level inference gives
    invalid p-values when several rows belong to one image or sequence. Dependence
    *between* groups, such as consecutive sequences from one flight, is not corrected.

    **The tests.** Independent rows get Pearson's χ² and exact hypergeometric cells.
    Grouped rows get the second-order Rao-Scott F, its denominator df scaled by the Kish
    effective number of groups behind the factor's thinnest level, and exact Korn-Graubard
    cells on effective counts, each cell's variance bounded below by a Kish cluster-model
    variance whose correlations come from the class at other levels or from the level's
    other classes. See :func:`~dataeval.core.parity`.

    **Factor status**, in order of precedence:

    - ``untestable``: fewer than two observed levels or classes, fewer than two groups, or
      too few effective groups; the design cannot answer the question
    - ``associated``: ``is_associated``
    - ``weak``: significant, with V below ``score_threshold``
    - ``inconclusive``: not significant, and ``has_insufficient_data``; not enough evidence
      to call it a negative
    - ``not_detected``: not significant with sufficient data

    **Cell status (D1)**, in order of precedence:

    - ``untestable``: the cell's factor is untestable, or its variance is degenerate
    - ``too_thin``: ``is_sparse``; collect to find out, never a finding whatever its p
    - ``underrepresented``: ``is_deviant`` and lift at or below ``1/lift_threshold``; a
      collection candidate
    - ``overrepresented``: ``is_deviant`` and lift at or above ``lift_threshold``;
      shortcut evidence
    - ``no_finding``: otherwise, including significant cells inside the lift band

    No cell judgment depends on the factor's V: a large deficit in a factor with low V is
    still ``underrepresented``. P-values are Benjamini-Hochberg adjusted across the
    testable factors, and separately across the testable cells; cells share their
    table's margins, so this is not an unconditional false discovery rate guarantee.

    **Known limit of the grouped cell tests.** They can raise false alarms when a class
    comes in bursts (most rows of an image or sequence are that class) at one level of a
    factor only, and that level's other classes do not burst; with two classes there are
    no others. A cell of that class at that level that by chance holds none of its bursts
    looks like a real shortage, and no estimate from the data tells the two apart. Look
    for it as a rare class seen in one or a few crowded images, all at one level, or a
    factor whose levels each have their own bursty class. In simulated nulls, the share of
    tables with a falsely flagged cell (nominal 5%) was 11-17% with
    two classes and about two bursty images at one of 3 to 10 levels, 3-10% for a single
    such cell in a 2×2 to 3×4 table, and 26-33% and 49-54% when each of 20 or 40 levels burst
    in its own class. Read an ``underrepresented`` cell of such a class at such a level as
    a reason to look at its images, not as a finding.

    **Cell power at small group counts.** The cluster-model floor costs power: in simulated
    100-image detection datasets (one class at a tenth of its share at a level holding 20% of
    the images) and 60-sequence ones (one class at 0.15 of its share), the factor test finds
    the gap 80% and 83% of the time, but its cell is reported ``underrepresented`` in only
    20% and 8%. At 300 images and 200 sequences the cell is reported every time. See "The
    few-groups trade" on the Dataset Bias and Coverage concept page.

    Cramér's V is descriptive. A fixed cutoff corresponds to different effect sizes across
    factor cardinalities, and its sampling uncertainty is not reported.

    References
    ----------
    Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
    Journal of the Korean Statistical Society, 42(3), 323-328.

    Rao, J. N. K., & Scott, A. J. (1984). On chi-squared tests for multiway contingency tables
    with cell proportions estimated from survey data. Annals of Statistics, 12(1), 46-60.

    Korn, E. L., & Graubard, B. I. (1998). Confidence intervals for proportions with small
    expected number of positive counts estimated from survey data. Survey Methodology, 24(2),
    193-201.

    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a
    practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B, 57(1), 289-300.

    Examples
    --------
    Initialize the Parity class:

    >>> parity = Parity()

    Specifying custom thresholds:

    >>> parity = Parity(score_threshold=0.4, p_value_threshold=0.01, lift_threshold=1.5)

    Using configuration:

    >>> config = Parity.Config(score_threshold=0.4, p_value_threshold=0.01)
    >>> parity = Parity(config=config)
    """

    class Config(EvaluatorConfig):
        """
        Configuration for Parity evaluator.

        Attributes
        ----------
        score_threshold : float, default 0.3
            Cramér's V at or above which a significant factor is ``associated``.
        p_value_threshold : float, default 0.05
            Largest adjusted p-value that counts as significant.
        lift_threshold : float, default 1.25
            Lift band for actionable cells; at least 1, and 1 disables the band.
        label : str or Sequence[str] or None, default None
            Factor(s) to condition on instead of the class labels. None reads
            :attr:`~dataeval.Metadata.class_labels`, which requires the metadata be
            viewed at its label level; naming a factor is how the same question is
            asked at a coarser view, where there is no single class label per row.
        assume_independent : bool, default False
            Declare each row its own independent item, for a container that records no
            sampling unit.
        """

        score_threshold: UnitInterval = DEFAULT_PARITY_SCORE_THRESHOLD
        p_value_threshold: OpenUnitInterval = DEFAULT_PARITY_P_VALUE_THRESHOLD
        lift_threshold: Annotated[float, Field(ge=1.0)] = DEFAULT_PARITY_LIFT_THRESHOLD
        label: str | Sequence[str] | None = None
        assume_independent: bool = False

    metadata: MetadataLike
    score_threshold: float
    p_value_threshold: float
    lift_threshold: float
    label: str | Sequence[str] | None
    assume_independent: bool
    group_source: str | None = None
    test: str | None = None
    config: Config

    def __init__(
        self,
        score_threshold: float | None = None,
        p_value_threshold: float | None = None,
        lift_threshold: float | None = None,
        label: str | Sequence[str] | None = None,
        assume_independent: bool | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    @set_metadata(state=_STATE)
    def evaluate(self, data: AnnotatedDataset[Any] | MetadataLike) -> ParityOutput:
        """
        Test each factor and each (factor, level, class) cell against the class labels.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or MetadataLike
            Either an annotated dataset (which will be converted to Metadata) or any object
            implementing the MetadataLike protocol.

        Returns
        -------
        ParityOutput
            The ``factors`` and ``cells`` frames, and the sampling unit used.

        Raises
        ------
        ValueError
            If the metadata has no factors, or nothing establishes the independent
            sampling unit and ``assume_independent`` is False.

        Examples
        --------
        An object detection dataset, so its detections are grouped by image:

        >>> import polars as pl
        >>> from dataeval import Metadata
        >>> metadata = Metadata(dataset)

        >>> result = Parity().evaluate(metadata)
        >>> result.test, result.group_source, result.group_level, result.n_rows, result.n_groups
        ('rao_scott_f', 'level', 'unit', 93, 50)
        >>> result.factors.select("factor_name", "status", "score", "p_value_adjusted", "has_insufficient_data")
        shape: (4, 5)
        ┌─────────────┬──────────────┬──────────┬──────────────────┬───────────────────────┐
        │ factor_name ┆ status       ┆ score    ┆ p_value_adjusted ┆ has_insufficient_data │
        │ ---         ┆ ---          ┆ ---      ┆ ---              ┆ ---                   │
        │ cat         ┆ enum         ┆ f64      ┆ f64              ┆ bool                  │
        ╞═════════════╪══════════════╪══════════╪══════════════════╪═══════════════════════╡
        │ angle       ┆ not_detected ┆ 0.111165 ┆ 0.2179           ┆ false                 │
        │ location    ┆ associated   ┆ 0.498447 ┆ 0.000021         ┆ true                  │
        │ time_of_day ┆ weak         ┆ 0.286942 ┆ 0.005495         ┆ true                  │
        │ weather     ┆ not_detected ┆ 0.165487 ┆ 0.131313         ┆ false                 │
        └─────────────┴──────────────┴──────────┴──────────────────┴───────────────────────┘

        On 50 images no cell is actionable yet: the association with ``location`` is clear,
        but each of its cells is either too thin to rely on or not significant after the
        adjustment. The thin cells with the largest deficits are where to collect:

        >>> result.cells.filter(pl.col("is_actionable")).height
        0
        >>> location = result.cells.filter(pl.col("factor_name") == "location")
        >>> location.select("level", "class_name", "status", "observed", "expected", "p_value_adjusted").head(3)
        shape: (3, 6)
        ┌──────────┬────────────┬──────────┬──────────┬──────────┬──────────────────┐
        │ level    ┆ class_name ┆ status   ┆ observed ┆ expected ┆ p_value_adjusted │
        │ ---      ┆ ---        ┆ ---      ┆ ---      ┆ ---      ┆ ---              │
        │ cat      ┆ cat        ┆ enum     ┆ i64      ┆ f64      ┆ f64              │
        ╞══════════╪════════════╪══════════╪══════════╪══════════╪══════════════════╡
        │ suburban ┆ person     ┆ too_thin ┆ 1        ┆ 6.236559 ┆ 1.0              │
        │ urban    ┆ plane      ┆ too_thin ┆ 5        ┆ 10.16129 ┆ 1.0              │
        │ rural    ┆ person     ┆ too_thin ┆ 0        ┆ 3.44086  ┆ 1.0              │
        └──────────┴────────────┴──────────┴──────────┴──────────┴──────────────────┘
        """
        return self.from_metadata(data if is_metadata_like(data) else Metadata(data))

    @set_metadata(state=_STATE)
    def from_metadata(self, metadata: MetadataLike) -> ParityOutput:
        """
        Measure the association between each factor and the class axis in metadata you already hold.

        :meth:`evaluate` builds a :class:`~dataeval.Metadata` from a dataset and calls this.
        Call it directly when the metadata is already built, for example to share one
        between several evaluators, so it is not built again.

        .. versionadded:: 1.2

        Parameters
        ----------
        metadata : MetadataLike
            A :class:`~dataeval.Metadata`, or any container implementing
            :class:`~dataeval.protocols.MetadataLike`.

        Returns
        -------
        ParityOutput
            The same output :meth:`evaluate` returns for the dataset the metadata describes.

        Raises
        ------
        ValueError
            If the metadata has no factors, or nothing establishes the independent
            sampling unit and ``assume_independent`` is False.

        See Also
        --------
        evaluate : Measure a dataset, building its metadata first.

        Examples
        --------
        >>> from dataeval import Metadata
        >>> metadata = Metadata(dataset)
        >>> result = Parity().from_metadata(metadata)
        """
        self.metadata = metadata

        axis = resolve_label_axis(self.metadata, self.label)
        # Recorded before anything is computed from it: the three scalar members
        # `set_metadata` stamps read through this, and so does the output's own field.
        self._axis_record = record = axis_record(self.metadata, axis)
        factor_data, factor_names, _ = factors_excluding(self.metadata, axis.excluded)

        if not factor_names:
            raise ValueError("No factors found in provided metadata.")

        unit = independence_groups(self.metadata, self.assume_independent)
        self.group_source = unit.source
        result = parity(factor_data, axis.values, unit.groups, assume_independent=unit.groups is None)
        self.test = result["test"]

        flags = result["insufficient"] & result["testable"]
        thin = [name for name, flag in zip(factor_names, flags, strict=True) if flag]
        if thin:
            _logger.warning(
                f"Factors {thin} hold tables too thin to trust the reference distribution "
                "(Cochran's criterion on the expected counts, or too few effective groups); "
                "where they are not significant, that is inconclusive rather than a negative.",
            )

        significant = pl.col("p_value_adjusted") <= self.p_value_threshold
        is_associated = pl.col("testable") & significant & (pl.col("score") >= self.score_threshold)
        factor_status = (
            pl
            .when(~pl.col("testable"))
            .then(pl.lit("untestable"))
            .when(pl.col("is_associated"))
            .then(pl.lit("associated"))
            .when(significant)
            .then(pl.lit("weak"))
            .when(pl.col("has_insufficient_data"))
            .then(pl.lit("inconclusive"))
            .otherwise(pl.lit("not_detected"))
            .cast(FACTOR_STATUS)
        )
        factors = (
            pl
            .DataFrame(
                {
                    "factor_name": factor_names,
                    "score": result["scores"],
                    "p_value": result["p_values"],
                    "p_value_adjusted": result["p_values_adjusted"],
                    "has_insufficient_data": result["insufficient"],
                    "design_effect": result["design_effects"],
                    "effective_groups": result["effective_groups"],
                    "df_num": result["df_num"],
                    "df_den": result["df_den"],
                    "testable": result["testable"],
                },
                schema_overrides={"factor_name": lexical_categorical()},
            )
            .with_columns(is_associated=is_associated)
            .with_columns(status=factor_status)
            .select(
                "factor_name",
                "status",
                "score",
                "p_value",
                "p_value_adjusted",
                "is_associated",
                "has_insufficient_data",
                "design_effect",
                "effective_groups",
                "df_num",
                "df_den",
            )
        )

        cells = result["cells"]
        level_names = factor_code_names(self.metadata, factor_data, factor_names)
        cell_status = (
            pl
            .when(~pl.col("testable"))
            .then(pl.lit("untestable"))
            .when(pl.col("is_sparse"))
            .then(pl.lit("too_thin"))
            .when(pl.col("is_deviant") & (pl.col("lift") <= 1 / self.lift_threshold))
            .then(pl.lit("underrepresented"))
            .when(pl.col("is_deviant") & (pl.col("lift") >= self.lift_threshold))
            .then(pl.lit("overrepresented"))
            .otherwise(pl.lit("no_finding"))
            .cast(CELL_STATUS)
        )
        cells_df = (
            pl
            .DataFrame(
                {
                    "factor_name": [factor_names[f] for f in cells["factor"]],
                    "level": [
                        level_names[f].get(int(code), str(code))
                        for f, code in zip(cells["factor"], cells["level"], strict=True)
                    ],
                    "class_name": [axis.names.get(int(c), str(c)) for c in cells["class_label"]],
                    "observed": cells["observed"],
                    "expected": cells["expected"],
                    "deficit": cells["deficit"],
                    "lift": cells["lift"],
                    "p_value": cells["p_values"],
                    "p_value_adjusted": cells["p_values_adjusted"],
                    "is_sparse": cells["sparse"],
                    "testable": cells["testable"],
                },
                schema_overrides={
                    "factor_name": lexical_categorical(),
                    "level": lexical_categorical(),
                    "class_name": lexical_categorical(),
                    "observed": pl.Int64,
                },
            )
            .with_columns(is_deviant=pl.col("testable") & (pl.col("p_value_adjusted") <= self.p_value_threshold))
            .with_columns(status=cell_status)
            .with_columns(is_actionable=pl.col("status").is_in(["underrepresented", "overrepresented"]))
            .sort(["factor_name", "status", "deficit"], descending=[False, False, True])
            .select(
                "factor_name",
                "level",
                "class_name",
                "status",
                "is_actionable",
                "observed",
                "expected",
                "deficit",
                "lift",
                "p_value",
                "p_value_adjusted",
                "is_deviant",
                "is_sparse",
            )
        )

        return ParityOutput(
            factors=factors,
            cells=cells_df,
            class_axis=record,
            test=result["test"],
            group_source=unit.source,
            group_level=unit.level,
            n_rows=result["n_rows"],
            n_groups=result["n_groups"],
        )

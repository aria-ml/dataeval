__all__ = []

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Generic, Literal, NamedTuple, Self, TypeVar, cast, overload

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval import Embeddings, Metadata
from dataeval._helpers import _get_index2label, _get_item_indices, lexical_categorical, reject_filtered_metadata
from dataeval._log import get_logger
from dataeval.core import (
    ClusterResult,
    ClusterStats,
    ColumnKind,
    StatsResult,
    cluster,
    combine_stats_results,
    compute_cluster_stats,
)
from dataeval.core._combination_scores import _named_groups
from dataeval.data import SequenceFrames
from dataeval.exceptions import MaiteShapeError
from dataeval.flags import ImageStats
from dataeval.protocols import (
    AnnotatedDataset,
    ArrayLike,
    Dataset,
    FeatureExtractor,
    MetadataLike,
    MultiobjectTrackingDataset,
    Threshold,
    ThresholdLike,
)
from dataeval.quality._factor_stats import eligible_factors, factor_kind, factor_levels, factor_stats
from dataeval.quality._outliers_neighbors import neighbor_checks, restratified, score_neighbors
from dataeval.quality._outliers_rows import (
    _CONTEXT_COLUMNS,
    _EXTRA_COLUMNS,
    _POPULATION_SCHEMA,
    DEFAULT_OUTLIERS_OUTLIER_THRESHOLD,
    FINDINGS_SCHEMA,
    _Fit,
    _fit_outliers,
    _percentile_of,
    _Population,
    _populations_frame,
    _Strata,
    _unfitted,
    classes_of,
    coded,
    has_level_schema,
    reached_levels,
    row_positions,
    strata_codes,
)
from dataeval.quality._outliers_scores import (
    _CATEGORICAL_KINDS,
    _ScoredCheck,
    _ScoreInput,
    flag_scored,
    rarity_within,
    read_score_input,
    score_combinations,
    strata_by_level,
)
from dataeval.quality._shared import (
    LABEL_KIND,
    TARGET_LEVELS,
    TRACKING_LEVEL_NAMES,
    LevelPlan,
    add_dataset_index,
    checked_compute_stats,
    checked_sole_metadata,
    drop_null_index_columns,
    is_metadata,
    reported_level,
    resolve_levels,
    row_kinds,
    selected_by_flags,
    yields_frames,
)
from dataeval.types import (
    ArrayND,
    ClusterConfigMixin,
    DataFrameOutput,
    Evaluator,
    EvaluatorConfig,
    FactorLevel,
    RemovalPlan,
    SourceIndex,
    StatsMap,
    set_metadata,
)
from dataeval.utils._array import flatten_samples, to_numpy
from dataeval.utils.data import iter_images
from dataeval.utils.thresholds import ZScoreThreshold, resolve_threshold

DEFAULT_OUTLIERS_FLAGS = ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
DEFAULT_OUTLIERS_CLUSTER_THRESHOLD: Threshold = ZScoreThreshold(upper_multiplier=2.5, lower_multiplier=None)

_logger = get_logger(__name__)


OutlierReasons = Sequence[str]
SingleOutliersMap = Mapping[int, Sequence[str]]
SingleTargetOutliersMap = Mapping[SourceIndex, Sequence[str]]
MultiOutliersMap = Mapping[int, Mapping[int, Sequence[str]]]
MultiTargetOutliersMap = Mapping[int, Mapping[SourceIndex, Sequence[str]]]
TOutliers = TypeVar("TOutliers", SingleOutliersMap, SingleTargetOutliersMap, MultiOutliersMap, MultiTargetOutliersMap)


def _row_address(row: Mapping[str, Any], *, has_level: bool) -> SourceIndex:
    """Return the address an outliers row names, spelled as the result spelled it.

    The level is carried rather than filled in: an unstated one is the task-generic reading, and stating it here
    would make ``outliers[SourceIndex(0, 3)]`` (the spelling a caller writes) miss the row it names, since two
    spellings of one address do not hash alike.
    """
    return SourceIndex(row["item_index"], row.get("target_index"), row["level"] if has_level else None)


def _join_classes_at_view(frame: pl.DataFrame, metadata: Any) -> pl.DataFrame:
    """Give each flagged row the classes it has at the metadata's view, one output row per distinct class."""
    has_level = "level" in frame.columns
    addresses = [_row_address(row, has_level=has_level) for row in frame.iter_rows(named=True)]
    levels, positions = row_positions(metadata, addresses)
    index2label = _get_index2label(metadata)
    names = [
        [None] if not classes else [index2label[c] for c in classes]
        for classes in classes_of(metadata, levels, positions)
    ]
    return frame.with_columns(pl.Series("class_name", names, dtype=pl.List(pl.Utf8))).explode("class_name")


def _join_labels_by_item(frame: pl.DataFrame, metadata: MetadataLike) -> pl.DataFrame:
    """Join each flagged row to the labels of a plain MetadataLike, which has no level schema."""
    item_ids = _get_item_indices(metadata)
    index2label = _get_index2label(metadata)
    # Whatever the metadata calls a class: a pivoted one groups by its own axis here,
    # which is what handing this method a metadata rather than a dataset is for. The
    # ``class_name`` column then holds that axis's group names, so a reader who did not
    # build the metadata should check ``metadata.class_axis_source`` before reading them
    # as classes.
    class_names = [index2label[label] for label in metadata.class_labels]

    labels_df = pl.DataFrame({"item_index": item_ids, "class_name": class_names})

    # Join the Issues with the Labels. When some item holds more than one label, a box is joined to its
    # own label -- the label's position within its item is the box's target index, as _build_class_ids
    # reads it -- while an item's own row is joined once to each distinct class of the item, as a
    # Metadata's rows above its view are. Labels one per item, such as a class axis read at the item
    # level, name no box, so every row takes its item's label.
    distinct = labels_df.unique(maintain_order=True)
    per_box = labels_df["item_index"].n_unique() < labels_df.height
    if per_box and "target_index" in frame.columns:
        # A column that holds only nulls has the Null dtype, which cannot be a join key beside an Int64.
        frame = frame.with_columns(pl.col("target_index").cast(pl.Int64))
        own_label = labels_df.with_columns(target_index=pl.int_range(pl.len(), dtype=pl.Int64).over("item_index"))
        joined_df = pl.concat([
            frame.filter(pl.col("target_index").is_null()).join(distinct, on="item_index", how="left"),
            frame.filter(pl.col("target_index").is_not_null()).join(
                own_label, on=["item_index", "target_index"], how="left"
            ),
        ])
    else:
        joined_df = frame.join(distinct, on="item_index", how="left")
    return joined_df


class OutliersOutput(DataFrameOutput, Generic[TOutliers]):
    """
    Output class for :class:`.Outliers` lint detector.

    DataFrame of outlier issues with columns:

    - dataset_index: int - Index of the originating dataset (only present for multi-dataset output)
    - item_index: int - Index of the outlier item (local to each dataset)
    - target_index: int | None - Which row within the item, as its level names it (None for
      item-level outliers). This is a target/detection index where `level` is absent, and
      that level's own key where it is present — ``unit_index`` for a video frame,
      ``track_id`` for a track. The column is omitted when every outlier is item-level.
    - level: str | None - The level the outlier was detected at, carried through from the
      statistics' source index. Omitted when no statistic stated one, which is what every
      producer emits for an image dataset — there, an item-level row and a target-level row
      are all an address can name, and `target_index` already tells them apart.
    - metric_name: str - Name of the metric that flagged this image/target
    - metric_value: float - Value of the metric for this image/target
    - direction: str - ``"upper"`` or ``"lower"``: which limit the value crossed
    - bound: float - The limit it crossed
    - percentile: float - Where the value ranks within its reference population, as a
      percentage. Quartiles and quintiles are readings of this column, not separate ones.
    - population_mean, population_std: float - The reference population's own figures
    - population: str - The stratum or class the row was judged in, such as ``weather=rain`` or
      ``class_label=cat``, null where it was judged with its whole level. Present only on a result
      judged within strata, by ``strata=``, :meth:`stratified` or :meth:`classwise`.
    - method: str - The check that flagged the row: ``"rarity"`` for a categorical value rare within
      its stratum, ``"mahalanobis"`` for an unusual combination of factors, ``"residual"`` for a
      factor unusual given another, ``"neighbors"`` for a value unusual beside its sequence neighbors,
      ``"order"`` for a timestamp out of order, null for a single-column threshold. Present only when a
      row has one.
    - observed: str - The categorical value a rarity row found rare, such as ``weather=snow``. Present
      only on a result with rarity rows.

    A rarity row reports the one-sided 95 % Clopper-Pearson upper bound on the value's share of its
    stratum as ``metric_value``, the share it is tested against (0.05) as ``bound``, ``direction``
    ``"lower"``, and no ``percentile``, ``population_mean`` or ``population_std``.

    A combination row names its group as ``metric_name``, such as ``rel_size, rel_offset_y``, and
    reports the robust Mahalanobis distance as ``metric_value``, always with ``direction``
    ``"upper"``. A residual row is named ``brightness given hour`` and reports the signed robust z
    of the one factor among the rows nearest in the other. Both carry ``percentile``,
    ``population_mean`` and ``population_std`` of the score within the population it was judged in.

    A neighbor row is named ``brightness along sequence`` (a frame along its video) or ``box_size along track`` (a
    detection along its track). It reports the raw value as ``metric_value``, where its neighbors put it as
    ``population_mean``, its sequence's spread as ``population_std``, and the crossed limit in the factor's units as
    ``bound``. An order row reports the value as ``metric_value`` and the value it is out of order against as
    ``bound``, with no ``percentile``, ``population_mean``, ``population_std`` or ``population``.

    Rows are thresholded **within a level**: a metric's distribution is fitted separately
    over each kind of row the statistics name, so a per-frame reading is never compared
    against a spread that includes per-sequence ones.

    **The reference population is the one the threshold was fitted to** -- that metric, at that
    level, and under :meth:`classwise`, within that class -- with NaN readings left out. So the
    context columns describe the same comparison that flagged the row rather than a second,
    looser one, and a value that is unremarkable globally still reads as extreme beside the class
    it was judged against.

    No z-score column is reported. A z is the statistic that flagged the row only under
    :class:`~dataeval.utils.thresholds.ZScoreThreshold`; beside an adaptive, modified-z or IQR
    bound it is an unrelated summary of the same value, and printing one next to those invites
    reading it as the reason. ``percentile`` means the same thing under every threshold, and
    ``(metric_value - population_mean) / population_std`` is the z for anyone who wants it.

    Attributes
    ----------
    calculation_results : StatsResult or Sequence[StatsResult] or None
        The original calculation result(s) from :func:`image_stats`. Used internally
        for re-detection via :meth:`classwise`, :meth:`itemwise`, and :meth:`with_threshold`.
    outlier_threshold : ThresholdLike or dict or None
        Threshold configuration used to detect outliers. Preserved across
        re-detection calls unless overridden via :meth:`with_threshold`.
    cluster_stats : ClusterStats or None
        Pre-computed cluster statistics for cluster-based outlier re-detection
        via :meth:`with_threshold`.
    cluster_threshold : ThresholdLike or None
        Threshold configuration used for cluster-based outlier detection.
        Preserved across re-detection calls unless overridden via
        :meth:`with_threshold`.
    dataset_steps : list[int] or None
        Cumulative image counts per dataset, used to number cluster rows. Statistics rows are
        numbered by steps recomputed from `calculation_results`. None for single-dataset output.
    restricted_to : frozenset[FactorLevel or None] or None
        The kinds of row this output re-detects: None for an item's own row, ``"instance"`` for one of
        its labels, otherwise a level name. None means every row. A level view, and a :meth:`~Outliers.from_stats`
        result built with ``per_image=False`` or ``per_target=False``, set it, so re-detection
        reproduces the rows the result describes.
    tracking : bool
        Whether the result describes tracking data, which decides what the level views select.
    strata : _Strata or None
        Which population each statistics row is judged in, kept so re-detection judges the rows the same way.
        None means every row is judged with its whole level.
    score_input : _ScoreInput or None
        The categorical factors, combinations and residuals of a metadata result, one frame per level, kept so
        :meth:`stratified`, :meth:`classwise` and :meth:`itemwise` can judge them again within new populations.
        None on a result of a dataset or of :meth:`~Outliers.from_stats`.
    scored : tuple[_ScoredCheck, ...]
        The scores of each combination, residual and neighbor check, kept so :meth:`with_threshold` refits them
        without scoring again.
    order : pl.DataFrame or None
        The rows a monotonic factor's order rule flagged under ``neighbors=``, kept as they are on every
        re-detection, since a rule has no threshold to refit. Read them in the result's rows with ``method``
        ``"order"``.
    """

    # Class-level defaults, not set only in __init__: an output pickled before these existed has
    # no such instance attribute, and DataFrameOutput.__getattr__ refuses names starting with "_".
    restricted_to: frozenset[FactorLevel | None] | None = None
    tracking: bool = False
    strata: _Strata | None = None
    score_input: _ScoreInput | None = None
    _populations: pl.DataFrame | None = None
    _rarity: tuple[pl.DataFrame, list[_Population]] | None = None
    scored: tuple[_ScoredCheck, ...] = ()
    _order: pl.DataFrame | None = None

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        calculation_results: StatsResult | Sequence[StatsResult] | None = None,
        outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None = None,
        cluster_stats: ClusterStats | None = None,
        cluster_threshold: ThresholdLike | None = None,
        dataset_steps: Sequence[int] | None = None,
        restricted_to: frozenset[FactorLevel | None] | None = None,
        tracking: bool = False,
        strata: _Strata | None = None,
        populations: pl.DataFrame | None = None,
        score_input: _ScoreInput | None = None,
        rarity: tuple[pl.DataFrame, list[_Population]] | None = None,
        scored: Sequence[_ScoredCheck] = (),
        order: pl.DataFrame | None = None,
    ) -> None:
        super().__init__(data)
        self.calculation_results = calculation_results
        self.outlier_threshold = outlier_threshold
        self.cluster_stats = cluster_stats
        self.cluster_threshold = cluster_threshold
        self.dataset_steps = dataset_steps
        self.restricted_to = restricted_to
        self.tracking = tracking
        self.strata = strata
        self._populations = populations
        self.score_input = score_input
        self._rarity = rarity
        self.scored = tuple(scored)
        self._order = order

    @property
    def populations(self) -> pl.DataFrame:
        """The populations statistical checks were asked to judge, and whether each was.

        A population is the rows of one metric at one level, or within one stratum under :meth:`stratified` or
        one class under :meth:`classwise`. A
        statistical threshold is fitted only to a population with at least 20 real values; a smaller one yields
        no findings and is listed here with ``judged=False``. A
        :class:`~dataeval.utils.thresholds.ConstantThreshold` needs no data and judges every population.
        Rarity judges a categorical factor only within a stratum of at least 93 rows, the fewest in which one
        occurrence can be rare at 95 % confidence. A combination of ``d`` factors needs at least 20 rows and
        10 rows per dimension, and a residual (``given=``) at least 31 rows for its window.
        A neighbor check (``neighbors=``) needs at least 20 scored rows in a pool.

        Every combination and residual you name is listed at its level, even when none of its rows has a value
        (``n=0``, ``judged=False``). A single-column metric is left out of a population where it has no real
        value: every factor is read at every level, and has no value at the levels it is not measured at.

        Returns
        -------
        pl.DataFrame
            One row per population, with columns:

            - level: str | None - The level the population sits at, None for an item's own row
            - metric_name: str - The metric judged
            - method: str | None - The check that judged it, None for a single-column threshold
            - population: str | None - The stratum or class the population holds, such as ``weather=rain`` or
              ``class_label=cat``, or None for the whole level
            - n: int - How many real values the population has
            - judged: bool - Whether a threshold was fitted to it
            - reason: str | None - Why it was not judged (``"below 20 rows"``, ``"below 93 rows"``,
              ``"no spread"``; for a combination or residual also ``"exact fit"``, ``"degenerate
              covariance"``, ``"too few rows"`` and ``"no upper bound"``), None when it was

            Multi-dataset populations are pooled across the datasets. A result from
            :meth:`~Outliers.from_clusters` has no populations and returns an empty frame.

        Examples
        --------
        >>> from dataeval import Metadata
        >>> md = Metadata.from_factors({"temperature": [20.1, 19.8, 20.3, 20.0, 19.9, 45.0]})
        >>> Outliers().evaluate(md).populations.select("metric_name", "n", "judged", "reason")
        shape: (1, 4)
        ┌─────────────┬─────┬────────┬───────────────┐
        │ metric_name ┆ n   ┆ judged ┆ reason        │
        │ ---         ┆ --- ┆ ---    ┆ ---           │
        │ cat         ┆ i64 ┆ bool   ┆ str           │
        ╞═════════════╪═════╪════════╪═══════════════╡
        │ temperature ┆ 6   ┆ false  ┆ below 20 rows │
        └─────────────┴─────┴────────┴───────────────┘

        .. versionadded:: 1.2
        """
        return pl.DataFrame(schema=_POPULATION_SCHEMA) if self._populations is None else self._populations

    @property
    def _combined(self) -> tuple[StatsMap, list[SourceIndex], list[int]] | None:
        """Extract combined stats, source_index, and dataset_steps from stored calculation_results."""
        if self.calculation_results is None:
            return None
        return combine_stats_results(self.calculation_results)

    def __len__(self) -> int:
        cols = ["item_index"]
        if "dataset_index" in self.data().columns:
            cols.insert(0, "dataset_index")
        if "target_index" in self.data().columns:
            cols.append("target_index")
        if "level" in self.data().columns:
            # Part of the row's identity, not a description of it: two rows differing only
            # in level name different rows, at different levels of the same item.
            cols.append("level")
        return self.data().select(cols).n_unique()

    def _kept(self, source_index: Sequence[SourceIndex]) -> NDArray[np.bool_] | None:
        """Mark the source-index entries this output re-detects, or return None when it re-detects all of them."""
        if self.restricted_to is None:
            return None
        return np.fromiter(
            (si.kind in self.restricted_to for si in source_index), dtype=np.bool_, count=len(source_index)
        )

    def _kinds(self) -> set[FactorLevel | None]:
        """Return the kinds of row this output covers: its stored statistics' when it kept them, else its frame's."""
        results = self.calculation_results
        if results is None:
            kinds: set[FactorLevel | None] = set(row_kinds(self.data()).to_list())  # type: ignore[arg-type]
        else:
            parts = [results] if isinstance(results, Mapping) else results
            kinds = {si.kind for part in parts for si in part["source_index"]}
        return kinds if self.restricted_to is None else kinds & self.restricted_to

    def _in_between_kinds(self) -> list[str]:
        """Return the levels between an item and its labels that this output covers, such as frames or tracks."""
        return sorted(str(kind) for kind in self._kinds() if kind not in (None, LABEL_KIND))

    def _refuse_in_between_kinds(self) -> None:
        """Raise when this output covers a level between an item and its labels, whose rows have no class."""
        between = self._in_between_kinds()
        if between:
            raise ValueError(
                f"aggregate_by_class: {between} rows have no class -- a frame or a track is not a label. "
                "Narrow the result to rows that do first, with `.targets` (or `.detections` for tracking data)."
            )

    def _metric_totals_by_level(self) -> pl.DataFrame:
        """Count the addresses each metric flagged at each level."""
        schema: Any = {"level": pl.Utf8, "metric_name": lexical_categorical(), "Total": pl.UInt32}
        frame = self.data()
        if frame.is_empty():
            return pl.DataFrame(schema=schema)
        addresses = [name for name in ("item_index", "target_index") if name in frame.columns]
        return (
            frame
            .with_columns(row_kinds(frame).alias("level"))
            .group_by(["level", "metric_name"])
            .agg(pl.struct(addresses).n_unique().alias("Total"))
            .select("level", "metric_name", "Total")
            .sort(["Total", "level", "metric_name"], descending=[True, False, False], nulls_last=False)
            .cast(schema)
        )

    # ------------------------------------------------------------------
    # Convenience accessor
    # ------------------------------------------------------------------

    @property
    def outliers(self) -> TOutliers:  # noqa: C901
        """Outlier items as a mapping of index to flagged metric names.

        When ``per_target=False``:

            - Single-dataset: ``dict[int, list[str]]`` keyed by ``item_index``
            - Multi-dataset: ``dict[int, dict[int, list[str]]]`` outer key is ``dataset_index``

        When ``per_target=True``:

            - Single-dataset: ``dict[SourceIndex, list[str]]`` keyed by :class:`SourceIndex`
            - Multi-dataset: ``dict[int, dict[SourceIndex, list[str]]]`` outer key is ``dataset_index``
        """
        df = self.data()
        is_cross = "dataset_index" in df.columns
        has_level = "level" in df.columns
        # A stated level makes the row an address even with no key: an unkeyed one names
        # the item's own row, and which level that is is what the column says.
        has_targets = has_level or ("target_index" in df.columns and df["target_index"].null_count() < len(df))

        if is_cross:
            if has_targets:
                result_cross_target: dict[int, dict[SourceIndex, list[str]]] = {}
                for row in df.iter_rows(named=True):
                    address = _row_address(row, has_level=has_level)
                    result_cross_target.setdefault(row["dataset_index"], {}).setdefault(address, []).append(
                        row["metric_name"]
                    )
                return result_cross_target  # type: ignore[return-value]
            result_cross: dict[int, dict[int, list[str]]] = {}
            for row in df.iter_rows(named=True):
                result_cross.setdefault(row["dataset_index"], {}).setdefault(row["item_index"], []).append(
                    row["metric_name"]
                )
            return result_cross  # type: ignore[return-value]

        if has_targets:
            result_target: dict[SourceIndex, list[str]] = {}
            for row in df.iter_rows(named=True):
                result_target.setdefault(_row_address(row, has_level=has_level), []).append(row["metric_name"])
            return result_target  # type: ignore[return-value]

        result_single: dict[int, list[str]] = {}
        for row in df.iter_rows(named=True):
            result_single.setdefault(row["item_index"], []).append(row["metric_name"])
        return result_single  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

    def aggregate_by_class(self, metadata: MetadataLike) -> pl.DataFrame:
        """
        Return a Polars DataFrame summarizing outliers per class and metric.

        Creates a pivot table showing the count of outlier images for each combination
        of class and metric. A flagged box is counted under its own class, or under its
        item's class when the metadata labels items rather than boxes. A flagged item is
        counted once under each distinct class among its labels. Includes a Total
        row showing the total number of outliers per metric across all classes, and a
        Total column showing the total number of outliers per class across all metrics.

        Parameters
        ----------
        metadata : MetadataLike
            Metadata object containing class labels and image-to-class mappings for the dataset.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            - class_name: cat - Name of the class
            - <metric_name>: int - Count of outliers for each metric (one column per metric)
            - Total: int - Total outlier count for the class across all metrics

            The last row is "Total" showing the sum across all classes for each metric.
            Rows are sorted by Total in descending order (excluding the Total row).

        Raises
        ------
        ValueError
            If the issues contain multiple DataFrames (from multiple datasets).
            If the result holds rows at a level between an item and its labels, which have no class.

        Examples
        --------
        >>> from dataeval import Metadata
        >>> from dataeval.flags import ImageStats
        >>> from dataeval.quality import Outliers

        >>> outliers = Outliers(flags=ImageStats.VISUAL, outlier_threshold="modzscore")
        >>> results = outliers.evaluate(dataset)
        >>> metadata = Metadata(dataset)
        >>> summary = results.aggregate_by_class(metadata)
        >>> summary
        shape: (4, 6)
        ┌────────────┬────────────┬──────────┬──────────┬───────────┬───────┐
        │ class_name ┆ brightness ┆ contrast ┆ darkness ┆ sharpness ┆ Total │
        │ ---        ┆ ---        ┆ ---      ┆ ---      ┆ ---       ┆ ---   │
        │ cat        ┆ u32        ┆ u32      ┆ u32      ┆ u32       ┆ u32   │
        ╞════════════╪════════════╪══════════╪══════════╪═══════════╪═══════╡
        │ person     ┆ 2          ┆ 2        ┆ 2        ┆ 2         ┆ 8     │
        │ plane      ┆ 2          ┆ 2        ┆ 2        ┆ 2         ┆ 8     │
        │ boat       ┆ 1          ┆ 1        ┆ 1        ┆ 1         ┆ 4     │
        │ Total      ┆ 5          ┆ 5        ┆ 5        ┆ 5         ┆ 20    │
        └────────────┴────────────┴──────────┴──────────┴───────────┴───────┘
        """
        if "dataset_index" in self.data().columns:
            raise ValueError("Aggregation by class only works with output from a single dataset.")

        self._refuse_in_between_kinds()

        schema: Any = {"class_name": lexical_categorical(), "Total": pl.UInt32}

        # Handle empty DataFrame case
        if self.data().shape[0] == 0:
            return pl.DataFrame(schema=schema)

        frame = self.data()
        joined_df = (
            _join_classes_at_view(frame, metadata)
            if has_level_schema(metadata)
            else _join_labels_by_item(frame, metadata)
        )

        # Create the Summary Pivot (classes as rows, metrics as columns)
        summary_df = (
            joined_df
            .group_by(["class_name", "metric_name"])
            .len()  # Count occurrences
            .pivot(on="metric_name", index="class_name", values="len")
            .fill_null(0)
        )

        # Get metric columns (all columns except class_name)
        metric_cols = sorted([col for col in summary_df.columns if col != "class_name"])

        # Add a Total column (sum across all metrics for each class)
        if metric_cols:
            summary_df = summary_df.with_columns(pl.sum_horizontal(metric_cols).cast(pl.UInt32).alias("Total"))
        else:
            summary_df = summary_df.with_columns(pl.lit(0, dtype=pl.UInt32).alias("Total"))

        column_order = ["class_name"] + metric_cols + ["Total"]

        # Sort by Total in descending order
        summary_df = summary_df.select(column_order).sort(["Total", "class_name"], descending=[True, False])

        # Create a Total row (sum across all classes for each metric)
        total_row = (
            summary_df
            .select(pl.col(metric_cols + ["Total"]).sum())
            .with_columns(pl.lit("Total").alias("class_name"))
            .select(column_order)
        )

        # Concatenate the summary with the total row
        return pl.concat([summary_df, total_row]).cast(schema)

    def aggregate_by_metric(self) -> pl.DataFrame:
        """
        Return a Polars DataFrame summarizing outlier counts per metric.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            - metric_name: str - Name of the metric
            - Total: int - Number of images flagged by this metric

            When the statistics hold frames or tracks, the summary is by level, even when nothing
            was flagged: the columns are ``level``, ``metric_name`` and ``Total``, and ``Total``
            counts each flagged frame, track, detection or item once. ``level`` is the level's name,
            ``"instance"`` for an item's labels, and null for an item's own row.

        Examples
        --------
        >>> outliers = Outliers(flags=ImageStats.PIXEL, outlier_threshold="zscore")
        >>> results = outliers.evaluate(dataset)
        >>> summary = results.aggregate_by_metric()
        >>> summary
        shape: (4, 2)
        ┌─────────────┬───────┐
        │ metric_name ┆ Total │
        │ ---         ┆ ---   │
        │ cat         ┆ u32   │
        ╞═════════════╪═══════╡
        │ entropy     ┆ 4     │
        │ mean        ┆ 4     │
        │ std         ┆ 4     │
        │ var         ┆ 4     │
        └─────────────┴───────┘
        """
        if "dataset_index" in self.data().columns:
            raise ValueError("Aggregation by metric only works with output from a single dataset.")

        if self._in_between_kinds():
            return self._metric_totals_by_level()

        # Handle empty DataFrame case
        if self.data().shape[0] == 0:
            return pl.DataFrame(schema={"metric_name": lexical_categorical(), "Total": pl.UInt32})

        # Group by metric_name and count unique images
        return (
            self
            .data()
            .group_by("metric_name")
            .agg(pl.col("item_index").n_unique().alias("Total"))
            .sort(["Total", "metric_name"], descending=[True, False])
        )

    def aggregate_by_item(self) -> pl.DataFrame:  # noqa: C901
        """
        Return a Polars DataFrame summarizing outliers per item (item_index, target_index pair) and metric.

        Creates a pivot table showing whether each item is flagged by each metric (1 if flagged, 0 if not).
        Includes a Total column showing the total number of metrics that flagged each item.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            - item_index: int - Item identifier
            - target_index: int or None - Target identifier (Only with per_target outliers)
            - <metric_name>: int - Binary indicator (1 or 0) for each metric
            - count: int - Total number of metrics that flagged this item

        Raises
        ------
        ValueError
            If the issues contain multiple DataFrames (from multiple datasets).

        Examples
        --------
        >>> outliers = Outliers(outlier_threshold=("modzscore", 3.0))
        >>> results = outliers.evaluate(dataset, per_target=True)
        >>> summary = results.aggregate_by_item()
        >>> summary.head(10)
        shape: (10, 22)
        ┌────────────┬────────────┬────────────┬────────────┬───┬──────────┬──────────┬────────────┬───────┐
        │ item_index ┆ target_ind ┆ instance_b ┆ instance_c ┆ … ┆ unit_std ┆ unit_var ┆ unit_zeros ┆ Total │
        │ ---        ┆ ex         ┆ rightness  ┆ ontrast    ┆   ┆ ---      ┆ ---      ┆ ---        ┆ ---   │
        │ i64        ┆ ---        ┆ ---        ┆ ---        ┆   ┆ u32      ┆ u32      ┆ u32        ┆ u32   │
        │            ┆ i64        ┆ u32        ┆ u32        ┆   ┆          ┆          ┆            ┆       │
        ╞════════════╪════════════╪════════════╪════════════╪═══╪══════════╪══════════╪════════════╪═══════╡
        │ 0          ┆ null       ┆ 0          ┆ 0          ┆ … ┆ 0        ┆ 0        ┆ 1          ┆ 1     │
        │ 2          ┆ null       ┆ 0          ┆ 0          ┆ … ┆ 0        ┆ 0        ┆ 1          ┆ 1     │
        │ 7          ┆ null       ┆ 0          ┆ 0          ┆ … ┆ 1        ┆ 1        ┆ 0          ┆ 8     │
        │ 7          ┆ 0          ┆ 1          ┆ 1          ┆ … ┆ 0        ┆ 0        ┆ 0          ┆ 8     │
        │ 11         ┆ null       ┆ 0          ┆ 0          ┆ … ┆ 1        ┆ 1        ┆ 0          ┆ 8     │
        │ 11         ┆ 0          ┆ 1          ┆ 1          ┆ … ┆ 0        ┆ 0        ┆ 0          ┆ 8     │
        │ 18         ┆ null       ┆ 0          ┆ 0          ┆ … ┆ 1        ┆ 1        ┆ 0          ┆ 8     │
        │ 18         ┆ 0          ┆ 1          ┆ 1          ┆ … ┆ 0        ┆ 0        ┆ 0          ┆ 8     │
        │ 18         ┆ 1          ┆ 1          ┆ 1          ┆ … ┆ 0        ┆ 0        ┆ 0          ┆ 8     │
        │ 19         ┆ 0          ┆ 0          ┆ 0          ┆ … ┆ 0        ┆ 0        ┆ 0          ┆ 2     │
        └────────────┴────────────┴────────────┴────────────┴───┴──────────┴──────────┴────────────┴───────┘
        """
        if "dataset_index" in self.data().columns:
            raise ValueError("Aggregation by item only works with output from a single dataset.")

        # Check if target_index column exists
        has_target_id = "target_index" in self.data().columns
        # Part of the row's identity, as it is for __len__ and the .outliers accessor: two
        # rows differing only in level name different rows, and grouping without it would
        # report a per-frame and a per-track finding on one line.
        has_level = "level" in self.data().columns

        index_cols = ["item_index"]
        if has_target_id:
            index_cols.append("target_index")
        if has_level:
            index_cols.append("level")

        # Build schema for known types
        schema: Any = {name: pl.Utf8 if name == "level" else pl.Int64 for name in index_cols} | {"Total": pl.UInt32}

        # Handle empty DataFrame case
        if self.data().shape[0] == 0:
            return pl.DataFrame(schema=schema)

        # Create a binary indicator for each (item_index, [target_index,] metric_name) combination
        # Group by item_index, [target_index,] and metric_name, then pivot

        grouped = (
            self
            .data()
            .group_by(index_cols + ["metric_name"])
            .agg(pl.len().alias("Total"))  # Count occurrences (should be 1 per combination)
            .with_columns(pl.lit(1, dtype=pl.UInt32).alias("flagged"))  # Create binary indicator
        )

        # Note: Polars 1.0.0 pivot cannot handle null values in index columns, so we use a placeholder
        temp_null_placeholder = -1
        temp_level_placeholder = ""

        # Replace null target_index with placeholder before pivot (if target_index exists)
        if has_target_id:
            grouped = grouped.with_columns(pl.col("target_index").fill_null(temp_null_placeholder))
        # The level column is the same story one dtype over: it is null wherever a
        # statistic stated no level, and the empty string is not a level name.
        if has_level:
            grouped = grouped.with_columns(pl.col("level").cast(pl.Utf8).fill_null(temp_level_placeholder))

        pivoted = grouped.pivot(on="metric_name", index=index_cols, values="flagged")

        # Get metric columns
        metric_cols = sorted([col for col in pivoted.columns if col not in index_cols])

        # Build expressions for columns
        expressions = []
        if has_target_id:
            expressions.append(
                pl
                .when(pl.col("target_index") == temp_null_placeholder)
                .then(None)
                .otherwise(pl.col("target_index"))
                .alias("target_index"),
            )
        if has_level:
            expressions.append(
                pl.when(pl.col("level") == temp_level_placeholder).then(None).otherwise(pl.col("level")).alias("level"),
            )

        if metric_cols:
            expressions.extend([pl.col(metric_cols).fill_null(0), pl.sum_horizontal(metric_cols).alias("Total")])
        else:
            expressions.append(pl.lit(0).alias("Total"))

        column_order = index_cols + metric_cols + ["Total"]

        return pivoted.with_columns(expressions).select(column_order).cast(schema).sort(index_cols)

    _UNSET = object()  # sentinel for distinguishing "not provided" from None

    def _redetect(  # noqa: C901
        self,
        strata: Any = _UNSET,
        outlier_threshold: Any = _UNSET,
        cluster_threshold: Any = _UNSET,
        rarity: Any = _UNSET,
        scored: Any = _UNSET,
    ) -> Self:
        """Re-run outlier detection with optional strata or threshold override, keeping this output's strata.

        Only the rows this output is restricted to are re-detected, so a level view stays at its level. Rarity rows
        and populations are kept as they are unless `rarity` replaces them (None drops them): a threshold does not
        change them, and a new stratification judges them anew. Combination scores are refitted under the
        threshold, from `scored` when a new stratification rescored them.

        Raises
        ------
        ValueError
            If this output was not created from an evaluation with stored
            statistics or cluster stats.
        """
        if self.calculation_results is None and self.cluster_stats is None:
            raise ValueError(
                "Re-detection requires statistics stored from evaluate(), from_stats(), or from_clusters()."
            )

        outlier_dfs: list[pl.DataFrame] = []
        populations: pl.DataFrame | None = None

        # Stats-based re-detection
        threshold = self.outlier_threshold if outlier_threshold is self._UNSET else outlier_threshold
        strata = self.strata if strata is self._UNSET else strata
        rarity = _narrowed(self._rarity if rarity is self._UNSET else rarity, self.restricted_to)
        scored = _kept_checks(self.scored if scored is self._UNSET else scored, self.restricted_to)
        order = _narrowed_rows(self._order, self.restricted_to)
        if self.calculation_results is not None:
            combined = self._combined
            if combined is None:
                raise ValueError("Combined stats or source_index is None, cannot re-detect outliers.")
            stats_map, source_index, ds_steps = combined

            keep = self._kept(source_index)
            restricted = strata
            if keep is not None:
                stats_map, source_index, restricted = _restrict(stats_map, source_index, strata, keep)

            stats_df, rows = _detect_outliers(stats_map, source_index, threshold, restricted)
            score_df, score_rows = self._flag_scored(scored, threshold)
            populations = _populations_frame(rows + ([] if rarity is None else rarity[1]) + score_rows)

            if ds_steps:
                stats_df = add_dataset_index(stats_df, ds_steps)

            # Rarity and combinations are judged only on a metadata, which spans one dataset, so their rows need no
            # dataset index.
            outlier_dfs.append(stats_df)
            outlier_dfs.extend(df for df in (None if rarity is None else rarity[0], score_df, order) if df is not None)

        # Cluster-based re-detection. Cluster rows are item rows, so a view that excludes the item
        # end must not gain them back.
        ct = self.cluster_threshold if cluster_threshold is self._UNSET else cluster_threshold
        resolved_ct = resolve_threshold(ct) if ct is not None else DEFAULT_OUTLIERS_CLUSTER_THRESHOLD
        if self.cluster_stats is not None and (self.restricted_to is None or None in self.restricted_to):
            cluster_df = Outliers._find_outliers_adaptive(self.cluster_stats, resolved_ct)
            if self.dataset_steps:
                cluster_df = add_dataset_index(cluster_df, self.dataset_steps)
            outlier_dfs.append(cluster_df)

        merged = Outliers._merge_outlier_dfs(outlier_dfs)
        merged = drop_null_index_columns(merged, _DROPPABLE)

        return OutliersOutput(  # type: ignore[return-value]
            merged,
            calculation_results=self.calculation_results,
            outlier_threshold=threshold,
            cluster_stats=self.cluster_stats,
            cluster_threshold=ct,
            dataset_steps=self.dataset_steps,
            restricted_to=self.restricted_to,
            tracking=self.tracking,
            strata=strata,
            populations=populations,
            score_input=self.score_input,
            rarity=rarity,
            scored=scored,
            order=order,
        )._attach(self._dataset, self._metadata)

    def _flag_scored(
        self, scored: Sequence[_ScoredCheck], threshold: Any
    ) -> tuple[pl.DataFrame | None, list[_Population]]:
        """Refit the stored combination scores under `threshold`; no frame when there are none."""
        if not scored or self.score_input is None:
            return None, []
        return flag_scored(scored, threshold, self.score_input.address)

    def _rescored(self, strata: Mapping[FactorLevel, Any]) -> tuple[_ScoredCheck, ...]:
        """Score the stored combinations again, and re-code the stored neighbor checks, within `strata`."""
        if self.score_input is None:
            return ()
        return (*score_combinations(self.score_input, strata), *restratified(self.scored, strata))

    def classwise(self, metadata: MetadataLike) -> Self:
        """Re-detect outliers using per-class thresholds.

        Computes outlier thresholds within each class separately rather than
        globally. This catches within-class anomalies that global detection misses,
        and avoids false positives where a sample is only unusual because its class
        is inherently different.

        For image classification datasets, each image is assigned to its class.
        For object detection datasets, target-level stats use the target's class,
        while image-level stats for images with multiple classes fall back to
        global detection.

        On a :class:`~dataeval.Metadata`, each row takes its class at the metadata's
        :attr:`~dataeval.Metadata.view`: a row at or below the view takes the class of its
        ancestor there, and a row above it takes the one class among its rows at the view.
        Rows with several classes or none form one population, and levels the class does
        not reach, such as frames on a detection view, are judged whole.

        A statistical threshold is fitted to a class only when it holds at least 20
        rows with a value. A smaller class yields no findings, and :attr:`populations`
        lists it with ``judged=False``. Each flagged row names its class in the
        ``population`` column, such as ``class_label=cat``, or ``class_label=null`` for a
        row with no one class. Any strata from :meth:`stratified` are replaced. Combinations
        and residuals asked for with ``combinations=`` and ``given=`` are scored again within
        each class; categorical factors are not judged by rarity. Neighbor checks asked for with
        ``neighbors=`` are judged again within the new populations without rescoring, and order rows
        are kept.

        Parameters
        ----------
        metadata : MetadataLike
            Metadata object containing class labels.

        Returns
        -------
        OutliersOutput
            New output with per-class detected outliers. Supports all the same
            aggregation methods (``aggregate_by_class``, ``aggregate_by_metric``, etc.).

        Raises
        ------
        ValueError
            If this output was not created from an evaluation with stored statistics; or if it
            holds combinations and `metadata` holds other items, or other rows at a level they
            were scored at, than this result was computed on.

        Examples
        --------
        Twenty images in each of three classes, where the cars are bright and one boat is as
        bright as a car:

        >>> rng = np.random.default_rng(0)
        >>> images = rng.random((60, 3, 16, 16))
        >>> images[20:40] = images[20:40] * 0.5 + 0.5
        >>> images[50] = images[50] * 0.5 + 0.5
        >>> md = Metadata.from_factors(
        ...     {"hour": np.arange(60) % 24}, np.repeat([0, 1, 2], 20), index2label={0: "plane", 1: "car", 2: "boat"}
        ... )

        Judged against every image, the bright boat is unremarkable. Judged against the boats,
        it is not:

        >>> result = Outliers(flags=ImageStats.PIXEL_MEAN | ImageStats.PIXEL_STD).evaluate(images)
        >>> len(result)
        0
        >>> result.classwise(md).aggregate_by_class(md)
        shape: (2, 4)
        ┌────────────┬──────┬─────┬───────┐
        │ class_name ┆ mean ┆ std ┆ Total │
        │ ---        ┆ ---  ┆ --- ┆ ---   │
        │ cat        ┆ u32  ┆ u32 ┆ u32   │
        ╞════════════╪══════╪═════╪═══════╡
        │ boat       ┆ 1    ┆ 1   ┆ 2     │
        │ Total      ┆ 1    ┆ 1   ┆ 2     │
        └────────────┴──────┴─────┴───────┘
        """
        combined = self._combined
        if combined is None:
            raise ValueError("classwise requires statistics stored from evaluate() or from_stats().")
        _, source_index, _ = combined
        if self.score_input is not None and has_level_schema(metadata):
            self._refuse_other_items(metadata, "classwise")
        scored = () if self.score_input is None else self._rescored(_level_class_strata(self.score_input, metadata))
        return self._redetect(_class_strata(source_index, metadata), rarity=None, scored=scored)

    def stratified(self, metadata: Metadata, *columns: str) -> Self:
        """Re-detect outliers within strata of categorical metadata factors.

        Each threshold is fitted within each stratum separately, so a value is judged against
        the rows that share its stratum rather than against its whole level. Several columns
        cross: a stratum is one combination of their values. The strata replace any this
        result was judged with, by ``strata=`` or :meth:`classwise`.

        A stratum reaches its own level and the levels below it, where each row takes the
        value of its ancestor: a detection takes its frame's ``weather``. A null value is a
        population of its own, ``weather=null``. Levels no stratum reaches are judged whole,
        and their rows have a null ``population``.

        On the result of evaluating a :class:`~dataeval.Metadata`, each categorical factor at a
        level the strata reach is also judged by rarity, as ``strata=`` judges it: a value is
        flagged when it makes up less than 5 % of its stratum at 95 % confidence, after a
        Benjamini-Hochberg correction across the strata. A stratum needs at least 93 rows to be
        judged this way, and a value rare in every stratum is not flagged, since it is rare in
        the dataset rather than out of place in one stratum. Each rarity row names the value in
        an ``observed`` column and ``"rarity"`` in a ``method`` column. Combinations and
        residuals asked for with ``combinations=`` and ``given=`` are scored again within the
        new strata. Neighbor checks asked for with ``neighbors=`` are judged again within the new
        populations without rescoring, and order rows are kept.

        Parameters
        ----------
        metadata : Metadata
            The metadata of the data this result was computed on, holding the strata.
        *columns : str
            Categorical factors, or the index columns ``item_index``, ``unit_index`` and
            ``track_id``. A frame or a track is a stratum within its video, labeled
            ``item_index=0, unit_index=3``.

        Returns
        -------
        OutliersOutput
            New output whose rows name the stratum they were judged in, in a
            ``population`` column such as ``weather=rain``.

        Raises
        ------
        ValueError
            If no column is named; if `metadata` has no level schema, or holds a different
            number of items than the stored statistics, or a different number of rows at a level
            this result judged factors at; if a column is continuous; if the
            strata reach no level of the stored statistics or of its categorical factors; or if
            this output holds no statistics.
        KeyError
            If a column is not a factor of `metadata`.

        Examples
        --------
        Rain readings sit near 0 and clear ones near 50. Item 3 reads 6, which is ordinary
        beside every reading and extreme beside the rain:

        >>> from dataeval import Metadata
        >>> rng = np.random.default_rng(0)
        >>> weather = ["rain"] * 30 + ["clear"] * 30
        >>> x = np.r_[rng.normal(0, 1, 30), rng.normal(50, 1, 30)]
        >>> x[3] = 6.0
        >>> md = Metadata.from_factors({"x": x, "weather": weather})
        >>> result = Outliers().evaluate(md)
        >>> len(result)
        0
        >>> result.stratified(md, "weather").data().select("item_index", "metric_value", "population")
        shape: (1, 3)
        ┌────────────┬──────────────┬──────────────┐
        │ item_index ┆ metric_value ┆ population   │
        │ ---        ┆ ---          ┆ ---          │
        │ i64        ┆ f64          ┆ cat          │
        ╞════════════╪══════════════╪══════════════╡
        │ 3          ┆ 6.0          ┆ weather=rain │
        └────────────┴──────────────┴──────────────┘

        .. versionadded:: 1.2
        """
        if not columns:
            raise ValueError("stratified names the strata to judge within; to judge each level whole, use itemwise().")
        if not has_level_schema(metadata):
            raise ValueError("stratified reads strata from a dataeval.Metadata with a level schema.")
        combined = self._combined
        if combined is None:
            raise ValueError("stratified requires statistics stored from evaluate() or from_stats().")
        self._refuse_other_items(metadata)
        _, source_index, _ = combined
        score_input = self.score_input
        if score_input is None:
            strata = strata_codes(metadata, columns, *row_positions(metadata, source_index))
            return self._redetect(strata, rarity=None)
        scored_levels = score_input.scored_levels()
        judged: list[FactorLevel] = [*score_input.judged(columns), *scored_levels]
        strata = strata_codes(metadata, columns, *row_positions(metadata, source_index), also=judged)
        rarity = rarity_within(metadata, columns, score_input)
        scored = self._rescored(strata_by_level(metadata, columns, scored_levels))
        return self._redetect(strata, rarity=rarity, scored=scored)

    def _refuse_other_items(self, metadata: Any, caller: str = "stratified") -> None:
        """Raise when `metadata` holds other items, or other rows at a level, than this result describes."""
        results = self.calculation_results
        stored = results.get("image_count") if isinstance(results, Mapping) else None
        metadata._structure()
        held = metadata._store.height(metadata.item_level)
        if stored is not None and stored != held:
            raise ValueError(
                f"{caller}: the metadata holds {held} items and the stored statistics describe {stored}; "
                "pass the metadata of the data this result was computed on."
            )
        frames = {} if self.score_input is None else self.score_input.frames
        judged = {level: frame.height for level, frame in frames.items()}
        rows = {level: metadata._store.height(level) for level in judged}
        if rows != judged:
            raise ValueError(
                f"{caller}: the metadata holds {rows} rows at the levels this result judged, which held {judged}; "
                "pass the metadata of the data this result was computed on."
            )

    def itemwise(self) -> Self:
        """Re-detect outliers using global thresholds (across all items).

        This is the inverse of :meth:`classwise` -- it re-runs detection without
        per-class grouping, using the full dataset distribution for thresholds. It also
        drops any strata and the rarity rows judged within them, and scores combinations
        and residuals again over each whole level. Neighbor checks asked for with ``neighbors=``
        are judged again over each whole level without rescoring, and order rows are kept.

        Returns
        -------
        OutliersOutput
            New output with globally detected outliers.

        Raises
        ------
        ValueError
            If this output was not created from an evaluation with stored statistics.

        Examples
        --------
        >>> outliers = Outliers(flags=ImageStats.PIXEL)
        >>> result = outliers.evaluate(dataset, per_class=True, metadata=metadata)
        >>> global_result = result.itemwise()
        """
        return self._redetect(None, rarity=None, scored=self._rescored({}))

    def _filtered_by_kind(self, *kinds: FactorLevel | None) -> Self:
        """Return these results narrowed to rows of the given kinds, still re-detectable at those kinds.

        Narrowing intersects with any restriction already in place, so a view of a view can only shrink.
        """
        wanted = frozenset(kinds)
        restricted = wanted if self.restricted_to is None else self.restricted_to & wanted
        frame = self.data()
        keep = pl.Series([kind in restricted for kind in row_kinds(frame).to_list()], dtype=pl.Boolean)
        populations = self.populations
        kept_levels = pl.Series([level in restricted for level in populations["level"].to_list()], dtype=pl.Boolean)
        return OutliersOutput(  # type: ignore[return-value]
            drop_null_index_columns(frame.filter(keep), _DROPPABLE),
            calculation_results=self.calculation_results,
            outlier_threshold=self.outlier_threshold,
            cluster_stats=self.cluster_stats,
            cluster_threshold=self.cluster_threshold,
            dataset_steps=self.dataset_steps,
            restricted_to=restricted,
            tracking=self.tracking,
            strata=self.strata,
            populations=populations.filter(kept_levels),
            score_input=self.score_input,
            rarity=_narrowed(self._rarity, restricted),
            scored=_kept_checks(self.scored, restricted),
            order=_narrowed_rows(self._order, restricted),
        )._attach(self._dataset, self._metadata)

    def _tracking_level(self, kind: FactorLevel | None) -> Self:
        """Return the view of one tracking level, which is empty on a result that is not tracking."""
        return self._filtered_by_kind(kind) if self.tracking else self._filtered_by_kind()

    @property
    def items(self) -> Self:
        """Filtered OutliersOutput holding only the rows that name one piece of media.

        An item's own row for an image dataset, a frame for a tracking one, so this reads the same
        for either. It matches :attr:`.DuplicatesOutput.items`. The view keeps re-detecting at that
        level: ``result.items.with_threshold(2.0)`` is still about items.

        Returns
        -------
        OutliersOutput
            The same kind of output, narrowed.
        """
        return self._filtered_by_kind("unit" if self.tracking else None)

    @property
    def targets(self) -> Self:
        """Filtered OutliersOutput holding only rows that name one of an item's labels.

        A box for an object detection dataset, a detection for a tracking one. The view keeps
        re-detecting at that level.

        Returns
        -------
        OutliersOutput
            The same kind of output, narrowed.
        """
        return self._filtered_by_kind(LABEL_KIND)

    @property
    def sequences(self) -> Self:
        """Filtered OutliersOutput holding only whole-sequence rows. Empty for a result that is not tracking.

        Returns
        -------
        OutliersOutput
            The same kind of output, narrowed.
        """
        return self._tracking_level(None)

    @property
    def frames(self) -> Self:
        """Filtered OutliersOutput holding only frame rows. Empty for a result that is not tracking.

        Returns
        -------
        OutliersOutput
            The same kind of output, narrowed.
        """
        return self._tracking_level("unit")

    @property
    def tracks(self) -> Self:
        """Filtered OutliersOutput holding only track rows. Empty for a result that is not tracking.

        Returns
        -------
        OutliersOutput
            The same kind of output, narrowed.
        """
        return self._tracking_level("track")

    @property
    def detections(self) -> Self:
        """Filtered OutliersOutput holding only detection rows. Empty for a result that is not tracking.

        Returns
        -------
        OutliersOutput
            The same kind of output, narrowed.
        """
        return self._tracking_level(LABEL_KIND)

    def with_threshold(
        self,
        outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None = _UNSET,  # type: ignore[arg-type]
        cluster_threshold: ThresholdLike | None = _UNSET,  # type: ignore[arg-type]
    ) -> Self:
        """Re-detect outliers using a different threshold configuration.

        Re-runs detection on the stored statistics with the new threshold,
        without recomputing the underlying image statistics. This enables
        quick sensitivity analysis and threshold experimentation.

        Can be chained with :meth:`classwise` for per-class detection
        with a different threshold. Rarity rows are kept as they are: they are
        judged against their own share, not against this threshold. Combination and
        residual scores are refitted under the new threshold without scoring them again;
        a scalar reaches them only when it is statistical and has no limits.

        Parameters
        ----------
        outlier_threshold : ThresholdLike, dict, or None
            New threshold configuration for stats-based outliers. Accepts
            the same formats as :class:`Outliers`:

            - ``None``: ``AdaptiveThreshold(3.5)`` (Double-MAD with asymmetric bounds)
            - ``float``: symmetric multiplier for modified z-score
            - ``str``: named threshold type (``"zscore"``, ``"iqr"``, etc.)
            - ``tuple``: named threshold with bounds, e.g. ``("zscore", 2.5)``
            - :class:`~dataeval.protocols.Threshold`: fully configured
              threshold
            - ``Mapping[str, ThresholdLike]``: per-metric thresholds
        cluster_threshold : ThresholdLike or None
            New threshold configuration for cluster-based outlier detection.
            Accepts the same formats as ``outlier_threshold``. Only applies
            when cluster statistics are stored from :meth:`evaluate` or
            :meth:`~Outliers.from_clusters`.

        Returns
        -------
        OutliersOutput
            New output with re-detected outliers using the new threshold.

        Raises
        ------
        ValueError
            If no arguments are provided, or if this output was not created
            from an evaluation with stored statistics or cluster stats.

        Examples
        --------
        >>> outliers = Outliers(flags=ImageStats.PIXEL)
        >>> result = outliers.evaluate(dataset)

        Loosen the threshold:

        >>> lenient = result.with_threshold(4.0)

        Switch to IQR method:

        >>> iqr_result = result.with_threshold("iqr")

        Per-metric overrides:

        >>> custom = result.with_threshold({"brightness": 2.0, "contrast": ("zscore", 3.0)})

        Chain with classwise:

        >>> per_class = result.classwise(metadata).with_threshold(2.0)

        Adjust cluster threshold:

        >>> strict_clusters = result.with_threshold(cluster_threshold=1.5)
        """
        if outlier_threshold is self._UNSET and cluster_threshold is self._UNSET:
            raise ValueError("At least one of outlier_threshold or cluster_threshold must be provided.")
        return self._redetect(outlier_threshold=outlier_threshold, cluster_threshold=cluster_threshold)

    def _check_metrics(self, wanted: Sequence[str]) -> None:
        """Refuse metric names this result was not measured on, when it kept the statistics that say so."""
        combined = self._combined
        if combined is None and self.cluster_stats is None:
            return
        known = set(self.data()["metric_name"].cast(pl.Utf8).to_list())
        # Every check judged, including a rarity or combination check that flagged nothing.
        known |= set(self.populations["metric_name"].cast(pl.Utf8).to_list())
        known |= {check.metric_name for check in self.scored}
        if combined is not None:
            known |= set(combined[0])
        if self.cluster_stats is not None:
            known.add("cluster_distance")
        unknown = sorted(set(wanted) - known)
        if unknown:
            raise ValueError(f"prune: {unknown} are not metrics this result was measured on; it holds {sorted(known)}.")

    def prune(
        self,
        *,
        metrics: str | Sequence[str] | None = None,
        min_flags: int = 1,
    ) -> RemovalPlan:
        """
        Turn these outliers into a plan of what to remove.

        Every choice the policy makes is an argument: which metrics are worth removing a row over, and how many of
        them have to agree. The result itself only reports.

        Parameters
        ----------
        metrics : str or Sequence[str] or None, default None
            The metrics whose flags count. None counts every metric.
        min_flags : int, default 1
            How many distinct metrics, among those that count, have to flag a row before it is removed.

        Returns
        -------
        RemovalPlan
            The flagged rows: whole items, or detections when the statistics were measured per target, at the
            level each row was flagged at. Pass it to :class:`~dataeval.data.Indices` with ``exclude=True``.

        Raises
        ------
        ValueError
            If the result spans several datasets, if `min_flags` is less than 1, or if `metrics` names a metric
            this result was not measured on. The last is checked only when the result kept its statistics or
            cluster statistics; without them an unknown name cannot be told from one that flagged nothing.

        Notes
        -----
        A metric that was measured but flagged nothing is not an error: it contributes no rows.

        Apply the plan to the dataset this result was computed on. Its item indices are positions in that dataset.

        On tracking data a plan can name frames, which :class:`~dataeval.data.Indices` cannot remove yet. Leave
        them out before you apply the plan: ``RemovalPlan(a for a in plan if a.kind != "unit")``.

        Examples
        --------
        >>> from dataeval.data import Indices, View
        >>> from dataeval.flags import ImageStats

        >>> result = Outliers(flags=ImageStats.VISUAL).evaluate(dataset)
        >>> plan = result.prune(min_flags=2)
        >>> cleaned = View(dataset, Indices(plan, exclude=True))
        """
        frame = self.data()
        if "dataset_index" in frame.columns:
            raise ValueError(
                "prune only works with output from a single dataset, since a plan names rows of one dataset. "
                "Evaluate each dataset on its own and prune each result."
            )
        if min_flags < 1:
            raise ValueError(f"min_flags must be at least 1; got {min_flags}.")
        if metrics is not None:
            wanted = [metrics] if isinstance(metrics, str) else list(metrics)
            self._check_metrics(wanted)
            frame = frame.filter(pl.col("metric_name").cast(pl.Utf8).is_in(wanted))
        has_level = "level" in frame.columns
        flagged: dict[SourceIndex, set[str]] = {}
        for row in frame.iter_rows(named=True):
            flagged.setdefault(_row_address(row, has_level=has_level), set()).add(str(row["metric_name"]))
        return RemovalPlan(address for address, names in flagged.items() if len(names) >= min_flags)


# Convenience type aliases for parameterized output
SingleOutliersOutput = OutliersOutput[SingleOutliersMap]
SingleTargetOutliersOutput = OutliersOutput[SingleTargetOutliersMap]
MultiOutliersOutput = OutliersOutput[MultiOutliersMap]
MultiTargetOutliersOutput = OutliersOutput[MultiTargetOutliersMap]


# The columns every outliers frame drops when no row holds a value for them.
_DROPPABLE = ["target_index", "level", "method", "population", "observed"]

# The measurements, which describe a row rather than identify it. Everything else in the frame
# is part of which row this is, and so is what the frame is ordered on.
_UNSORTED_COLUMNS = frozenset({"metric_value", *_CONTEXT_COLUMNS, *_EXTRA_COLUMNS})


def _narrowed_rows(frame: pl.DataFrame | None, restricted: frozenset[FactorLevel | None] | None) -> pl.DataFrame | None:
    """Keep the rows of a stored findings frame at the kinds of row an output is restricted to."""
    if frame is None or restricted is None:
        return frame
    keep = pl.Series([kind in restricted for kind in row_kinds(frame).to_list()], dtype=pl.Boolean)
    return frame.filter(keep)


def _narrowed(
    rarity: tuple[pl.DataFrame, list[_Population]] | None, restricted: frozenset[FactorLevel | None] | None
) -> tuple[pl.DataFrame, list[_Population]] | None:
    """Keep the rarity rows and populations at the kinds of row an output is restricted to."""
    if rarity is None or restricted is None:
        return rarity
    frame, populations = rarity
    kept = _narrowed_rows(frame, restricted)
    return cast(pl.DataFrame, kept), [population for population in populations if population.level in restricted]


def _kept_checks(
    scored: Sequence[_ScoredCheck], restricted: frozenset[FactorLevel | None] | None
) -> tuple[_ScoredCheck, ...]:
    """Keep the combination checks at the kinds of row an output is restricted to."""
    return tuple(check for check in scored if restricted is None or check.kind in restricted)


def _restrict(
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    strata: _Strata | None,
    keep: NDArray[np.bool_],
) -> tuple[StatsMap, list[SourceIndex], _Strata | None]:
    """Keep the entries `keep` marks, in the statistics, the source index and the strata `_detect_outliers` indexes."""
    indices = np.flatnonzero(keep)
    return (
        {name: values[indices] for name, values in stats.items()},
        [source_index[i] for i in indices.tolist()],
        None if strata is None else strata._replace(codes=strata.codes[indices]),
    )


def _extend_context(context: dict[str, list[Any]], fit: _Fit) -> None:
    """Append one fit's flagged rows to the running context columns, in frame order."""
    context["direction"].extend(np.where(fit.above[fit.mask], "upper", "lower").tolist())
    context["bound"].extend(fit.bound[fit.mask].tolist())
    context["percentile"].extend(fit.percentile[fit.mask].tolist())
    context["population_mean"].extend(fit.mean[fit.mask].tolist())
    context["population_std"].extend(fit.std[fit.mask].tolist())


def _build_class_ids(  # noqa: C901
    source_index: Sequence[SourceIndex],
    metadata: MetadataLike,
) -> NDArray[np.intp]:
    """Map each source_index entry to a class ID for per-class outlier detection.

    For image-level entries (target is None):
        - If all targets in the image share one class, assign that class.
        - If the image has targets of multiple classes, assign -1 (global fallback).
    For target-level entries (target is not None):
        - Look up the target's class via the metadata mapping.

    Parameters
    ----------
    source_index : Sequence[SourceIndex]
        List of source index entries from image_stats.
    metadata : MetadataLike
        Metadata object with class_labels and optional item_indices.

    Returns
    -------
    NDArray[np.intp]
        Array of class IDs aligned with source_index, -1 for unclassifiable entries.
    """
    item_indices = _get_item_indices(metadata)
    class_labels = metadata.class_labels
    n = len(source_index)
    class_ids = np.full(n, -1, dtype=np.intp)

    # Build mapping: image_id -> list of global indices into class_labels
    image_to_label_indices: dict[int, list[int]] = {}
    for k, img_idx in enumerate(item_indices):
        image_to_label_indices.setdefault(int(img_idx), []).append(k)

    # Precompute unique classes per image
    image_to_unique_classes: dict[int, set[int]] = {}
    for img_idx, label_indices in image_to_label_indices.items():
        image_to_unique_classes[img_idx] = {int(class_labels[k]) for k in label_indices}

    for i, si in enumerate(source_index):
        kind = si.kind
        if kind is None:
            # Image-level: assign class only if image has a single unique class
            classes = image_to_unique_classes.get(si.item, set())
            if len(classes) == 1:
                class_ids[i] = next(iter(classes))
        elif kind == LABEL_KIND and si.key is not None:
            # Target-level: look up the specific target's class
            label_indices = image_to_label_indices.get(si.item, [])
            if si.key < len(label_indices):
                class_ids[i] = int(class_labels[label_indices[si.key]])
        # A key at any other level is that level's own key — a unit_index or a track_id —
        # and names no label, so there is no class to look up. Left at -1, the global
        # bucket, rather than indexing the label list with it and reporting the class of
        # whichever detection happened to sit at that number.

    return class_ids


def _class_labels(class_ids: NDArray[np.intp], metadata: MetadataLike) -> dict[int, str | None]:
    """Label classwise buckets: the class name, and class_label=null for rows with no one class."""
    index2label = _get_index2label(metadata)
    labels: dict[int, str | None] = {
        int(c): f"class_label={index2label.get(int(c), int(c))}" for c in np.unique(class_ids) if c >= 0
    }
    labels[-1] = "class_label=null"
    return labels


def _class_strata(source_index: Sequence[SourceIndex], metadata: MetadataLike) -> _Strata:
    """Stratify the statistics rows by class, for per-class detection."""
    if has_level_schema(metadata):
        return _view_class_strata(source_index, metadata)
    class_ids = _build_class_ids(source_index, metadata)
    return _Strata(class_ids, _class_labels(class_ids, metadata), ("<class>",))


def _view_class_strata(source_index: Sequence[SourceIndex], metadata: Any) -> _Strata:
    """Stratify by the class each row has at the metadata's view; a row with several classes or none is ``<axis>=null``.

    A level the class does not reach is judged whole: one off the view's branch, and on tracking data every level
    above the view (a sequence, a frame or a track on the detection view), as per-class detection always judged them.
    """
    levels, positions = row_positions(metadata, source_index)
    classes = classes_of(metadata, levels, positions)
    if "sequence" in metadata.levels:
        above = {level for level in levels if metadata._store.schema.is_ancestor(level, metadata.view)}
        classes = [None if level in above else found for level, found in zip(levels, classes, strict=True)]
    index2label = _get_index2label(metadata)
    axis = metadata.class_axis
    labels = [
        None if found is None else f"{axis}={index2label.get(found[0], found[0]) if len(found) == 1 else 'null'}"
        for found in classes
    ]
    return coded(labels, ("<class>",))


def _level_class_strata(score_input: _ScoreInput, metadata: MetadataLike) -> dict[FactorLevel, Any]:
    """Stratify the rows of each level with combinations by class, None where the class does not reach the level."""
    found: dict[FactorLevel, Any] = {}
    for level in score_input.scored_levels():
        addresses = [score_input.address(level, row) for row in range(score_input.frames[level].height)]
        strata = _class_strata(addresses, metadata)
        labels = {code: label for code, label in strata.labels.items() if label is not None}
        found[level] = None if np.all(strata.codes == -1) else (pl.Series(strata.codes.astype(str)), labels)
    return found


def _resolve_outlier_threshold(
    outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None,
) -> Threshold | Mapping[str, Threshold]:
    """Eagerly resolve an outlier_threshold value so that None is replaced with the default."""
    if isinstance(outlier_threshold, Mapping):
        return {k: resolve_threshold(v) for k, v in outlier_threshold.items()}
    if outlier_threshold is not None:
        return resolve_threshold(outlier_threshold)
    return resolve_threshold(None)


def _resolve_metric_threshold(
    outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None,
    metric_name: str,
) -> Threshold:
    """Resolve the Threshold object for a given metric name.

    Priority:
    1. If outlier_threshold is a dict and metric_name is in it, use that entry.
    2. If outlier_threshold is a non-dict ThresholdLike, use it for all metrics.
    3. Otherwise, use default AdaptiveThreshold (Double-MAD with asymmetric bounds).
    """
    if isinstance(outlier_threshold, Mapping):
        value = outlier_threshold.get(metric_name)
        if value is not None:
            return resolve_threshold(value)
    elif outlier_threshold is not None:
        return resolve_threshold(outlier_threshold)
    return DEFAULT_OUTLIERS_OUTLIER_THRESHOLD


def _place(into: _Fit, where: NDArray[np.bool_], part: _Fit) -> None:
    """Write one bucket's fit into the slots it occupies in the level-wide arrays."""
    for whole, piece in zip(into, part, strict=True):
        whole[where] = piece


def _fit_level(
    level_values: NDArray[Any],
    threshold: Threshold,
    level_codes: NDArray[np.intp],
) -> tuple[_Fit, list[tuple[int, int, str | None]]]:
    """Fit one level's values, in one population per stratum code.

    Parameters
    ----------
    level_values : NDArray
        Metric values for entries at this level.
    threshold : Threshold
        Threshold instance to compute bounds.
    level_codes : NDArray[np.intp]
        Each entry's population code at this level, -1 for the whole level.

    Returns
    -------
    _Fit
        Arrays over ``level_values``, each entry described against the population it was
        thresholded within -- the level as a whole, or its own class.
    list[tuple[int, int, str or None]]
        One ``(code, n, reason)`` per population: its code, its count of real values, and why it
        was not judged, or None.

    Notes
    -----
    Entries with no class to look up (code -1) form their own bucket, which is the whole level
    when no entry has a class.
    """
    fit = _unfitted(len(level_values))
    buckets: list[tuple[int, int, str | None]] = []
    for code in np.unique(level_codes).tolist():
        bucket = level_codes == code
        part, n, reason = _fit_outliers(level_values[bucket], threshold)
        _place(fit, bucket, part)
        buckets.append((code, n, reason))
    return fit, buckets


def _population_label(strata: _Strata | None, code: int, codes_at_level: NDArray[np.intp]) -> str | None:
    """Return the label of one bucket: None for the whole level, the strata's label otherwise."""
    if strata is None or (code == -1 and np.all(codes_at_level == -1)):
        return None
    return strata.labels.get(code)


def _masks_by_level(source_index: Sequence[SourceIndex]) -> list[NDArray[np.bool_]]:
    """One boolean mask per kind of row the source index names.

    Grouped by :attr:`~dataeval.types.SourceIndex.kind`, which says what an address names
    without a dataset to resolve it against and reads two spellings of one row as one kind.
    On the two-way split every producer emits today this returns exactly the item mask and
    the label mask, in that order; on a source index that states levels it returns one mask
    per level, so each is thresholded against its own distribution.
    """
    # Insertion-ordered, so the item level keeps coming first on the two-way split, which
    # is the order image_stats emits and the order the rows are already in. Numbered as
    # the kinds are seen rather than by looking each up in a list of the distinct ones,
    # which is a linear scan per entry over a source index holding one entry per detection.
    codes: dict[FactorLevel | None, int] = {}
    as_array = np.fromiter(
        (codes.setdefault(src_idx.kind, len(codes)) for src_idx in source_index),
        dtype=np.intp,
        count=len(source_index),
    )
    return [as_array == code for code in range(len(codes))]


def _detect_outliers(  # noqa: C901
    stats: StatsMap,
    source_index: Sequence[SourceIndex],
    outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None,
    strata: _Strata | None = None,
) -> tuple[pl.DataFrame, list[_Population]]:
    """Detect outliers from pre-computed statistics, optionally per stratum.

    Parameters
    ----------
    stats : StatsMap
        Dictionary mapping metric names to value arrays.
    source_index : Sequence[SourceIndex]
        Source index entries aligned with stat values.
    outlier_threshold : ThresholdLike or dict or None
        Threshold configuration for outlier detection.
    strata : _Strata or None
        Each entry's population, aligned with `source_index`, or None to judge each level whole.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: item_index, target_index, metric_name, metric_value.
    list[_Population]
        Every population each metric was judged in at each level, and whether it was.

    Notes
    -----
    Columns holding NaN follow one rule: **a NaN is never an outlier.** It records a
    measurement that was not made, not an unusual value, and the two must not be
    conflated — an absent band would otherwise read as the most anomalous thing in the
    dataset.

    Three cases follow from it:

    - **Entirely NaN** — no outliers. Named band groups make this ordinary rather than
      pathological: a group no datum can satisfy is measured over an all-NaN slice, so
      ``image_stats(channels={"nir": 3})`` against 3-channel data yields a full
      ``nir_mean`` column of NaN.
    - **Partly NaN** — thresholded over the values it does have. Every statistical
      threshold reduces NaN-aware, so the surviving values set the bounds and the NaN
      entries are excluded from the result rather than compared against them.
    - **Too few real values** — no outliers. A statistical threshold is fitted only to a
      population of at least ``_MIN_POPULATION`` real values, so nothing is flagged off a
      sample too small to describe a distribution. The population is listed as not judged.
    """
    item_ids: list[int] = []
    target_ids: list[int | None] = []
    levels: list[str | None] = []
    metric_names: list[str] = []
    metric_values: list[float] = []
    context: dict[str, list[Any]] = {name: [] for name in _CONTEXT_COLUMNS}
    row_populations: list[str | None] = []
    populations: list[_Population] = []
    codes = np.full(len(source_index), -1, dtype=np.intp) if strata is None else strata.codes

    if len(source_index) > 0:
        # One mask per kind of row the source index names, rather than the two an
        # ``(item, target)`` address could ever describe. A threshold is a distribution
        # fitted to the values under it, so pooling a per-frame reading with a
        # per-sequence one would compare each against a spread neither belongs to.
        level_masks = _masks_by_level(source_index)

        for stat, values in stats.items():
            if values.ndim == 1 and np.issubdtype(values.dtype, np.number):
                threshold = _resolve_metric_threshold(outlier_threshold, stat)

                for level_mask in level_masks:
                    if not np.any(level_mask):
                        continue

                    level_values = values[level_mask]
                    level_indices = np.flatnonzero(level_mask)
                    level_codes = codes[level_mask]
                    fit, buckets = _fit_level(level_values, threshold, level_codes)
                    # The level's kind, which names the level for every row but an item's own. A bucket
                    # with no real value is not listed: the metric was not measured there, as a metadata
                    # factor is not at the levels it does not live at.
                    level = source_index[level_indices[0]].kind
                    label_of = {code: _population_label(strata, code, level_codes) for code, _, _ in buckets}
                    populations.extend(
                        _Population(level, stat, None, label_of[code], n, reason is None, reason)
                        for code, n, reason in buckets
                        if n > 0
                    )

                    if np.any(fit.mask):
                        outlier_indices = level_indices[fit.mask]
                        item_ids.extend(source_index[idx].item for idx in outlier_indices)
                        target_ids.extend(source_index[idx].key for idx in outlier_indices)
                        levels.extend(reported_level(source_index[idx]) for idx in outlier_indices)
                        metric_names.extend([stat] * len(outlier_indices))
                        metric_values.extend(values[outlier_indices].tolist())
                        row_populations.extend(label_of[code] for code in level_codes[fit.mask].tolist())
                        _extend_context(context, fit)

    if not item_ids:
        return pl.DataFrame(schema=FINDINGS_SCHEMA), populations

    df = pl.DataFrame(
        {
            "item_index": pl.Series(item_ids, dtype=pl.Int64),
            "target_index": pl.Series(target_ids, dtype=pl.Int64),
            "level": pl.Series(levels, dtype=lexical_categorical()),
            "metric_name": pl.Series(metric_names, dtype=lexical_categorical()),
            "metric_value": pl.Series(metric_values, dtype=pl.Float64),
            **{name: pl.Series(context[name], dtype=dtype) for name, dtype in _CONTEXT_COLUMNS.items()},
            "method": pl.Series([None] * len(item_ids), dtype=_EXTRA_COLUMNS["method"]),
            "population": pl.Series(row_populations, dtype=_EXTRA_COLUMNS["population"]),
            "observed": pl.Series([None] * len(item_ids), dtype=_EXTRA_COLUMNS["observed"]),
        },
    )

    # Level sits between the key and the metric: it is part of which row this is, so two
    # readings of one item at different levels are ordered rather than left tied.
    return df.sort(["item_index", "target_index", "level", "metric_name"]), populations


def _annotated_metadata(data: Any) -> Metadata | None:
    """Build the metadata of a dataset with labels and datum metadata; None for images alone, which have none."""
    if not isinstance(data, AnnotatedDataset):
        return None
    try:
        return Metadata(data)
    except MaiteShapeError:
        return None


def _add_measured_factors(
    built: Metadata, stats: StatsResult, source_index: Sequence[SourceIndex] | None = None
) -> list[str]:
    """Add a stats result's scalar statistics to `built` as factors, returning the ordered ones added."""
    before = set(built.factor_names)
    # A factor is one value per row; the vector-valued statistics (histogram, percentiles) are not thresholded.
    scalar = {name: values for name, values in stats["stats"].items() if np.ndim(values) == 1}
    built.add_factors(cast(Any, {**stats, "stats": scalar}), source_index=source_index)
    # A stat that is a flag rather than a quantity, such as an invalid box, is not thresholded.
    return [name for name in built.factor_names if name not in before and factor_kind(built, name) == "ordered"]


def _roll_up(built: Metadata, added: Sequence[str], plan: LevelPlan) -> tuple[Metadata, list[str]]:
    """Roll the frame and detection statistics up to the videos and tracks the plan judges, by their median."""
    at = factor_levels(built)
    names: list[str] = []
    pairs: tuple[tuple[FactorLevel, FactorLevel, str], ...] = (
        ("unit", "sequence", "unit_"),
        ("instance", "track", "instance_"),
    )
    for source, target, prefix in pairs:
        stems = {name: name.removeprefix(prefix) for name in added if at[name] == source}
        if target not in plan.levels or not stems:
            continue
        exprs = [pl.col(name).median().alias(f"{target}_{stem}") for name, stem in stems.items()]
        built = built.agg(source, target, *exprs)
        names.extend(f"{target}_{stem}" for stem in stems.values())
    return built, names


def _refuse_unjudged(measured: Mapping[str, FactorLevel], plan: LevelPlan, n_rows: int) -> None:
    """Refuse a video plan that judges none of the measured statistics, saying why."""
    if not n_rows:
        # `_video_frames` refused a view that keeps no frames, and any kept frame is a row unless per_image=False,
        # so only detections were measured, and there are none.
        raise ValueError(
            "Outliers.evaluate: this dataset has no detections to judge; pass per_image=True to judge its frames."
        )
    if not measured:
        raise ValueError(
            "Outliers.evaluate: the statistics `flags` asks for hold no ordered value to threshold; "
            "include ImageStats.PIXEL or ImageStats.VISUAL, for example."
        )
    # Called only for its raise: no measured statistic lives at a planned level.
    _factors_at_planned_levels(measured, plan, {level: level for level in measured.values()}, named=False)


def _refuse_foreign_metadata(metadata: MetadataLike | None, built: Metadata) -> None:
    """Refuse a `metadata=` that is not the Metadata of the tracking dataset evaluated.

    Checked before it supplies classes or strata, whose rows would otherwise be matched to the wrong frames and
    detections, or to none, without a word.
    """
    if metadata is None:
        return
    if not has_level_schema(metadata) or "sequence" not in cast(Metadata, metadata).levels:
        raise ValueError(
            "Outliers.evaluate: metadata= must be the Metadata of this tracking dataset, Metadata(dataset); "
            "this one has no tracking levels."
        )
    held, expected = dict(cast(Metadata, metadata).level_counts), dict(built.level_counts)
    if held != expected:
        raise ValueError(
            f"Outliers.evaluate: metadata= holds {held} rows by level and this tracking dataset holds {expected}; "
            "pass the metadata of the data evaluated, Metadata(dataset)."
        )


def _video_frames(data: Any) -> SequenceFrames:
    """Present a tracking dataset as every one of its frames; a frame view is measured as given."""
    frames = data if isinstance(data, SequenceFrames) else SequenceFrames(data)
    # Read before `n_dropped`, which a selector that cannot plan only learns by walking -- and
    # asking for the map is what walks it.
    kept = len(frames.frame_map)
    if not kept:
        raise ValueError(f"Outliers.evaluate: {frames.selector!r} kept no frames, so there is nothing to judge.")
    if frames.n_dropped:
        _logger.info(
            "Outliers: measuring %d of %d frame(s); %r dropped the rest. Every unweighted statistic "
            "therefore describes the frames kept rather than the source stream.",
            kept,
            frames.n_source_frames,
            frames.selector,
        )
    return frames


def _check_metadata_evaluate_preconditions(data: Any, other: Sequence[Any], has_extractor: bool) -> None:
    """Validate a metadata `evaluate()` call before any factor is read."""
    if other:
        raise ValueError("Outliers.evaluate: metadata evaluation takes one metadata, not several.")
    if has_extractor:
        raise ValueError(
            "Outliers.evaluate: cluster-based detection embeds images, and a metadata holds none. "
            "Pass the dataset for cluster detection."
        )
    if not all(hasattr(data, name) for name in ("levels", "rows_at", "at", "item_level", "label_level")):
        raise TypeError("Outliers.evaluate: thresholding factors needs a dataeval.Metadata with a level schema.")


def _refuse_no_flag(levels: str | Sequence[str] | None, per_image: bool | None, per_target: bool | None) -> None:
    """Refuse flags that judge nothing. A dataset reaches the same refusal after its own preconditions."""
    if levels is None and not (True if per_image is None else per_image) and not per_target:
        raise ValueError("At least one of per_image or per_target must be True.")


def _plan_for(
    levels: str | Sequence[str] | None,
    per_image: bool | None,
    per_target: bool | None,
    tracking: bool,
    subject: str,
) -> LevelPlan:
    """Resolve which levels to judge, refusing a tracking name on an image dataset."""
    if subject == "dataset" and not tracking and levels is not None:
        named = {levels} if isinstance(levels, str) else set(levels)
        tracking_only = sorted(named & set(TRACKING_LEVEL_NAMES))
        if tracking_only:
            raise ValueError(
                f"Outliers.evaluate: levels {tracking_only} are tracking levels, which an image dataset does not "
                "have; it accepts ['item', 'target']. Pass a tracking dataset, or a Metadata of one "
                "(Metadata(dataset) plus add_factors), to judge them."
            )
    return resolve_levels(levels, per_image, per_target, tracking, caller="Outliers.evaluate", subject=subject)


def _level_names(data: Any, tracking: bool) -> dict[FactorLevel, str]:
    """Map each of the metadata's levels to the name `levels=` spells it with."""
    if tracking:
        return {level: level for level in data.levels}
    # The item level last, so it keeps the name where a single-level metadata puts both ends on one level.
    return {data.label_level: "target", data.item_level: "item"}


def _refuse_lone_target(plan: LevelPlan, data: Any, tracking: bool) -> None:
    """Refuse `target` alone on a metadata whose item and label levels are one level; `item` names it."""
    if tracking or data.item_level != data.label_level:
        return
    if "target" in plan.levels and "item" not in plan.levels:
        raise ValueError(
            "Outliers.evaluate: this metadata has one level, which `levels` calls 'item'; "
            "judge it with per_image=True or levels='item'."
        )


def _include_hint(names: Iterable[str]) -> str:
    """Spell how to include levels: by `levels=` and by the flag that implies them."""
    wanted = sorted(set(names))
    flags = sorted({"per_target=True" if name in TARGET_LEVELS else "per_image=True" for name in wanted})
    return f"pass levels={wanted} or {' and '.join(flags)}"


def _factors_at_planned_levels(
    factor_levels: Mapping[str, FactorLevel],
    plan: LevelPlan,
    names: Mapping[FactorLevel, str],
    named: bool,
    *,
    others: bool = False,
) -> dict[str, FactorLevel]:
    """Return the factors that live at a planned level.

    A named factor that lives elsewhere raises. A factor left out of the default selection is
    dropped, unless that would leave nothing to threshold and no other check (`others`) runs.
    """
    kept: dict[str, FactorLevel] = {
        name: level for name, level in factor_levels.items() if names.get(level) in plan.levels
    }
    left_out = {name: names.get(level, level) for name, level in factor_levels.items() if name not in kept}
    if named and left_out:
        # The hint keeps the levels of the named factors already judged, so following it leaves them judged.
        wanted = set(left_out.values()) | {names[level] for level in kept.values()}
        raise ValueError(
            f"Outliers.evaluate: {sorted(left_out)} were named but live at {sorted(set(left_out.values()))}, "
            f"which this call does not judge; {_include_hint(wanted)}."
        )
    if not kept and not others:
        raise ValueError(
            f"Outliers.evaluate: no ordered factor lives at {sorted(plan.levels)}. The ordered factors live at "
            f"{sorted(set(left_out.values()))}; {_include_hint(left_out.values())}."
        )
    return kept


def _categorical_factors(
    data: Any,
    factors: Sequence[str] | None,
    strata: Sequence[str] | None,
    plan: LevelPlan,
    names: Mapping[FactorLevel, str],
) -> dict[str, FactorLevel]:
    """Return the categorical factors at the judged levels that rarity reads: the ones named, or every one.

    Without strata a named categorical factor is left to :func:`eligible_factors` to refuse. Every one is kept when
    `factors` is None, so :meth:`OutliersOutput.stratified` can judge them later.
    """
    if factors is not None and strata is None:
        return {}
    found: dict[str, FactorLevel] = {
        name: level
        for name, level in factor_levels(data).items()
        if (factors is None or name in factors) and factor_kind(data, name) == "categorical"
    }
    return _factors_at_planned_levels(found, plan, names, named=factors is not None, others=True)


def _rarity_runs(data: Any, categorical: Mapping[str, FactorLevel], strata: Sequence[str] | None, named: bool) -> bool:
    """Whether the strata reach a categorical factor that is not itself a stratum, refusing named ones they miss.

    Raises
    ------
    ValueError
        When a categorical factor named in `factors=` lives at a level the strata do not reach.
    """
    if strata is None:
        return False
    judged: dict[str, FactorLevel] = {name: level for name, level in categorical.items() if name not in strata}
    reached = set(reached_levels(data, strata, dict.fromkeys(judged.values())))
    missed = {name: level for name, level in judged.items() if level not in reached}
    if named and missed:
        raise ValueError(
            f"Outliers.evaluate: categorical factors {missed} (each at its own level) are judged by rarity within "
            f"a stratum, and strata {list(strata)} do not reach their levels. A stratum reaches its own level and "
            "the levels below it."
        )
    return bool(reached)


class _Requested(NamedTuple):
    """The combination, residual and neighbor checks an evaluate call asked for, as given."""

    combinations: Sequence[Sequence[str]] | Literal["all"] | None
    given: Mapping[str, str] | None
    kinds: Mapping[str, ColumnKind] | None
    neighbors: Sequence[str] | Literal["all"] | None


_METADATA_ROUTE = (
    "pass the Metadata itself as `data`: Metadata(dataset), with the statistics to judge added by add_factors "
    "(for a tracking dataset, placed on its rows by SequenceFrames.to_source)."
)


def _refuse_factor_arguments(factors: Sequence[str] | None, checks: _Requested) -> None:
    """Refuse the arguments that read metadata factors on a dataset, image or tracking, naming the Metadata route."""
    if factors is not None:
        raise ValueError(f"Outliers.evaluate: `factors` names metadata factors to threshold; {_METADATA_ROUTE}")
    if any(value is not None for value in checks):
        raise ValueError(
            f"Outliers.evaluate: combinations=, given=, kinds= and neighbors= read metadata factors; {_METADATA_ROUTE}"
        )


def _refuse_video_arguments(
    data: Any,
    other: Sequence[Any],
    extractor: FeatureExtractor | None,
    factors: Sequence[str] | None,
    checks: _Requested,
) -> None:
    """Refuse what a tracking dataset does not support, before any frame is decoded."""
    if not all(yields_frames(dataset) for dataset in (data, *other)):
        raise ValueError(
            "Outliers.evaluate: cannot combine tracking datasets with image datasets in one call. A tracking "
            "dataset's item index is a video and an image dataset's is an image, so the rows would address two "
            "different things under one column."
        )
    if other:
        raise ValueError(
            "Outliers.evaluate: several datasets are not supported for tracking datasets yet; evaluate each."
        )
    if extractor is not None:
        raise ValueError(
            "Outliers.evaluate: cluster-based detection on a tracking dataset is not supported; the frames of one "
            "video cluster together, so a frame's cluster distance mostly says which video it is from. Use "
            "`from_clusters` over `Embeddings(SequenceFrames(dataset))`."
        )
    _refuse_factor_arguments(factors, checks)


def _family(data: Any, name: str, kinds: Mapping[str, ColumnKind]) -> str | None:
    """How core reads a column: ``"categorical"``, ``"numeric"``, or None when it reads it not at all."""
    kind = kinds.get(name) or {"ordered": "numeric", "categorical": "categorical"}.get(factor_kind(data, name))
    return None if kind is None else "categorical" if kind in _CATEGORICAL_KINDS else "numeric"


def _column_level(data: Any, levels: Mapping[str, FactorLevel], name: str) -> FactorLevel:
    """Return the level a column lives at: a factor's own, or the level whose own frame holds a list column."""
    if name in levels:
        return levels[name]
    for level in data.levels:
        if name in data._store.frame(level).columns:
            return level
    raise KeyError(f"Outliers.evaluate: {name!r} is not a column of this metadata. Its factors are {sorted(levels)}.")


def _deepest(data: Any, group: Sequence[str], own: Mapping[str, FactorLevel]) -> FactorLevel:
    """Return the deepest level among a group's columns, refusing two that lie on different branches."""
    for (first, at), (second, other) in itertools.combinations(own.items(), 2):
        if not (data._levels.propagates_to(at, other) or data._levels.propagates_to(other, at)):
            raise ValueError(
                f"Outliers.evaluate: {list(group)} holds {first!r} at {at!r} and {second!r} at {other!r}, levels on "
                "different branches of the level schema, so no row holds both. Group columns on one chain of levels."
            )
    return next(level for level in own.values() if all(data._levels.propagates_to(o, level) for o in own.values()))


def _scoring_level(
    data: Any,
    levels: Mapping[str, FactorLevel],
    group: Sequence[str],
    plan: LevelPlan,
    names: Mapping[FactorLevel, str],
) -> FactorLevel:
    """Return the level a group is scored at, the deepest of its columns', refusing one the call does not judge."""
    level = _deepest(data, group, {name: _column_level(data, levels, name) for name in group})
    if names.get(level) not in plan.levels:
        raise ValueError(
            f"Outliers.evaluate: {list(group)} is scored at {level!r}, the deepest level among its columns, which "
            f"this call does not judge; {_include_hint([names.get(level, level)])}."
        )
    return level


def _rarity_reaches(data: Any, strata: Sequence[str] | None, level: FactorLevel) -> bool:
    """Whether a stratum reaches `level`, so a categorical group there can be judged by rarity."""
    return strata is not None and bool(reached_levels(data, strata, [level]))


def _named_checks(
    data: Any,
    levels: Mapping[str, FactorLevel],
    checks: _Requested,
    strata: Sequence[str] | None,
    plan: LevelPlan,
    names: Mapping[FactorLevel, str],
) -> dict[FactorLevel, list[tuple[str, ...]]]:
    """Place each named group at its scoring level, refusing a single column and an unstratified categorical group."""
    named = _named_groups(cast(Sequence[Sequence[str]] | None, checks.combinations)) or []
    # A sample column (a list of numbers per row) is not judged on its own elsewhere: core scores it by distribution.
    single = [group for group in named if len(group) == 1 and (checks.kinds or {}).get(group[0]) != "sample"]
    if single:
        raise ValueError(
            f"Outliers.evaluate: combinations {single} name one column; a single column is already judged on its own. "
            "Only a list column read as kinds={name: 'sample'} is scored alone, by its distribution."
        )
    found: dict[FactorLevel, list[tuple[str, ...]]] = {}
    for group in named:
        level = _scoring_level(data, levels, group, plan, names)
        if all(_family(data, name, checks.kinds or {}) == "categorical" for name in group):
            _refuse_categorical_group(data, group, strata, level)
        found.setdefault(level, []).append(group)
    return found


def _refuse_categorical_group(
    data: Any, group: Sequence[str], strata: Sequence[str] | None, level: FactorLevel
) -> None:
    """Refuse a categorical group that no stratum reaches, or that holds a stratum column."""
    if not _rarity_reaches(data, strata, level):
        raise ValueError(
            f"Outliers.evaluate: {list(group)} are categorical, so the group is judged by rarity within a "
            f"stratum; pass strata= that reach {level!r}."
        )
    shared = [name for name in group if name in (strata or ())]
    if shared:
        raise ValueError(
            f"Outliers.evaluate: {list(group)} holds {shared}, also named in strata=. A categorical group is judged "
            f"by rarity within each stratum, where {shared} holds a single value. Leave {shared} out of the group "
            "or out of strata=."
        )


def _refuse_unknown_kinds(checks: _Requested, levels: Mapping[str, FactorLevel], factors: Sequence[str] | None) -> None:
    """Refuse a `kinds=` key that names no column of a group or `given` pair, which would otherwise be ignored."""
    if checks.combinations == "all":
        known = {name for name in levels if factors is None or name in factors}
    else:
        named = _named_groups(cast(Sequence[Sequence[str]] | None, checks.combinations)) or []
        known = {name for group in named for name in group}
    known |= {name for pair in (checks.given or {}).items() for name in pair}
    unknown = sorted(set(checks.kinds or {}) - known)
    if unknown:
        raise KeyError(
            f"Outliers.evaluate: kinds= names {unknown}, which no group of combinations= or pair of given= holds. "
            f"Their columns are {sorted(known)}."
        )


def _searched_pairs(
    data: Any,
    levels: Mapping[str, FactorLevel],
    checks: _Requested,
    factors: Sequence[str] | None,
    strata: Sequence[str] | None,
    judged: Sequence[FactorLevel],
) -> dict[FactorLevel, list[tuple[str, ...]]]:
    """Every pair of one family among the factors that live at each judged level, within `factors`.

    Categorical pairs are kept for a later stratification, and judged only where a stratum reaches their level.

    Raises
    ------
    ValueError
        When no judged level has a pair to run now and no residual is asked for.
    """
    found: dict[FactorLevel, list[tuple[str, ...]]] = {}
    for level in judged:
        family = {
            name: _family(data, name, checks.kinds or {})
            for name, own in levels.items()
            if own == level and (factors is None or name in factors)
        }
        pairs = [
            (first, second)
            for first, second in itertools.combinations(family, 2)
            if family[first] is not None and family[first] == family[second]
        ]
        if pairs:
            found[level] = pairs
    runs = any(
        _family(data, group[0], checks.kinds or {}) == "numeric" or _rarity_reaches(data, strata, level)
        for level, groups in found.items()
        for group in groups
    )
    if not runs and not checks.given:
        raise ValueError(
            f"Outliers.evaluate: combinations='all' finds no pair of factors to score at {list(judged)}; a level needs "
            "two numeric factors, or two categorical ones and strata= that reach it. Name groups in combinations=, "
            "or add given=."
        )
    return found


def _judged_levels(data: Any, plan: LevelPlan, names: Mapping[FactorLevel, str]) -> list[FactorLevel]:
    """Return the metadata's levels that the plan judges."""
    return [level for level in data.levels if names.get(level) in plan.levels]


def _combination_checks(
    data: Any,
    checks: _Requested,
    factors: Sequence[str] | None,
    strata: Sequence[str] | None,
    plan: LevelPlan,
    names: Mapping[FactorLevel, str],
) -> tuple[dict[FactorLevel, list[tuple[str, ...]]], dict[FactorLevel, dict[str, str]]]:
    """Place each group and residual the call asked for at the level it is scored at."""
    levels = factor_levels(data)
    _refuse_unknown_kinds(checks, levels, factors)
    if checks.combinations == "all":
        judged = _judged_levels(data, plan, names)
        groups = _searched_pairs(data, levels, checks, factors, strata, judged)
    else:
        groups = _named_checks(data, levels, checks, strata, plan, names)
    given: dict[FactorLevel, dict[str, str]] = {}
    for y, x in (checks.given or {}).items():
        given.setdefault(_scoring_level(data, levels, (y, x), plan, names), {})[y] = x
    return {level: list(dict.fromkeys(found)) for level, found in groups.items()}, given


def _attach_neighbors(result: OutliersOutput[Any], neighbors: Mapping[FactorLevel, Sequence[str]]) -> None:
    """Score the neighbor checks once and store them on a fresh metadata result, before its first re-detection."""
    if not neighbors or result.score_input is None:
        return
    scored, order = score_neighbors(result.score_input, neighbors)
    result.scored = tuple(scored)
    result._order = order


def _strata_names(strata: str | Sequence[str] | None, per_class: bool) -> list[str] | None:
    """Read `strata=` as a list of names, refusing it beside per_class=True."""
    if strata is None:
        return None
    if not strata:
        raise ValueError("Outliers.evaluate: strata= names no strata; pass None to judge each level whole.")
    if per_class:
        raise ValueError(
            "Outliers.evaluate: strata= and per_class=True both stratify; pass one. "
            "per_class is strata by the class label, spelled classwise()."
        )
    return [strata] if isinstance(strata, str) else list(strata)


class Outliers(Evaluator):
    r"""
    Computes statistical outliers of a dataset using various statistical tests applied to each image.

    Supports two complementary detection methods:

    1. **Image statistics-based**: Computes pixel-level statistics (brightness, contrast, etc.)
       and flags images with unusual values using configurable :class:`~dataeval.protocols.Threshold`
       objects.
    2. **Cluster-based**: Uses embeddings from a neural network to cluster images and identifies
       outliers based on distance from cluster centers in embedding space.

    Both methods can be used together or independently based on the ``flags`` parameter.
    A tracking dataset is judged by image statistics only, in video terms: its videos,
    frames, tracks and detections. See :meth:`evaluate`.

    Parameters
    ----------
    flags : ImageStats, default ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
        Statistics to compute for image statistics-based outlier detection. Set to
        ``ImageStats.NONE`` to skip image statistics and use only cluster-based detection
        (requires ``extractor``).
    outlier_threshold : ThresholdLike, dict, or None, default None
        Threshold configuration for image statistics-based outlier detection.

        - ``None``: uses ``AdaptiveThreshold()`` with default multiplier (3.5),
          which computes both z-score and modified z-score bounds and takes the
          wider (more lenient) bound on each side.
        - ``float``: symmetric multiplier for the default method (modified z-score
          via ``resolve_threshold``)
        - ``str``: named threshold type (e.g., ``"zscore"``, ``"iqr"``,
          ``"adaptive"``) with defaults
        - ``tuple[float | None, float | None]``: asymmetric ``(lower, upper)`` multipliers
        - ``tuple[str, ThresholdBounds]``: named threshold with bounds, e.g.
          ``("zscore", 2.5)`` or ``("iqr", (1.0, 3.0))``
        - :class:`~dataeval.protocols.Threshold`: a fully configured threshold
          (e.g., ``ZScoreThreshold``, ``IQRThreshold``, ``ConstantThreshold``,
          ``AdaptiveThreshold``)
        - ``Mapping[str, ThresholdLike]``: per-metric thresholds keyed by metric name.
          Metrics not in the dict use the default (``AdaptiveThreshold()``).
    extractor : FeatureExtractor, optional
        Feature extractor for cluster-based outlier detection. When provided, embeddings
        are extracted and clustered to find semantic/visual outliers in embedding space.
    batch_size : int or None, default None
        Batch size for feature extraction during cluster-based detection. If None, uses DataEval
        default. Must be set by either parameter or global default if extractor is provided.
    cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
        Clustering algorithm for cluster-based detection.
    cluster_threshold : ThresholdLike or None, default None
        Threshold configuration for cluster-based outlier detection. When None,
        defaults to ``ZScoreThreshold(upper_multiplier=2.5)``.
        Accepts the same formats as ``outlier_threshold``.
        Only used when ``extractor`` is provided.
    n_clusters : int, optional
        Expected number of clusters. For HDBSCAN, this is a hint that adjusts
        min_cluster_size. For KMeans, this is the exact number of clusters.
    config : Outliers.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Attributes
    ----------
    stats : StatsResult
        Statistics computed during the last evaluate() call.
        Contains dimension, pixel, and/or visual statistics based on the flags.
        On a tracking dataset these are the frame view's statistics, addressed by position in
        the frame view rather than by video; the result's rows are the same statistics
        addressed in the source dataset.
    flags : ImageStats
        Statistics to compute for outlier detection.
    outlier_threshold : ThresholdLike | Mapping[str, ThresholdLike] | None
        Threshold configuration for outlier detection.
    extractor : FeatureExtractor | None
        Feature extractor for cluster-based detection.
    cluster_threshold : ThresholdLike | None
        Threshold configuration for cluster-based detection.
    cluster_algorithm : Literal["kmeans", "hdbscan"]
        Clustering algorithm to use.
    n_clusters : int | None
        Expected number of clusters.

    See Also
    --------
    :term:`Duplicates`

    Notes
    -----
    **Threshold Methods:**

    - ``AdaptiveThreshold`` (default): Uses tail-weighted Double-MAD (separate MAD for
      data below and above the median) with automatic multiplier scaling for heavy
      tails to produce asymmetric bounds. Default multiplier: 3.5.
    - ``ModifiedZScoreThreshold``: Based on median absolute deviation. Default multiplier: 3.5.
      Modified z score = :math:`0.6745 * |x_i - x̃| / MAD`
    - ``ZScoreThreshold``: Based on standard deviation from mean. Default multiplier: 3.0.
      Z score = :math:`|x_i - \mu| / \sigma`
    - ``IQRThreshold``: Based on interquartile range. Default multiplier: 1.5.
      Outliers are outside :math:`[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]`
    - ``ConstantThreshold``: Hard lower/upper bounds (data-independent).

    All threshold types support asymmetric lower/upper multipliers via
    ``lower_multiplier`` and ``upper_multiplier`` parameters.

    **Pixel Units:**

    ``PIXEL`` statistics are measured in the units the data is stored in, as
    :func:`~dataeval.core.image_stats` measures them by default, so :meth:`evaluate` and
    :meth:`from_stats` given statistics from ``image_stats`` agree. Only a
    ``ConstantThreshold`` depends on the unit. To compare images stored at different bit
    depths on one scale, compute the statistics with ``normalize_pixel_values=True`` and
    pass them to :meth:`from_stats`.

    .. versionchanged:: 1.2
        :meth:`evaluate` no longer normalizes pixel values to ``[0, 1]`` before measuring.

    **Cluster-based Detection:**

    Uses adaptive distance-based detection that accounts for varying cluster densities.
    A :class:`~dataeval.protocols.Threshold` is applied per-cluster to the distance
    distribution (default: ``ZScoreThreshold(upper_multiplier=2.5)``), and points whose
    distance exceeds the upper bound are flagged as outliers.

    References
    ----------
    [1] How to Detect and Handle Outliers.
        Iglewicz, B., & Hoaglin, D. C. (1993). ASQC Quality Press.
        (Origin of the modified z-score and of the 3.5 multiplier used by
        ``ModifiedZScoreThreshold``. Note this class defaults to
        ``AdaptiveThreshold()``, not to the modified z-score.)

    Examples
    --------
    Basic image statistics-based outlier detection (default: modified z-score):

    >>> outliers = Outliers()
    >>> result = outliers.evaluate(dataset)

    Using a specific threshold method:

    >>> from dataeval.utils.thresholds import ZScoreThreshold
    >>> outliers = Outliers(outlier_threshold=ZScoreThreshold(2.5))

    Asymmetric thresholds (stricter on lower, lenient on upper):

    >>> from dataeval.utils.thresholds import IQRThreshold
    >>> outliers = Outliers(outlier_threshold=IQRThreshold(lower_multiplier=1.0, upper_multiplier=3.0))

    Hard bounds:

    >>> from dataeval.utils.thresholds import ConstantThreshold
    >>> outliers = Outliers(outlier_threshold=ConstantThreshold(lower=0.1, upper=0.9))

    Named threshold type with bounds (no need to import threshold classes):

    >>> outliers = Outliers(outlier_threshold="iqr")
    >>> outliers = Outliers(outlier_threshold=("zscore", 2.5))
    >>> outliers = Outliers(outlier_threshold=("modzscore", (1.0, 3.0)))

    Named threshold type with bounds and limits  (no need to import threshold classes):

    >>> outliers = Outliers(outlier_threshold=("zscore", 4.0, (0.0, 1.0)))

    Per-metric thresholds:

    >>> outliers = Outliers(outlier_threshold={"mean": 2.0, "brightness": ("zscore", 2.0)})

    Cluster-based detection with embeddings:

    >>> from dataeval.extractors import FlattenExtractor

    >>> outliers = Outliers(flags=ImageStats.NONE, extractor=FlattenExtractor(), cluster_threshold=2.0)
    >>> result = outliers.evaluate(train_ds)  # Only cluster_distance metric

    Using configuration:

    >>> config = Outliers.Config(outlier_threshold=2.5)
    >>> outliers = Outliers(config=config)
    """

    class Config(EvaluatorConfig, ClusterConfigMixin):
        """
        Configuration for Outliers detector.

        Attributes
        ----------
        flags : ImageStats, default ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
            Statistics to compute for image statistics-based outlier detection.
        outlier_threshold : ThresholdLike | Mapping[str, ThresholdLike] | None, default None
            Threshold configuration. When None, uses ``AdaptiveThreshold(3.5)``
            (Double-MAD with asymmetric bounds). See :class:`Outliers` for full description.
        extractor : FeatureExtractor or None, default None
            Feature extractor for cluster-based outlier detection.
        batch_size : int or None, default None
            Batch size for feature extraction during cluster-based detection. If None, uses DataEval
            default. Must be set by either parameter or global default if extractor is provided.
        cluster_algorithm : {"kmeans", "hdbscan"}, default "hdbscan"
            Clustering algorithm for cluster-based detection.
        cluster_threshold : ThresholdLike or None, default None
            Threshold configuration for cluster-based detection. When None,
            defaults to ``ZScoreThreshold(upper_multiplier=2.5)``.
        n_clusters : int or None, default None
            Expected number of clusters.
        """

        flags: ImageStats = DEFAULT_OUTLIERS_FLAGS
        outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None = None
        cluster_threshold: ThresholdLike | None = None

    stats: StatsResult
    flags: ImageStats
    outlier_threshold: Threshold | Mapping[str, Threshold]
    extractor: FeatureExtractor | None
    batch_size: int | None
    cluster_algorithm: Literal["kmeans", "hdbscan"]
    cluster_threshold: ThresholdLike | None
    n_clusters: int | None
    config: Config

    def __init__(
        self,
        flags: ImageStats | None = None,
        outlier_threshold: ThresholdLike | Mapping[str, ThresholdLike] | None = None,
        extractor: FeatureExtractor | None = None,
        batch_size: int | None = None,
        cluster_algorithm: Literal["kmeans", "hdbscan"] | None = None,
        cluster_threshold: ThresholdLike | None = None,
        n_clusters: int | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())
        self.outlier_threshold = _resolve_outlier_threshold(self.outlier_threshold)

    def _get_outliers(
        self,
        stats: StatsMap,
        source_index: Sequence[SourceIndex],
        strata: _Strata | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        frame, populations = _detect_outliers(stats, source_index, self.outlier_threshold, strata)
        return frame, _populations_frame(populations)

    @overload
    def from_stats(
        self,
        stats: StatsResult,
        *,
        per_image: bool = True,
        per_target: Literal[False] = ...,
    ) -> SingleOutliersOutput: ...

    @overload
    def from_stats(
        self,
        stats: StatsResult,
        *,
        per_image: bool = True,
        per_target: Literal[True],
    ) -> SingleTargetOutliersOutput: ...

    @overload
    def from_stats(
        self,
        stats: Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: Literal[False] = ...,
    ) -> MultiOutliersOutput: ...

    @overload
    def from_stats(
        self,
        stats: Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: Literal[True],
    ) -> MultiTargetOutliersOutput: ...

    @set_metadata(state=["outlier_threshold"])
    def from_stats(
        self,
        stats: StatsResult | Sequence[StatsResult],
        *,
        per_image: bool = True,
        per_target: bool = False,
    ) -> SingleOutliersOutput | SingleTargetOutliersOutput | MultiOutliersOutput | MultiTargetOutliersOutput:
        """
        Return indices of Outliers with the issues identified for each.

        Parameters
        ----------
        stats : StatsResult | Sequence[StatsResult]
            The output(s) from image_stats() with ImageStats.DIMENSION, PIXEL, or VISUAL flags
        per_image : bool, default True
            Whether to include item-level (image) outliers in the results.
        per_target : bool, default False
            Whether to include target-level outliers in the results.
            When True, the ``.outliers`` accessor uses :class:`SourceIndex` keys;
            when False, it uses plain ``int`` item indices.

            Both flags describe the two-way reading of an address that states no level —
            an item's own row, or one of its labels. A statistic addressed at a level
            *between* those, such as a video frame, is neither and is always included; the
            ``.outliers`` accessor then uses :class:`SourceIndex` keys whatever `per_target`
            says, since a plain item index cannot name such a row.
            Re-detection (`with_threshold`, `classwise`, `itemwise`) covers the same rows.

        Returns
        -------
        OutliersOutput
            Output class containing a DataFrame of outlier issues with columns:
            - item_index: int - Index of the outlier item
            - target_index: int | None - Index of the target within the item (None for item-level outliers)
            - metric_name: str - Name of the metric that flagged this item/target
            - metric_value: float - Value of the metric for this item/target

            For multiple datasets, a ``dataset_index`` column identifies the originating dataset
            and ``item_index`` values are local to each dataset.

        Example
        -------
        Evaluate the dataset using pre-computed stats:

        >>> from dataeval.core import image_stats
        >>> from dataeval.flags import ImageStats
        >>> from dataeval.utils.thresholds import ZScoreThreshold

        >>> stats = image_stats(images, stats=ImageStats.PIXEL)
        >>> outliers = Outliers(outlier_threshold=ZScoreThreshold(2.5))
        >>> results = outliers.from_stats(stats)
        >>> results.head(10).select("item_index", "metric_name", "metric_value")
        shape: (10, 3)
        ┌────────────┬─────────────┬──────────────┐
        │ item_index ┆ metric_name ┆ metric_value │
        │ ---        ┆ ---         ┆ ---          │
        │ i64        ┆ cat         ┆ f64          │
        ╞════════════╪═════════════╪══════════════╡
        │ 7          ┆ entropy     ┆ 0.0          │
        │ 7          ┆ mean        ┆ 0.98         │
        │ 7          ┆ std         ┆ 0.0          │
        │ 7          ┆ var         ┆ 0.0          │
        │ 8          ┆ skew        ┆ 0.062311     │
        │ 11         ┆ entropy     ┆ 0.0          │
        │ 11         ┆ mean        ┆ 0.98         │
        │ 11         ┆ std         ┆ 0.0          │
        │ 11         ┆ var         ┆ 0.0          │
        │ 18         ┆ entropy     ┆ 0.0          │
        └────────────┴─────────────┴──────────────┘
        """
        combined_stats, combined_source_index, dataset_steps = combine_stats_results(stats)

        # An address that states a level is selected by neither flag and is always kept: the flags
        # name the two ends of the two-way reading -- an item's own row and one of its labels -- and
        # a level between them is not either, so a caller who supplied one has already said which
        # rows they mean. The kinds that survive are remembered, so a re-detection reproduces them.
        tracking = any(si.kind in ("unit", "track") for si in combined_source_index)
        restricted: frozenset[FactorLevel | None] | None = None
        if not (per_image and per_target):
            keep = np.array(
                [selected_by_flags(si, per_image, per_target) for si in combined_source_index], dtype=np.bool_
            )
            restricted = frozenset(si.kind for si, kept in zip(combined_source_index, keep, strict=True) if kept)
            combined_stats, combined_source_index, _ = _restrict(combined_stats, combined_source_index, None, keep)

        outliers_df, populations = self._get_outliers(combined_stats, combined_source_index)
        outliers_df = drop_null_index_columns(outliers_df, _DROPPABLE)

        if dataset_steps:
            outliers_df = add_dataset_index(outliers_df, dataset_steps)

        return OutliersOutput(
            outliers_df,
            calculation_results=stats,
            outlier_threshold=self.outlier_threshold,
            restricted_to=restricted,
            tracking=tracking,
            populations=populations,
        )

    @set_metadata(state=["cluster_threshold", "cluster_algorithm", "n_clusters"])
    def from_clusters(
        self,
        embeddings: ArrayND[float],
        cluster_result: ClusterResult,
        cluster_threshold: ThresholdLike | None = None,
    ) -> SingleOutliersOutput:
        """
        Find outliers using cluster-based adaptive distance detection.

        Identifies outliers based on their distance from cluster centers in
        embedding space. Points that are unusually far from their nearest
        cluster center are flagged as outliers. This method is particularly
        effective for finding semantic or visual outliers in image embeddings.

        Parameters
        ----------
        embeddings : ArrayND[float]
            The embedding vectors used for clustering, shape (n_samples, n_features).
            Should be the same embeddings passed to the cluster() function.
        cluster_result : ClusterResult
            Clustering results from the cluster() function, containing cluster
            assignments and related metadata.
        cluster_threshold : ThresholdLike or None, default None
            Threshold configuration for cluster-based outlier detection.
            Accepts the same formats as ``outlier_threshold``. When None,
            uses the detector's configured ``cluster_threshold``.

        Returns
        -------
        SingleOutliersOutput
            Output containing a DataFrame of outlier issues. For each outlier,
            ``metric_value`` is the raw distance to its nearest cluster center, and the
            number of standard deviations it sits from its cluster's mean is
            ``(metric_value - population_mean) / population_std``.

        See Also
        --------
        :func:`~dataeval.core.cluster` : Function to compute clusters from embeddings
        :func:`~dataeval.core.compute_cluster_stats` : Computes statistics for adaptive detection
        :meth:`~dataeval.quality.Outliers.from_stats` : Find outliers from pre-computed image statistics
        :meth:`~dataeval.quality.Outliers.evaluate` : Find outliers by computing statistics from images

        Notes
        -----
        This method uses adaptive distance-based outlier detection that accounts
        for varying cluster densities. It significantly reduces false outliers
        compared to using HDBSCAN's binary -1 labels, especially for image
        embeddings with varying density distributions.

        The threshold parameter allows experimentation with different sensitivity
        levels and methods without recomputing clusters.
        """
        # Convert embeddings to numpy array and flatten if needed
        embeddings_array = flatten_samples(to_numpy(embeddings))

        # Compute cluster statistics
        cs = compute_cluster_stats(
            embeddings=embeddings_array,
            cluster_labels=cluster_result["clusters"],
        )

        ct = cluster_threshold if cluster_threshold is not None else self.cluster_threshold
        resolved_ct = resolve_threshold(ct) if ct is not None else DEFAULT_OUTLIERS_CLUSTER_THRESHOLD

        # Find outliers using adaptive method
        outlier_issues = self._find_outliers_adaptive(cluster_stats=cs, threshold=resolved_ct)

        return OutliersOutput(outlier_issues, cluster_stats=cs, cluster_threshold=ct)

    @staticmethod
    def _merge_outlier_dfs(outliers_dfs: Sequence[pl.DataFrame]) -> pl.DataFrame:  # noqa: C901
        """Merge a list of outlier DataFrames into one, normalizing columns."""
        if len(outliers_dfs) == 0:
            return pl.DataFrame(
                schema={
                    "item_index": pl.Int64,
                    "target_index": pl.Int64,
                    "metric_name": lexical_categorical(),
                    "metric_value": pl.Float64,
                    **_CONTEXT_COLUMNS,
                    **_EXTRA_COLUMNS,
                },
            )
        if len(outliers_dfs) == 1:
            return outliers_dfs[0]

        # Determine which optional columns are present in any DataFrame
        has_target_id = any("target_index" in df.columns for df in outliers_dfs)
        has_level = any("level" in df.columns for df in outliers_dfs)
        # Kept rather than selected away: every frame carries it once the caller has been
        # through add_dataset_index, and dropping it here turned a cross-dataset result
        # back into a single-dataset one — item indices with nothing saying whose they are.
        has_dataset_id = all("dataset_index" in df.columns for df in outliers_dfs)

        column_order = ["dataset_index"] if has_dataset_id else []
        column_order.append("item_index")
        if has_target_id:
            column_order.append("target_index")
        if has_level:
            column_order.append("level")
        column_order.extend(["metric_name", "metric_value"])
        # Carried the same way as `level`: a frame built by hand may not have them, and the
        # merged result should still hold the context every detector-built frame does.
        described = _CONTEXT_COLUMNS | _EXTRA_COLUMNS
        context = [name for name in described if any(name in df.columns for df in outliers_dfs)]
        column_order.extend(context)

        normalized_dfs: list[pl.DataFrame] = []
        for df in outliers_dfs:
            if has_target_id and "target_index" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("target_index"))
            if has_level and "level" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=lexical_categorical()).alias("level"))
            missing = [name for name in context if name not in df.columns]
            df = df.with_columns(pl.lit(None, dtype=described[name]).alias(name) for name in missing)
            normalized_dfs.append(df.select(column_order))
        # Sorted on the columns that survived, not on a fixed list: an item-level-only
        # result has no target_index to sort by, and naming it raised ColumnNotFoundError.
        sort_columns = [name for name in column_order if name not in _UNSORTED_COLUMNS]
        return pl.concat(normalized_dfs).sort(sort_columns)

    @staticmethod
    def _find_outliers_adaptive(
        cluster_stats: ClusterStats,
        threshold: Threshold,
    ) -> pl.DataFrame:
        """
        Find outliers using pre-calculated cluster statistics.

        This method uses pre-calculated cluster centers and distance statistics to identify
        outliers. A :class:`Threshold` is applied per-cluster to the distance distribution,
        and points whose distance exceeds the upper bound are flagged as outliers.

        Parameters
        ----------
        cluster_stats : dict
            Pre-calculated cluster centers, distance statistics, and nearest cluster indices.
            Should contain keys: 'distances', 'nearest_cluster_idx', 'cluster_distances_mean',
            'cluster_distances_std'.
        threshold : Threshold
            Threshold instance to compute per-cluster upper bounds from distance arrays.

        Returns
        -------
        pl.DataFrame
            DataFrame with outlier details containing columns:

            - item_index: int - Index of the outlier
            - metric_name: str - Always "cluster_distance"
            - metric_value: float - Distance to the nearest cluster center
            - direction, bound, percentile, population_mean, population_std - the frame of
              reference, taken from the point's own cluster

        Notes
        -----
        ``metric_value`` is the distance itself, as every other metric reports the value that
        was measured. The standard deviations it sits from its cluster's mean are
        ``(metric_value - population_mean) / population_std``.
        """
        distances = cluster_stats["distances"]
        nearest_cluster_idx = cluster_stats["nearest_cluster_idx"]
        count = len(distances)

        is_outlier = np.full(count, False, dtype=bool)
        bounds = np.full(count, np.nan, dtype=np.float64)
        percentiles = np.full(count, np.nan, dtype=np.float64)

        for cluster_id in np.unique(nearest_cluster_idx[nearest_cluster_idx >= 0]):
            mask = nearest_cluster_idx == cluster_id
            cluster_distances = distances[mask]
            _, upper = threshold(cluster_distances)
            percentiles[mask] = _percentile_of(cluster_distances, np.sort(cluster_distances))
            if upper is not None:
                is_outlier[mask] = cluster_distances > upper
                bounds[mask] = upper

        outlier_indices = np.nonzero(is_outlier)[0]
        schema: Any = {
            "item_index": pl.Int64,
            "metric_name": lexical_categorical(),
            "metric_value": pl.Float64,
            **_CONTEXT_COLUMNS,
        }
        if len(outlier_indices) == 0:
            return pl.DataFrame(schema=schema)

        clusters_of = nearest_cluster_idx[outlier_indices]
        means = np.asarray(cluster_stats["cluster_distances_mean"])[clusters_of]
        stds = np.asarray(cluster_stats["cluster_distances_std"])[clusters_of]
        # Cluster-based detection is always item-level, so we don't include target_index, and it
        # only ever flags points that are too far out, never too close in.
        return pl.DataFrame(
            {
                "item_index": pl.Series(outlier_indices.tolist(), dtype=pl.Int64),
                "metric_name": pl.Series(["cluster_distance"] * len(outlier_indices), dtype=lexical_categorical()),
                "metric_value": pl.Series(distances[outlier_indices].tolist(), dtype=pl.Float64),
                "direction": pl.Series(["upper"] * len(outlier_indices), dtype=lexical_categorical()),
                "bound": pl.Series(bounds[outlier_indices].tolist(), dtype=pl.Float64),
                "percentile": pl.Series(percentiles[outlier_indices].tolist(), dtype=pl.Float64),
                "population_mean": pl.Series(means.tolist(), dtype=pl.Float64),
                "population_std": pl.Series(stds.tolist(), dtype=pl.Float64),
            },
        ).sort(["item_index", "metric_name"], descending=[False, False])

    def _get_cluster_outliers(
        self,
        data: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    ) -> tuple[pl.DataFrame, ClusterStats]:
        """Extract embeddings, cluster them, and return cluster-based outlier DataFrame and stats."""
        embeddings = Embeddings(data, self.extractor, batch_size=self.batch_size)

        cluster_result = cluster(
            embeddings,
            algorithm=self.cluster_algorithm,
            n_clusters=self.n_clusters,
        )

        cs = compute_cluster_stats(
            embeddings=np.asarray(embeddings),
            cluster_labels=cluster_result["clusters"],
        )

        resolved_ct = (
            resolve_threshold(self.cluster_threshold)
            if self.cluster_threshold is not None
            else DEFAULT_OUTLIERS_CLUSTER_THRESHOLD
        )
        return self._find_outliers_adaptive(cluster_stats=cs, threshold=resolved_ct), cs

    # A MOT dataset yields a (stream, target, metadata) item whose first element is not an image array, so it satisfies
    # neither image spelling and a type checker would refuse it although `evaluate` judges it in video terms.
    _DatasetInput = Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]]
    _EvaluateInput = _DatasetInput | MultiobjectTrackingDataset

    @overload
    def evaluate(  # pyright: ignore[reportOverlappingOverload]
        self,
        data: _EvaluateInput,
        *other: _EvaluateInput,
        levels: str | Sequence[str],
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        factors: Sequence[str] | None = None,
        strata: str | Sequence[str] | None = None,
        combinations: Sequence[Sequence[str]] | Literal["all"] | None = None,
        given: Mapping[str, str] | None = None,
        kinds: Mapping[str, ColumnKind] | None = None,
        neighbors: Sequence[str] | Literal["all"] | None = None,
    ) -> OutliersOutput[Any]: ...

    @overload
    def evaluate(  # pyright: ignore[reportOverlappingOverload]
        self,
        data: MultiobjectTrackingDataset,
        *,
        levels: None = None,
        per_image: bool | None = None,
        per_target: bool | None = None,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        factors: Sequence[str] | None = None,
        strata: str | Sequence[str] | None = None,
        combinations: Sequence[Sequence[str]] | Literal["all"] | None = None,
        given: Mapping[str, str] | None = None,
        kinds: Mapping[str, ColumnKind] | None = None,
        neighbors: Sequence[str] | Literal["all"] | None = None,
    ) -> SingleTargetOutliersOutput: ...

    @overload
    def evaluate(  # pyright: ignore[reportOverlappingOverload]
        self,
        data: _EvaluateInput,
        *,
        levels: None = None,
        per_image: bool | None = None,
        per_target: Literal[False] | None = ...,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        factors: Sequence[str] | None = None,
        strata: str | Sequence[str] | None = None,
        combinations: Sequence[Sequence[str]] | Literal["all"] | None = None,
        given: Mapping[str, str] | None = None,
        kinds: Mapping[str, ColumnKind] | None = None,
        neighbors: Sequence[str] | Literal["all"] | None = None,
    ) -> SingleOutliersOutput: ...

    @overload
    def evaluate(  # pyright: ignore[reportOverlappingOverload]
        self,
        data: _EvaluateInput,
        *,
        levels: None = None,
        per_image: bool | None = None,
        per_target: Literal[True],
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        factors: Sequence[str] | None = None,
        strata: str | Sequence[str] | None = None,
        combinations: Sequence[Sequence[str]] | Literal["all"] | None = None,
        given: Mapping[str, str] | None = None,
        kinds: Mapping[str, ColumnKind] | None = None,
        neighbors: Sequence[str] | Literal["all"] | None = None,
    ) -> SingleTargetOutliersOutput: ...

    @overload
    def evaluate(
        self,
        data: _EvaluateInput,
        *other: _EvaluateInput,
        levels: None = None,
        per_image: bool | None = None,
        per_target: Literal[False] | None = ...,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        factors: Sequence[str] | None = None,
        strata: str | Sequence[str] | None = None,
        combinations: Sequence[Sequence[str]] | Literal["all"] | None = None,
        given: Mapping[str, str] | None = None,
        kinds: Mapping[str, ColumnKind] | None = None,
        neighbors: Sequence[str] | Literal["all"] | None = None,
    ) -> MultiOutliersOutput: ...

    @overload
    def evaluate(
        self,
        data: _EvaluateInput,
        *other: _EvaluateInput,
        levels: None = None,
        per_image: bool | None = None,
        per_target: Literal[True],
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        factors: Sequence[str] | None = None,
        strata: str | Sequence[str] | None = None,
        combinations: Sequence[Sequence[str]] | Literal["all"] | None = None,
        given: Mapping[str, str] | None = None,
        kinds: Mapping[str, ColumnKind] | None = None,
        neighbors: Sequence[str] | Literal["all"] | None = None,
    ) -> MultiTargetOutliersOutput: ...

    @set_metadata(
        state=[
            "flags",
            "outlier_threshold",
            "cluster_threshold",
            "cluster_algorithm",
            "n_clusters",
        ],
    )
    def evaluate(
        self,
        data: _EvaluateInput,
        *other: _EvaluateInput,
        levels: str | Sequence[str] | None = None,
        per_image: bool | None = None,
        per_target: bool | None = None,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        factors: Sequence[str] | None = None,
        strata: str | Sequence[str] | None = None,
        combinations: Sequence[Sequence[str]] | Literal["all"] | None = None,
        given: Mapping[str, str] | None = None,
        kinds: Mapping[str, ColumnKind] | None = None,
        neighbors: Sequence[str] | Literal["all"] | None = None,
    ) -> SingleOutliersOutput | SingleTargetOutliersOutput | MultiOutliersOutput | MultiTargetOutliersOutput:
        """
        Return indices of Outliers with the issues identified for each.

        Computes outliers using image statistics and/or cluster-based detection,
        depending on configuration. When both methods are enabled, results are
        combined into a single DataFrame. Supports single or multiple datasets.

        Parameters
        ----------
        data : Dataset
            The images to scan for outliers. Accepts any object satisfying the
            :class:`~dataeval.protocols.Dataset` interface (indexed access via
            ``__getitem__`` and ``__len__``). Each item may be:

            - a bare image array, i.e. ``Dataset[ArrayLike]``; or
            - a full MAITE ``(image, target, metadata)`` tuple, i.e.
              ``Dataset[tuple[ArrayLike, Any, Any]]``.

            A plain array or list of image arrays also structurally satisfies
            :class:`~dataeval.protocols.Dataset` and is accepted directly.

            A multi-object-tracking dataset is measured frame by frame, every frame, and judged in
            video terms: its videos, frames, tracks and detections (see `levels`). To measure fewer
            frames, pass a :class:`~dataeval.data.SequenceFrames` whose selector chooses them, such
            as ``SequenceFrames(dataset, Stride(5))``; it is judged in the same terms, and every
            unweighted statistic then describes the frames kept. Statistics are computed once per
            measured frame, plus once per detection with ``per_target=True`` or a track or
            detection level in `levels`. Each video's stream
            is read more than once per call; selecting fewer frames reduces the statistics
            computed, not the reads.

            `data` may instead be a :class:`~dataeval.Metadata` as a whole, whose ordered
            factors at the judged levels (see `levels`) are thresholded, each against the rows
            at the level it was measured at (a track against tracks, a frame against frames).
            Only raw values are read; bins never are.

            .. versionadded:: 1.2
        *other : Dataset
            Zero or more additional datasets for cross-dataset outlier
            detection, each accepting the same forms as ``data``. When provided,
            the output includes a ``dataset_index`` column identifying each
            item's originating dataset.
        levels : str or Sequence[str] or None, default None
            Which levels to judge. An image or detection dataset, or a :class:`~dataeval.Metadata` of
            one, accepts ``"item"`` and ``"target"``; a tracking dataset or a tracking metadata
            accepts ``"sequence"`` (a video), ``"unit"`` (a frame), ``"track"`` and ``"instance"``
            (a detection).

            This and ``per_image``/``per_target`` are two spellings of one thing, so passing both
            raises. Without either, a tracking dataset or metadata judges its sequence and frame
            levels, as :class:`~dataeval.quality.Duplicates` does, so judge track factors with
            ``levels="track"``. ``per_target=True`` adds every detection and track.

            On a tracking dataset, a video's value of a statistic is its median over the video's
            measured frames, named ``sequence_<statistic>``, and a track's is its median over the
            track's measured detections, named ``track_<statistic>``. A statistic measured at one
            level has its plain name, such as ``brightness``, unless the tracking metadata
            already holds that name (the frame size it records as ``width`` and ``height``, or a
            key of a video's datum metadata); it is then named ``<statistic>_added``, such as
            ``width_added`` and ``sequence_width_added``. When frames and detections are both
            measured, as with ``per_target=True`` or a track or detection level in `levels`, the
            names carry their level: ``unit_brightness``, ``instance_brightness`` and
            ``unit_width``. A statistic's name therefore depends on what was measured;
            :attr:`OutliersOutput.populations` lists every name judged in its ``metric_name``
            column, so filter on those. By
            default each level is judged against all of its rows in the dataset. A statistical
            threshold needs at least 20 rows, so with a statistical threshold the sequence level is
            judged only with 20 or more videos; with fewer, :attr:`OutliersOutput.populations`
            shows it with ``judged`` false and the reason ``"below 20 rows"``.

            Because frames are judged against every frame of the dataset, a video that is dark
            throughout, such as one shot at night, is a sequence row and also a row for each of
            its frames. To judge the videos against each other and each frame against its own
            video, make two calls: ``levels=["sequence", "track"]``, then ``levels="unit",
            strata="item_index"``. The second call names only the frame level because `strata`
            applies to every level judged: with a statistical threshold, a video's own stratum
            holds one sequence row, too few to judge.

            .. versionadded:: 1.2
        per_image : bool or None, default None
            Whether to compute statistics for full items (images/videos).
            When True, item-level outliers will be detected. None is True, unless `levels` is given.
        per_target : bool or None, default None
            Whether to compute statistics for individual targets/detections.
            When True, the ``.outliers`` accessor uses :class:`SourceIndex` keys;
            when False, it uses plain ``int`` item indices.
            Has no effect for datasets without targets or for cluster-based detection.
            None is False, unless `levels` is given. On a tracking dataset, True measures every
            detection of every measured frame and adds the track and detection levels.
        per_class : bool, default False
            Whether to compute outlier thresholds within each class separately,
            rather than globally across the entire dataset. When True, ``metadata``
            must be provided, except for a tracking dataset, which reads its classes from
            ``metadata`` when given and from the metadata it builds otherwise; there only
            detections have a class, so the other levels are judged as one population. Only
            applies to image statistics-based detection,
            not cluster-based detection.
        metadata : MetadataLike or None, default None
            Metadata object containing class labels. Required when ``per_class=True``, except
            for a tracking dataset, which reads its classes and strata from this when given and
            from the metadata it builds otherwise; this one is read, not changed, and must be the
            ``Metadata(dataset)`` of the dataset evaluated, with the same rows at every level.
            This is a secondary, per-class grouping input for image-statistics
            detection only; it is not an alternative to ``data`` and does not
            itself supply the images to scan.
        factors : Sequence[str] or None, default None
            Metadata factors to threshold, when `data` is a :class:`~dataeval.Metadata`. None
            thresholds every *ordered* factor -- numeric or temporal by its raw dtype. A
            categorical factor (strings, booleans, or integers with a declared vocabulary) has
            no location or scale, so it is judged by rarity within `strata` instead: every one
            at the levels the strata reach when `factors` is None, or the ones named. Without
            `strata`, naming one raises.
        strata : str or Sequence[str] or None, default None
            Categorical metadata factors to judge within, or the index columns ``item_index``,
            ``unit_index`` and ``track_id`` (a video, a frame or a track of one video). Each threshold is
            fitted within each stratum, and every flagged row names its stratum in a
            ``population`` column. Several names cross. A bare string is one name. See
            :meth:`OutliersOutput.stratified`, which this equals on the result of a call
            without it. On an image dataset, `metadata` must be a :class:`~dataeval.Metadata`
            that holds the strata; a tracking dataset reads them from `metadata` when given and
            from the metadata it builds otherwise. Cluster-based detection is not stratified.

            On a tracking dataset the strata apply to every level judged, and a stratum reaches
            its own level and the levels below it. ``strata="item_index"`` therefore makes each
            video a sequence population of one row and, with a statistical threshold, leaves it
            unjudged and judges a video's frames only when it has 20 or more measured frames. See
            `levels` for the two calls that judge videos together and frames within their video.

            On a :class:`~dataeval.Metadata`, the categorical factors (see `factors`) are judged
            by rarity within the strata, apart from the strata themselves; see
            :meth:`OutliersOutput.stratified`. A metadata of only categorical factors can be
            evaluated this way.

            .. versionadded:: 1.2
        combinations : Sequence[Sequence[str]] or "all" or None, default None
            Groups of metadata factors to judge together, when `data` is a :class:`~dataeval.Metadata`:
            a row is flagged when its combination of values is unusual even though each value is
            ordinary on its own, such as a large box near the horizon. Each group of two or more numeric
            factors is scored by its robust Mahalanobis distance, at the deepest level among its
            factors, with the factors of coarser levels carried down to its rows. The factors of a group
            must lie on one chain of levels. A group of categorical factors is judged by rarity, and
            needs `strata` that reach its level.

            ``"all"`` judges every pair of factors that live at each judged level, within `factors` when
            given, skipping levels with fewer than two and, where no stratum reaches a level, the
            categorical pairs there. Scoring every pair is slow on a large metadata, and the result keeps
            every score, so it is not cheap to keep or pickle.

            Each flagged row has ``method`` ``"mahalanobis"``, the group as ``metric_name`` (such as
            ``"rel_size, rel_offset_y"``), the distance as ``metric_value`` and ``direction``
            ``"upper"``: a small distance is not unusual. A group is judged in a population of at least
            20 rows, and 10 rows per dimension.

            .. versionadded:: 1.2
        given : Mapping[str, str] or None, default None
            ``{y: x}`` judges numeric factor ``y`` against the rows nearest to it in numeric factor ``x``,
            when `data` is a :class:`~dataeval.Metadata`: a bright frame at night is ordinary in
            brightness and in hour, and unusual as a brightness given its hour. The score is the signed
            robust z of ``y`` against its neighbors in ``x``, so both directions flag. Each row has
            ``method`` ``"residual"`` and ``metric_name`` ``"y given x"``. A population needs at least 31
            rows.

            The score checks of `combinations` and `given` take a scalar `outlier_threshold` only when it
            is statistical and has no ``lower_limit`` or ``upper_limit``, since a constant or a limit is
            in the units of the factors; otherwise they use the default. Name a check in a mapping, such
            as ``{"rel_size, rel_offset_y": 4.0}``, to set its threshold.

            .. versionadded:: 1.2
        kinds : Mapping[str, ColumnKind] or None, default None
            How to read a column in `combinations` and `given`, such as ``{"camera_id": "categorical"}``,
            as :func:`~dataeval.core.combination_scores` reads it. A column with a declared vocabulary
            is read as categorical without an entry here. Each level reads only the entries for its
            own columns.

            .. versionadded:: 1.2
        neighbors : Sequence[str] or "all" or None, default None
            Metadata factors to judge against their neighbors along a sequence, when `data` is a
            :class:`~dataeval.Metadata` of a tracking dataset: a frame's value against the frames before and after it in
            its video, a detection's against the detections before and after it in its track, ordered by
            ``unit_index``. A value is compared with the median of each side and with the line between them, so a
            scene cut, a fade or a slow drift is not flagged and a value that departs from all of them is; see
            :func:`~dataeval.core.sequence_residuals`. ``"all"`` judges every ordered factor at the judged ``"unit"``
            and ``"instance"`` levels, within `factors` when given.

            Each flagged row has ``method`` ``"neighbors"``, ``metric_name`` such as ``"brightness along sequence"``
            or ``"box_size along track"``, the raw value as ``metric_value``, and the value its neighbors put it at as
            ``population_mean``. A timestamp (a date, datetime or duration factor, or ``time_s`` or ``pts``) is also
            checked for order, and a row out of order is flagged with ``method`` ``"order"`` and the value it is out
            of order against as ``bound``. A timestamp held as plain numbers under another name is read as an
            ordinary factor, so cast it to a Datetime or Duration before adding it. The scores of every sequence are
            judged together, each divided by its sequence's spread, and a pool needs at least 20 scored rows.

            .. versionadded:: 1.2

        Returns
        -------
        SingleOutliersOutput or MultiOutliersOutput
            Output class containing a DataFrame of outlier issues with columns:

            - item_index: int - Index of the outlier item
            - target_index: int | None - Index of the target within the item
              (None for item-level outliers, omitted if all are item-level)
            - metric_name: str - Name of the metric that flagged this item/target.
              Includes "cluster_distance" when extractor is provided.
            - metric_value: float - Value of the metric for this item/target.
              For cluster_distance, this is the raw distance to the nearest cluster center; the
              number of standard deviations it sits from its cluster's mean is
              ``(metric_value - population_mean) / population_std``.

            For multi-dataset input, includes a ``dataset_index`` column.

            For a :class:`~dataeval.Metadata` input, a row judged at a level between the
            item and the label level -- a track, a video frame -- names that level in a
            ``level`` column, and its ``target_index`` holds the level's own key:
            ``track_id`` for a track, ``unit_index`` for a unit.

            For a tracking dataset, or a :class:`~dataeval.data.SequenceFrames` over one, rows are
            addressed in the source dataset. ``item_index`` is the video's position in it, and
            ``target_index`` holds the frame's own ``frame_index`` for a frame, the ``track_id``
            for a track, the detection's position among its video's detections for a detection,
            and null for a video. ``level`` is ``"unit"`` for a frame and ``"track"`` for a
            track, and null for a video and for a detection. The result holds the source dataset
            and the tracking metadata built for it, and :attr:`OutliersOutput.sequences`,
            :attr:`~OutliersOutput.frames`, :attr:`~OutliersOutput.tracks` and
            :attr:`~OutliersOutput.detections` select the rows of one level. A statistic whose name
            the tracking metadata already holds (the frame size it records as ``width`` and
            ``height``, or a key of a video's datum metadata) has the ``metric_name``
            ``<statistic>_added``; :attr:`OutliersOutput.populations` lists every name judged.

        Raises
        ------
        ValueError
            If ``flags`` is ``ImageStats.NONE`` and no ``extractor`` is provided.
            If both ``per_image`` and ``per_target`` are False.
            If ``per_class`` is True and ``metadata`` is None for an image dataset.
            If `levels` is given together with `per_image` or `per_target`, is empty, or names a
            level the data does not have; if `levels` names a tracking level for an image dataset.
            If `factors` is given without a Metadata as `data`, or is empty; if a named factor
            is ineligible, categorical without `strata`, or lives at a level `levels` or the
            flags do not judge; if a named categorical factor lives at a level `strata` do not
            reach;
            if the levels judged hold no ordered factor and no other check runs; if a Metadata is
            evaluated with an ``extractor``.
            If `strata` is given with ``per_class=True``, names a continuous factor, reaches
            none of the judged levels, or is given for an image dataset without a
            :class:`~dataeval.Metadata` as `metadata`.
            For a tracking dataset, or a :class:`~dataeval.data.SequenceFrames` over one: if an
            ``extractor`` is set, since cluster-based detection is not supported there; if
            `other` is given, or tracking and image datasets are mixed in one call; if ``flags``
            is ``ImageStats.NONE`` or measures no ordered statistic; if the frame view keeps no
            frames; if only detections are judged and the dataset has none; if `metadata` has no
            tracking levels or holds other row counts by level than the dataset; or if a video
            repeats a ``frame_index``.
            If `combinations`, `given` or `kinds` is given for a dataset; if a group has one column,
            mixes levels on different branches of the level schema, or is scored at a level the call
            does not judge; if a categorical group has no stratum reaching its level, or holds a
            column also named in `strata`; if
            ``combinations="all"`` finds no pair and `given` is empty; or if
            :func:`~dataeval.core.combination_scores` refuses a group or a `given` entry.
            If `neighbors` is given for a dataset or for a metadata with no sequence level, names a factor
            that holds no numbers or times, lives at a level with no order or that the call does not judge,
            or is ``"all"`` and finds no factor.
        KeyError
            If a named factor, stratum, or column of `combinations` or `given` is not one of the
            metadata's; if a key of `kinds` is not a column of `combinations` or `given`; if a name in
            `neighbors` is not a factor of the metadata.
        TypeError
            If `factors` is a single string rather than a list of names, or `combinations` (or one of
            its groups) is a string other than ``"all"``, or `neighbors` is a string other than ``"all"``.

        Examples
        --------
        Basic outlier detection:

        >>> outliers = Outliers(outlier_threshold=2.5)
        >>> results = outliers.evaluate(images)
        >>> results.head(6).select("item_index", "metric_name", "metric_value")
        shape: (6, 3)
        ┌────────────┬─────────────┬──────────────┐
        │ item_index ┆ metric_name ┆ metric_value │
        │ ---        ┆ ---         ┆ ---          │
        │ i64        ┆ cat         ┆ f64          │
        ╞════════════╪═════════════╪══════════════╡
        │ 0          ┆ zeros       ┆ 0.000081     │
        │ 2          ┆ zeros       ┆ 0.000081     │
        │ 7          ┆ brightness  ┆ 249.900009   │
        │ 7          ┆ contrast    ┆ 0.0          │
        │ 7          ┆ darkness    ┆ 249.900009   │
        │ 7          ┆ entropy     ┆ 0.0          │
        └────────────┴─────────────┴──────────────┘

        Every row also carries the frame of reference it was judged against:

        >>> results.head(4).select("metric_name", "direction", "bound", "percentile")
        shape: (4, 4)
        ┌─────────────┬───────────┬────────────┬────────────┐
        │ metric_name ┆ direction ┆ bound      ┆ percentile │
        │ ---         ┆ ---       ┆ ---        ┆ ---        │
        │ cat         ┆ cat       ┆ f64        ┆ f64        │
        ╞═════════════╪═══════════╪════════════╪════════════╡
        │ zeros       ┆ upper     ┆ 0.000048   ┆ 92.0       │
        │ zeros       ┆ upper     ┆ 0.000048   ┆ 92.0       │
        │ brightness  ┆ upper     ┆ 114.368277 ┆ 96.0       │
        │ contrast    ┆ lower     ┆ 0.886324   ┆ 4.0        │
        └─────────────┴───────────┴────────────┴────────────┘

        Evaluate two or more datasets (cross-dataset detection):

        >>> outliers = Outliers()
        >>> results = outliers.evaluate(train_ds, test_ds)
        >>> results = outliers.evaluate(train_ds_area1, train_ds_area2, train_ds_area3, test_ds)  # or more

        Threshold metadata factors, each against the rows at its own level. A statistical
        threshold needs at least 20 rows at a level:

        >>> from dataeval import Metadata
        >>> readings = [20.1, 19.8, 20.3, 20.0, 19.9, 20.2, 20.1, 19.7, 20.0, 20.2, 19.9, 20.1]
        >>> readings += [19.8, 20.0, 20.3, 19.9, 20.1, 20.0, 19.8, 20.2, 20.1, 19.9, 20.0, 45.0]
        >>> md = Metadata.from_factors({"temperature": readings})
        >>> Outliers().evaluate(md).data().select("item_index", "metric_name", "metric_value")
        shape: (1, 3)
        ┌────────────┬─────────────┬──────────────┐
        │ item_index ┆ metric_name ┆ metric_value │
        │ ---        ┆ ---         ┆ ---          │
        │ i64        ┆ cat         ┆ f64          │
        ╞════════════╪═════════════╪══════════════╡
        │ 23         ┆ temperature ┆ 45.0         │
        └────────────┴─────────────┴──────────────┘

        Judge one factor given another. Frame 299 is as bright as noon at 02:00, while each of its
        brightness and its hour is ordinary on its own:

        >>> rng = np.random.default_rng(0)
        >>> hour = rng.integers(0, 24, 300).astype(float)
        >>> brightness = 0.5 + 0.4 * np.cos(2 * np.pi * (hour - 12) / 24) + rng.normal(0, 0.03, 300)
        >>> hour[-1], brightness[-1] = 2.0, 0.9
        >>> md = Metadata.from_factors({"brightness": brightness, "hour": hour})
        >>> result = Outliers().evaluate(md, given={"brightness": "hour"})
        >>> result.data().select("item_index", "metric_name", "metric_value", "direction")
        shape: (1, 4)
        ┌────────────┬───────────────────────┬──────────────┬───────────┐
        │ item_index ┆ metric_name           ┆ metric_value ┆ direction │
        │ ---        ┆ ---                   ┆ ---          ┆ ---       │
        │ i64        ┆ cat                   ┆ f64          ┆ cat       │
        ╞════════════╪═══════════════════════╪══════════════╪═══════════╡
        │ 299        ┆ brightness given hour ┆ 9.017146     ┆ upper     │
        └────────────┴───────────────────────┴──────────────┴───────────┘

        Judge a frame against the frames before and after it in its video. Frame 30 of video 2 is ordinary among
        every frame and far brighter than its neighbors:

        >>> result = Outliers().evaluate(video_metadata, factors=["brightness"], neighbors=["brightness"])
        >>> result.data().select("item_index", "target_index", "metric_name", "metric_value", "method")
        shape: (1, 5)
        ┌────────────┬──────────────┬───────────────────────────┬──────────────┬───────────┐
        │ item_index ┆ target_index ┆ metric_name               ┆ metric_value ┆ method    │
        │ ---        ┆ ---          ┆ ---                       ┆ ---          ┆ ---       │
        │ i64        ┆ i64          ┆ cat                       ┆ f64          ┆ cat       │
        ╞════════════╪══════════════╪═══════════════════════════╪══════════════╪═══════════╡
        │ 2          ┆ 30           ┆ brightness along sequence ┆ 162.805554   ┆ neighbors │
        └────────────┴──────────────┴───────────────────────────┴──────────────┴───────────┘

        Judge a tracking dataset in video terms. In 24 videos of 30 frames, the last of which is
        dark throughout, that video is one row per flagged statistic at the sequence level:

        >>> result = Outliers().evaluate(tracking_dataset)  # doctest: +SKIP
        >>> result.sequences.data().select("item_index", "metric_name", "metric_value")  # doctest: +SKIP
        shape: (3, 3)
        ┌────────────┬─────────────────────┬──────────────┐
        │ item_index ┆ metric_name         ┆ metric_value │
        │ ---        ┆ ---                 ┆ ---          │
        │ i64        ┆ cat                 ┆ f64          │
        ╞════════════╪═════════════════════╪══════════════╡
        │ 23         ┆ sequence_brightness ┆ 40.5         │
        │ 23         ┆ sequence_darkness   ┆ 40.5         │
        │ 23         ┆ sequence_mean       ┆ 40.5         │
        └────────────┴─────────────────────┴──────────────┘

        Each of its frames is flagged too, against every frame of the dataset. Judged against
        the frames of its own video instead, none is:

        >>> frames = Outliers().evaluate(tracking_dataset, levels="unit", strata="item_index")  # doctest: +SKIP
        >>> len(frames)  # doctest: +SKIP
        0
        """
        # Every input, not just the first: a filtered metadata among ``other`` misaligns
        # the same way, and is the one an evaluator handed several datasets is likeliest
        # to reach for.
        for candidate in (metadata, data, *other):
            reject_filtered_metadata(candidate, "Outliers")
        strata = _strata_names(strata, per_class)
        checks = _Requested(combinations, given, kinds, neighbors)

        if is_metadata(data):
            return self._evaluate_metadata(
                data,
                other,
                levels=levels,
                per_image=per_image,
                per_target=per_target,
                per_class=per_class,
                metadata=metadata,
                factors=factors,
                strata=strata,
                checks=checks,
            )
        if any(yields_frames(dataset) for dataset in (data, *other)):
            _refuse_video_arguments(data, other, self.extractor, factors, checks)
            return self._evaluate_video(
                data,
                levels=levels,
                per_image=per_image,
                per_target=per_target,
                per_class=per_class,
                metadata=metadata,
                strata=strata,
            )
        _refuse_factor_arguments(factors, checks)

        plan = _plan_for(levels, per_image, per_target, tracking=False, subject="dataset")
        # The video branch above took every tracking dataset, so what remains is an image dataset.
        images = cast("list[Outliers._DatasetInput]", [data, *other])
        if other:
            return self._evaluate_multi(
                images,
                per_image=plan.per_image,
                per_target=plan.per_target,
                per_class=per_class,
                metadata=metadata,
                strata=strata,
            )

        return self._evaluate_single(
            images[0],
            per_image=plan.per_image,
            per_target=plan.per_target,
            per_class=per_class,
            metadata=metadata,
            strata=strata,
        )

    def _validate_evaluate_inputs(
        self,
        per_image: bool,
        per_target: bool,
        per_class: bool,
        metadata: MetadataLike | None,
        strata: Sequence[str] | None,
    ) -> None:
        """Validate common inputs for single and multi-dataset evaluation."""
        if self.flags == ImageStats.NONE and self.extractor is None:
            raise ValueError("Either flags must not be ImageStats.NONE or extractor must be provided.")
        if not (per_image or per_target):
            raise ValueError("At least one of per_image or per_target must be True.")
        if per_class and metadata is None:
            raise ValueError("metadata must be provided when per_class=True.")
        if strata is not None and not has_level_schema(metadata):
            raise ValueError(
                "Outliers.evaluate: strata= on a dataset reads its factors from a Metadata with a level schema; "
                "pass metadata=Metadata(dataset)."
            )

    @staticmethod
    def _dataset_strata(
        source_index: Sequence[SourceIndex],
        per_class: bool,
        metadata: MetadataLike | None,
        strata: Sequence[str] | None,
    ) -> _Strata | None:
        """Stratify a dataset's statistics rows by class or by the strata named, or not at all."""
        if strata is not None:
            return strata_codes(metadata, strata, *row_positions(metadata, source_index))
        if per_class and metadata is not None:
            return _class_strata(source_index, metadata)
        return None

    def _evaluate_metadata(
        self,
        data: Any,
        other: Sequence[Any],
        *,
        levels: str | Sequence[str] | None,
        per_image: bool | None,
        per_target: bool | None,
        per_class: bool,
        metadata: MetadataLike | None,
        factors: Sequence[str] | None,
        strata: Sequence[str] | None,
        checks: "_Requested",
        built: bool = False,
    ) -> OutliersOutput[Any]:
        """Threshold a metadata's ordered factors, each against the rows at its own level.

        The factors are laid out as a stats result and handed to :meth:`from_stats`, so the
        per-level fitting and every re-detection method behave exactly as they do for image
        statistics. Which levels are judged is decided by the plan `levels=` or the flags resolve to.
        """
        labels = checked_sole_metadata(data, metadata, "Outliers")
        _check_metadata_evaluate_preconditions(data, other, self.extractor is not None and not built)
        tracking = "sequence" in data.levels
        _refuse_no_flag(levels, per_image, per_target)
        plan = _plan_for(levels, per_image, per_target, tracking, "metadata")
        _refuse_lone_target(plan, data, tracking)
        names = _level_names(data, tracking)
        categorical = _categorical_factors(data, factors, strata, plan, names)
        rarity = _rarity_runs(data, categorical, strata, named=factors is not None)
        groups, given = _combination_checks(data, checks, factors, strata, plan, names)
        judged = _judged_levels(data, plan, names)
        neighbors = neighbor_checks(data, checks.neighbors, factors, judged)
        # A named categorical factor needs rarity; with factors=None any other check may run alone.
        scores = bool(groups or given or neighbors)
        chosen = _factors_at_planned_levels(
            eligible_factors(data, factors, others=rarity or (scores and factors is None)),
            plan,
            names,
            named=factors is not None,
            others=rarity or scores,
        )
        result = self.from_stats(factor_stats(data, chosen), per_image=True, per_target=True)
        result._attach(metadata=data)
        result.tracking = tracking
        result.score_input = read_score_input(data, categorical, groups, given, checks.kinds, neighbors)
        _attach_neighbors(result, neighbors)
        if strata is not None:
            return result.stratified(data, *strata)
        if per_class:
            return result.classwise(labels)
        return result._redetect(scored=result._rescored({})) if scores else result

    def _evaluate_annotated(
        self,
        data: Any,
        built: Metadata,
        *,
        per_image: bool,
        per_target: bool,
        per_class: bool,
        metadata: MetadataLike | None,
        strata: Sequence[str] | None,
    ) -> OutliersOutput[Any]:
        """Judge a dataset through the metadata a user would have built for it, and attach that metadata.

        Equivalent to ``Metadata(data)`` with ``add_factors`` of the dataset's statistics, then
        :meth:`evaluate` of that metadata, so a dataset can never be judged in a way the metadata could not.
        `metadata` supplies the classes and strata, as it does for any dataset.
        """
        self.stats = checked_compute_stats(
            [data],
            stats=self.flags,
            caller=type(self).__name__,
            per_image=per_image,
            per_target=per_target,
            normalize_pixel_values=False,
        )[0]
        added = _add_measured_factors(built, self.stats)
        result = self._evaluate_metadata(
            built,
            (),
            levels=None,
            per_image=per_image,
            per_target=per_target,
            per_class=False,
            metadata=None,
            factors=added,
            strata=None,
            checks=_Requested(None, None, None, None),
            built=True,
        )
        if strata is not None:
            result = result.stratified(cast(Metadata, metadata), *strata)
        elif per_class:
            result = result.classwise(cast(MetadataLike, metadata))
        if self.extractor is not None:
            result.cluster_stats = self._get_cluster_outliers(data)[1]
            result.cluster_threshold = self.cluster_threshold
            result = result._redetect()
        return result._attach(data, built)

    def _evaluate_video(
        self,
        data: Any,
        *,
        levels: str | Sequence[str] | None,
        per_image: bool | None,
        per_target: bool | None,
        per_class: bool,
        metadata: MetadataLike | None,
        strata: Sequence[str] | None,
    ) -> OutliersOutput[Any]:
        """Judge a tracking dataset in video terms, through the tracking metadata of its source, and attach both.

        Equivalent to ``Metadata(dataset)`` with ``add_factors`` of the frame view's statistics, placed on the source
        rows by :meth:`~dataeval.data.SequenceFrames.to_source`, then :meth:`evaluate` of that metadata. `metadata`
        supplies the classes and strata when given, and the metadata built here does otherwise.
        """
        _refuse_no_flag(levels, per_image, per_target)
        plan = _plan_for(levels, per_image, per_target, tracking=True, subject="dataset")
        if self.flags == ImageStats.NONE:
            raise ValueError("Outliers.evaluate: flags must not be ImageStats.NONE for a tracking dataset.")
        frames = _video_frames(data)
        self.stats = checked_compute_stats(
            [frames],
            stats=self.flags,
            caller=type(self).__name__,
            per_image=plan.per_image,
            per_target=plan.per_target,
            normalize_pixel_values=False,
        )[0]
        built = Metadata(frames.source)
        added = _add_measured_factors(built, self.stats, frames.to_source(self.stats["source_index"]))
        built, rolled = _roll_up(built, added, plan)
        # `levels` measures frames even when it judges none (only per_image=False skips them), and a track level
        # measures its detections, so a statistic can live at a level this call does not judge.
        at = factor_levels(built)
        measured: dict[str, FactorLevel] = {name: at[name] for name in [*added, *rolled]}
        judged = [name for name, level in measured.items() if level in plan.levels]
        if not judged:
            _refuse_unjudged(measured, plan, len(self.stats["source_index"]))
        result = self._evaluate_metadata(
            built,
            (),
            levels=levels,
            per_image=per_image,
            per_target=per_target,
            per_class=False,
            metadata=None,
            factors=judged,
            strata=None,
            checks=_Requested(None, None, None, None),
            built=True,
        )
        _refuse_foreign_metadata(metadata, built)
        labels = built if metadata is None else metadata
        if strata is not None:
            result = result.stratified(cast(Metadata, labels), *strata)
        elif per_class:
            result = result.classwise(labels)
        return result._attach(frames.source, built)

    def _evaluate_single(
        self,
        data: _DatasetInput,
        *,
        per_image: bool = True,
        per_target: bool = True,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        strata: Sequence[str] | None = None,
    ) -> SingleOutliersOutput:
        """Single-dataset evaluate implementation."""
        self._validate_evaluate_inputs(per_image, per_target, per_class, metadata, strata)

        if self.flags != ImageStats.NONE and (built := _annotated_metadata(data)) is not None:
            return self._evaluate_annotated(
                data,
                built,
                per_image=per_image,
                per_target=per_target,
                per_class=per_class,
                metadata=metadata,
                strata=strata,
            )

        outliers_dfs: list[pl.DataFrame] = []
        stats_result: StatsResult | None = None
        stored_cluster_stats: ClusterStats | None = None
        row_strata: _Strata | None = None
        populations: pl.DataFrame | None = None

        if self.flags != ImageStats.NONE:
            self.stats = checked_compute_stats(
                [data],
                stats=self.flags,
                caller=type(self).__name__,
                per_image=per_image,
                per_target=per_target,
                normalize_pixel_values=False,
            )[0]
            stats_result = self.stats
            source_index = self.stats["source_index"]
            row_strata = self._dataset_strata(source_index, per_class, metadata, strata)
            stats_df, populations = self._get_outliers(self.stats["stats"], source_index, row_strata)
            outliers_dfs.append(stats_df)

        if self.extractor is not None:
            cluster_df, stored_cluster_stats = self._get_cluster_outliers(data)
            outliers_dfs.append(cluster_df)

        return OutliersOutput(  # type: ignore[return-value]
            drop_null_index_columns(self._merge_outlier_dfs(outliers_dfs), _DROPPABLE),
            calculation_results=stats_result,
            outlier_threshold=self.outlier_threshold,
            cluster_stats=stored_cluster_stats,
            cluster_threshold=self.cluster_threshold,
            strata=row_strata,
            populations=populations,
        )._attach(data)

    def _evaluate_multi(
        self,
        datasets: Sequence[_DatasetInput],
        *,
        per_image: bool = True,
        per_target: bool = True,
        per_class: bool = False,
        metadata: MetadataLike | None = None,
        strata: Sequence[str] | None = None,
    ) -> MultiOutliersOutput:
        """Multi-dataset evaluate: compute stats per dataset, then combine."""
        self._validate_evaluate_inputs(per_image, per_target, per_class, metadata, strata)

        # Compute dataset_steps from dataset lengths (needed for index remapping)
        dataset_steps: list[int] = []
        cumulative = 0
        for ds in datasets:
            cumulative += len(ds)
            dataset_steps.append(cumulative)

        # Stats-based: compute stats per dataset, then combine
        stats_results: list[StatsResult] = []
        if self.flags != ImageStats.NONE:
            stats_results = checked_compute_stats(
                datasets,
                stats=self.flags,
                caller=type(self).__name__,
                per_image=per_image,
                per_target=per_target,
                normalize_pixel_values=False,
            )
            self.stats = stats_results[-1]

        outliers_dfs: list[pl.DataFrame] = []
        row_strata: _Strata | None = None
        populations: pl.DataFrame | None = None

        if stats_results:
            # combined_steps count statistics rows, target rows included; dataset_steps count images, for cluster rows.
            combined_stats, combined_source_index, combined_steps = combine_stats_results(stats_results)
            row_strata = self._dataset_strata(combined_source_index, per_class, metadata, strata)

            # Populations are pooled across the datasets, as the thresholds are.
            outliers_df, populations = self._get_outliers(combined_stats, combined_source_index, row_strata)
            outliers_df = drop_null_index_columns(outliers_df, _DROPPABLE)
            outliers_df = add_dataset_index(outliers_df, combined_steps)
            outliers_dfs.append(outliers_df)

        # Cluster-based: early fusion — combine all images, cluster globally
        stored_cluster_stats: ClusterStats | None = None
        if self.extractor is not None:
            all_images = [img for ds in datasets for img in iter_images(ds)]
            embeddings = Embeddings(all_images, self.extractor, batch_size=self.batch_size)

            cluster_result = cluster(
                embeddings,
                algorithm=self.cluster_algorithm,
                n_clusters=self.n_clusters,
            )

            stored_cluster_stats = compute_cluster_stats(
                embeddings=np.asarray(embeddings),
                cluster_labels=cluster_result["clusters"],
            )

            resolved_ct = (
                resolve_threshold(self.cluster_threshold)
                if self.cluster_threshold is not None
                else DEFAULT_OUTLIERS_CLUSTER_THRESHOLD
            )
            cluster_df = self._find_outliers_adaptive(stored_cluster_stats, resolved_ct)
            cluster_df = add_dataset_index(cluster_df, dataset_steps)
            outliers_dfs.append(cluster_df)

        return OutliersOutput(  # type: ignore[return-value]
            drop_null_index_columns(self._merge_outlier_dfs(outliers_dfs), _DROPPABLE),
            calculation_results=stats_results if stats_results else None,
            outlier_threshold=self.outlier_threshold,
            cluster_stats=stored_cluster_stats,
            cluster_threshold=self.cluster_threshold,
            dataset_steps=dataset_steps,
            strata=row_strata,
            populations=populations,
        )

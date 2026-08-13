__all__ = ["Metadata"]

import copy
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from typing import Any, Literal, NoReturn, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval._log import get_logger
from dataeval._structurers import (
    FactorsStructurer,
    RowLayout,
    SourceIndexRows,
    StructuredData,
    Structurer,
    TaskOverride,
    safe_column_name,
    select_structurer,
)
from dataeval.core._bin import bin_data, digitize_data, is_continuous
from dataeval.core._compute_stats import StatsResult
from dataeval.exceptions import NotFittedError, ShapeMismatchError
from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    DatumMetadata,
    FeatureExtractor,
    ProgressCallback,
)
from dataeval.types import Array1D, FactorInfo, FactorLevel, FactorLevelSchema, SourceIndex
from dataeval.utils._internal import as_numpy
from dataeval.utils._validate import requires_maite_dataset

_logger = get_logger(__name__)


def _is_stats_result(candidate: Any) -> bool:
    """Whether a mapping is a :class:`~dataeval.core.StatsResult` rather than a factor mapping.

    ``StatsResult`` is a :class:`~typing.TypedDict`, so at runtime it is an ordinary dict
    and there is nothing to check with ``isinstance``. Both keys are required and both are
    checked structurally: a caller's factor mapping could plausibly hold a factor named
    ``stats`` or ``source_index``, but not one whose ``stats`` is itself a mapping *and*
    whose ``source_index`` is a sequence of :class:`~dataeval.types.SourceIndex`. Being
    strict here matters more than being permissive — a false positive silently discards
    every factor the caller passed.

    The first entry stands for the sequence. A stats result's index is homogeneous by
    construction, and this runs on every ``add_factors`` call, including the common one
    where the argument really is a factor mapping holding one entry per detection.
    """
    if not isinstance(candidate, Mapping) or not candidate.keys() >= {"stats", "source_index"}:
        return False
    source_index = candidate["source_index"]
    return (
        isinstance(candidate["stats"], Mapping)
        and isinstance(source_index, Sequence)
        and (not source_index or isinstance(source_index[0], SourceIndex))
    )


def _unpack_stats_result(
    factors: Any,
    source_index: Sequence[SourceIndex] | None,
    *,
    level: Any = None,
) -> tuple[Mapping[str, Array1D[Any]], Sequence[SourceIndex] | None]:
    """Accept a whole stats result wherever a factor mapping is accepted.

    :func:`~dataeval.core.compute_stats` and :func:`~dataeval.core.compute_ratios` return
    the statistics and the labels that place them in one object, and separating them again
    at every call site is busywork that also invites passing one without the other. When
    the result is recognised, its ``stats`` become the factors and its ``source_index``
    the placement — unless the caller passed an explicit one, which wins so that a
    hand-corrected index remains usable.

    The bookkeeping keys — ``object_count``, ``invalid_box_count``, ``image_count`` —
    describe the run rather than the images and are not factors, so they are dropped.

    Raises
    ------
    ValueError
        When a level is named as well. The result already says what each value describes,
        and honouring one of the two silently would discard a real contradiction.
    """
    if not _is_stats_result(factors):
        return factors, source_index
    if level is not None and level != "auto":
        raise ValueError(
            f"`level` and the source_index carried by this stats result are mutually exclusive; "
            f"the result already labels each value with what it describes. Pass the result's "
            f"['stats'] mapping instead to place its values at level={level!r}.",
        )
    return factors["stats"], source_index if source_index is not None else factors["source_index"]


def _reject_length_mismatch(factors: Mapping[str, Any], source_index: Sequence[SourceIndex]) -> None:
    """Reject factors that do not hold exactly one value per source-index entry.

    Shared by both constructors: the source index is the placement, so a factor that is
    not as long as it names rows the caller never described, whichever spelling was used
    to get here.
    """
    mismatched = {name: len(values) for name, values in factors.items() if len(values) != len(source_index)}
    if mismatched:
        raise ShapeMismatchError(
            f"All factors must have one value per source_index entry ({len(source_index)}); got {mismatched}.",
        )


def _flatten_column_vector(values: NDArray[Any]) -> NDArray[Any]:
    """Flatten an ``(N, 1)`` column of single values to ``(N,)``, leaving any other shape alone."""
    return values.reshape(-1) if values.ndim == 2 and values.shape[1] == 1 else values


def _holds_no_values(values: NDArray[Any]) -> bool:
    """Whether an array of a factor's values at one level holds no value at all.

    An empty array is not "no values" in this sense: a level that has no rows holds
    nothing for every factor alike, which says something about the dataset rather than
    about this factor, and ``level="combined"`` promises its column names regardless.
    """
    if values.size == 0:
        return False
    if values.dtype.kind in "fc":
        return bool(np.isnan(values).all())
    if values.dtype.kind == "O":
        return all(value is None or (isinstance(value, float) and np.isnan(value)) for value in values)
    # Integer, boolean and fixed-width string arrays have no null to be made of.
    return False


def _drop_vacuous_splits(
    columns: list[tuple[str, FactorLevel, NDArray[Any]]],
) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
    """Split one factor's level columns into those carrying values and those carrying none.

    A statistic that only applies to some of the levels a source index spans still has
    to be as long as the index to be placed at all, so it arrives padded with nulls at
    the levels it does not describe — ``compute_stats(per_background=True)`` measures the
    scene behind an image's annotations, which is a property of the image and of nothing
    inside it, and so has nulls on every instance row. Splitting that array by level
    yields one real column and one holding nothing. The empty one is not a factor: it
    cannot be binned, and left in place it reaches every evaluator that reads
    :attr:`~dataeval.Metadata.factor_names` — :class:`~dataeval.bias.Balance` reports it
    as a row of zero mutual information rather than omitting it.

    Only ever separates out a *part* of a split. A factor whose every level came back
    empty is kept whole, so that a factor a caller passed in can never vanish entirely —
    an all-null factor is then theirs to see in the frame rather than ours to hide.

    Pure, and returns the discarded names rather than recording them, so that
    :meth:`Metadata.add_factors` can still abandon the whole call on a later validation
    failure without having already written to :attr:`~dataeval.Metadata.dropped_factors`.
    """
    kept = [column for column in columns if not _holds_no_values(column[2])]
    if not kept:
        return columns, []
    return kept, [name for name, _, _ in columns if not any(name == kept_name for kept_name, _, _ in kept)]


def _binned(name: str) -> str:
    return f"{name}↕"


def _digitized(name: str) -> str:
    return f"{name}#"


def _to_col(name: str, info: FactorInfo, binned: bool = True) -> str:
    if binned and info.is_binned:
        return _binned(name)
    if info.is_digitized:
        return _digitized(name)
    return name


def _float_col(name: str, info: FactorInfo, schema: Mapping[str, pl.DataType]) -> str:
    """Column holding a float-castable representation of a factor.

    Raw values wherever they are numeric, so continuous factors keep their real
    values. A categorical factor's raw column holds strings, which have no float
    form at all, so it resolves to its digitized companion instead.
    """
    dtype = schema.get(name)
    return name if dtype is not None and dtype.is_numeric() else _to_col(name, info, binned=False)


def _build_index2label(
    provided: Mapping[int, str] | None,
    observed_labels: Iterable[Any],
) -> dict[int, str]:
    """Map each class index to a name, backfilling observed labels missing from ``provided``.

    When ``provided`` is given it is the source of truth; any observed label without an
    entry gets an ``UNDEFINED_CLASS_<i>`` placeholder. Otherwise labels name themselves.
    """
    if provided is not None:
        index2label = {int(k): str(v) for k, v in provided.items()}
        for lbl in observed_labels:
            index2label.setdefault(int(lbl), f"UNDEFINED_CLASS_{int(lbl)}")
        return index2label
    return {int(lbl): str(int(lbl)) for lbl in observed_labels}


def _resolve_legacy_level(level: str, aliases: Mapping[str, FactorLevel], stacklevel: int, unit_type: str) -> str:
    """Translate a retired level spelling, warning at the caller's line.

    Shared by the two paths that can be handed one: :meth:`Metadata._resolve_level`,
    which has a structurer and so knows the task's full alias map, and
    :meth:`Metadata._load_factors`, which is choosing the level a
    :class:`FactorsStructurer` will be built with and so has only the base map. One
    function so the two cannot word the deprecation differently.
    """
    alias = aliases.get(level)
    if alias is None:
        return level
    # Naive "+ s" pluralization: correct for every current unit_type ("image",
    # "frame", "item"). A future unit_type needing an irregular plural should be
    # given one at its declaration site rather than inflected here.
    warnings.warn(
        f"Level {level!r} is deprecated and will stop resolving in a future "
        f"release. It is no longer a level name; pass {alias!r} instead "
        f"(this dataset's units are {unit_type}s).",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return alias


class Metadata(Array, FeatureExtractor):
    """Collection of binned metadata using Polars DataFrames.

    Processes dataset metadata by automatically binning continuous factors and digitizing
    categorical factors for analysis and visualization workflows.

    This class also implements the :class:`~dataeval.protocols.FeatureExtractor` protocol,
    allowing it to be used directly with drift detectors that accept feature extractors.

    Rows are organized by *level* — a granularity at which one row means one entity.
    The dataframe holds rows at every level at once, each tagged by a ``level`` column.
    Which levels exist depends on the task, which is detected from the
    ``(item, target)`` types of the bound dataset:

    ======================================  ==========  ==============  ================================  =========
    Task                                    Item level  Instance level  Levels                            unit_type
    ======================================  ==========  ==============  ================================  =========
    ``IC``  image classification            unit        instance        unit, instance                    image
    ``OD``  object detection (image)        unit        instance        unit, instance                    image
    ``MOT`` multi-object tracking (video)   sequence    instance        sequence, unit, track, instance   frame
    ======================================  ==========  ==============  ================================  =========

    An *instance* is one labelled thing inside an item: a detection for object detection
    or multi-object tracking, the image itself for classification. The label level is
    always distinct from the item level, so an item carrying no label — an unlabeled
    image, or one with no detections — still has an item row and keeps every factor on
    it. :attr:`levels` enumerates the names this instance accepts,
    :data:`~dataeval.types.FactorLevel` types them, and :attr:`item_level`,
    :attr:`label_level` and :attr:`unit_type` name what the schema singles out.

    Factors are stored once, at their own level, and propagate *downwards* only; rows
    above a factor's level, on a sibling branch of it, or with no ancestor at it carry
    null values, and factors are never aggregated upwards. For multi-object tracking the
    levels form a **diamond** rather than a chain — ``unit`` (a frame) and ``track`` are
    siblings under ``sequence``, and an instance descends from both.

    Which rows the array-shaped accessors project — :attr:`factor_data`,
    :attr:`factor_names`, :attr:`is_discrete`, :attr:`shape` — is a separate, movable
    choice called the :attr:`view`. It defaults to :attr:`label_level`, so that a
    projection lines up with :attr:`class_labels`; :meth:`at` returns the same metadata
    read at another level.

    Each factor is **binned at its own level** — the level whose rows hold one value per
    entity — and the resulting bins propagate downwards like any other value. See
    :ref:`binning-levels` for a worked example.

    See :doc:`/concepts/MetadataLevels` for the model behind all of this: what follows
    from the diamond, why binning at a factor's own level is what makes results
    comparable across levels, and why the level vocabulary is modality-neutral.

    Parameters
    ----------
    dataset : AnnotatedDataset or None, default None
        Dataset that provides original targets and metadata for processing. When None,
        creates an unbound instance that can be used as a reusable feature extractor.
        Use :meth:`bind` to attach a dataset later, or pass data directly to :meth:`__call__`.
    task : str or None, default None
        Explicit task override, e.g. ``"IC"``, ``"OD"``, or ``"MOT"``. When None the task is
        inferred from the first datum. Supply this for datasets whose target protocol
        MAITE has not defined yet, or when inference would be ambiguous.
    continuous_factor_bins : Mapping[str, int | Sequence[float]] | None, default None
        Mapping from continuous factor names to bin counts or explicit bin edges.
        When None, uses automatic discretization. A bin count is applied to the factor's
        values at its own level, so ``{"brightness": 10}`` on a unit-level factor means
        ten bins over the units.
    auto_bin_method : Literal["uniform_width", "uniform_count", "clusters"], default "uniform_width"
        Binning strategy for continuous factors without explicit bins. Default "uniform_width"
        provides intuitive equal-width intervals for most distributions. Every strategy reads
        the factor's distribution at its own level (see :ref:`binning-levels`); this matters
        most for "uniform_count" and "clusters", whose cut points depend on how the values
        are distributed rather than only on their range.
    exclude : Sequence[str] | None, default None
        Factor names to exclude from processing. Cannot be used with `include` parameter.
        When None, processes all available factors.
    include : Sequence[str] | None, default None
        Factor names to include in processing. Cannot be used with `exclude` parameter.
        When None, processes all available factors.
    view : str or None, default None
        Level the array-shaped accessors project. When None the view follows
        :attr:`label_level`, which is what keeps :attr:`factor_data` aligned with
        :attr:`class_labels`. See :attr:`view`.
    inherited : bool, default True
        Whether factors defined *above* the view count. When False only factors native
        to the view survive, instead of ancestor factors being replicated onto its rows.
        See :attr:`factor_data` for why that replication can skew a marginal
        distribution, and :attr:`inherited` for why widening the view is usually the
        better answer.

    Raises
    ------
    ValueError
        When both exclude and include parameters are specified simultaneously.

    Warns
    -----
    UserWarning
        When the bound dataset is empty and no `task` was given, since the task cannot
        then be inferred and image classification is assumed.

    Notes
    -----
    Structuring and binning are lazy: reading any attribute that describes the bound
    dataset triggers structure analysis on first access, and reading anything derived
    from factor values triggers binning. Nothing is computed at construction time, so
    an unbound instance is cheap to build and reuse.

    Example
    -------
    Using as a feature extractor with drift detection:

    >>> from dataeval import Metadata
    >>> from dataeval.shift import DriftUnivariate
    >>>
    >>> # Create reusable extractor (no dataset bound)
    >>> extractor = Metadata(continuous_factor_bins={"brightness": 10})
    >>>
    >>> # Use with drift detector
    >>> drift = DriftUnivariate(extractor=extractor).fit(train_dataset)
    >>> result = drift.predict(test_dataset)

    Using with a bound dataset:

    >>> # Create with dataset bound
    >>> metadata = Metadata(train_dataset, continuous_factor_bins={"brightness": 10})
    >>> train_factors = metadata()  # Extract from bound dataset
    >>> test_factors = metadata(test_dataset)  # Extract from new dataset
    """

    @requires_maite_dataset("dataset", expected="any_target")
    def __init__(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]] | None = None,
        *,
        task: TaskOverride | None = None,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
        auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
        exclude: str | Sequence[str] | None = None,
        include: str | Sequence[str] | None = None,
        view: FactorLevel | Literal["image"] | None = None,
        inherited: bool = True,
    ) -> None:
        self._class_labels: NDArray[np.intp]
        self._dropped_factors: dict[str, list[str]]
        self._dataframe: pl.DataFrame
        self._raw: Sequence[Mapping[str, Any]]

        self._warned_level_rename = False
        self._reset_structure()

        self._dataset = dataset
        self._task: TaskOverride | None = task
        self._count = len(dataset) if dataset is not None and isinstance(dataset, Sized) else 0
        self._continuous_factor_bins = dict(continuous_factor_bins) if continuous_factor_bins else {}
        self._auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = auto_bin_method

        if exclude is not None and include is not None:
            raise ValueError("Filters for `exclude` and `include` are mutually exclusive.")

        self._exclude = {exclude} if isinstance(exclude, str) else set(exclude or ())
        self._include = {include} if isinstance(include, str) else set(include or ())
        # Validated lazily against the bound dataset's schema, since there is no schema
        # to validate against until structuring; see _structure. A retired spelling like
        # "image" may be sitting here transiently: the cast documents that _adopt
        # resolves it (via _resolve_level) before anything treats this as a real level.
        self._view: FactorLevel | None = cast("FactorLevel | None", view)
        self._inherited = inherited
        self._target_factors_only = False

        self._warn_if_task_unknowable()

    def _reset_structure(self) -> None:
        """Clear everything structuring produced, leaving an unstructured instance."""
        # A bare Structurer, not None: its class defaults *are* what the level accessors
        # should answer before there is a dataset to structure, so the unstructured
        # answers stay declared in one place rather than restated as fallbacks here.
        self._structurer: Structurer = Structurer()
        self._layout: RowLayout = RowLayout(())
        self._factors_by_level: dict[FactorLevel, set[str]] = {}
        # Two containers answering two questions, deliberately not one.
        #
        # ``_factors`` is which factors the current view analyses. It is derived, and
        # rebuilt from scratch by _build_factors whenever the view or the factor
        # registry moves.
        #
        # ``_factor_cache`` is what binning has computed, keyed by name and independent
        # of any view, because a factor is binned once at its own level and its
        # companion column stays valid however the metadata is read.
        #
        # Holding the info on the visible set instead — the obvious single dict — loses
        # a factor's info every time it leaves the visible set, while the companion
        # column describing it stays in the dataframe. _bin() skips any factor that has
        # a companion column, so that info is never recomputed: the factor comes back
        # counted by factor_names and shape but absent from factor_data and is_discrete.
        self._factors: set[str] = set()
        self._factor_cache: dict[str, FactorInfo] = {}
        self._is_structured = False
        self._is_binned = False

    def _warn_if_task_unknowable(self) -> None:
        """Warn that an empty dataset will be structured as image classification."""
        dataset = self._dataset
        if self._task is not None or dataset is None:
            return
        if isinstance(dataset, Sized) and len(dataset) == 0:
            warnings.warn(
                "Cannot infer a task from an empty dataset; assuming image classification. "
                "Pass an explicit task, e.g. Metadata(dataset, task='IC'), to silence this warning.",
                UserWarning,
                stacklevel=4,
            )

    @classmethod
    def from_factors(
        cls,
        factors: Mapping[str, Array1D[Any]] | StatsResult,
        class_labels: Array1D[Any] | None = None,
        *,
        index2label: Mapping[int, str] | None = None,
        item_indices: Array1D[Any] | None = None,
        level: FactorLevel | Literal["image"] | None = None,
        source_index: Sequence[SourceIndex] | None = None,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
        auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
        exclude: str | Sequence[str] | None = None,
        include: str | Sequence[str] | None = None,
        inherited: bool = True,
    ) -> Self:
        """Build a :class:`Metadata` from raw factor arrays without a MAITE dataset.

        This is the "minimal data" constructor: use it when you already have a table of
        metadata factors (and, optionally, class labels) but do not own a full
        :term:`MAITE<Modular AI Trustworthy Engineering (MAITE)>` image dataset. The
        resulting instance is fully structured and can be passed directly to the bias
        evaluators (:class:`~dataeval.bias.Balance`, :class:`~dataeval.bias.Diversity`,
        :class:`~dataeval.bias.Parity`) and to :func:`~dataeval.data.split_dataset`,
        exactly like a :class:`Metadata` built from a dataset.

        Continuous factors are binned and categorical factors digitized using the
        same machinery as the dataset-backed constructor, so ``continuous_factor_bins``
        and ``auto_bin_method`` behave identically here.

        Parameters
        ----------
        factors : Mapping[str, ArrayLike]
            Mapping from factor name to a 1D array of that factor's raw (un-binned)
            values. Every array must have the same length ``N`` — one entry per
            target/detection, or one per `source_index` entry when one is given.

            A whole :class:`~dataeval.core.StatsResult` — the return of
            :func:`~dataeval.core.compute_stats` or
            :func:`~dataeval.core.compute_ratios` — may be passed here directly, in
            which case its ``stats`` and ``source_index`` are used and its bookkeeping
            keys ignored.
        class_labels : ArrayLike or None, default None
            Integer class label per row, length ``N`` — or one per label-level row when
            `source_index` spans two levels. When None, a single class (all zeros) is
            assumed; supply real labels for any class-aware analysis
            (e.g. :class:`~dataeval.bias.Parity`), which is otherwise degenerate.
        index2label : Mapping[int, str] or None, default None
            Optional mapping from integer class index to human-readable name. When
            None, labels are their own names. Missing observed labels are backfilled.
        item_indices : ArrayLike or None, default None
            Optional array of length ``N`` mapping each row back to its source image
            index (used, e.g., for object-detection where multiple detections share
            an image). When None a 1:1 mapping is assumed (one factor row per image).
            Mutually exclusive with `source_index`, which already carries an item index
            per value.
        level : str or None, default None
            Level the supplied rows sit at, ``unit`` or ``instance``. When None the rows
            are treated as ``unit``-level, matching the historical behavior. A
            factors-only instance built this way has this single level and no separate
            item level, so an unlabeled row is not representable here the way it is for a
            dataset-backed instance. Mutually exclusive with `source_index`, which sets
            the level of each value itself.
        source_index : Sequence[SourceIndex] or None, default None
            Labels describing what each value in every factor array refers to, as
            returned by :func:`~dataeval.core.compute_stats`. This is how object
            detection statistics are imported without the dataset: an index carrying both
            per-item entries (``target`` is None) and per-label entries builds two levels,
            ``unit`` and ``instance``, and splits each factor into ``unit_<name>`` and
            ``instance_<name>`` — the same names :meth:`add_factors` gives them. An index
            carrying one kind builds the single level it describes.
        continuous_factor_bins : Mapping[str, int | Sequence[float]] or None, default None
            Bin counts or explicit edges for continuous factors. When None, uses
            automatic discretization via ``auto_bin_method``.
        auto_bin_method : {"uniform_width", "uniform_count", "clusters"}, default "uniform_width"
            Binning strategy for continuous factors without explicit bins.
        exclude : str or Sequence[str] or None, default None
            Factor names to exclude. Mutually exclusive with ``include``.
        include : str or Sequence[str] or None, default None
            Factor names to include. Mutually exclusive with ``exclude``.
        inherited : bool, default True
            Whether factors defined above the view count. A no-op on a single-level
            instance, which has nothing above its rows. It does apply when `source_index`
            carries both kinds of entry: the ``unit``-level half of each factor is
            readable from the ``instance`` rows only while this is True.

        Returns
        -------
        Metadata
            A structured Metadata instance backed by the provided factors.

        Raises
        ------
        ShapeMismatchError
            When factor arrays (or ``class_labels`` / ``item_indices``) do not all
            share the same length.
        ValueError
            When both ``exclude`` and ``include`` are specified, or when ``source_index``
            is combined with ``level`` or ``item_indices``.

        Notes
        -----
        Multi-dimensional values (vector-valued statistics such as ``histogram``,
        ``percentiles`` or ``center``) have no single-column representation and are
        skipped with a warning; the skipped names are recorded in
        :attr:`dropped_factors`, exactly as :meth:`add_factors` records them.

        Example
        -------
        >>> import numpy as np
        >>> from dataeval import Metadata
        >>> from dataeval.bias import Balance
        >>>
        >>> factors = {"age_bin": np.array([0, 1, 0, 2]), "weather": np.array([1, 1, 0, 0])}
        >>> labels = np.array([0, 1, 0, 1])
        >>> md = Metadata.from_factors(factors, labels, index2label={0: "cat", 1: "dog"})
        >>> md.factor_data.shape
        (4, 2)
        >>> result = Balance().evaluate(md)

        Import object detection statistics with no dataset bound. The source index the
        stats carry splits them across the two levels it describes:

        >>> from dataeval.core import compute_stats
        >>> from dataeval.flags import ImageStats
        >>> stats = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)
        >>> md = Metadata.from_factors(stats)
        >>> md.levels
        ('unit', 'instance')
        >>> sorted(md.factor_names)
        ['instance_mean', 'unit_mean']
        """
        factors, source_index = _unpack_stats_result(factors, source_index, level=level)
        inst = cls(
            None,
            continuous_factor_bins=continuous_factor_bins,
            auto_bin_method=auto_bin_method,
            exclude=exclude,
            include=include,
            inherited=inherited,
        )
        inst._load_factors(
            factors,
            class_labels,
            index2label=index2label,
            item_indices=item_indices,
            level=level,
            source_index=source_index,
        )
        return inst

    def _load_factors(
        self,
        factors: Mapping[str, Array1D[Any]],
        class_labels: Array1D[Any] | None,
        *,
        index2label: Mapping[int, str] | None,
        item_indices: Array1D[Any] | None,
        level: FactorLevel | Literal["image"] | None = None,
        source_index: Sequence[SourceIndex] | None = None,
    ) -> None:
        """Populate structured state directly from raw factor arrays (see from_factors)."""
        # Vector-valued statistics have no single-column form. Dropping them here rather
        # than letting a flatten silently produce a wrong-length column keeps a mapping
        # straight from compute_stats usable, and reports the same way add_factors reports
        # it. Column vectors keep working: _split_by_dimensionality flattens them.
        factor_arrays, skipped = self._split_by_dimensionality({str(k): v for k, v in factors.items()})

        if source_index is not None:
            self._load_factors_by_source_index(
                factor_arrays,
                class_labels,
                index2label=index2label,
                level=level,
                item_indices=item_indices,
                source_index=source_index,
            )
        else:
            self._load_factors_by_length(
                factor_arrays,
                class_labels,
                index2label=index2label,
                level=level,
                item_indices=item_indices,
            )
        # Recorded only once the structure exists, and only once every validation above has
        # passed: recording is a mutation, and a rejected call must leave no trace of itself
        # behind.
        self._record_multidimensional(skipped)

    def _load_factors_by_length(  # noqa: C901
        self,
        factor_arrays: Mapping[str, NDArray[Any]],
        class_labels: Array1D[Any] | None,
        *,
        index2label: Mapping[int, str] | None,
        level: FactorLevel | Literal["image"] | None,
        item_indices: Array1D[Any] | None,
    ) -> None:
        """Populate structured state from factor arrays that all describe one level.

        Nothing in bare arrays distinguishes an item from a label, so every factor sits at
        the same level and the rows are numbered by position.

        Raises
        ------
        ShapeMismatchError
            When the factors, ``class_labels`` and ``item_indices`` do not agree on a
            single row count.
        """
        lengths = {len(v) for v in factor_arrays.values()}
        if len(lengths) > 1:
            raise ShapeMismatchError(f"All factor arrays must have the same length; got lengths {sorted(lengths)}.")
        factor_len = next(iter(lengths)) if factor_arrays else None

        if class_labels is not None:
            labels = as_numpy(class_labels, dtype=np.intp).reshape(-1)
            n = len(labels)
            if factor_len is not None and factor_len != n:
                raise ShapeMismatchError(f"class_labels length {n} does not match factor length {factor_len}.")
        elif factor_len is not None:
            n = factor_len
            labels = np.zeros(n, dtype=np.intp)
        else:
            n = 0
            labels = np.array([], dtype=np.intp)

        if item_indices is None:
            srcidx = np.arange(n, dtype=np.intp)
        else:
            srcidx = as_numpy(item_indices, dtype=np.intp).reshape(-1)
            if len(srcidx) != n:
                raise ShapeMismatchError(f"item_indices length {len(srcidx)} does not match row count {n}.")

        # A factors-only instance has a single level, which is therefore both the
        # item level and the label level. Structuring goes through the same
        # StructuredData bundle as the dataset path, so the reserved columns have
        # exactly one producer and cannot drift between the two constructors. No
        # structurer instance exists yet to resolve a retired spelling against — this
        # call is what builds one — so this resolves against the base Structurer's
        # alias map rather than a task-specific one.
        # caller -> from_factors -> _load_factors -> here -> _resolve_legacy_level.
        # test_from_factors_blames_caller pins this.
        requested = _resolve_legacy_level(
            level or "unit", Structurer.legacy_level_aliases, stacklevel=5, unit_type=Structurer.unit_type
        )
        structurer = FactorsStructurer(requested)  # type: ignore[arg-type]
        data = structurer.build_from_arrays(factor_arrays, labels, srcidx)

        self._index2label = _build_index2label(index2label, np.unique(labels))
        # Items, not rows, matching :attr:`item_count`'s contract and the source-index
        # path. ``item_indices`` exists so that several rows can share one item, and a
        # count of rows disagrees with it on exactly the tables it is for. It is also what
        # tells :func:`~dataeval.data.split_dataset` an object detection table from a
        # classification one — counted as rows, a table of detections never reaches the
        # grouped split and two detections of one image can land in different folds.
        self._count = int(len(np.unique(srcidx)))
        self._adopt(structurer, data)

    def _load_factors_by_source_index(
        self,
        factor_arrays: Mapping[str, NDArray[Any]],
        class_labels: Array1D[Any] | None,
        *,
        index2label: Mapping[int, str] | None,
        level: FactorLevel | Literal["image"] | None,
        item_indices: Array1D[Any] | None,
        source_index: Sequence[SourceIndex],
    ) -> None:
        """Populate structured state from factor arrays labelled by a source index.

        The source index supplies what `level` and `item_indices` supply on the other
        path — which level each value belongs to and which item it came from — so all
        three together is a contradiction rather than a redundancy, and is rejected
        instead of one silently winning.

        Raises
        ------
        ValueError
            When `level` or `item_indices` is given alongside the source index.
        ShapeMismatchError
            When a factor does not have one value per source-index entry.
        """
        for name, value in (("level", level), ("item_indices", item_indices)):
            if value is not None:
                raise ValueError(
                    f"`{name}` and `source_index` are mutually exclusive; the source index already "
                    f"says which level each value sits at and which item it came from.",
                )

        _reject_length_mismatch(factor_arrays, source_index)

        rows = SourceIndexRows.parse(source_index)
        structurer = FactorsStructurer(rows=rows)
        labels = None if class_labels is None else as_numpy(class_labels, dtype=np.intp).reshape(-1)
        data = structurer.build_from_source_index(factor_arrays, labels)

        self._index2label = _build_index2label(index2label, np.unique(data.class_labels))
        # Items, not rows: several labels can name the same item, and item_count that
        # counted rows would disagree with item_indices on the very datasets — one item,
        # several detections — this path exists to carry. Counted by adjacent change
        # rather than np.unique, which would re-sort what parse already left sorted.
        named_items = rows.item_ids if len(rows.item_positions) else rows.label_items
        self._count = int(np.count_nonzero(np.diff(named_items))) + 1 if len(named_items) else 0
        self._adopt(structurer, data)

    def __repr__(self) -> str:  # noqa: C901
        bound = self._dataset is not None
        parts = [f"bound={bound}"]
        if self._task is not None:
            parts.append(f"task={self._task!r}")
        if self._continuous_factor_bins:
            parts.append(f"continuous_factor_bins={self._continuous_factor_bins!r}")
        parts.append(f"auto_bin_method={self._auto_bin_method!r}")
        if self._exclude:
            parts.append(f"exclude={self._exclude!r}")
        if self._include:
            parts.append(f"include={self._include!r}")
        if bound:
            parts.append(f"n={self._count}")
        return f"Metadata({', '.join(parts)})"

    def __str__(self) -> str:
        bound = self._dataset is not None
        factors = sorted(self._factors) if self._is_structured else []
        factor_str = f", factors={factors}" if factors else ""
        task_str = f", task={self._structurer.task}" if self._is_structured else ""
        unit_str = f", units={self._structurer.unit_type}" if self._is_structured else ""
        return f"Metadata(n={self._count}, bound={bound}{task_str}{unit_str}{factor_str})"

    @property
    def is_bound(self) -> bool:
        """Whether this instance is bound to a dataset.

        Returns
        -------
        bool
            True if a dataset is bound, False otherwise.
        """
        return self._dataset is not None

    @property
    def _is_fitted(self) -> bool:
        """Whether factor data is available — either a dataset is bound or state was loaded directly."""
        return self._dataset is not None or self._is_structured

    @requires_maite_dataset("dataset", expected="any_target")
    def bind(self, dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]]) -> Self:
        """Bind this instance to a dataset.

        Attaches a dataset to this Metadata instance for metadata extraction.
        Any previously processed metadata is cleared.

        Parameters
        ----------
        dataset : AnnotatedDataset
            Dataset to bind for metadata extraction.

        Returns
        -------
        Self
            Returns self for method chaining.

        Warns
        -----
        UserWarning
            When the dataset is empty and no ``task`` was given.

        Notes
        -----
        An explicitly chosen :attr:`view` is cleared along with the rest of the level
        state: it names a level of the schema being discarded, and the new dataset
        need not have one by that name.

        Example
        -------
        >>> from dataeval import Metadata
        >>>
        >>> extractor = Metadata(continuous_factor_bins={"brightness": 10})
        >>> _ = extractor.bind(train_dataset)
        """
        self._dataset = dataset
        self._count = len(dataset) if isinstance(dataset, Sized) else 0
        self._reset_structure()
        self._view = None
        self._warned_level_rename = False
        self._warn_if_task_unknowable()
        return self

    @property
    def _levels(self) -> FactorLevelSchema:
        """Level schema of the bound dataset, or a bare default before structuring."""
        return self._structurer.levels

    @property
    def _item_level(self) -> FactorLevel:
        """Level one dataset item corresponds to."""
        return self._structurer.item_level

    @property
    def _label_level(self) -> FactorLevel:
        """Level whose rows carry ``class_label``."""
        return self._structurer.label_level

    @property
    def _view_level(self) -> FactorLevel:
        """Resolved view — the explicit one, or the label level when none was set."""
        return self._label_level if self._view is None else self._view

    def __array__(self) -> NDArray[np.int64]:
        """NumPy array representation of binned metadata.

        Returns
        -------
        NDArray[np.int64]
            Binned metadata as a NumPy array of shape (n_samples, n_factors).

        Notes
        -----
        This property triggers factor binning analysis on first access.
        Use this for interoperability with libraries expecting NumPy arrays.
        """
        return self.factor_data

    def __len__(self) -> int:
        """Return the number of rows in the binned metadata array.

        Returns
        -------
        int
            Number of rows at the ``instance`` level — the rows :attr:`factor_data`
            returns and the rows this instance iterates and indexes. This is the
            detection count for object detection, not the dataset item count; use
            :attr:`item_count` for the latter.

        Raises
        ------
        NotFittedError
            If no dataset is bound.

        Notes
        -----
        This is ``shape[0]``. :class:`Metadata` implements the
        :class:`~dataeval.protocols.Array` protocol, so its length, its shape and
        what it yields when iterated all have to describe the same rows.
        """
        return self.shape[0]

    @property
    def ndim(self) -> int:
        """Number of dimensions of the binned metadata array.

        Returns
        -------
        int
            Number of dimensions.

        Raises
        ------
        NotFittedError
            If no dataset is bound.
        """
        return len(self.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the binned metadata array.

        Returns
        -------
        tuple[int, ...]
            Shape of the binned metadata as (n_samples, n_factors), where
            n_samples is the number of rows at the :attr:`view` level — the rows
            :attr:`factor_data` actually returns. At the default view that is the
            detection count for object detection, not the dataset item count;
            :attr:`item_count` reports the item count.

        Raises
        ------
        NotFittedError
            If no dataset is bound.

        Notes
        -----
        This property triggers dataset structure analysis on first access.
        """
        if not self._is_fitted:
            raise NotFittedError("No dataset bound. Call bind() first.")
        self._structure()
        # Counted rather than measured off factor_names, which would sort the names
        # only to discard the sorted list; len() and ndim both route through here.
        return (self._layout.counts.get(self._view_level, 0), sum(1 for name in self._factors if self._filter(name)))

    def __iter__(self) -> Iterator[NDArray[np.int64]]:
        """Iterate over rows of the binned metadata.

        Yields
        ------
        Iterator[NDArray[np.int64]]
            Rows of the binned metadata array, one at a time.

        Raises
        ------
        NotFittedError
            If no dataset is bound.
        """
        if not self._is_fitted:
            raise NotFittedError("No dataset bound. Call bind() first.")
        yield from self.factor_data

    def __getitem__(self, index: int | str | slice) -> Array:  # noqa: C901
        """Get binned metadata for specific indices or factors.

        Parameters
        ----------
        index : int, str, or slice
            Index or slice to select specific rows (by integer index)
            or columns (by factor name) from the binned metadata.

        Returns
        -------
        Array

            Binned metadata for the specified indices or factors.

        Raises
        ------
        NotFittedError
            If no dataset is bound.
        KeyError
            If a specified factor name is not found in the metadata.
        """
        if not self._is_fitted:
            raise NotFittedError("No dataset bound. Call bind() first.")

        data = self.factor_data

        if isinstance(index, int):
            return data[index]
        if isinstance(index, str):
            if index not in self.factor_names:
                raise KeyError(f"Factor '{index}' not found in metadata.")
            col_index = self.factor_names.index(index)
            return data[:, col_index]
        if isinstance(index, slice):
            return data[index]
        raise TypeError("Index must be an int, str, or slice.")

    def new(self, dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]]) -> Self:
        """Create new Metadata instance with a different dataset.

        Generate a new Metadata object using the same configuration
        but with a different dataset.

        Parameters
        ----------
        dataset : AnnotatedDataset
            Dataset that provides metadata for the new Metadata instance.

        Returns
        -------
        Metadata
            New Metadata object configured identically to the current instance,
            including an explicitly chosen :attr:`view`. A view left at its default
            stays at the default, so the new instance follows its own dataset's
            :attr:`label_level` rather than inheriting this one's.
        """
        return self.__class__(
            dataset,
            task=self._task,
            continuous_factor_bins=self._continuous_factor_bins,
            auto_bin_method=self._auto_bin_method,
            exclude=list(self._exclude) if self._exclude else None,
            include=list(self._include) if self._include else None,
            view=self._view,
            inherited=self._inherited,
        )

    def __call__(self, data: Any | None = None) -> Array:
        """Extract metadata factors from data.

        Implements the :class:`~dataeval.protocols.FeatureExtractor` protocol,
        allowing this instance to be used directly with drift detectors.

        Parameters
        ----------
        data : Any or None, default None
            Dataset to extract metadata from. If None, uses the bound dataset.

        Returns
        -------
        Array
            Binned metadata array of shape (n_samples, n_factors).

        Raises
        ------
        NotFittedError
            If data is None and no dataset is bound.

        Example
        -------
        >>> from dataeval import Metadata
        >>>
        >>> metadata = Metadata(train_dataset, continuous_factor_bins={"brightness": 10})
        >>>
        >>> # Extract from bound dataset
        >>> train_factors = metadata()
        >>>
        >>> # Extract from new dataset
        >>> test_factors = metadata(test_dataset)
        """
        if data is None:
            if self._dataset is None:
                raise NotFittedError("No dataset bound. Provide data or call bind() first.")
            # Return factors for bound dataset
            return self.factor_data

        # Check if same as bound dataset (by identity)
        if self._dataset is not None and data is self._dataset:
            return self.factor_data

        # Compute metadata for new data using this config
        return self.new(data).factor_data

    @property
    def raw(self) -> Sequence[Mapping[str, Any]]:
        """Original metadata dictionaries extracted from the dataset.

        Access the unprocessed metadata as it was provided in the original dataset before
        any binning, filtering, or transformation operations.

        Returns
        -------
        Sequence[Mapping[str, Any]]
            List of metadata dictionaries, one per dataset item, containing the original key-value
            pairs as provided in the source data

        """
        self._structure()
        return self._raw

    @property
    def levels(self) -> tuple[FactorLevel, ...]:
        """Level names for the bound dataset, coarsest first.

        Returns
        -------
        tuple[str, ...]
            Every level name this instance recognizes, in schema order. These are the
            names :meth:`rows_at`, :meth:`add_factors` and :attr:`level_counts` accept.

        Notes
        -----
        The parent relationships behind this ordering are an implementation detail of
        the structuring layer and are deliberately not exposed here; factors propagate
        downwards along them, which :attr:`factor_data` documents in the terms a caller
        needs.
        """
        self._structure()
        return tuple(self._levels)

    @property
    def level_counts(self) -> Mapping[FactorLevel, int]:
        """Number of dataframe rows at each level.

        Returns
        -------
        Mapping[Level, int]
            Mapping of level name to row count, in schema order.
        """
        self._structure()
        return dict(self._layout.counts)

    @property
    def item_level(self) -> FactorLevel:
        """Level at which one row corresponds to one dataset item.

        Returns
        -------
        str
            One of :attr:`levels`. ``rows_at(md.item_level)`` is the task-generic
            spelling of "one row per dataset item", which is what
            :attr:`item_count` counts.
        """
        self._structure()
        return self._item_level

    @property
    def label_level(self) -> FactorLevel:
        """Level whose rows carry a class label.

        Returns
        -------
        str
            One of :attr:`levels` — the level :attr:`class_labels`, ``score`` and
            ``box`` describe. A structural fact about where the annotations sit,
            which is why it is read-only; :attr:`view` is the knob that decides
            which rows get projected.
        """
        self._structure()
        return self._label_level

    @property
    def unit_type(self) -> str:
        """What one row at the ``unit`` level holds, in the dataset's own vocabulary.

        Returns
        -------
        str
            ``"image"`` for image classification and object detection, ``"frame"``
            for multi-object tracking, ``"item"`` for a factors-only instance.

        Notes
        -----
        Descriptive only. It names the medium so that messages and reports can speak
        the caller's language, and is never consulted by structuring, binning or
        projection. It is deliberately a plain :class:`str` rather than a member of
        :data:`~dataeval.types.FactorLevel`: a new modality supplies a new value here
        and the level vocabulary stays closed and task-independent.
        """
        self._structure()
        return self._structurer.unit_type

    @property
    def multi_target(self) -> bool:
        """Whether one dataset item can carry more than one label.

        Returns
        -------
        bool
            True for object detection and multi-object tracking, False for image classification.

        Notes
        -----
        A declared property of the task, not a measurement. It is deliberately not
        derived from the row counts — an object detection dataset with exactly one
        detection per image has as many label rows as item rows — nor from
        ``label_level != item_level``, which is true of every task now that
        classification labels sit on their own ``instance`` rows.
        """
        self._structure()
        return self._structurer.multi_target

    @property
    def view(self) -> FactorLevel:
        """Level the array-shaped accessors project.

        :attr:`factor_data`, :attr:`factor_names`, :attr:`is_discrete` and
        :attr:`shape` describe the rows at this level, and so do :func:`len`,
        iteration and indexing. The dataframe itself is unaffected: it always holds
        every level, and :meth:`rows_at` reaches any of them regardless of the view.

        .. deprecated::
            ``"target"`` is accepted with a warning and resolves to the
            ``"instance"`` level.

        .. deprecated::
            ``"image"`` is accepted with a warning and resolves to the ``"unit"``
            level. Removed in v1.2.0.

        Returns
        -------
        FactorLevel
            One of :attr:`levels`. Defaults to :attr:`label_level`, which is what
            keeps a projection aligned with :attr:`class_labels`.

        Raises
        ------
        ValueError
            When assigned a level that is not part of this dataset's schema.

        Notes
        -----
        Moving the view *up* is how a factor is read once per entity rather than once
        per descendant — see :attr:`factor_data` for why that distinction changes a
        marginal distribution. Factors do not travel upwards, so a view above a
        factor's own level does not see it at all, and :attr:`class_labels` above
        :attr:`label_level` raises rather than inventing a label per item.

        Bin values never move with the view. Every factor is binned once, at its own
        level (see :ref:`binning-levels`), so changing the view changes which rows are
        counted and never what any of them says.

        Prefer :meth:`at` where the alternative is assigning to this and assigning
        back: evaluators hold a reference to the metadata, so a view mutated in place
        changes what an already-constructed evaluator will read.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.view
        'instance'
        >>> metadata.factor_data.shape[0] == metadata.level_counts["instance"]
        True
        """
        self._structure()
        return self._view_level

    @view.setter
    def view(self, level: FactorLevel | Literal["target", "image"]) -> None:
        self._structure()
        resolved = self._resolve_level(level)
        if resolved != self._view_level:
            self._view = resolved
            self._reset_view_dependent_state()

    def at(self, level: FactorLevel | Literal["target", "image"]) -> Self:
        """Return this metadata read at another level.

        Parameters
        ----------
        level : FactorLevel
            Level to project, one of :attr:`levels`.

            .. deprecated::
                ``"target"`` is accepted with a warning and resolves to the
                ``"instance"`` level.

            .. deprecated::
                ``"image"`` is accepted with a warning and resolves to the
                ``"unit"`` level. Removed in v1.2.0.

        Returns
        -------
        Metadata
            A copy whose :attr:`view` is ``level``, sharing this instance's structuring
            and binning work. The original is untouched.

        Raises
        ------
        ValueError
            When the level is not part of this dataset's schema.

        Notes
        -----
        The copy is independent from the moment it is made: adding factors to one does
        not add them to the other. Use it rather than assigning to :attr:`view` when
        the metadata is being handed to an evaluator, so that two evaluators can read
        two levels of the same dataset at once.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> # One row per detection, image factors replicated across each image's detections
        >>> metadata.factor_data.shape[0]
        93
        >>> # One row per image, image factors counted once each
        >>> metadata.at("unit").factor_data.shape[0]
        50
        """
        self._structure()
        resolved = self._resolve_level(level)

        view = copy.copy(self)
        # Everything mutable is copied out: a view that shares the factor dict with its
        # source would have its FactorInfo overwritten the next time either one binned.
        view._factors = set(self._factors)
        view._factor_cache = dict(self._factor_cache)
        view._factors_by_level = {name: set(names) for name, names in self._factors_by_level.items()}
        view._dropped_factors = {name: list(reasons) for name, reasons in self._dropped_factors.items()}
        view._dataframe = self._dataframe.clone()
        view._view = resolved
        # The move can bring factors into the visible set that the source never binned —
        # anything defined below its view. ``_is_binned`` is the source's claim that
        # nothing is left to process, and it does not carry: without clearing it the copy
        # reports those factors in factor_names and shape while factor_data and
        # is_discrete, which read factor_info, silently omit them. Companion columns
        # survive in the clone and _bin() skips a factor that has one, so this re-bins
        # only what the move made visible.
        view._is_binned = False
        # A copy is a fresh object in the user's hands, so it gets its own once-per-
        # instance budget for the FactorInfo.level rename — the same reasoning as bind().
        view._warned_level_rename = False
        view._build_factors()
        return view

    def _reset_view_dependent_state(self) -> None:
        """Rebuild the factor set after something moved which rows or factors are visible.

        Only the visible set is rebuilt; bins are kept. A factor is binned once, at its
        own level (see :ref:`binning-levels`), so its companion column stays valid
        across a view move, and a factor that leaves the visible set and later returns
        is restored from ``_factor_cache`` rather than re-binned.

        ``_is_binned`` is cleared regardless, for the same reason :meth:`at` clears it:
        the move can expose factors that have never been binned, and the flag means
        "nothing left to process", which is no longer true. :meth:`_bin` skips any
        factor that already has a companion column, so the re-bin covers only what
        became visible.
        """
        self._is_binned = False
        self._build_factors()

    @property
    def exclude(self) -> set[str]:
        """Factor names excluded from metadata processing.

        Returns
        -------
        set[str]
            Set of factor names that are filtered out during processing.
            Empty set when no exclusions are active.

        """
        return self._exclude

    @exclude.setter
    def exclude(self, value: str | Sequence[str]) -> None:
        """Set factor names to exclude from processing.

        Automatically clears include filter and resets binning state when exclusion list changes.

        Parameters
        ----------
        value : str | Sequence[str]
            Factor name or names to exclude from metadata analysis.
        """
        exclude = {value} if isinstance(value, str) else set(value)
        if self._exclude != exclude:
            self._exclude = exclude
            self._include = set()
            self._is_binned = False

    @property
    def include(self) -> set[str]:
        """Factor names included in metadata processing.

        Returns
        -------
        set[str]
            Set of factor names that are processed during analysis. Empty set when no inclusion filter is active.
        """
        return self._include

    @include.setter
    def include(self, value: str | Sequence[str]) -> None:
        """Set factor names to include in processing.

        Automatically clears exclude filter and resets binning state when
        inclusion list changes.

        Parameters
        ----------
        value : str | Sequence[str]
            Factor name or names to include in metadata analysis.
        """
        include = {value} if isinstance(value, str) else set(value)
        if self._include != include:
            self._include = include
            self._exclude = set()
            self._is_binned = False

    @property
    def continuous_factor_bins(self) -> Mapping[str, int | Sequence[float]]:
        """Binning configuration for continuous factors.

        Returns
        -------
        Mapping[str, int | Sequence[float]]
            Mapping of factor names to either the number of bins
            (int) or explicit bin edges (sequence of floats).
        """
        return self._continuous_factor_bins

    @continuous_factor_bins.setter
    def continuous_factor_bins(self, bins: Mapping[str, int | Sequence[float]]) -> None:
        """Update binning configuration for continuous factors.

        Triggers re-binning when configuration changes to ensure data
        consistency with new bin specifications.

        Parameters
        ----------
        bins : Mapping[str, int | Sequence[float]]
            Mapping of factor names to bin counts or explicit edges.
        """
        if self._continuous_factor_bins != bins:
            self._continuous_factor_bins = dict(bins)
            self._reset_bins(bins)

    @property
    def auto_bin_method(self) -> Literal["uniform_width", "uniform_count", "clusters"]:
        """Automatic binning strategy for continuous factors.

        Returns
        -------
        {"uniform_width", "uniform_count", "clusters"}
            Current method used for automatic discretization of continuous
            factors that lack explicit bin specifications.
        """
        return self._auto_bin_method

    @auto_bin_method.setter
    def auto_bin_method(self, method: Literal["uniform_width", "uniform_count", "clusters"]) -> None:
        """Set automatic binning strategy for continuous factors.

        Triggers re-binning with the new method when strategy changes to
        ensure consistent discretization across all factors.

        Parameters
        ----------
        method : {"uniform_width", "uniform_count", "clusters"}
            Binning strategy to apply for continuous factors without
            explicit bin configurations.
        """
        if self._auto_bin_method != method:
            self._auto_bin_method = method
            self._reset_bins()

    @property
    def inherited(self) -> bool:
        """Whether factors defined above the :attr:`view` are counted.

        Returns
        -------
        bool
            True (default) if a factor from an ancestor level is replicated onto the
            view's rows and analysed there, False if only factors native to the view
            survive.

        Notes
        -----
        Settable, like :attr:`exclude` and :attr:`include`, which it sits alongside as a
        factor-selection filter: assigning to it rebuilds the factor list in place so a
        single instance can be re-read both ways. Pass it to the constructor instead when
        the value is fixed for the instance's lifetime.

        Setting this False is usually the wrong half of the fix. The problem it
        addresses is that a unit-level factor read from detection rows has its
        marginal distribution weighted by detections-per-image; dropping the factor
        answers that by discarding it, whereas ``md.at("unit")`` answers it by reading
        the factor where there is one value per unit. Reach for this only when the
        goal really is "instance-native factors and nothing else".

        On a task where no factor is native to the view — image classification puts
        essentially all per-item metadata at the ``unit`` level — this leaves no
        factors at all, and the evaluators will say so.
        """
        return self._inherited

    @inherited.setter
    def inherited(self, value: bool) -> None:
        if self._inherited != value:
            self._inherited = value
            self._reset_view_dependent_state()

    @property
    def target_factors_only(self) -> bool:
        """Whether factors above the target level are dropped, on multi-target tasks.

        .. deprecated::
            Two knobs in one, and neither of them this. Use ``md.at(level)`` to choose
            which rows are read, and :attr:`inherited` to choose whether ancestor
            factors count. Removed in v1.2.0.

        Notes
        -----
        Retains its v1.1 semantics exactly, including the part that reads like a bug:
        it is a no-op unless :attr:`multi_target`, so on image classification it has
        never done anything. :attr:`inherited` does not carry that exemption over — it
        means what it says on every task — which is why this is kept as its own flag
        rather than forwarded to it.
        """
        warnings.warn(
            "Metadata.target_factors_only is deprecated and will be removed in v1.2.0. "
            "Use Metadata.at(level) to choose the rows and Metadata.inherited to choose "
            "whether ancestor factors count.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._target_factors_only

    @target_factors_only.setter
    def target_factors_only(self, value: bool) -> None:
        warnings.warn(
            "Metadata.target_factors_only is deprecated and will be removed in v1.2.0. "
            "Use Metadata.at(level) to choose the rows and Metadata.inherited to choose "
            "whether ancestor factors count.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._target_factors_only != value:
            self._target_factors_only = value
            self._reset_view_dependent_state()

    @property
    def dataframe(self) -> pl.DataFrame:
        """Processed DataFrame containing rows at every level of the dataset.

        Access the main data structure. Every row carries a ``level`` column
        naming the level it belongs to; use :meth:`rows_at` to filter to a specific
        level.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns for level, item_index, target_index, class_label,
            scores, bounding boxes (when applicable), a ``level`` tag naming the
            level each row belongs to, that level's own key columns, and all
            processed metadata factors.

        See Also
        --------
        :meth:`~dataeval.Metadata.rows_at` : Filter to any level
        :attr:`~dataeval.Metadata.view` : The level the array accessors project
        :meth:`~dataeval.Metadata.at` : This metadata read at another level

        Notes
        -----
        Factor binning occurs automatically when accessing factor-related data.

        Rows are ordered by level, coarsest first, so for an object detection
        dataset all unit rows precede all instance rows. The legacy ``item_index``
        and ``target_index`` columns are retained: ``target_index`` remains null
        on rows above :attr:`label_level`.
        """
        self._structure()
        return self._dataframe

    @property
    def dropped_factors(self) -> Mapping[str, Sequence[str]]:
        """Factors removed during preprocessing with removal reasons.

        Returns
        -------
        Mapping[str, Sequence[str]]
            Mapping of dropped factor names to lists of reasons
            why they were excluded from the final dataset.

        Notes
        -----
        Common removal reasons include incompatible data types, excessive
        missing values, or insufficient variation.

        Two reasons are recorded by :meth:`add_factors` itself:

        - ``"multi_dimensional"`` — the factor was not a 1-D array, so it has no
          single-column form.
        - ``"no_values_at_level"`` — a factor split across levels held nothing at one of
          them, so that half was not written. The name recorded is the generated
          ``<level>_<name>``, and the factor's other level(s) were kept. A factor whose
          every level came back empty is kept whole rather than recorded here.
        """
        self._structure()
        return self._dropped_factors

    @property
    def factor_data(self) -> NDArray[np.int64]:
        """Factor data with continuous values discretized into bins.

        Access fully processed factor data where both categorical and
        continuous factors are converted to integer bin indices.

        Returns
        -------
        NDArray[np.int64]
            Array with shape (n_samples, n_factors) containing binned integer
            data ready for categorical analysis algorithms. Returns empty array
            when no factors are available.

        Notes
        -----
        Use this for algorithms requiring purely discrete input data.

        Rows are taken at the :attr:`view` level, which defaults to
        :attr:`label_level` so that they align with :attr:`class_labels` — one row per
        detection for object detection, one row per image for image classification.

        A factor defined above the view is *replicated* onto these rows, once per
        descendant. Its bin values are correct — binning happens at the factor's own
        level (see :ref:`binning-levels`) — but its marginal distribution here is
        weighted by how many descendants each entity has, so a unit-level factor on
        a detection dataset counts crowded images more heavily than sparse ones.
        ``md.at("unit").factor_data`` reads it once per unit instead.

        A factor the view's rows cannot *all* read is omitted from these columns, and from
        :attr:`factor_names`, rather than represented as a gap — a partly null column has no
        binning, since discretizing it would compare None against a number. Both cases come
        from the tracking schema's diamond: a factor on a sibling branch (per-frame read
        from track rows, or the reverse) and a per-track factor read from instance rows when
        some detection is untracked. The factor stays in :attr:`dataframe` throughout, and
        ``md.at`` on its own level reads it in full.
        """
        info_by_name = self._factor_info
        return self._project([_to_col(name, info) for name, info in info_by_name.items()], np.int64)

    @property
    def factor_names(self) -> Sequence[str]:
        """Names of all processed metadata factors.

        Returns
        -------
        Sequence[str]
            List of factor names that passed filtering and preprocessing steps.
            Order matches columns in factor_data.

        Notes
        -----
        Factor names respect include/exclude filtering settings.
        """
        self._structure()
        return sorted(filter(self._filter, self._factors))

    @property
    def _factor_info(self) -> Mapping[str, FactorInfo]:
        """:attr:`factor_info` without the rename warning, for internal callers.

        Everything in this class that needs a factor's type or companion column
        reads this. The warning belongs on the paths that put a
        :class:`FactorInfo` in *user* hands — :attr:`factor_info` and
        :meth:`filter_by_factor` — not on ``factor_data``, which every bias
        evaluator calls and whose callers may never touch ``FactorInfo`` at all.

        An instance with nothing visible to analyse answers without binning, so that
        the array accessors can shape an empty projection off the row layout alone
        rather than materializing a dataframe they have nothing to read from.
        """
        if not self.factor_names:
            return {}
        self._bin()
        # Visible *and* processed: a factor is in ``_factors`` from the moment it is
        # registered and in the cache only once it has a companion column, so the
        # intersection is exactly the set factor_data can project.
        return {name: self._factor_cache[name] for name in self.factor_names if name in self._factor_cache}

    @property
    def factor_info(self) -> Mapping[str, FactorInfo]:
        """Type information and processing status for each factor.

        Returns
        -------
        Mapping[str, FactorInfo]
            Mapping of factor names to FactorInfo objects containing
            data type classification, processing flags (binned, digitized),
            and the level the factor is defined at.

        Notes
        -----
        Only includes factors that survived preprocessing and filtering.

        On tasks whose label level used to be reported as ``"target"`` this warns
        once per instance that :attr:`FactorInfo.level` now reports the level's
        real name; see :class:`FactorInfo`.
        """
        info = self._factor_info
        self._warn_level_rename()
        return info

    @property
    def is_discrete(self) -> Sequence[bool]:
        """Whether each factor is discrete (True) or continuous (False).

        Returns
        -------
        Sequence[bool]
            Boolean sequence with length equal to factor_names, where True
            indicates a discrete factor (categorical or discrete numeric)
            and False indicates a continuous factor.

        Notes
        -----
        This property is part of the :class:`~dataeval.protocols.MetadataLike`
        and aligns with scientific computing conventions where discrete factors
        are treated differently from continuous ones in statistical analyses.
        """
        return [info.factor_type != "continuous" for info in self._factor_info.values()]

    @property
    def raw_data(self) -> NDArray[Any]:
        """Raw factor values before binning or digitization.

        .. deprecated::
            This is ``rows_at(md.view).select(factor_names).to_numpy()``, and going
            through the dataframe keeps the per-factor dtypes that this array flattens
            to ``object`` the moment factors of different types are mixed. Use
            :meth:`rows_at` for raw values, or :meth:`filter_by_factor` for a float
            array. Removed in v1.2.0.

        Returns
        -------
        NDArray[Any]
            Array with shape (n_samples, n_factors) containing original factor
            values, taken at the :attr:`view` level. Returns empty array when no
            factors are available.
        """
        warnings.warn(
            "Metadata.raw_data is deprecated and will be removed in v1.2.0. Use "
            "Metadata.rows_at(md.view).select(md.factor_names).to_numpy() instead, or "
            "Metadata.filter_by_factor() for a float array.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.factor_names:
            return np.array([], dtype=np.float64)

        return self.rows_at(self._view_level).select(self.factor_names).to_numpy()

    @property
    def class_labels(self) -> NDArray[np.intp]:
        """Class labels as integer indices, one per row at :attr:`label_level`.

        Returns
        -------
        NDArray[np.intp]
            Array of class indices corresponding to dataset labels, one per
            label-level row: one per detection for object detection, one per
            labelled image for image classification.

        Raises
        ------
        ValueError
            When the :attr:`view` sits above :attr:`label_level`, where there is no
            label per row to return.

        Notes
        -----
        Use index2label property to get human-readable label names.

        This aligns with :attr:`factor_data` at the default view. A view moved above
        the label level has several labels per row — or none — and rather than pick
        one, this raises: the alternative is silently handing an evaluator a label
        array that does not correspond to its factor rows. Read the labels through
        :meth:`rows_at` when a coarser view genuinely needs them.
        """
        self._structure()
        view = self._view_level
        if view != self._label_level:
            raise ValueError(
                f"class_labels is defined at the {self._label_level!r} level, but this metadata is "
                f"viewed at {view!r}, which has no label per row. Use md.at({self._label_level!r}) "
                f'for the labels, or read them from rows_at({view!r})["class_label"].',
            )
        return self._class_labels

    @property
    def index2label(self) -> Mapping[int, str]:
        self._structure()
        return self._index2label

    @property
    def item_indices(self) -> NDArray[np.intp]:
        """Dataset item each row of the current :attr:`view` came from.

        Returns
        -------
        NDArray[np.intp]
            One index per row at :attr:`view`, mapping that row back to its source
            item in the original dataset. At the default view this is one entry per
            detection for object detection and one per labelled image for image
            classification; at :attr:`item_level` it is one entry per item.

        Notes
        -----
        Read from the view's own rows rather than fixed at :attr:`label_level`, so
        that it stays the same length as :attr:`factor_data` however the view is
        moved. A caller pairing the two — which is the only thing this is for — would
        otherwise get a silent length mismatch at any view other than the default,
        where :attr:`class_labels` raises for exactly that situation.
        """
        self._structure()
        return self.rows_at(self._view_level)["item_index"].to_numpy().astype(np.intp, copy=False)

    @property
    def item_count(self) -> int:
        """Total number of items in the dataset.

        Returns
        -------
        int
            Count of unique items in the source dataset, regardless of
            how many targets/detections each item contains.
        """
        if self._count == 0:
            self._structure()
        return self._count

    def rows_at(self, level: FactorLevel | Literal["target", "image"]) -> pl.DataFrame:
        """Dataframe rows belonging to a single level.

        Parameters
        ----------
        level : str
            Level to filter to, one of :attr:`levels`.

            .. deprecated::
                ``"image"`` is accepted with a warning and resolves to the
                ``"unit"`` level. Removed in v1.2.0.

        Returns
        -------
        pl.DataFrame
            Rows at the requested level. Columns belonging to other levels are
            present but null.

        Raises
        ------
        ValueError
            When the level is not part of this dataset's schema.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.rows_at("unit").height
        50
        """
        self._structure()
        return self._dataframe.filter(pl.col("level") == self._resolve_level(level))

    @property
    def image_data(self) -> pl.DataFrame:
        """Dataframe containing only image-level rows.

        .. deprecated::
            The name only tells the truth when a dataset item is an image, so it is
            defined for image-based tasks alone and raises for every other task. Use
            ``rows_at(md.item_level)``. Removed in v1.2.0.

        Returns
        -------
        pl.DataFrame
            Image-level metadata, exactly as previous releases returned it — which,
            on image classification, means the *labelled* rows rather than the image
            rows. See Notes.

        Raises
        ------
        ValueError
            When the bound dataset's items are not images.

        Notes
        -----
        Bug-for-bug with v1.1, on purpose. There, a classification dataset had a
        single block of rows and this property returned it; the level restructure
        split that block into image rows and instance rows, and returning the image
        rows here would silently hand existing callers nulls where ``class_label``,
        ``score`` and ``target_index`` used to be. So it still returns the labelled
        rows for a single-target task and the image rows for a multi-target one.
        ``rows_at(md.item_level)`` is the spelling that means image rows on every task.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.rows_at(metadata.item_level).select("item_index", "time_of_day", "weather", "location").head(5)
        shape: (5, 4)
        ┌────────────┬─────────────┬─────────┬──────────┐
        │ item_index ┆ time_of_day ┆ weather ┆ location │
        │ ---        ┆ ---         ┆ ---     ┆ ---      │
        │ i64        ┆ str         ┆ str     ┆ str      │
        ╞════════════╪═════════════╪═════════╪══════════╡
        │ 0          ┆ dawn        ┆ rainy   ┆ suburban │
        │ 1          ┆ day         ┆ rainy   ┆ rural    │
        │ 2          ┆ dawn        ┆ clear   ┆ maritime │
        │ 3          ┆ dusk        ┆ rainy   ┆ maritime │
        │ 4          ┆ dusk        ┆ clear   ┆ suburban │
        └────────────┴─────────────┴─────────┴──────────┘
        """
        warnings.warn(
            "Metadata.image_data is deprecated and will be removed in v1.2.0. Use "
            "Metadata.rows_at(md.item_level) for image rows. Note that on a "
            "single-target task this property returns the labelled rows, not the "
            "image rows, to match what v1.1 returned.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._structure()
        if self._item_level != "unit":
            raise ValueError(
                "Metadata.image_data is only defined for image-based tasks, but this dataset has "
                f"items at the {self._item_level!r} level. "
                'Use Metadata.rows_at("unit") for the image rows, '
                "or Metadata.rows_at(md.item_level) for item-level rows.",
            )
        return self.rows_at(self._item_level if self.multi_target else self._label_level)

    @property
    def target_data(self) -> pl.DataFrame:
        """Dataframe containing only label-level rows.

        .. deprecated::
            One spelling per level does not scale, and this one names a level that no
            longer exists. Use ``rows_at(md.label_level)`` for the rows the labels are
            on, or ``rows_at(md.view)`` for the rows the array accessors project.
            Removed in v1.2.0.

        Returns
        -------
        pl.DataFrame
            Dataframe with label-level metadata. Each row represents a single
            labelled thing with its associated class, score, and bounding box
            information: a detection for object detection, the image itself for
            classification.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.rows_at(metadata.label_level).select("item_index", "target_index", "class_label").head(5)
        shape: (5, 3)
        ┌────────────┬──────────────┬─────────────┐
        │ item_index ┆ target_index ┆ class_label │
        │ ---        ┆ ---          ┆ ---         │
        │ i64        ┆ i64          ┆ i64         │
        ╞════════════╪══════════════╪═════════════╡
        │ 0          ┆ 0            ┆ 0           │
        │ 1          ┆ 0            ┆ 3           │
        │ 1          ┆ 1            ┆ 2           │
        │ 1          ┆ 2            ┆ 1           │
        │ 2          ┆ 0            ┆ 1           │
        └────────────┴──────────────┴─────────────┘
        """
        warnings.warn(
            "Metadata.target_data is deprecated and will be removed in v1.2.0. Use "
            "Metadata.rows_at(md.label_level) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._structure()
        return self.rows_at(self._label_level)

    def get_image_factors(self, image_idx: int) -> dict[str, Any]:
        """Get all factors for a specific image.

        .. deprecated::
            A single row lookup is one dataframe filter, and phrasing it as a method
            per level does not scale past the two levels that happen to exist today.
            Use ``rows_at("unit").filter(pl.col("item_index") == image_idx)``.
            Removed in v1.2.0.

        Parameters
        ----------
        image_idx : int
            Index of the image to retrieve factors for

        Returns
        -------
        dict[str, Any]
            Dictionary mapping factor names to their values for the specified image

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> factors = metadata.get_image_factors(0)
        >>> factors["time_of_day"]
        'dawn'
        >>> factors["weather"]
        'rainy'
        >>> factors["location"]
        'suburban'
        """
        warnings.warn(
            "Metadata.get_image_factors() is deprecated and will be removed in v1.2.0. Use "
            'Metadata.rows_at("unit").filter(pl.col("item_index") == image_idx) instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self._structure()
        row = self.rows_at(self._item_level).filter(pl.col("item_index") == image_idx)
        if row.height == 0:
            raise ValueError(f"No image found with index {image_idx}")
        return row.to_dicts()[0]

    def get_target_factors(self, image_idx: int, target_idx: int) -> dict[str, Any]:
        """Get all factors for a specific target within an item.

        .. deprecated::
            A single row lookup is one dataframe filter, and phrasing it as a method
            per level does not scale past the two levels that happen to exist today.
            Use ``target_data.filter(...)``. Removed in v1.2.0.

        Parameters
        ----------
        image_idx : int
            Index of the item containing the target
        target_idx : int
            Index of the target within the item (0-indexed per item)

        Returns
        -------
        dict[str, Any]
            Dictionary mapping factor names to their values for the specified target

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> factors = metadata.get_target_factors(1, 1)
        >>> factors["item_index"]
        1
        >>> factors["target_index"]
        1
        >>> factors["class_label"]
        2
        """
        warnings.warn(
            "Metadata.get_target_factors() is deprecated and will be removed in v1.2.0. Use "
            'Metadata.rows_at(md.label_level).filter((pl.col("item_index") == image_idx) & '
            '(pl.col("target_index") == target_idx)) instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self._structure()
        rows = self.rows_at(self._label_level)
        row = rows.filter((pl.col("item_index") == image_idx) & (pl.col("target_index") == target_idx))
        if row.height == 0:
            raise ValueError(f"No target found with item_index={image_idx}, target_index={target_idx}")
        return row.to_dicts()[0]

    def _empty_projection(self, dtype: Any) -> NDArray[Any]:
        """Build a no-column projection that still has the view's row count.

        Shaped, not bare. :func:`len`, :attr:`shape` and iteration all describe the
        view's rows, and a dataset can perfectly well have rows and no usable factors;
        returning ``array([])`` there makes those three disagree.
        """
        self._structure()
        return np.empty((self._layout.counts.get(self._view_level, 0), 0), dtype=dtype)

    def _project(self, columns: Sequence[str], dtype: Any) -> NDArray[Any]:
        """Read the named columns off the view's rows as a dense array.

        The one place that answers "which rows does an array-shaped accessor read, and
        what does an empty one look like": the view's rows only — the dataframe also
        holds rows at every other level, which align neither with these factors nor
        with :attr:`class_labels` — and a shaped empty projection when there is nothing
        to select, since selecting no columns leaves a frame with no arrays to stack,
        which older polars cannot convert.

        Callers must have binned first, since :meth:`_bin` replaces ``_dataframe``.
        """
        if not columns:
            return self._empty_projection(dtype)
        return self.rows_at(self._view_level).select(columns).to_numpy().astype(dtype, copy=False)

    def _filter(self, factor: str | tuple[str, Any]) -> bool:
        factor = factor[0] if isinstance(factor, tuple) else factor
        return factor in self.include if self.include else factor not in self.exclude

    def _reset_bins(self, cols: Iterable[str] | None = None) -> None:
        """Drop the companion columns of ``cols`` (or of every factor) and their info.

        Deliberately not guarded on ``_is_binned``. That flag says "there is nothing
        left to process", which the include/exclude setters clear to force a re-bin
        while leaving every companion column in place — so a guard on it would skip
        exactly the calls that exist to remove those columns. :meth:`_bin` skips a
        factor whose companion column is present, so a companion left behind is a
        factor that never re-bins: ``continuous_factor_bins`` assignments are silently
        ignored, and a factor replaced by :meth:`add_factors` keeps the old column's
        digitization while its info reads ``None``.
        """
        columns = set(self._dataframe.columns)
        for col in cols or self._dataframe.columns:
            # Both spellings are checked because a factor that changes type between
            # bins moves from one to the other and would otherwise leave the old behind.
            for companion in {_binned(col), _digitized(col)} & columns:
                self._dataframe.drop_in_place(companion)
                # Dropping the column drops the info describing it. That pairing is the
                # whole invariant: _bin() decides what to process by looking for the
                # column, and factor_info answers by looking in the cache.
                self._factor_cache.pop(col, None)
        self._is_binned = False

    def _unreadable_at(self, level: FactorLevel, view: FactorLevel) -> str | None:
        """Why a factor defined at ``level`` cannot be read from ``view``'s rows, or None.

        Sole arbiter of what enters factor analysis, so the two ways the level graph's
        diamond can put a factor out of reach are decided together and phrased once.

        Parameters
        ----------
        level : str
            Level the factor is defined at.
        view : str
            Level whose rows would be projected.

        Returns
        -------
        str or None
            A reason suitable for a log line, or None when every ``view`` row can read it.
        """
        if not self._levels.propagates_to(level, view):
            return (
                f"{level!r} does not propagate to {view!r} — they are on separate branches of the "
                "level graph, so these rows have no value for it"
            )
        if self._layout.partial_ancestry(level, view):
            return (
                f"not every {view!r} row has a {level!r} ancestor, so the column is partly null "
                f"and cannot be binned; read it at {level!r} instead"
            )
        return None

    def _factor_level(self, name: str) -> FactorLevel:
        """Level a factor is defined at, or the item level when unknown.

        A factor is stored once, at its own level, and read from descendant rows by
        propagation. That "once" is an enforced invariant rather than a convention:
        :meth:`StructuredData.__post_init__` rejects a name declared at two levels, and
        :meth:`_register_factor_levels` clears a name from every level before adding it
        to one. So at most one level matches, and ``highest`` is picking from a
        one-element list — kept because it is the safe answer if the invariant is ever
        relaxed, values being visible from a coarse level's descendants but not the
        other way round.
        """
        levels: list[FactorLevel] = [level for level, names in self._factors_by_level.items() if name in names]
        return self._levels.highest(levels) if levels else self._item_level

    def _resolve_level(self, level: FactorLevel | Literal["target", "image"], stacklevel: int = 4) -> FactorLevel:
        """Validate a caller-supplied level name, translating any retired spelling.

        Sole entry point for a level name that came from a caller: :meth:`rows_at` and
        :meth:`add_factors` both route through here, so the two public spellings of the
        deprecation cannot drift apart. Everything internal calls
        ``self._levels.validate`` directly, which knows only about levels that exist.

        Which retired names resolve is declared per task, by the structurer's
        ``legacy_level_aliases``. A task that never used a retired spelling has no
        entry for it, so the name falls through to ``validate`` and is rejected as
        unknown in the same words as any other name that is not a level.

        Parameters
        ----------
        level : str
            Level name as the caller spelled it.
        stacklevel : int, default 4
            Frames between the warning and the user's line. The warning itself is
            raised one level down, in :func:`_resolve_legacy_level`, so the default
            counts their call, the public method, this helper, and that helper — right
            for every caller that reaches this directly. :meth:`_resolve_requested_level`
            passes one more, since it sits between :meth:`add_factors` and this.

        Returns
        -------
        str
            The level the caller named, or the level a retired spelling now resolves to.

        Raises
        ------
        ValueError
            When the level is not part of this dataset's schema.
        """
        resolved = _resolve_legacy_level(
            level, self._structurer.legacy_level_aliases, stacklevel, unit_type=self._structurer.unit_type
        )
        try:
            return self._levels.validate(resolved)
        except ValueError as exc:
            # FactorLevelSchema knows the level vocabulary but not the medium, so the
            # unit-type clause is added here rather than pushed down into validate().
            raise ValueError(f"{exc} (this dataset's units are {self._structurer.unit_type}s)") from None

    def _warn_level_rename(self) -> None:
        """Announce the ``FactorInfo.level`` rename, once per instance.

        ``FactorInfo.level`` used to report a name that is no longer a level — the
        structurer's ``legacy_level_aliases`` says which — and now reports the level's
        real name. Nothing can intercept an
        ``info.level == "target"`` comparison to warn at the point it silently
        turns false, so the warning is raised where the ``FactorInfo`` objects are
        handed to a caller instead. Only a task that declares a retired spelling
        warns; one that never used one has nothing to announce.

        Called directly by each such handout point rather than from
        :attr:`_factor_info`, for two reasons: the once-per-instance budget must be
        spent on a call the user actually made, not on an internal read from
        ``factor_data``; and ``stacklevel=3`` then points at the user's line from
        every one of them.
        """
        aliases = self._structurer.legacy_level_aliases
        if self._warned_level_rename or not aliases:
            return
        self._warned_level_rename = True
        retired = ", ".join(repr(name) for name in aliases)
        now = ", ".join(repr(level) for level in dict.fromkeys(aliases.values()))
        warnings.warn(
            f"FactorInfo.level now reports {now} for {self._structurer.task} rows and no longer reports "
            f"{retired}. Comparisons against {retired} will silently fail; compare against {now}.",
            DeprecationWarning,
            stacklevel=3,
        )

    def _combined_length(self) -> int | None:
        """Length an *inferred* v1.1 ``"combined"`` array has here, or None when there is none.

        Stricter than what :meth:`_resolve_combined` accepts, on purpose. Inference is a
        guess made from a length alone, so it is offered only where the guess is worth
        making: a two-level schema over a multi-target task. A classification dataset
        carries one label per image, so its combined length is merely twice the image
        count — far more likely a caller's mistake than a deliberate two-level array, and
        v1.1 did not infer it there either. An explicit ``level="combined"`` is the caller
        asserting the layout rather than the code guessing it, so that spelling is refused
        only where the split is structurally impossible.

        A schema with a third level, as tracking's frames and tracks are, has no combined
        length under either spelling.
        """
        if len(self._levels) != 2 or not self.multi_target:
            return None
        counts = self._layout.counts
        return counts.get(self._item_level, 0) + counts.get(self._label_level, 0)

    def _reject_unmatched_length(self, factor_len: int, combined_len: int | None) -> NoReturn:
        """Report a factor length that names neither a level nor a combined array."""
        counts = self._layout.counts
        expected = ", ".join(f"{level}={counts.get(level, 0)}" for level in self._levels)
        if combined_len is not None:
            expected += f", {self._item_level}+{self._label_level}={combined_len}"
        raise ShapeMismatchError(
            "The lists/arrays in the provided factors have a different length "
            f"than any level of the current metadata. Expected one of ({expected}), got {factor_len}.",
        )

    def _infer_factor_level(self, factor: Array1D[Any]) -> FactorLevel | Literal["combined"]:
        """Infer the destination of a single factor array from its length.

        A level's own row count wins over the combined length, so a factor that could be
        read either way lands on a level rather than being split.

        Raises
        ------
        ShapeMismatchError
            When the length matches no level and no combined length.
        """
        factor_len = len(factor)
        counts = self._layout.counts
        matches: list[FactorLevel] = [level for level in self._levels if counts.get(level, 0) == factor_len]

        if not matches:
            combined_len = self._combined_length()
            if combined_len is not None and factor_len == combined_len:
                return "combined"
            self._reject_unmatched_length(factor_len, combined_len)
        if len(matches) > 1:
            # Levels routinely coincide in size — a fully labelled classification dataset has
            # one label per image, and so does an object detection dataset with one
            # detection per image — so this cannot raise; that would break code that works on
            # every other dataset. The coarsest level wins, matching what add_factors has
            # always done.
            chosen = self._levels.highest(matches)
            if not all(self._rows_correspond(chosen, other) for other in matches if other != chosen):
                warnings.warn(
                    f"A factor length of {factor_len} matches the {matches} levels, which currently have "
                    f"the same number of rows but do not correspond one-to-one; storing it at the "
                    f"{chosen!r} level. Pass an explicit level= to add_factors to choose.",
                    UserWarning,
                    # caller -> add_factors -> _resolve_factor_levels -> here. One frame
                    # shallower than _warn_inferred_combined, which _resolve_destinations
                    # reaches from _resolve_factor_levels rather than from the loop.
                    # test_ambiguity_warning_points_at_the_caller pins this.
                    stacklevel=4,
                )
            return chosen
        return matches[0]

    def _rows_correspond(self, coarse: FactorLevel, fine: FactorLevel) -> bool:
        """Whether each ``fine`` row has its own ``coarse`` row, in the same order.

        When it does, the two levels are interchangeable as a destination: the values
        land on the same target rows either way, so there is nothing for the caller to
        disambiguate. When it does not — three detections spread 0/1/2 across three
        images — the choice changes the data and has to be surfaced.
        """
        for level, size, ancestor_pos in self._layout.blocks:
            if level != fine:
                continue
            positions = ancestor_pos.get(coarse)
            return positions is not None and np.array_equal(positions, np.arange(size, dtype=np.intp))
        return False

    def _validate_factor_lengths(self, factors: Mapping[str, Array1D[Any]], level: FactorLevel) -> None:
        """Validate that factor lengths match the specified level's row count."""
        expected_len = self._layout.counts.get(level, 0)
        mismatched = {k: len(v) for k, v in factors.items() if len(v) != expected_len}
        if mismatched:
            raise ShapeMismatchError(
                f"All {level}-level factors must have length {expected_len} ({level} row count); got {mismatched}.",
            )

    def _split_source_index(self, source_index: Sequence[SourceIndex]) -> dict[FactorLevel, NDArray[np.intp]]:
        """Group source-index positions by the level each entry describes.

        :func:`~dataeval.core.compute_stats` labels every value it returns with a
        :class:`~dataeval.types.SourceIndex`, so a factor array spanning several levels
        is split by those labels rather than by a positional convention.

        Parameters
        ----------
        source_index : Sequence[SourceIndex]
            One entry per value in each factor array being added.

        Returns
        -------
        dict[str, NDArray[np.intp]]
            For each level the source index covers, the positions within it that
            describe that level's rows, ordered as that level's rows are.

        Raises
        ------
        ValueError
            When the source index carries per-channel entries, which have no
            single-column representation, or when the two kinds of entry cannot be
            told apart because items and labels share a level.
        ShapeMismatchError
            When a level's entry count does not match that level's row count.

        Notes
        -----
        A :class:`~dataeval.types.SourceIndex` distinguishes exactly two kinds of
        value — one per dataset item, where ``target`` is None, and one per label —
        so it can address :attr:`item_level` and :attr:`label_level` and no others.
        That is a property of the type, not of this method: a schema with a third level
        between them — multi-object tracking's ``unit`` level sits between ``sequence``
        (item) and ``instance`` (label) — has no representable form here, and per-frame
        values (e.g. from :func:`~dataeval.core.compute_stats` run per frame) have to be
        added with an explicit ``level="unit"`` instead. Widening this means widening
        :class:`~dataeval.types.SourceIndex` first.
        """
        # Parsing — sort order, per-channel rejection, duplicate-row rejection — is shared
        # with the dataset-free constructor rather than reimplemented, so the two spellings
        # of "place these values by their labels" cannot drift apart. Only the parts that
        # need this metadata's own rows to check against live here.
        rows = SourceIndexRows.parse(source_index)

        # The two kinds are told apart only by which level they land on, so a schema whose
        # items and labels coincide merges them into one over-long group, which then
        # surfaces as a row-count mismatch. Say what actually happened instead. Read off the
        # parse, which has already separated the two kinds, rather than rescanning.
        if self._item_level == self._label_level and rows.spans_two_levels:
            raise ValueError(
                f"source_index mixes per-item entries (target=None) with per-label entries, but this "
                f"metadata's items and its labels are both at the {self._item_level!r} level, so the "
                "two cannot be placed apart. Add each kind in its own call with an explicit level=.",
            )

        candidates: tuple[tuple[FactorLevel, NDArray[np.intp], NDArray[np.intp], NDArray[np.intp] | None], ...] = (
            (self._item_level, rows.item_positions, rows.item_ids, None),
            (self._label_level, rows.label_positions, rows.label_items, rows.label_targets),
        )
        order: dict[FactorLevel, NDArray[np.intp]] = {}
        for level, positions, items, targets in candidates:
            if len(positions) == 0:
                continue
            self._reject_unmatched_rows(level, items, targets)
            order[level] = positions
        return order

    def _reject_unmatched_rows(
        self,
        level: FactorLevel,
        items: NDArray[np.intp],
        targets: NDArray[np.intp] | None,
    ) -> None:
        """Reject a source index whose entries do not name this level's rows exactly.

        Counting the entries is not enough. An index that names one row twice and another
        not at all has the right count and every value lands somewhere, just not where the
        caller said — the failure mode a source index exists to prevent. Matching the keys
        catches it, and costs one comparison per row.

        Raises
        ------
        ShapeMismatchError
            When the entry count does not match the level's row count.
        ValueError
            When the counts match but the entries name different rows.
        """
        counts = self._layout.counts
        expected_len = counts.get(level, 0)
        if len(items) != expected_len:
            raise ShapeMismatchError(
                f"source_index describes {len(items)} {level}-level values but the "
                f"metadata has {expected_len} {level} rows. Row counts are {dict(counts)}; "
                "note that a dataset item whose target was empty contributes no rows, so "
                "Metadata.item_indices, not range(item_count), lists the items that have them.",
            )

        # The two key columns, not rows_at's whole frame: that widens with every factor the
        # caller has already added, and none of them are compared here.
        frame = (
            self._dataframe
            .lazy()
            .filter(pl.col("level") == level)
            # Null marks a row that is not a target, and -1 stands in for it exactly as it
            # does in SourceIndexRows. Left null, an integer column comes back from
            # to_numpy() as float NaN: the comparison below still reports the mismatch,
            # but formatting it then raises "cannot convert float NaN to integer" in place
            # of the error the caller needs.
            .select("item_index", pl.col("target_index").fill_null(-1))
            .collect()
        )
        actual_items = frame["item_index"].to_numpy()
        actual_targets = None if targets is None else frame["target_index"].to_numpy()
        mismatched = actual_items != items
        if actual_targets is not None:
            mismatched |= actual_targets != targets
        if np.any(mismatched):
            # Formatted from the first few alone, since only the first few are named: a
            # rejected million-row index must not spend longer building its error than the
            # call would have taken to succeed.
            worst = np.flatnonzero(mismatched)[:5]
            named = [(int(items[i]), None if targets is None else int(targets[i])) for i in worst]
            expected = [
                (
                    int(actual_items[i]),
                    None if actual_targets is None or actual_targets[i] < 0 else int(actual_targets[i]),
                )
                for i in worst
            ]
            raise ValueError(
                f"source_index names {level}-level rows this metadata does not have. It has the right "
                f"number of {level} entries, but {int(np.count_nonzero(mismatched))} of them name {named} "
                f"where the metadata's rows are {expected}. Every row at a level must be named exactly once.",
            )

    def has_targets(self) -> bool:
        """Check if the source dataset has targets.

        .. deprecated::
            Renamed for what it actually reports. Use :attr:`multi_target`.
            Removed in v1.2.0.

        Returns
        -------
        bool
            True for object detection, False for image classification — unchanged
            from v1.1.

        Notes
        -----
        No expression over the row counts reproduces this, which is why the
        replacement is a property and not one. ``level_counts["instance"] !=
        level_counts["unit"]`` is False for a detection dataset with one detection
        per image and True for a classification dataset with an unlabeled item, so it
        gets the answer wrong in both directions. Nor is it ``label_level !=
        item_level``: every task now names its labelled level ``instance`` and its
        item level ``unit``, so that comparison is true even for classification.
        """
        warnings.warn(
            "Metadata.has_targets() is deprecated and will be removed in v1.2.0. Use Metadata.multi_target instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.multi_target

    def _build_factors(self) -> None:
        """Build the set of factor names visible at the current view."""
        if not self._is_structured:
            self._factors = set()
            return

        view = self._view_level
        # ``target_factors_only`` is the retired spelling and keeps its exemption for
        # single-target tasks; ``inherited`` is the current one and has none. Either
        # narrowing to the view's own factors is enough.
        # Read off the structurer rather than through the ``multi_target`` property,
        # which would re-enter _structure() from inside the tail of _structure().
        legacy_narrowing = self._target_factors_only and self._structurer.multi_target
        if not self._inherited or legacy_narrowing:
            names = set(self._factors_by_level.get(view, ()))
        else:
            names = {name for level_names in self._factors_by_level.values() for name in level_names}

        # A factor the view's rows cannot all read stays in the dataframe but out of factor
        # analysis, for either of two reasons the level graph's diamond makes real:
        #
        # - Off the branch entirely: ``unit`` and ``track`` are siblings, so neither
        #   propagates to the other and a per-frame factor has no value on a track row.
        # - On the branch but not for every row: a detection no tracker linked has no track
        #   ancestor, so a per-track factor is null there. Binning cannot represent that —
        #   discretizing sorts the values, and None does not order against a float — so the
        #   factor is excluded here rather than left to fail mid-analysis. It is still read
        #   in full at its own level, via ``md.at("track")``.
        visible: set[str] = set()
        for name in names:
            level = self._factor_level(name)
            unreadable = self._unreadable_at(level, view)
            if unreadable is None:
                visible.add(name)
            else:
                _logger.debug(
                    "Factor %r, defined at level %r, is excluded from factor analysis at view level %r: %s",
                    name,
                    level,
                    view,
                    unreadable,
                )

        # Purely derived: nothing is carried over from the outgoing set, because a
        # factor's binning lives in ``_factor_cache`` and survives the rebuild there.
        self._factors = {
            k for k in visible if not isinstance(self._dataframe.schema.get(k), pl.List | pl.Struct | pl.Array)
        }

    def _structure(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if self._is_structured:
            return

        if self._dataset is None:
            raise NotFittedError("No dataset bound. Call bind() first.")

        structurer = select_structurer(self._dataset, self._task)
        datum_count = len(self._dataset) if isinstance(self._dataset, Sized) else 0
        _logger.info("Processing metadata for %d dataset items using %r", datum_count, structurer)

        data = structurer.build(self._dataset, progress_callback=progress_callback)

        unique_labels = np.unique(data.class_labels) if len(data.class_labels) else np.array([], dtype=np.intp)
        self._index2label = _build_index2label(self._dataset.metadata.get("index2label", None), unique_labels)
        self._adopt(structurer, data)

        _logger.debug(
            "Metadata structured as %s: %s rows per level, %d factors, %d dropped",
            structurer.task,
            self._layout.counts,
            sum(len(names) for names in self._factors_by_level.values()),
            sum(len(v) for v in self._dropped_factors.values()),
        )

    def _adopt(self, structurer: Structurer, data: StructuredData) -> None:
        """Take on everything a :class:`StructuredData` describes.

        The dataset path and the factors-only path build their bundle very
        differently and then adopt it identically, so the field list lives here once:
        a field added to :class:`StructuredData` is wired up in one place rather than
        remembered in two. Each caller keeps only what is genuinely its own — where
        ``index2label`` comes from, and the dataset item count.
        """
        self._structurer = structurer
        self._layout = data.layout
        self._factors_by_level = {level: set(names) for level, names in data.factors.items()}
        # A declared level that produced no factors still has to be a key, so that
        # _factor_level and _build_factors can look any level up unconditionally.
        for level in self._levels:
            self._factors_by_level.setdefault(level, set())

        # A view chosen at construction is resolved here, at the first moment there is a
        # schema to resolve it against — and before _build_factors below reads it. It
        # goes through _resolve_level rather than validate() so that a retired spelling
        # passed to the constructor deprecates rather than raises. The warning points at
        # whatever first triggered structuring, not at the constructor call, because the
        # constructor frame is long gone by then.
        if self._view is not None:
            self._view = self._resolve_level(self._view, stacklevel=3)

        self._raw = data.raw
        self._class_labels = data.class_labels
        # ``data.item_indices`` is deliberately not stored: it is the label level's
        # ``item_index`` column, which every block already carries, and :attr:`item_indices`
        # reads it from the view's own rows so that it cannot disagree with factor_data.
        self._dropped_factors = {name: list(reasons) for name, reasons in data.dropped_factors.items()}
        self._dataframe = pl.DataFrame(data.to_rows())
        self._is_structured = True

        # Reuse the canonical factor builder so List/Struct columns are filtered out
        # identically on both paths.
        self._build_factors()

    def _add_level_column(
        self,
        df: pl.DataFrame,
        col_name: str,
        values: NDArray[Any],
        level: FactorLevel,
    ) -> pl.DataFrame:
        """Add a column defined at ``level``, propagated down and null above.

        A binned column travels exactly as the raw factor it was derived from: it
        is written once at the factor's own level and gathered onto descendant rows,
        which is what keeps a factor's bin assignment identical however it is read.
        :meth:`RowLayout.expand` is the same gather
        the structuring layer applies to raw values,
        so there is one propagation rule rather than two.

        Parameters
        ----------
        df : pl.DataFrame
            Frame to add the column to.
        col_name : str
            Name of the column to write.
        values : NDArray[np.int64]
            One bin or category index per row at ``level``, in that level's row order.
        level : str
            Level the values are defined at.
        """
        # Stated rather than inferred: every companion column is a bin or category
        # index, and inferring the dtype from the expanded column would read it off a
        # list that may lead with nulls.
        expanded = self._layout.expand(values, level)
        return df.with_columns(pl.Series(name=col_name, values=expanded, dtype=pl.Int64))

    def _classify_factor(
        self,
        col: str,
        data: NDArray,
        factor_bins: Mapping[str, int | Sequence[float]],
    ) -> tuple[NDArray[np.int64], FactorInfo]:
        """Bin or digitize one factor's native values, and say which was done.

        ``data`` holds the factor's values at its own level, one per entity, so
        every decision made here — the bin edges, the number of bins, the
        continuous/discrete verdict — is read off the factor's true distribution.
        """
        if col in factor_bins:
            return digitize_data(data, factor_bins[col]).astype(np.int64), FactorInfo("continuous", is_binned=True)

        _, ordinal = np.unique(data, return_inverse=True)
        if not np.issubdtype(data.dtype, np.number):
            return ordinal.astype(np.int64), FactorInfo("categorical", is_digitized=True)
        # No de-duplication argument: ``data`` carries one value per entity at the
        # factor's own level, so there are no propagated repeats for is_continuous
        # to mistake for discrete support.
        if is_continuous(data):
            _logger.warning(
                f"A user defined binning was not provided for {col}. "
                f"Using the {self.auto_bin_method} method to discretize the data. "
                "It is recommended that the user rerun and supply the desired "
                "bins using the continuous_factor_bins parameter.",
            )
            return bin_data(data, self.auto_bin_method).astype(np.int64), FactorInfo("continuous", is_binned=True)
        # Digitize discrete numeric factors so that factor_data always
        # contains non-negative integers (required by np.bincount in
        # downstream bias evaluators).
        return ordinal.astype(np.int64), FactorInfo("discrete", is_digitized=True)

    def _process_factor(
        self,
        df: pl.DataFrame,
        col: str,
        data: NDArray,
        factor_bins: Mapping[str, int | Sequence[float]],
        level: FactorLevel,
    ) -> tuple[pl.DataFrame, FactorInfo]:
        """Write one factor's companion column and return the info describing it.

        Which of the two companion spellings the column takes is not a fifth decision:
        :func:`_to_col` reads it off the :class:`FactorInfo` that
        :meth:`_classify_factor` just built, so the name and the flags cannot disagree.
        """
        values, info = self._classify_factor(col, data, factor_bins)
        info.level = level
        return self._add_level_column(df, _to_col(col, info), values, level), info

    def _bin(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Populate factor info and bin non-categorical factors.

        Every factor is binned at its own level — the level whose rows hold one
        value per entity — and the resulting column is then propagated downwards.
        Binning at the view instead would read each factor's distribution
        through however many descendants each entity happens to have, which moves
        the bin edges, changes how many bins survive the low-count collapse in
        the binner, and drops entities with no
        descendants from the binner's input entirely.
        """
        if self._is_binned:
            return

        factor_info: dict[str, FactorInfo] = {}
        df = self.dataframe.clone()
        factor_bins = self.continuous_factor_bins

        # Check for invalid keys
        invalid_keys = set(factor_bins.keys()) - set(df.columns)
        if invalid_keys:
            _logger.warning(
                f"The keys - {invalid_keys} - are present in the `continuous_factor_bins` dictionary "
                "but are not columns in the metadata DataFrame. Unknown keys will be ignored.",
            )

        column_set = set(df.columns)
        factors_to_process = [col for col in self.factor_names if not {_binned(col), _digitized(col)} & column_set]
        total_factors = len(factors_to_process)

        # Resolved up front, and one filter per level rather than one per factor: a
        # dataset has a handful of levels and may carry hundreds of factors.
        levels: dict[str, FactorLevel] = {col: self._factor_level(col) for col in factors_to_process}
        native_rows: dict[FactorLevel, pl.DataFrame] = {level: self.rows_at(level) for level in set(levels.values())}

        for i, col in enumerate(factors_to_process):
            level = levels[col]
            data = native_rows[level][col].to_numpy()
            df, info = self._process_factor(df, col, data, factor_bins, level)
            factor_info[col] = info

            if progress_callback:
                progress_callback(i + 1, total=total_factors)

        # Store the results
        self._dataframe = df
        self._factor_cache.update(factor_info)
        self._is_binned = True

    def _resolve_factor_name(self, name: str, taken: set[str], overwrite: bool, append_string: str) -> str:
        """Pick the dataframe column a new factor should be written to.

        Reserved columns are load-bearing — ``level`` in particular drives every level
        filter — so a colliding factor is renamed the same way :meth:`_structure` renames
        dataset metadata keys rather than allowed to overwrite one.
        """
        safe = safe_column_name(name)
        if safe != name:
            _logger.warning(
                f"The factor name '{name}' collides with a reserved metadata column and has been "
                f"stored as '{safe}' instead.",
            )

        if safe not in taken or overwrite:
            return safe

        candidate = f"{safe}{append_string}"
        suffix = 2
        while candidate in taken:
            candidate = f"{safe}{append_string}_{suffix}"
            suffix += 1
        return candidate

    def _register_factor_levels(self, factors: Sequence[tuple[str, FactorLevel]]) -> None:
        """Record which level each newly added factor is defined at.

        A factor is stored once, at one level, so stale membership is cleared before
        re-registering: overwriting at a new level must not leave it claimed by the old one.
        Any cached binning is dropped for the same reason — the name now describes
        different values, so info computed from the old ones does not carry.
        """
        for name, level in factors:
            for names in self._factors_by_level.values():
                names.discard(name)
            self._factors_by_level.setdefault(level, set()).add(name)
            self._factor_cache.pop(name, None)

    @staticmethod
    def _split_by_dimensionality(
        factors: Mapping[str, Array1D[Any]],
    ) -> tuple[dict[str, NDArray[Any]], list[str]]:
        """Separate the factors with a single-column form from those without one.

        A column vector has one: ``(N, 1)`` holds one value per row and is flattened to
        it. That is what a dataframe column or a ``reshape(-1, 1)`` pipeline produces, so
        rejecting it would drop real data over a shape carrying no extra information. Only
        an array that is genuinely several values per row — a histogram, a percentile
        vector, a centre coordinate — has nowhere to go.

        A *leading* singleton axis is left alone rather than flattened, and a 1-D array is
        passed through untouched however short it is. Both guard the same edge: on a
        one-row dataset ``(1, K)`` reads equally as one row of K values and as K rows of
        one value, and the first is what :func:`~dataeval.core.compute_stats` emits — so
        flattening would import ``center`` and ``histogram`` as K rows of data on the very
        dataset shape where they least resemble a real factor, while a blanket squeeze
        would additionally collapse a legitimate one-row ``mean`` to a scalar and drop it.

        Pure: it decides what will be dropped without recording it, so that
        :meth:`add_factors` can still abandon the whole call on a later validation
        failure without having already written to :attr:`dropped_factors`.
        """
        arrays = {name: _flatten_column_vector(as_numpy(values)) for name, values in factors.items()}
        kept = {name: values for name, values in arrays.items() if values.ndim == 1}
        return kept, [name for name in arrays if name not in kept]

    def _record_multidimensional(self, names: Sequence[str]) -> None:
        """Warn about factors that have no single-column form and record why they were dropped."""
        if not names:
            return
        _logger.warning(
            "Skipping multi-dimensional factors %s; a factor must be a 1-D array. "
            "Reduce vector-valued statistics to one value per row before adding them.",
            sorted(names),
        )
        for name in names:
            reasons = self._dropped_factors.setdefault(name, [])
            if "multi_dimensional" not in reasons:
                reasons.append("multi_dimensional")

    def _record_vacuous(self, names: Sequence[str]) -> None:
        """Record level splits that were discarded for holding no values at their level.

        Reported rather than silent: the column is genuinely unusable — it cannot be
        binned and carries nothing — but its absence is still a surprise to code that
        expected the split to produce both halves, and :attr:`dropped_factors` is where
        this metadata says what it did not keep and why.
        """
        if not names:
            return
        _logger.info(
            "Skipping level splits %s; the factor has no value at that level, so the "
            "column would be entirely null. Its other level(s) were kept.",
            sorted(names),
        )
        for name in names:
            reasons = self._dropped_factors.setdefault(name, [])
            if "no_values_at_level" not in reasons:
                reasons.append("no_values_at_level")

    def _resolve_by_source_index(
        self,
        factors: Mapping[str, NDArray[Any]],
        source_index: Sequence[SourceIndex],
    ) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
        """Place every value by its source-index label rather than by its position.

        Parameters
        ----------
        factors : Mapping[str, NDArray[Any]]
            One value per source-index entry, per factor.
        source_index : Sequence[SourceIndex]
            Label for each value.

        Returns
        -------
        tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]
            The columns to write, and any level splits discarded for holding no values.
        """
        _reject_length_mismatch(factors, source_index)
        return self._place(factors, self._split_source_index(source_index))

    def _place(
        self,
        factors: Mapping[str, NDArray[Any]],
        positions_by_level: Mapping[FactorLevel, NDArray[np.intp]],
        qualify: bool = False,
    ) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
        """Gather each factor onto the rows its positions name, and name the columns.

        Sole producer of the ``<level>_<name>`` rule, so every way of placing values by
        label — an explicit source index, and the retired ``"combined"`` spelling that
        defers to it — names its columns identically.

        Parameters
        ----------
        factors : Mapping[str, NDArray[Any]]
            One value per position, per factor.
        positions_by_level : Mapping[str, NDArray[np.intp]]
            For each level being written, the positions within each factor array that hold
            that level's rows, ordered as those rows are.
        qualify : bool, default False
            Force the ``<level>_`` prefix even where only one level is being written.
            Values spanning several levels are always prefixed — a single column cannot
            hold a value per image *and* a value per instance, and each half is a distinct
            measurement anyway. Pass True where the naming was promised to the caller
            independently of what the values turned out to cover, as ``level="combined"``
            promises it, so that a dataset whose label level happens to have no rows still
            names its columns the documented way.

            Also suppresses the vacuous-split drop below, for the same reason and to the
            same end: a caller told exactly which columns it will get must get them, even
            where one of them turns out to hold nothing.

        Returns
        -------
        tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]
            The columns to write, and the names of any level splits discarded for holding
            no values — for the caller to record, since this method stays pure.
        """
        prefixed = qualify or len(positions_by_level) > 1
        placed: list[tuple[str, FactorLevel, NDArray[Any]]] = []
        vacuous: list[str] = []
        for name, values in factors.items():
            columns: list[tuple[str, FactorLevel, NDArray[Any]]] = [
                (f"{factor_level}_{name}" if prefixed else name, factor_level, values[positions])
                for factor_level, positions in positions_by_level.items()
            ]
            if qualify:
                placed.extend(columns)
                continue
            kept, dropped = _drop_vacuous_splits(columns)
            placed.extend(kept)
            vacuous.extend(dropped)
        return placed, vacuous

    def _resolve_requested_level(
        self,
        level: FactorLevel | Literal["auto", "target", "combined", "image"],
        source_index: Sequence[SourceIndex] | None,
    ) -> FactorLevel | Literal["combined"] | None:
        """Turn ``add_factors``' ``level=`` argument into a destination, or None to infer.

        Both retired spellings a v1.1 caller can still pass are handled here —
        ``"target"``, which named a level that has since been renamed, and
        ``"combined"``, which never named a level at all — so that the vocabulary of
        retired names lives in one place and every warning about one is raised at the
        same depth below the user's call.

        Raises
        ------
        ValueError
            When both a level and a source index are given, or when the level is not
            part of this dataset's schema.
        """
        if source_index is not None and level != "auto":
            raise ValueError("`level` and `source_index` are mutually exclusive; source_index sets the level.")
        if level == "auto":
            return None
        if level == "combined":
            warnings.warn(
                f"level='combined' is deprecated and will be removed in v1.2.0. It is not a level "
                f"name; it described an array ordered the way compute_stats emits one — by "
                f"(item, target), each item's {self._item_level}-level value ahead of that item's "
                f"{self._label_level}-level ones. Pass source_index= from compute_stats instead, "
                "which labels each value with what it describes. Until then the array is split "
                f"into '{self._item_level}_<name>' and '{self._label_level}_<name>' factors.",
                DeprecationWarning,
                stacklevel=3,
            )
            return "combined"
        # One frame deeper than the other callers of _resolve_level, which reach it
        # straight from the public method.
        return self._resolve_level(level, stacklevel=5)

    def _resolve_combined(
        self,
        factors: Mapping[str, NDArray[Any]],
    ) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
        """Split a v1.1 ``"combined"`` array into one factor per level.

        ``"combined"`` was never a level. It was v1.1's name for an array ordered by
        ``(item, target)`` — each item's item-level value ahead of that item's label-level
        ones, exactly as :func:`~dataeval.core.compute_stats` emits them — which is what
        `source_index` replaces with an explicit label per value.

        The order is interleaved, *not* one item-level block followed by one label-level
        block. The two readings agree on nothing beyond the first value, so splitting
        positionally silently scatters every value onto the wrong row. Ranking the rows in
        that order and deferring the gather to :meth:`_place` keeps the deprecated
        spelling and its replacement placing identical data, and keeps the naming rule in
        one place rather than in two implementations that can drift apart.

        Raises
        ------
        ValueError
            When items and labels sit at the same level, where the split is ambiguous.
        ShapeMismatchError
            When a factor is not as long as the two levels' rows combined.
        """
        item_level, label_level = self._item_level, self._label_level
        if item_level == label_level:
            raise ValueError(
                f"level='combined' describes two levels, but this metadata's items and its labels "
                f"are both at the {item_level!r} level, so there is no split to make. Add each "
                "level's values in its own call, or pass source_index= to place them by label.",
            )
        if len(self._levels) != 2:
            # Two levels is not incidental to "combined": it was v1.1's name for an array
            # over the whole dataframe, and v1.1 had no schema with a third level. On a
            # schema that does — tracking puts ``image`` and ``track`` between ``sequence``
            # and ``instance`` — splitting item/label still type-checks and still produces
            # a plausible-looking pair of factors, while silently describing none of the
            # rows in between. Refuse rather than half-cover the dataframe.
            raise ValueError(
                f"level='combined' describes an array over exactly two levels, but this metadata "
                f"has {list(self._levels)}. An array of {item_level}-level values interleaved with "
                f"{label_level}-level ones would say nothing about the rows at the levels between "
                "them, so there is no array it can name here. Pass source_index= to place values by "
                "label, or level= to name the one level they belong to.",
            )

        counts = self._layout.counts
        head, tail = counts.get(item_level, 0), counts.get(label_level, 0)
        mismatched = {name: len(values) for name, values in factors.items() if len(values) != head + tail}
        if mismatched:
            raise ShapeMismatchError(
                f"All combined-level factors must have length {head + tail} "
                f"({item_level} count {head} + {label_level} count {tail}); got {mismatched}.",
            )
        # qualify=True rather than letting the data decide: the deprecation warning has
        # already promised '<item_level>_<name>' and '<label_level>_<name>', and a dataset
        # whose label level happens to have no rows — or whose values at one of them are
        # all null — must not silently rename or remove the columns out from under a
        # caller following that warning. It therefore keeps both halves unconditionally.
        return self._place(factors, self._combined_positions(), qualify=True)

    def _combined_positions(self) -> dict[FactorLevel, NDArray[np.intp]]:
        """Position within a v1.1 ``"combined"`` array of each row it described.

        The array was ordered by ``(item, target)`` with an item's own value ahead of that
        item's labels, so each row's position is simply its rank in that order — read off
        the two key columns rather than rebuilt as :class:`~dataeval.types.SourceIndex`
        objects and re-parsed, which would allocate one per row of the dataframe and sort
        it three more times to arrive back here.

        Every row of the dataframe is ranked, which is the whole of it: :meth:`_resolve_combined`
        has already established that the schema is exactly the item level and the label
        level, so there are no rows in between for a combined array not to describe.
        """
        frame = (
            self._dataframe
            .lazy()
            .select(
                "level",
                "item_index",
                # Null marks a per-item row. -1 both stands in for it and, sorting below
                # every real target, puts an item's value ahead of that item's labels.
                pl.col("target_index").fill_null(-1),
            )
            .collect()
        )
        order = np.lexsort((frame["target_index"].to_numpy(), frame["item_index"].to_numpy()))
        rank = np.empty(len(order), dtype=np.intp)
        rank[order] = np.arange(len(order), dtype=np.intp)

        is_item = frame["level"].to_numpy() == self._item_level
        return {self._item_level: rank[is_item], self._label_level: rank[~is_item]}

    def _resolve_factor_levels(
        self,
        factors: Mapping[str, NDArray[Any]],
        level: FactorLevel | Literal["combined"] | None,
        source_index: Sequence[SourceIndex] | None,
    ) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
        """Work out the level and values of every column ``add_factors`` is about to write.

        Returns one entry per column as ``(name, level, values)``, where ``values`` holds
        one value per row at ``level``. ``level`` is None when it is to be inferred per
        factor. Both a source index spanning several levels and the retired
        ``"combined"`` spelling yield several columns per factor, named ``<level>_<name>``.

        Alongside them, the names of any level splits discarded for holding no values, for
        the caller to record. Only the multi-level paths can produce any: an explicit
        `level` writes exactly the columns it was asked for, whatever they hold.

        The two arguments are mutually exclusive; :meth:`add_factors` rejects the
        combination before resolving either, so that the error names what the caller
        passed rather than what it resolved to.
        """
        if source_index is not None:
            return self._resolve_by_source_index(factors, source_index)

        if level == "combined":
            return self._resolve_combined(factors)

        if level is not None:
            self._validate_factor_lengths(factors, level)
            return [(name, level, values) for name, values in factors.items()], []

        # Each factor is inferred independently, so a mapping mixing unit-level and
        # instance-level arrays can be added in a single call. Written as a loop rather than
        # a comprehension so that the stacklevel of the ambiguity warning _infer_factor_level
        # may raise counts the same number of frames on every supported Python.
        destinations: list[tuple[str, FactorLevel | Literal["combined"], NDArray[Any]]] = []
        for name, values in factors.items():
            destinations.append((name, self._infer_factor_level(values), values))
        return self._resolve_destinations(destinations)

    def _resolve_destinations(
        self,
        destinations: Sequence[tuple[str, FactorLevel | Literal["combined"], NDArray[Any]]],
    ) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
        """Turn inferred destinations into columns, batching the combined ones.

        Batched rather than resolved one at a time because :meth:`_resolve_combined` ranks
        every row of the dataframe. The default call —
        ``add_factors(compute_stats(...)["stats"])`` — brings ~20 statistics of the same
        length, so per-factor resolution would repeat that ranking ~20 times.
        """
        resolved: list[tuple[str, FactorLevel, NDArray[Any]]] = [
            (name, level, values) for name, level, values in destinations if level != "combined"
        ]
        vacuous: list[str] = []
        combined = {name: values for name, level, values in destinations if level == "combined"}
        if combined:
            self._warn_inferred_combined(sorted(combined))
            placed, vacuous = self._resolve_combined(combined)
            resolved.extend(placed)
        return resolved, vacuous

    def _warn_inferred_combined(self, names: Sequence[str]) -> None:
        """Warn that factors were placed by the retired combined convention.

        Inference reaching ``"combined"`` means the values were placed by their position
        in an undeclared ordering rather than by a label. That is the same bet
        ``level="combined"`` makes, so it earns the same warning — the caller has a
        `source_index` available whenever the array came from
        :func:`~dataeval.core.compute_stats`, and passing it removes the guess.

        Raised once for the whole batch rather than once per factor: the call this fires
        on is ``add_factors(compute_stats(...)["stats"])``, which brings ~20 statistics of
        the same length, and twenty copies of one paragraph bury the one action it asks
        for.
        """
        warnings.warn(
            f"Factor(s) {list(names)} are as long as the {self._item_level} and {self._label_level} "
            "levels combined, so their values were placed by the ordering compute_stats emits — by "
            f"(item, target), each item's {self._item_level}-level value ahead of that item's "
            f"{self._label_level}-level ones — and each was split into '{self._item_level}_<name>' "
            f"and '{self._label_level}_<name>'. Inferring this is deprecated and will be removed in "
            "v1.2.0; pass source_index= from compute_stats, which labels each value instead.",
            DeprecationWarning,
            # caller -> add_factors -> _resolve_factor_levels -> _resolve_destinations ->
            # here. test_inference_warnings_point_at_the_caller pins this.
            stacklevel=5,
        )

    def add_factors(
        self,
        factors: Mapping[str, Array1D[Any]] | StatsResult,
        level: FactorLevel | Literal["auto", "target", "combined", "image"] = "auto",
        overwrite: bool = False,
        append_string: str = "_added",
        source_index: Sequence[SourceIndex] | None = None,
    ) -> None:
        """Add additional factors to metadata collection.

        Extend the current metadata with new factors at any level of the bound
        dataset's schema. Values are stored on the rows at that level and
        propagate downwards to descendant rows, so a factor added at the ``unit``
        level of an object detection dataset is visible from its instance rows.

        Parameters
        ----------
        factors : Mapping[str, Array1D[Any]]
            Mapping of factor names to their values. Factor length must match
            the row count of the specified level (see :attr:`level_counts`), or the
            length of `source_index` when one is given.

            A whole :class:`~dataeval.core.StatsResult` — the return of
            :func:`~dataeval.core.compute_stats` or
            :func:`~dataeval.core.compute_ratios` — may be passed here directly, in
            which case its ``stats`` become the factors and its ``source_index`` the
            placement. Its bookkeeping keys (``object_count``, ``invalid_box_count``,
            ``image_count``) describe the run rather than the images and are ignored.
        level : str, default "auto"
            Level at which to store the factors — one of :attr:`levels`, or
            ``"auto"`` to infer the level of each factor independently from its
            array length. This also fixes the level the factor is binned at, so a
            factor stored at the ``unit`` level is discretized over one value per unit
            (see :ref:`binning-levels`).

            Prefer naming the level. Inference reads the level off an array's length,
            and levels routinely hold the same number of rows, so a mapping that works
            on one dataset can land somewhere else on the next.

            .. deprecated::
                ``level="target"`` is accepted with a warning and resolves to
                the ``"instance"`` level.

            .. deprecated::
                ``level="combined"`` is accepted with a warning. It was never a level;
                it described an array ordered by ``(item, target)``, each item's
                item-level value ahead of that item's label-level ones. The array is
                split into ``<level>_<name>`` factors, one per level. Pass
                `source_index` instead, which labels each value rather than relying on
                the ordering. Inferring the same layout under ``level="auto"`` is
                deprecated on the same terms.

            .. deprecated::
                ``level="image"`` is accepted with a warning and resolves to the
                ``"unit"`` level. Removed in v1.2.0.
        overwrite : bool, default False
            Whether to overwrite factors of the same name already present in the metadata.
            When False, a colliding factor is stored under a new name instead (see `append_string`).
        append_string : str, default "_added"
            Suffix appended to a factor name that collides with an existing column when
            `overwrite` is False. If the suffixed name is also taken, an incrementing
            counter is appended (``brightness_added``, ``brightness_added_2``, ...).
        source_index : Sequence[SourceIndex] or None, default None
            Labels describing what each value in every factor array refers to, as
            returned by :func:`~dataeval.core.compute_stats`. Mutually exclusive with
            `level`, which it replaces: each value is placed by its label rather than by
            its position (see Notes).

        Raises
        ------
        ShapeMismatchError
            When factor lengths do not match the specified level's row count, the
            length of `source_index`, or the row counts `source_index` implies.
        ValueError
            When the level is not part of the dataset's schema, when both `level` and
            `source_index` are given, or when `source_index` carries per-channel entries.

        Warns
        -----
        UserWarning
            When ``level="auto"`` and an array length matches more than one level.

        Notes
        -----
        .. versionchanged:: 1.1
            The media-unit level was renamed from ``image`` to ``unit``, so a factor that
            `source_index` splits across levels is now generated as ``unit_<name>`` where it
            was ``image_<name>`` — ``compute_stats`` output piped through here yields
            ``unit_brightness`` rather than ``image_brightness``. Unlike the level name
            itself, the old generated name is not aliased: code that reads such a column by
            name has to be updated.

        Under ``level="auto"`` each factor is placed independently, so a mapping holding
        both unit-level and instance-level arrays can be added in one call. Levels can hold
        the same number of rows — an object detection dataset with one detection per image
        has as many instances as images — and an array length that matches several of them is
        stored at the coarsest match, with a warning. Pass `level` explicitly to choose.

        `source_index` is the way to pass :func:`~dataeval.core.compute_stats` output
        straight through. When it spans several levels — ``per_image`` and ``per_target``
        both enabled — each factor is split into one factor per level, named
        ``<level>_<name>`` (``unit_brightness``, ``instance_brightness``). Both halves stay
        visible to factor analysis, since unit-level values propagate down to instance rows.

        Multi-dimensional values (vector-valued statistics such as ``histogram``,
        ``percentiles`` or ``center``) have no single-column representation and are skipped
        with a warning; the skipped names are recorded in :attr:`dropped_factors`.

        Factor names that would collide with a reserved dataframe column — ``level``,
        ``item_index``, ``target_index``, ``class_label``, ``score``, ``box``, a level's
        own key column (``instance_index``, ``unit_index``, ``track_index``,
        ``sequence_index``) or the ``track_id`` identifier — are prefixed with
        ``metadata_``, matching how dataset
        metadata keys are treated.

        Either every factor in `factors` is added or none is — a validation failure on any
        factor leaves the metadata unchanged.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> # Add unit-level factors (e.g., from imagestats)
        >>> unit_factors = {
        ...     "brightness": np.random.rand(50),  # One per image
        ...     "contrast": np.random.rand(50),  # One per image
        ... }
        >>> metadata.add_factors(unit_factors, level="unit")
        >>>
        >>> # Add instance-level factors (e.g., detection confidence scores)
        >>> target_factors = {
        ...     "iou": np.random.rand(93),  # One per detection
        ... }
        >>> metadata.add_factors(target_factors, level="instance")
        >>>
        >>> # Or hand the whole compute_stats result over, source index included
        >>> from dataeval.core import compute_stats
        >>> from dataeval.flags import ImageStats
        >>> stats = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)
        >>> metadata.add_factors(stats)
        >>> sorted(n for n in metadata.factor_names if n.endswith("mean"))
        ['instance_mean', 'unit_mean']
        """
        self._structure()
        factors, source_index = _unpack_stats_result(factors, source_index, level=level)

        # Early return for empty factors
        if not factors:
            return

        resolved_level = self._resolve_requested_level(level, source_index)

        # Resolve, validate and materialize every column before touching any state, so that a
        # bad factor anywhere in the mapping leaves this Metadata instance exactly as it was.
        # That is also why the skipped names are only *recorded* after the resolve loop below:
        # recording them is a mutation, and a validation failure must not leave it behind.
        kept, skipped = self._split_by_dimensionality(factors)

        taken = set(self._dataframe.columns)
        resolved: list[tuple[str, FactorLevel, pl.Series]] = []
        placed, vacuous = self._resolve_factor_levels(kept, resolved_level, source_index)
        for name, factor_level, values in placed:
            # Rows at ``factor_level`` take their own value and descendant rows inherit
            # from their ancestor; rows at unrelated or higher levels get null.
            column = self._layout.expand(values, factor_level)
            col_name = self._resolve_factor_name(name, taken, overwrite, append_string)
            taken.add(col_name)
            # Anchored on the native values, exactly as _add_level_column does and for
            # the same reason: the expanded column leads with nulls wherever the factor's
            # level is not the first block, so inferring from it reads the dtype off
            # those. A factor added at a level with no rows infers Null and comes back
            # as a categorical; a narrow numeric one silently widens.
            dtype = pl.Series(values=values).dtype
            resolved.append((col_name, factor_level, pl.Series(name=col_name, values=column, dtype=dtype)))

        self._record_multidimensional(skipped)
        self._record_vacuous(vacuous)
        self._commit_factors(resolved)

    def _commit_factors(self, resolved: Sequence[tuple[str, FactorLevel, pl.Series]]) -> None:
        """Write resolved columns to the dataframe and register their levels.

        The half of :meth:`add_factors` that mutates. Everything before it resolves,
        validates and materializes without touching state, so that a bad factor
        anywhere in the mapping leaves the instance exactly as it was — which only
        holds while the two halves stay separated.
        """
        if not resolved:
            return

        # Drop any stale binned/digitized companion columns of the factors being replaced,
        # otherwise _bin() skips them and they disappear from factor_info.
        self._reset_bins([name for name, _, _ in resolved])

        self._dataframe = self.dataframe.with_columns([series for _, _, series in resolved])
        self._register_factor_levels([(name, factor_level) for name, factor_level, _ in resolved])

        self._is_binned = False
        self._build_factors()

    def filter_by_factor(self, condition: Callable[[str, FactorInfo], bool]) -> NDArray[np.float64]:
        """Filter metadata factors by factor name or FactorInfo.

        Parameters
        ----------
        condition : Callable[[str, FactorInfo], bool]
            A condition to include the factor in the output.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_samples, n_factors) where the factors
            are filtered by the user provided condition. Rows are taken at
            the ``instance`` level, so they align with :attr:`class_labels` and with
            :attr:`factor_data`.

        Notes
        -----
        Numeric factors contribute their raw values. A factor with no numeric form —
        a categorical one whose values are strings — contributes its digitized
        encoding instead, since the result is a single float array.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> # Keep only factors defined at the level the labels sit at
        >>> data = metadata.filter_by_factor(lambda _, fi: fi.level == metadata.label_level)
        """
        info_by_name = self._factor_info
        self._warn_level_rename()
        selected = [(name, info) for name, info in info_by_name.items() if condition(name, info)]
        schema = self.dataframe.schema if selected else {}
        return self._project([_float_col(name, info, schema) for name, info in selected], np.float64)

    def filter_by_factor_type(
        self,
        factor_type: Literal["categorical", "discrete", "continuous"],
    ) -> NDArray[np.float64]:
        """Filter metadata factors by factor type.

        .. deprecated::
            One predicate over :meth:`filter_by_factor` is the whole of this method, and
            keeping a named wrapper per :class:`FactorInfo` field does not scale as the
            class grows fields. Use
            ``filter_by_factor(lambda _, fi: fi.factor_type == factor_type)``.

        Parameters
        ----------
        factor_type : "categorical", "discrete" or "continuous"
            The factor type to include in the output.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_samples, n_factors) where the factors
            are filtered by the user provided factor type. Rows are taken at
            the ``instance`` level; see :meth:`filter_by_factor` for which
            representation of each factor the values come from.
        """
        warnings.warn(
            "Metadata.filter_by_factor_type() is deprecated and will be removed in v1.2.0. "
            "Use Metadata.filter_by_factor(lambda _, fi: fi.factor_type == factor_type) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.filter_by_factor(lambda _, fi: fi.factor_type == factor_type)

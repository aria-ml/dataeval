__all__ = []

import copy
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval._log import get_logger
from dataeval._metadata._aggregate import aggregate, validate
from dataeval._metadata._columns import (
    binned,
    digitized,
    drop_vacuous_splits,
    float_col,
    split_by_dimensionality,
    to_col,
)
from dataeval._metadata._deprecated import DeprecatedMetadataAPI
from dataeval._metadata._entry_legacy import infer_factor_level, resolve_combined, resolve_destinations
from dataeval._metadata._filters import evaluate, report_orphaned_rows
from dataeval._metadata._input import (
    build_index2label,
    reject_length_mismatch,
    resolve_legacy_level,
    unpack_stats_result,
)
from dataeval._metadata._keyed import resolve_keyed
from dataeval._metadata._links import to_series
from dataeval._metadata._loading import _load_factors
from dataeval._metadata._serialize import restore as _restore
from dataeval._metadata._serialize import save as _save
from dataeval._metadata._store import LevelStore
from dataeval._metadata._structurers import (
    RowLayout,
    SourceIndexRows,
    StructuredData,
    Structurer,
    TaskOverride,
    safe_column_name,
    select_structurer,
)
from dataeval.core._bin import bin_data, digitize_data, is_continuous, level_budget
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
from dataeval.utils._validate import requires_maite_dataset

_logger = get_logger(__name__)


class _ResolvedFactor(NamedTuple):
    """One column :meth:`Metadata.add_factors` has resolved and is about to commit.

    ``native`` holds one value per row at ``level`` — the only form, since descendant
    rows read it through the store's gather rather than from a full-height copy.
    """

    name: str
    level: FactorLevel
    native: pl.Series


def _reject_unusable_key(
    key: str | None,
    level: FactorLevel | Literal["auto", "target", "combined", "image"],
    source_index: Sequence[SourceIndex] | None,
) -> None:
    """Refuse a ``key=`` that cannot name rows: it needs a named level, and excludes ``source_index=``."""
    if key is None:
        return
    if source_index is not None:
        raise ValueError(
            "key= and source_index= are two ways of saying which row each value belongs to; pass "
            "one. key= matches values against a column, source_index= labels each value.",
        )
    if level in {"auto", "combined"}:
        raise ValueError(
            f"key={key!r} matches against a column of one level's rows, so that level has to be "
            "named: pass level= as well.",
        )


class Metadata(DeprecatedMetadataAPI, Array, FeatureExtractor):
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
        self._dropped_factors: dict[str, list[str]]
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
        # Validated lazily: there is no schema until structuring. The cast covers a
        # retired spelling sitting here until _adopt resolves it.
        self._view: FactorLevel | None = cast("FactorLevel | None", view)
        self._inherited = inherited
        self._target_factors_only = False

        self._warn_if_task_unknowable()

    def _reset_structure(self) -> None:
        """Clear everything structuring produced, leaving an unstructured instance."""
        # A bare Structurer rather than None: its class defaults are what the level
        # accessors should answer before there is a dataset to structure.
        self._structurer: Structurer = Structurer()
        # Unread here; retained only because ``dataeval-flow`` reconstructs a cached
        # Metadata by setting it (cache.py:1069). FE-7 retires it.
        self._layout: RowLayout = RowLayout(())
        # Memoized flat frame, dropped by the ``_store`` setter on every rebind.
        self._flat: pl.DataFrame | None = None
        self._store = LevelStore.empty(self._structurer.levels)
        self._factors_by_level: dict[FactorLevel, set[str]] = {}
        # Two containers, deliberately not one: ``_factors`` is what the current view
        # analyses, ``_factor_cache`` what binning computed. Merging them would lose a
        # factor's info whenever it left the visible set while its companion column
        # stayed — and _bin() skips a factor that has one, so it would never be recomputed.
        self._factors: set[str] = set()
        self._factor_cache: dict[str, FactorInfo] = {}
        self._is_structured = False
        self._is_binned = False
        # Set by where()/having() and never cleared: a filtered instance still holds its
        # whole dataset, so anything pairing these rows with embeddings has to be able to ask.
        self._is_filtered = False
        # Whether a filter kept only part of an item's rows. Recorded as the filter runs,
        # since answering later would need the rows it removed.
        self._cut_below_items = False
        # Which factors came from agg(), and the level each was rolled up from. Describes
        # the factor rather than the column, so it survives a re-bin.
        self._aggregated_from: dict[str, FactorLevel] = {}
        # Set by load(): distinguishes "the dataset carried no metadata" from "this
        # instance cannot say", since a saved file does not hold the raw dicts.
        self._raw_omitted = False

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

    @property
    def _store(self) -> LevelStore:
        """The normalized store: each level's own rows, and the edges between them.

        Immutable, so a view made by :meth:`at` shares one rather than copying it; every
        writer rebinds this rather than mutating it.
        """
        return self._level_store

    @_store.setter
    def _store(self, store: LevelStore) -> None:
        """Rebind the store, retiring the flat frame derived from the previous one.

        A property rather than a plain field so the memoized frame — the largest object
        either holds — cannot outlive the store it describes.
        """
        self._level_store = store
        self._flat = None

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
        factors, source_index = unpack_stats_result(factors, source_index, level=level)
        inst = cls(
            None,
            continuous_factor_bins=continuous_factor_bins,
            auto_bin_method=auto_bin_method,
            exclude=exclude,
            include=include,
            inherited=inherited,
        )
        _load_factors(
            inst,
            factors,
            class_labels,
            index2label=index2label,
            item_indices=item_indices,
            level=level,
            source_index=source_index,
        )
        return inst

    @classmethod
    def load(
        cls,
        path: Path | str,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]] | None = None,
        *,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
        auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
        exclude: str | Sequence[str] | None = None,
        include: str | Sequence[str] | None = None,
        view: FactorLevel | None = None,
        inherited: bool = True,
    ) -> Self:
        """
        Read metadata previously written by :meth:`save`, skipping the walk over the dataset.

        Structuring reads every item of a dataset — decoding images, unpacking targets,
        accumulating tracks — and is the expensive half of building metadata. A saved file
        holds the rows that walk produced, so loading one gives back an instance that
        behaves as though the dataset had been structured, without reading it.

        The dataset is passed rather than stored, because it cannot be written to a file
        and should not be. Supplying it makes the loaded instance bindable, lets the item
        counts be checked against each other, and is what anything reading images
        alongside the metadata needs; omitting it gives a working but unbound instance.

        Binning is **not** restored. The file holds each factor's values, and the binning
        configuration given here is applied lazily on first read, so one file serves every
        set of bins a caller might want from it.

        Parameters
        ----------
        path : Path or str
            File written by :meth:`save`.
        dataset : ImageClassificationDataset, ObjectDetectionDataset or None, default None
            The dataset the metadata was built from, bound to the loaded instance.
            Its item count is checked against the file's.
        continuous_factor_bins : Mapping[str, int or Sequence[float]] or None, default None
            Bin counts or explicit edges per factor, applied when factors are first read.
        auto_bin_method : {"uniform_width", "uniform_count", "clusters"}, default "uniform_width"
            Binning strategy for continuous factors with no explicit bins.
        exclude : str, Sequence[str] or None, default None
            Factor names to exclude from analysis.
        include : str, Sequence[str] or None, default None
            Factor names to restrict analysis to. Mutually exclusive with ``exclude``.
        view : str or None, default None
            Level to read from. Defaults to the level carrying class labels.
        inherited : bool, default True
            Whether factors defined above the view are analysed at it.

        Returns
        -------
        Metadata
            An instance holding the saved rows, configured as asked here.

        Raises
        ------
        MetadataFormatError
            When the file is not readable as this version's metadata format — including
            when an older or newer dataeval wrote it. See the Notes.
        ValueError
            When ``dataset`` holds a different number of items than the file was saved
            for, or when both ``exclude`` and ``include`` are given.

        See Also
        --------
        save

        Notes
        -----
        **This is a cache, not an interchange format.** What is written is the library's
        internal per-level layout, and that layout may change in any release. A file is
        stamped with the format version and the level graph it was written against, and
        loading refuses anything it does not recognize rather than guessing — so a stale
        file raises :class:`~dataeval.exceptions.MetadataFormatError`, which a caching
        caller is meant to catch and recompute from::

            try:
                metadata = Metadata.load(path, dataset)
            except MetadataFormatError:
                metadata = Metadata(dataset)
                metadata.save(path)

        Do not use it to move metadata between dataeval versions, and do not treat it as
        a record: reach for :meth:`dataframe` and a parquet file for either.

        The per-item dictionaries behind :attr:`raw` are not written — they hold whatever
        the dataset put there, of unbounded size — so :attr:`raw` on a loaded instance
        raises rather than answering as though the dataset carried none.

        Example
        -------
        >>> md = Metadata(dataset)
        >>> md.save(save_dir / "metadata.dem")
        >>> reloaded = Metadata.load(save_dir / "metadata.dem", dataset)
        >>> sorted(reloaded.factor_names) == sorted(md.factor_names)
        True
        >>> reloaded.item_count == md.item_count
        True

        The rows come back at every level, not just the one being read:

        >>> reloaded.level_counts == md.level_counts
        True
        """
        inst = cls(
            dataset,
            continuous_factor_bins=continuous_factor_bins,
            auto_bin_method=auto_bin_method,
            exclude=exclude,
            include=include,
            view=view,
            inherited=inherited,
        )
        _restore(inst, path)
        return inst

    def save(self, path: Path | str) -> None:
        """
        Write the structured rows to a file that :meth:`load` can read back.

        Structures first when it has to, so saving a freshly constructed instance writes
        the dataset's rows rather than an empty file.

        What is written is the level rows, the positional links between them, and which
        factor sits at which level. What is not written is the dataset itself, the binning
        configuration and the columns derived from it, and the per-item dictionaries
        behind :attr:`raw` — see :meth:`load` for what that means when reading back.

        The file is written to a temporary name and renamed into place, so a reader sees
        either the previous file or the new one, never a partial one.

        Parameters
        ----------
        path : Path or str
            Destination file. Parent directories are created if needed, and an existing
            file is replaced.

        Raises
        ------
        NotFittedError
            When the instance has no bound dataset and no structured rows, so there is
            nothing to write.

        See Also
        --------
        load

        Notes
        -----
        **This is a cache, not an interchange format** — it holds dataeval's internal
        layout, and only the dataeval that wrote it promises to read it back. To keep
        metadata for anything else, write :attr:`dataframe` to a parquet file.

        A filtered instance saves and loads as filtered, so
        :attr:`is_filtered` still reports True on the way back and the evaluators that
        refuse a filtered metadata still refuse it.

        Example
        -------
        >>> md = Metadata(dataset)
        >>> md.save(save_dir / "od.dem")
        >>> (save_dir / "od.dem").exists()
        True
        """
        _save(self, path)

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
        # Counted rather than read off factor_names, which would sort the names only to
        # discard the sorted list; len() and ndim both route through here.
        return (self._store.height(self._view_level), sum(1 for name in self._factors if self._filter(name)))

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
            return self.factor_data

        if self._dataset is not None and data is self._dataset:
            return self.factor_data

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

        Raises
        ------
        ValueError
            When the instance came from :meth:`load`. The dictionaries are not written to
            a saved file, and an empty list here would read as a dataset that carried no
            metadata rather than as an instance that cannot say.
        """
        self._structure()
        if self._raw_omitted:
            raise ValueError(
                "This metadata was loaded from a file, which does not hold the per-item "
                "metadata dictionaries — they are unbounded in size and hold arbitrary "
                "values. Build the metadata from its dataset to read them: Metadata(dataset).raw",
            )
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
        return dict(self._store.counts)

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

        Notes
        -----
        This is ``"instance"`` for every task built from a dataset, but **not** for
        :meth:`from_factors`, which has no dataset to impose a shape and puts both the
        item level and the label level at whichever level the caller asked for. A
        single-level factors bundle therefore reports ``"unit"`` here. Code that gates
        on where the labels sit should read this rather than compare against
        ``"instance"``.
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

        .. deprecated:: 1.1
            ``"target"`` is accepted with a warning and resolves to the
            ``"instance"`` level. Removed in v1.2.0.

        .. deprecated:: 1.1
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

            .. deprecated:: 1.1
                ``"target"`` is accepted with a warning and resolves to the
                ``"instance"`` level. Removed in v1.2.0.

            .. deprecated:: 1.1
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
        # ``_store`` is deliberately shared: it is immutable, so a writer on either side
        # rebinds its own field. What is copied are the mutable containers describing how
        # this instance *reads* the store, which a view must own or its FactorInfo is
        # overwritten the next time either binned.
        view._factors = set(self._factors)
        view._factor_cache = dict(self._factor_cache)
        view._factors_by_level = {name: set(names) for name, names in self._factors_by_level.items()}
        view._dropped_factors = {name: list(reasons) for name, reasons in self._dropped_factors.items()}
        view._aggregated_from = dict(self._aggregated_from)
        view._view = resolved
        # The move can expose factors the source never binned — anything below its view —
        # so its "nothing left to process" claim does not carry. _bin() skips a factor
        # that already has a companion column, so this re-bins only what the move exposed.
        view._is_binned = False
        # A copy is a fresh object in the user's hands, so it gets its own once-per-
        # instance budget for the FactorInfo.level rename, as bind() does.
        view._warned_level_rename = False
        view._build_factors()
        return view

    def _filtered(self, keep: dict[FactorLevel, NDArray[np.intp]], level: FactorLevel) -> Self:
        """Build the metadata over a set of surviving rows, sharing nothing mutable.

        The same copy discipline as :meth:`at`, plus the restricted store. Clearing
        ``_is_binned`` re-bins what the filter exposed: dropping rows that lacked an
        ancestor can turn a partly-null column into a total one. Bin edges already
        computed are kept — filtering is not re-structuring.
        """
        filtered = copy.copy(self)
        filtered._factors = set(self._factors)
        filtered._factor_cache = dict(self._factor_cache)
        filtered._factors_by_level = {name: set(names) for name, names in self._factors_by_level.items()}
        filtered._dropped_factors = {name: list(reasons) for name, reasons in self._dropped_factors.items()}
        filtered._aggregated_from = dict(self._aggregated_from)
        filtered._cut_below_items = self._cut_below_items or self._cuts_below_items(keep)
        filtered._store = self._store.restrict(keep)
        filtered._is_filtered = True
        filtered._is_binned = False
        filtered._warned_level_rename = False
        # The surviving items, not the dataset's: this describes the rows the metadata
        # holds, and __repr__ reporting the whole dataset's would name rows that are gone.
        filtered._count = filtered._store.height(self._item_level)
        report_orphaned_rows(self._store, filtered._store, keep, level)
        filtered._build_factors()
        return filtered

    def where(self, predicate: pl.Expr, level: FactorLevel | Literal["target", "image"] | None = None) -> Self:
        """Keep the rows at ``level`` that satisfy ``predicate``, and what depends on them.

        Filters downwards: the rows that survive at ``level`` keep their descendants, and a
        descendant whose parent was dropped goes with it. It does **not** filter upwards —
        every sequence and every frame above the cut stays — and it does not filter that
        level's siblings, which for tracking means ``md.where(..., level="unit")`` leaves
        every track row in place, including tracks whose every observation was in a dropped
        frame. Those rows are counted and reported on the ``dataeval`` logger.

        Parameters
        ----------
        predicate : pl.Expr
            A polars expression answering one boolean per row at ``level``, evaluated
            against that level's resolved rows — so it may read any factor defined there or
            at one of its ancestors, spelled as it is in :attr:`dataframe`.
        level : str or None, default None
            Level whose rows the predicate is answered for. Defaults to :attr:`view`.

        Returns
        -------
        Metadata
            A new metadata over the surviving rows. Neither this instance nor the copy can
            see the other's later writes.

        Raises
        ------
        ValueError
            When the predicate names a column these rows have no value for, or does not
            answer one boolean per row.

        See Also
        --------
        :meth:`having` : Keep the rows that *have* a matching row below them
        :meth:`at` : The same rows read at another level

        Notes
        -----
        A filter can only add factors to :attr:`factor_data`, never silently remove them.
        Dropping the rows that had no ancestor at a level can turn a partly-null factor
        into a total one, which brings it into the analysis; nothing takes one out.

        The bin edges are those computed before the filter, which is why this bins the
        *unfiltered* rows on the way through rather than leaving it to the first read of
        :attr:`factor_data` on the result. Deferring it would compute the edges from the
        survivors, so the same filter would answer differently depending on whether anything
        had happened to read the source's factors first. Re-structuring the filtered dataset
        recomputes them deliberately, and so answers differently again — see
        :meth:`selected_items` for the dataset-side counterpart.

        Examples
        --------
        >>> import polars as pl
        >>> metadata = Metadata(dataset)
        >>> len(metadata.at("unit")), len(metadata)
        (50, 93)
        >>> # Keep the night-time images, and with them only their detections
        >>> night = metadata.where(pl.col("time_of_day") == "night", level="unit")
        >>> len(night.at("unit")), len(night)
        (16, 33)
        """
        self._structure()
        self._bin()
        resolved = self._view_level if level is None else self._resolve_level(level)
        mask = evaluate(self._store, resolved, predicate)
        return self._filtered(self._store.surviving_where(resolved, mask), resolved)

    def having(self, predicate: pl.Expr, level: FactorLevel | Literal["target", "image"] | None = None) -> Self:
        """Keep the rows *above* ``level`` that have a row at ``level`` satisfying ``predicate``.

        The upward filter, and the one that cuts across the level graph's diamond. Each
        level above ``level`` keeps the rows some matching row points at; every other level,
        including ``level`` itself, then keeps the rows whose parents all survived.

        For tracking that has a consequence worth stating plainly: a detection whose frame
        holds a match but whose *track* does not is dropped, because its track did not
        survive. ``md.having(pl.col("class_label") == person, level="instance")`` therefore
        keeps the tracks that contain a person and drops the car travelling through the
        same frames.

        Parameters
        ----------
        predicate : pl.Expr
            A polars expression answering one boolean per row at ``level``, evaluated
            against that level's resolved rows.
        level : str or None, default None
            Level whose rows are matched. Defaults to :attr:`view`.

        Returns
        -------
        Metadata
            A new metadata over the surviving rows.

        Raises
        ------
        ValueError
            When ``level`` has no ancestors, since the seed only travels upwards and there
            would be nothing for it to reach; when the predicate names a column these rows
            have no value for; or when it does not answer one boolean per row.

        See Also
        --------
        :meth:`where` : Keep the matching rows themselves, and what depends on them

        Notes
        -----
        The bin edges are those computed before the filter, for the reason
        :meth:`where` gives.

        Examples
        --------
        >>> import polars as pl
        >>> metadata = Metadata(dataset)
        >>> len(metadata.at("unit"))
        50
        >>> # The images holding at least one car, and every detection in them
        >>> with_cars = metadata.having(pl.col("class_label") == 1, level="instance")
        >>> len(with_cars.at("unit")), len(with_cars)
        (21, 47)
        """
        self._structure()
        self._bin()
        resolved = self._view_level if level is None else self._resolve_level(level)
        if not self._levels.ancestors(resolved):
            raise ValueError(
                f"having() keeps the rows above {resolved!r} that have a matching row at it, but "
                f"{resolved!r} is the coarsest level of this dataset and has nothing above it. Use "
                f"where(..., level={resolved!r}) to keep the matching rows themselves.",
            )
        mask = evaluate(self._store, resolved, predicate)
        return self._filtered(self._store.surviving_having(resolved, mask), resolved)

    def agg(
        self,
        from_level: FactorLevel | Literal["target", "image"],
        to_level: FactorLevel | Literal["target", "image"],
        *exprs: pl.Expr,
        unique_by: FactorLevel | Literal["target", "image"] | None = None,
    ) -> Self:
        """Roll ``from_level``'s rows up into a new factor on each ``to_level`` row.

        The counterpart to reading a coarse factor from a fine level. That direction
        replicates a value downwards and loses nothing; this one collapses many rows into
        one and has to be told what the fan-out means, which is what ``unique_by`` is for.

        Rows with no ancestor at ``to_level`` take no part — an untracked detection belongs
        to no track — and a ``to_level`` row with nothing beneath it answers null rather
        than zero, since nothing was measured there.

        Parameters
        ----------
        from_level : str
            Level whose rows are rolled up.
        to_level : str
            Level receiving one value per row. Must sit strictly above ``from_level``.
        *exprs : pl.Expr
            Aggregating polars expressions, evaluated over the ``from_level`` rows beneath
            each ``to_level`` row. Each contributes one factor, named by its output name.
        unique_by : str or None, default None
            Count each ``unique_by`` entity once within a group, keeping its first row in
            row order. Required by any expression reading a column defined above
            ``from_level``, and legal for ``from_level`` itself or any level above it —
            including one that is not below ``to_level``, which is what makes
            ``unique_by="unit"`` the answer for an instance-to-track roll-up.

        Returns
        -------
        Metadata
            A new metadata carrying the aggregated factors at ``to_level``. Neither this
            instance nor the copy can see the other's later writes.

        Raises
        ------
        ValueError
            When ``to_level`` does not sit above ``from_level``; when ``unique_by`` is
            neither ``from_level`` nor above it; when an expression reads a column defined
            above ``from_level`` and no ``unique_by`` was given; or when no expression was
            passed.

        See Also
        --------
        :meth:`where` : Keep the rows matching a predicate
        :meth:`having` : Keep the rows that have a matching row below them

        Notes
        -----
        A resulting factor's :class:`~dataeval.types.FactorInfo` records
        ``aggregated_from``, so a reader can tell a rolled-up value from one measured at
        ``to_level`` directly before comparing the two.

        Grouping is positional: a row's group is its parent's row position, so nothing is
        hashed or joined and the result scatters straight into an array as long as
        ``to_level`` has rows.

        Examples
        --------
        >>> import polars as pl
        >>> metadata = Metadata(dataset)
        >>> counted = metadata.agg("instance", "unit", pl.len().alias("n_detections"))
        >>> counted.at("unit").rows_at("unit")["n_detections"].sum()
        93
        """
        self._structure()
        if not exprs:
            raise ValueError("agg needs at least one expression, e.g. agg(from, to, pl.len().alias('n')).")
        source = self._resolve_level(from_level)
        target = self._resolve_level(to_level)
        unique = None if unique_by is None else self._resolve_level(unique_by)
        validate(self._store, source, target, exprs, unique)

        rolled = copy.copy(self)
        rolled._factors = set(self._factors)
        rolled._factor_cache = dict(self._factor_cache)
        rolled._factors_by_level = {name: set(names) for name, names in self._factors_by_level.items()}
        rolled._dropped_factors = {name: list(reasons) for name, reasons in self._dropped_factors.items()}
        rolled._aggregated_from = dict(self._aggregated_from)

        store = self._store
        taken = set(store.columns)
        added: list[tuple[str, FactorLevel]] = []
        for series in aggregate(self._store, source, target, exprs, unique):
            name = self._resolve_factor_name(series.name, taken, overwrite=False, append_string="_agg")
            taken.add(name)
            store = store.with_column(target, series.rename(name))
            rolled._aggregated_from[name] = source
            added.append((name, target))
        rolled._store = store
        rolled._register_factor_levels(added)
        rolled._is_binned = False
        rolled._warned_level_rename = False
        rolled._build_factors()
        return rolled

    @property
    def is_filtered(self) -> bool:
        """Whether these rows are a subset produced by :meth:`where` or :meth:`having`.

        Returns
        -------
        bool
            True for a filtered instance, and for anything derived from one.

        Notes
        -----
        A filtered instance still holds its **whole** dataset, so anything computed from
        that dataset — embeddings above all — describes more rows than these. Pairing the
        two is a silent misalignment rather than an error, which is why the evaluators
        that take both refuse a filtered metadata outright. :meth:`selected_items` is how
        to bring the dataset side into correspondence.
        """
        return self._is_filtered

    def _cuts_below_items(self, keep: dict[FactorLevel, NDArray[np.intp]]) -> bool:
        """Whether a filter kept only part of some surviving item's rows.

        Asked here because it needs the pre-filter store. :meth:`selected_items` is the
        consumer: a dataset item is indivisible, so no subset reproduces such a filter.
        """
        item = self._item_level
        alive = np.zeros(self._store.height(item), dtype=np.bool_)
        alive[keep[item]] = True
        for level in self._store.frames:
            if level == item or not self._levels.is_ancestor(item, level):
                continue
            positions = self._store.link(level, item).positions()
            inherited = alive[np.maximum(positions, 0)] if alive.size else np.zeros(len(positions), dtype=np.bool_)
            whole = np.flatnonzero((positions >= 0) & inherited)
            if not np.array_equal(keep[level], whole):
                return True
        return False

    def selected_items(self) -> NDArray[np.intp]:
        """Dataset items these rows came from, for filtering the dataset to match.

        Hand the result to :class:`~dataeval.data.Indices` to build a
        :class:`~dataeval.data.View` over the same items, so that embeddings computed from
        the view line up with this metadata's rows.

        Returns
        -------
        NDArray[np.intp]
            Source item indices, ascending, one per surviving row at :attr:`item_level`.

        Raises
        ------
        ValueError
            When the filter kept only part of some item — some frames of a video, or some
            detections of an image. No dataset subset reproduces that, because an item is
            indivisible, so there is no correspondence to hand back.

        Notes
        -----
        This is for the *dataset* side, not for reproducing this metadata. Re-structuring
        the filtered dataset builds a different :class:`Metadata`: bin edges are computed
        from the values present, so a factor binned over the whole dataset and the same
        factor binned over the subset do not agree, and a bin index means something
        different in each. Keep this instance for the analysis; use the view for anything
        that has to be recomputed from images.

        Examples
        --------
        >>> import polars as pl
        >>> from dataeval.data import Indices, View
        >>> metadata = Metadata(dataset)
        >>> night = metadata.where(pl.col("time_of_day") == "night", level="unit")
        >>> items = night.selected_items()
        >>> len(items)
        16
        >>> matching_dataset = View(dataset, Indices(items.tolist()))
        """
        self._structure()
        if self._cut_below_items:
            raise ValueError(
                "This metadata was filtered below the item level, so no subset of the dataset "
                f"reproduces it: some {self._item_level!r} row kept only part of its rows. A dataset "
                "item is indivisible — it can hand back whole items only. Filter at "
                f"{self._item_level!r}, or with having(..., level=...) so that whole items survive, if "
                "the dataset and embeddings have to follow.",
            )
        return self._store.column(self._item_level, "item_index").to_numpy().astype(np.intp, copy=False)

    def _reset_view_dependent_state(self) -> None:
        """Rebuild the factor set after something moved which rows or factors are visible.

        Only the visible set is rebuilt; bins are kept, since a factor is binned once at
        its own level. ``_is_binned`` is cleared to re-bin whatever the move exposed, as
        :meth:`at` does.
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

        Derived, not stored: the store holds each level's rows once and this widens
        every level to every column and stacks them, which is what makes a factor
        defined once readable from each of its descendants. It is memoized until the
        store is rebound, so repeated reads are free and any write rebuilds it. Reading
        one level is :meth:`rows_at`, and reading a few columns of one level is cheaper
        still — neither goes through this.
        """
        self._structure()
        if self._flat is None:
            self._flat = self._store.flat()
        return self._flat

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
        return self._project([to_col(name, info) for name, info in info_by_name.items()], np.int64)

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

        The warning belongs on the paths that put a :class:`FactorInfo` in *user* hands,
        not on ``factor_data``, which every bias evaluator calls.
        """
        if not self.factor_names:
            return {}
        self._bin()
        # Visible *and* processed: a factor is in ``_factors`` from registration but in
        # the cache only once it has a companion column, so the intersection is exactly
        # what factor_data can project.
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
    def is_binned(self) -> Sequence[bool]:
        """Whether each factor's codes came from cutting a range, rather than from its own values.

        Returns
        -------
        Sequence[bool]
            Boolean sequence with length equal to :attr:`factor_names`, True where the
            factor's entries in :attr:`factor_data` are bin indices and False where each
            code stands for one value.

        Notes
        -----
        This property is part of the :class:`~dataeval.protocols.MetadataLike` protocol,
        and is the same answer :attr:`factor_info` gives through
        :attr:`~dataeval.types.FactorInfo.is_binned` — offered here as a flat sequence
        aligned with :attr:`factor_names`, which is the form the evaluators read.

        Not the same question as :attr:`is_discrete`. The two agree except for a discrete
        numeric factor carrying more levels than the sample supports, which is binned like a
        continuous one while remaining discrete; see :ref:`binning-levels`.

        .. versionadded:: 1.1
        """
        return [info.is_binned for info in self._factor_info.values()]

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
        Describes the *variable*, not what was done to it — a continuous factor stays
        continuous here after being binned. For the latter question, which is what the bias
        evaluators read, use :attr:`is_binned`.

        .. versionchanged:: 1.1
            No longer part of :class:`~dataeval.protocols.MetadataLike`, which now asks for
            :attr:`is_binned` instead. This property is unaffected and stays: it is a true
            description of the factors, and :class:`Metadata` has always offered more than
            the protocol requires.
        """
        return [info.factor_type != "continuous" for info in self._factor_info.values()]

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

        Read from the label level's own ``class_label`` column rather than kept
        alongside it. The two were always equal, and holding the labels twice would
        mean every relational operation has to subset the array in lockstep with the
        frame — where the cost of forgetting is labels silently misaligned against
        factor rows, which is the worst answer this class can give.
        """
        self._structure()
        view = self._view_level
        if view != self._label_level:
            raise ValueError(
                f"class_labels is defined at the {self._label_level!r} level, but this metadata is "
                f"viewed at {view!r}, which has no label per row. Use md.at({self._label_level!r}) "
                f'for the labels, or read them from rows_at({view!r})["class_label"].',
            )
        return self._store.column(view, "class_label").to_numpy().astype(np.intp, copy=False)

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
        # One column off the store, not the view's whole widened frame: avoiding that
        # widening is what the store exists for.
        return self._store.column(self._view_level, "item_index").to_numpy().astype(np.intp, copy=False)

    @property
    def item_count(self) -> int:
        """Total number of items in the dataset.

        Returns
        -------
        int
            Count of unique items these rows come from, regardless of how many
            targets/detections each item contains. On a filtered instance this is the
            items that survived, not the dataset's total: the count describes the rows
            this metadata holds, and anything sizing an array by it — the multi-label
            matrix in :func:`~dataeval.data.split_dataset`, above all — would otherwise
            reserve room for items that contribute no row.
        """
        # Deliberately not structured up front: a bound dataset already knows its item
        # count, and answering that should not cost the whole extraction. A filtered
        # instance is structured by construction — where()/having() both structure first.
        if self._is_filtered:
            return self._store.height(self._item_level)
        if self._count == 0:
            self._structure()
        return self._count

    def rows_at(self, level: FactorLevel | Literal["target", "image"]) -> pl.DataFrame:
        """Dataframe rows belonging to a single level.

        Parameters
        ----------
        level : str
            Level to filter to, one of :attr:`levels`.

            .. deprecated:: 1.1
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
        return self._store.resolve(self._resolve_level(level))

    def _empty_projection(self, dtype: Any) -> NDArray[Any]:
        """Build a no-column projection that still has the view's row count.

        Shaped, not bare: a dataset can have rows and no usable factors, and returning
        ``array([])`` there would make :func:`len`, :attr:`shape` and iteration disagree.
        """
        self._structure()
        return np.empty((self._store.height(self._view_level), 0), dtype=dtype)

    def _project(self, columns: Sequence[str], dtype: Any) -> NDArray[Any]:
        """Read the named columns off the view's rows as a dense array.

        Sole answer to which rows an array-shaped accessor reads: the view's only, since
        other levels align neither with these factors nor with :attr:`class_labels`.
        Callers must have binned first — :meth:`_bin` writes the columns this selects.
        """
        if not columns:
            return self._empty_projection(dtype)
        self._structure()
        return self._store.select(self._view_level, columns).to_numpy().astype(dtype, copy=False)

    def _filter(self, factor: str | tuple[str, Any]) -> bool:
        factor = factor[0] if isinstance(factor, tuple) else factor
        return factor in self.include if self.include else factor not in self.exclude

    def _reset_bins(self, cols: Iterable[str] | None = None) -> None:
        """Drop the companion columns of ``cols`` (or of every factor) and their info.

        Deliberately not guarded on ``_is_binned``: the include/exclude setters clear that
        flag while leaving companion columns in place, so a guard would skip exactly the
        calls that exist to remove them. A companion left behind never re-bins.
        """
        columns = set(self._store.columns)
        companions: set[str] = set()
        for col in cols or self._store.columns:
            # Both spellings, since a factor changing type between bins moves from one to
            # the other and would otherwise leave the old column behind.
            if found := {binned(col), digitized(col)} & columns:
                companions |= found
                # Column and info are dropped together: _bin() decides what to process by
                # looking for the column, and factor_info answers from the cache.
                self._factor_cache.pop(col, None)
        self._store = self._store.without_columns(companions)
        self._is_binned = False

    def _unreadable_at(self, level: FactorLevel, view: FactorLevel) -> str | None:
        """Why a factor defined at ``level`` cannot be read from ``view``'s rows, or None.

        Sole arbiter of what enters factor analysis, so the two ways the level graph's
        diamond can put a factor out of reach are decided together. Returns a reason
        suitable for a log line, or None when every ``view`` row can read it.
        """
        if not self._levels.propagates_to(level, view):
            return (
                f"{level!r} does not propagate to {view!r} — they are on separate branches of the "
                "level graph, so these rows have no value for it"
            )
        if self._store.partial_ancestry(level, view):
            return (
                f"not every {view!r} row has a {level!r} ancestor, so the column is partly null "
                f"and cannot be binned; read it at {level!r} instead"
            )
        return None

    def _factor_level(self, name: str) -> FactorLevel:
        """Level a factor is defined at, or the item level when unknown.

        At most one level matches — ``StructuredData.__post_init__`` and
        :meth:`_register_factor_levels` both enforce it — so ``highest`` picks from a
        one-element list. Kept as the safe answer if that invariant is ever relaxed.
        """
        levels: list[FactorLevel] = [level for level, names in self._factors_by_level.items() if name in names]
        return self._levels.highest(levels) if levels else self._item_level

    def _resolve_level(self, level: FactorLevel | Literal["target", "image"], stacklevel: int = 4) -> FactorLevel:
        """Validate a caller-supplied level name, translating any retired spelling.

        Sole entry point for a level name from a caller, so the public spellings of the
        deprecation cannot drift; internal callers use ``self._levels.validate`` directly.
        ``stacklevel`` counts the frames to the user's line — one more from
        :meth:`_resolve_requested_level`, which sits a call deeper than the rest.
        """
        resolved = resolve_legacy_level(
            level, self._structurer.legacy_level_aliases, stacklevel, unit_type=self._structurer.unit_type
        )
        try:
            return self._levels.validate(resolved)
        except ValueError as exc:
            # FactorLevelSchema knows the level vocabulary but not the medium, so the
            # unit-type clause is added here rather than inside validate().
            raise ValueError(f"{exc} (this dataset's units are {self._structurer.unit_type}s)") from None

    def _warn_level_rename(self) -> None:
        """Announce the ``FactorInfo.level`` rename, once per instance.

        Nothing can intercept an ``info.level == "target"`` comparison at the point it
        silently turns false, so the warning is raised where ``FactorInfo`` objects are
        handed to a caller. Each such handout point calls this directly rather than going
        through :attr:`_factor_info`, so the once-per-instance budget is spent on a call
        the user made and ``stacklevel=3`` points at their line.
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

    def _validate_factor_lengths(self, factors: Mapping[str, Array1D[Any]], level: FactorLevel) -> None:
        """Validate that factor lengths match the specified level's row count."""
        expected_len = self._store.height(level)
        mismatched = {k: len(v) for k, v in factors.items() if len(v) != expected_len}
        if mismatched:
            raise ShapeMismatchError(
                f"All {level}-level factors must have length {expected_len} ({level} row count); got {mismatched}.",
            )

    def _split_source_index(self, source_index: Sequence[SourceIndex]) -> dict[FactorLevel, NDArray[np.intp]]:
        """Group source-index positions by the level each entry describes.

        A :class:`~dataeval.types.SourceIndex` distinguishes two kinds of value — per item
        (``target`` is None) and per label — so it addresses :attr:`item_level` and
        :attr:`label_level` and no others. A level between them, such as tracking's
        ``unit``, needs an explicit ``level=`` instead.
        """
        # Parsing is shared with the dataset-free constructor, so the two spellings of
        # "place these values by their labels" cannot drift. Only the checks that need
        # this metadata's own rows live here.
        rows = SourceIndexRows.parse(source_index)

        # The two kinds are told apart only by which level they land on, so a schema whose
        # items and labels coincide would merge them into one over-long group and surface
        # as a row-count mismatch. Say what actually happened instead.
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

        Counting entries is not enough: an index naming one row twice and another not at
        all has the right count and lands every value somewhere, just not where the caller
        said. Matching the keys catches it for one comparison per row.
        """
        counts = self._store.counts
        expected_len = counts.get(level, 0)
        if len(items) != expected_len:
            raise ShapeMismatchError(
                f"source_index describes {len(items)} {level}-level values but the "
                f"metadata has {expected_len} {level} rows. Row counts are {dict(counts)}; "
                "note that a dataset item whose target was empty contributes no rows, so "
                "Metadata.item_indices, not range(item_count), lists the items that have them.",
            )

        # The two key columns, not rows_at's whole frame, which widens with every factor
        # already added and compares none of them.
        frame = self._store.select(level, ("item_index", "target_index")).select(
            "item_index",
            # -1 stands in for the null marking a non-target row, as in SourceIndexRows.
            # Left null, to_numpy() yields float NaN and formatting the error below then
            # raises "cannot convert float NaN to integer" instead.
            pl.col("target_index").fill_null(-1),
        )
        actual_items = frame["item_index"].to_numpy()
        actual_targets = None if targets is None else frame["target_index"].to_numpy()
        mismatched = actual_items != items
        if actual_targets is not None:
            mismatched |= actual_targets != targets
        if np.any(mismatched):
            # First few only: a rejected million-row index must not spend longer building
            # its error than the call would have taken to succeed.
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

    def _build_factors(self) -> None:
        """Build the set of factor names visible at the current view."""
        if not self._is_structured:
            self._factors = set()
            return

        view = self._view_level
        # ``target_factors_only`` is the retired spelling and keeps its exemption for
        # single-target tasks; ``inherited`` is the current one and has none. Read off the
        # structurer, not the ``multi_target`` property, which would re-enter _structure().
        legacy_narrowing = self._target_factors_only and self._structurer.multi_target
        if not self._inherited or legacy_narrowing:
            names = set(self._factors_by_level.get(view, ()))
        else:
            names = {name for level_names in self._factors_by_level.values() for name in level_names}

        # A factor the view's rows cannot all read stays in the dataframe but out of factor
        # analysis — either off the branch entirely, or null on some rows, which binning
        # cannot represent. _unreadable_at decides which; md.at(level) still reads it whole.
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

        # Purely derived: nothing carries over from the outgoing set, since a factor's
        # binning lives in ``_factor_cache`` and survives the rebuild there.
        self._factors = {k for k in visible if not isinstance(self._store.dtype_of(k), pl.List | pl.Struct | pl.Array)}

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
        self._index2label = build_index2label(self._dataset.metadata.get("index2label", None), unique_labels)
        self._adopt(structurer, data)

        _logger.debug(
            "Metadata structured as %s: %s rows per level, %d factors, %d dropped",
            structurer.task,
            self._store.counts,
            sum(len(names) for names in self._factors_by_level.values()),
            sum(len(v) for v in self._dropped_factors.values()),
        )

    def _adopt(self, structurer: Structurer, data: StructuredData) -> None:
        """Take on everything a :class:`StructuredData` describes.

        The dataset path and the factors-only path build their bundle differently and adopt
        it identically, so the field list lives here once. Each caller keeps only what is
        its own — where ``index2label`` comes from, and the dataset item count.
        """
        self._structurer = structurer
        self._layout = data.layout
        # The setter retires any memoized flat frame, so there is nothing to clear here.
        self._store = LevelStore.of(self._levels, data)
        self._factors_by_level = {level: set(names) for level, names in data.factors.items()}
        # A declared level that produced no factors still has to be a key, so any level
        # can be looked up unconditionally.
        for level in self._levels:
            self._factors_by_level.setdefault(level, set())

        # A view chosen at construction is resolved at the first moment there is a schema
        # to resolve it against, and before _build_factors reads it. Through _resolve_level
        # rather than validate() so a retired spelling deprecates rather than raises.
        if self._view is not None:
            self._view = self._resolve_level(self._view, stacklevel=3)

        self._raw = data.raw
        # ``class_labels`` and ``item_indices`` are deliberately not stored: both are
        # columns the store already holds, and the properties read them from there.
        self._dropped_factors = {name: list(reasons) for name, reasons in data.dropped_factors.items()}
        self._is_structured = True

        self._build_factors()

    def _classify_factor(
        self,
        col: str,
        data: NDArray,
        factor_bins: Mapping[str, int | Sequence[float]],
    ) -> tuple[NDArray[np.int64], FactorInfo]:
        """Bin or digitize one factor's native values, and say which was done.

        ``data`` holds one value per entity at the factor's own level, so every decision
        here is read off the factor's true distribution.
        """
        if col in factor_bins:
            return digitize_data(data, factor_bins[col]).astype(np.int64), FactorInfo("continuous", is_binned=True)

        distinct, ordinal = np.unique(data, return_inverse=True)
        if not np.issubdtype(data.dtype, np.number):
            return ordinal.astype(np.int64), FactorInfo("categorical", is_digitized=True)
        # No de-duplication argument: one value per entity means no propagated repeats
        # for is_continuous to mistake for discrete support.
        # No factor carries more levels than the sample can fill, whichever path bins it.
        budget = level_budget(data.shape[0])
        if is_continuous(data):
            _logger.warning(
                f"A user defined binning was not provided for {col}. "
                f"Using the {self.auto_bin_method} method to discretize the data. "
                "It is recommended that the user rerun and supply the desired "
                "bins using the continuous_factor_bins parameter.",
            )
            binned = bin_data(data, self.auto_bin_method, max_bins=budget)
            return binned.astype(np.int64), FactorInfo("continuous", is_binned=True)
        # Discrete, but not necessarily coarse: an integer factor can take a value per
        # entity and still read as discrete, and scoring one value at a time is what makes
        # such a factor report a correlation with anything it is measured against. Bin it
        # against the same budget a histogram would use.
        levels = int(distinct.size)
        if levels > budget:
            _logger.warning(
                f"Factor {col} reads as discrete but takes {levels} distinct values over "
                f"{data.shape[0]} entities, too many to score one value at a time. "
                f"Binning it with the {self.auto_bin_method} method. Supply explicit bins "
                "using the continuous_factor_bins parameter to control this.",
            )
            binned = bin_data(data, self.auto_bin_method, max_bins=budget)
            return binned.astype(np.int64), FactorInfo("discrete", is_binned=True)
        # Digitized so factor_data holds non-negative integers, which np.bincount in the
        # downstream bias evaluators requires.
        return ordinal.astype(np.int64), FactorInfo("discrete", is_digitized=True)

    def _process_factor(
        self,
        col: str,
        data: NDArray,
        factor_bins: Mapping[str, int | Sequence[float]],
        level: FactorLevel,
    ) -> tuple[pl.Series, FactorInfo]:
        """Build one factor's companion column and the info describing it.

        The column name is read off the :class:`FactorInfo` :meth:`_classify_factor` just
        built, so name and flags cannot disagree. The dtype is stated rather than
        inferred: a companion column is always a bin or category index.
        """
        values, info = self._classify_factor(col, data, factor_bins)
        info.level = level
        info.aggregated_from = self._aggregated_from.get(col)
        return pl.Series(name=to_col(col, info), values=values, dtype=pl.Int64), info

    def _bin(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Populate factor info and bin non-categorical factors.

        Every factor is binned at its own level — one value per entity — and the companion
        column is written there, reaching descendant rows by the same gather the raw values
        do. Binning at the view instead would read each factor's distribution through
        however many descendants each entity happens to have, moving the bin edges.
        """
        if self._is_binned:
            return

        factor_info: dict[str, FactorInfo] = {}
        factor_bins = self.continuous_factor_bins

        invalid_keys = set(factor_bins.keys()) - set(self._store.columns)
        if invalid_keys:
            _logger.warning(
                f"The keys - {invalid_keys} - are present in the `continuous_factor_bins` dictionary "
                "but are not columns in the metadata DataFrame. Unknown keys will be ignored.",
            )

        column_set = set(self._store.columns)
        factors_to_process = [col for col in self.factor_names if not {binned(col), digitized(col)} & column_set]
        total_factors = len(factors_to_process)

        # One frame lookup per level, not per factor: a dataset has a handful of levels
        # and may carry hundreds of factors.
        levels: dict[str, FactorLevel] = {col: self._factor_level(col) for col in factors_to_process}
        native = {level: self._store.frame(level) for level in set(levels.values())}

        store = self._store
        for i, col in enumerate(factors_to_process):
            level = levels[col]
            companion, info = self._process_factor(col, native[level][col].to_numpy(), factor_bins, level)
            store = store.with_column(level, companion)
            factor_info[col] = info

            if progress_callback:
                progress_callback(i + 1, total=total_factors)

        self._store = store
        self._factor_cache.update(factor_info)
        self._is_binned = True

    def _resolve_factor_name(self, name: str, taken: set[str], overwrite: bool, append_string: str) -> str:
        """Pick the dataframe column a new factor should be written to.

        Reserved columns are load-bearing — ``level`` drives every level filter — so a
        colliding factor is renamed rather than allowed to overwrite one.
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

    def _store_native_factors(self, resolved: Sequence[_ResolvedFactor]) -> None:
        """Write each added factor into its own level's frame, once, undigested.

        A factor added after structuring lands here and nowhere else; the expanded column
        the flat frame shows is derived from this one on the way out.
        """
        store = self._store
        for factor in resolved:
            store = store.with_column(factor.level, factor.native)
        self._store = store

    def _register_factor_levels(self, factors: Sequence[tuple[str, FactorLevel]]) -> None:
        """Record which level each newly added factor is defined at.

        A factor is stored once, at one level, so stale membership and any cached binning
        are cleared first: the name now describes different values.
        """
        for name, level in factors:
            for names in self._factors_by_level.values():
                names.discard(name)
            self._factors_by_level.setdefault(level, set()).add(name)
            self._factor_cache.pop(name, None)

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

        Reported rather than silent: the column is unusable, but its absence still
        surprises code that expected the split to produce both halves.
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

        Returns the columns to write, and any level splits discarded for holding no values.
        """
        reject_length_mismatch(factors, source_index)
        return self._place(factors, self._split_source_index(source_index))

    def _place(
        self,
        factors: Mapping[str, NDArray[Any]],
        positions_by_level: Mapping[FactorLevel, NDArray[np.intp]],
        qualify: bool = False,
    ) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
        """Gather each factor onto the rows its positions name, and name the columns.

        Sole producer of the ``<level>_<name>`` rule. Values spanning several levels are
        always prefixed; ``qualify`` forces the prefix for a single level too and
        suppresses the vacuous-split drop, so a caller promised exactly which columns it
        will get — as ``level="combined"`` promises — gets them even where one holds
        nothing. Returns the columns to write plus any discarded splits, staying pure.
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
            kept, dropped = drop_vacuous_splits(columns)
            placed.extend(kept)
            vacuous.extend(dropped)
        return placed, vacuous

    def _resolve_requested_level(
        self,
        level: FactorLevel | Literal["auto", "target", "combined", "image"],
        source_index: Sequence[SourceIndex] | None,
    ) -> FactorLevel | Literal["combined"] | None:
        """Turn ``add_factors``' ``level=`` argument into a destination, or None to infer.

        Both retired spellings a v1.1 caller can still pass — ``"target"`` and
        ``"combined"`` — are handled here, so the vocabulary of retired names lives in one
        place and every warning is raised at the same depth below the user's call.
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
        # One frame deeper than callers reaching _resolve_level from the public method.
        return self._resolve_level(level, stacklevel=5)

    def _resolve_factor_levels(
        self,
        factors: Mapping[str, NDArray[Any]],
        level: FactorLevel | Literal["combined"] | None,
        source_index: Sequence[SourceIndex] | None,
    ) -> tuple[list[tuple[str, FactorLevel, NDArray[Any]]], list[str]]:
        """Work out the level and values of every column ``add_factors`` is about to write.

        Returns one ``(name, level, values)`` per column, plus the names of any level
        splits discarded for holding no values. A ``level`` of None is inferred per factor;
        a multi-level source index and the retired ``"combined"`` spelling both yield
        several columns per factor, named ``<level>_<name>``.
        """
        if source_index is not None:
            return self._resolve_by_source_index(factors, source_index)

        if level == "combined":
            return resolve_combined(self, factors)

        if level is not None:
            self._validate_factor_lengths(factors, level)
            return [(name, level, values) for name, values in factors.items()], []

        # Each factor is inferred independently, so one call can mix levels. A loop rather
        # than a comprehension, so the stacklevel of infer_factor_level's ambiguity warning
        # counts the same number of frames on every supported Python.
        destinations: list[tuple[str, FactorLevel | Literal["combined"], NDArray[Any]]] = []
        for name, values in factors.items():
            destinations.append((name, infer_factor_level(self, values), values))
        return resolve_destinations(self, destinations)

    def add_factors(
        self,
        factors: Mapping[str, Array1D[Any]] | StatsResult,
        level: FactorLevel | Literal["auto", "target", "combined", "image"] = "auto",
        overwrite: bool = False,
        append_string: str = "_added",
        source_index: Sequence[SourceIndex] | None = None,
        key: str | None = None,
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
            Level at which to store the factors — one of :attr:`levels`, or ``"auto"``
            to infer it (deprecated, see below). This also fixes the level the factor is
            binned at, so a factor stored at the ``unit`` level is discretized over one
            value per unit (see :ref:`binning-levels`).

            **Name the level, or label the values with** `source_index`. Those are the
            two supported ways to say where values belong, and between them they cover
            everything the retired spellings did.

            .. deprecated:: 1.1
                ``"auto"`` — the current default — infers each factor's level from its
                array length, and warns when it does. A length identifies a level only
                by coincidence: levels routinely hold the same number of rows, so a
                mapping that lands correctly on one dataset can land somewhere else on
                the next, and the inference cannot tell the difference. Removed in
                v1.2.0, from when the destination has to be stated — by `level` or by
                `source_index`, either one.

            .. deprecated:: 1.1
                ``level="target"`` is accepted with a warning and resolves to
                the ``"instance"`` level.

            .. deprecated:: 1.1
                ``level="combined"`` is accepted with a warning. It was never a level;
                it described an array ordered by ``(item, target)``, each item's
                item-level value ahead of that item's label-level ones. The array is
                split into ``<level>_<name>`` factors, one per level. Pass
                `source_index` instead, which labels each value rather than relying on
                an ordering nothing declares — it carries the same information for the
                same two levels, and cannot place a value on the wrong row. Inferring
                the same layout under ``"auto"`` is deprecated on the same terms.

            .. deprecated:: 1.1
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
        key : str or None, default None
            Column of the named level's rows to match values against, instead of taking
            them in that level's row order. Requires `level`, and is mutually exclusive
            with `source_index` — the three are the ways of saying which row a value
            belongs to, and a call uses one.

            This is what attaches :func:`~dataeval.core.track_stats` output, which is
            indexed by sorted track id within one sequence while a metadata track row is
            keyed ``(item_index, track_index)`` in order of first appearance. The two
            orders coincide only by accident. Matching is on ``(item_index, key)``,
            because a track id restarts in each sequence; supply an ``item_index`` entry
            alongside the values when the dataset holds more than one item, which
            ``track_stats`` requires since it describes one sequence at a time.

            The key column itself is consumed rather than stored — it says which row a
            value belongs to, not anything about the row. ``track_stats`` returns it as
            ``track_ids``, and both that and the singular column name are accepted. A row
            no incoming key names is null, so the column still has one value per row.

        Raises
        ------
        ShapeMismatchError
            When factor lengths do not match the specified level's row count, the
            length of `source_index`, or the row counts `source_index` implies; or, under
            `key`, when they do not match the number of keys.
        ValueError
            When the level is not part of the dataset's schema, when both `level` and
            `source_index` are given, or when `source_index` carries per-channel entries.
            Under `key`: when it is not a column of that level's rows, when no values for
            it were supplied, when the keys are not unique, or when the dataset holds
            several items and the values do not say which they belong to.

        Warns
        -----
        DeprecationWarning
            When ``level="auto"`` — the default — infers a level from an array length,
            raised once per call naming each factor with the level it reached. Also when
            a retired `level` spelling is passed, or when inference reaches the retired
            combined layout.
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
        factors, source_index = unpack_stats_result(factors, source_index, level=level)

        if not factors:
            return

        _reject_unusable_key(key, level, source_index)

        resolved_level = self._resolve_requested_level(level, source_index)

        # Resolve, validate and materialize every column before touching any state, so a bad
        # factor anywhere in the mapping leaves this instance exactly as it was. The skipped
        # names are likewise only recorded after the resolve loop, since recording mutates.
        kept, skipped = split_by_dimensionality(factors)

        taken = set(self._store.columns)
        resolved: list[_ResolvedFactor] = []
        if key is not None:
            # _reject_unusable_key has already refused "auto" and "combined", the only
            # spellings resolving to something other than a level, so this is one.
            placed, vacuous = resolve_keyed(self, kept, cast("FactorLevel", resolved_level), key), []
        else:
            placed, vacuous = self._resolve_factor_levels(kept, resolved_level, source_index)
        for name, factor_level, values in placed:
            col_name = self._resolve_factor_name(name, taken, overwrite, append_string)
            taken.add(col_name)
            # One value per entity at the factor's own level: descendant rows read them by
            # the store's gather, so there is no expanded copy to build and no dtype to
            # infer off one leading with nulls.
            resolved.append(_ResolvedFactor(col_name, factor_level, to_series(col_name, values)))

        self._record_multidimensional(skipped)
        self._record_vacuous(vacuous)
        self._commit_factors(resolved)

    def _commit_factors(self, resolved: Sequence[_ResolvedFactor]) -> None:
        """Write resolved columns to the dataframe and register their levels.

        The half of :meth:`add_factors` that mutates. Its all-or-nothing guarantee holds
        only while the two halves stay separated.
        """
        if not resolved:
            return

        # Stale companion columns of the factors being replaced, otherwise _bin() skips
        # them and they disappear from factor_info.
        self._reset_bins([factor.name for factor in resolved])

        self._store_native_factors(resolved)
        self._register_factor_levels([(factor.name, factor.level) for factor in resolved])

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
        # Read off the store: ``self.dataframe.schema`` would build the whole flat frame
        # to answer "is this numeric" about a handful of names.
        schema = {name: self._store.dtype_of(name) for name, _ in selected}
        return self._project([float_col(name, info, schema) for name, info in selected], np.float64)

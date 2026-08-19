__all__ = []

import copy
import hashlib
import inspect
import json
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import replace
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
from dataeval._metadata._encoding import (
    FactorEncoding,
    apply_level_spec,
    declared_levels,
    encoding_to_json,
    encoding_to_mapping,
    read_encoding,
)
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
from dataeval.core._bin import (
    MIN_LEVEL_BUDGET,
    apply_bin_spec,
    bin_data,
    digitize_data,
    is_continuous,
    level_budget,
)
from dataeval.core._compute_stats import StatsResult
from dataeval.exceptions import NotFittedError, ShapeMismatchError
from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    DatumMetadata,
    FeatureExtractor,
    ProgressCallback,
)
from dataeval.types import Array1D, BinSpec, FactorInfo, FactorLevel, FactorLevelSchema, LevelSpec, SourceIndex
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


def _as_orderable(data: NDArray) -> NDArray:
    """Read a temporal column as the number it already is.

    A timestamp is totally ordered, so it cuts into intervals exactly like a number and
    should not be treated as an unorderable set of labels — a capture time is one of the
    most common per-row fields there is, and one distinct value per row would otherwise
    make it look like an identifier. The edges come out in the column's own unit, which is
    what a caller declaring ``continuous_factor_bins`` against it would supply.

    Floats rather than integers, so that ``NaT`` can be carried across as ``NaN`` — which
    is how every other column spells a missing value, and what the binning path reads to
    reserve the missing code. ``astype(np.int64)`` alone renders ``NaT`` as ``INT64_MIN``,
    an extreme *observed* magnitude nine quintillion below the data: the derived edges are
    placed to span it, every real timestamp collapses into one bin, and the missing code
    the record reserves is never used.

    **Nanosecond timestamps lose resolution here, deliberately.** A float64 carries 53 bits
    of mantissa, so near the current epoch — about 1.8e18 nanoseconds — consecutive
    representable values are 256 ns apart, and two capture times closer together than that
    become one number. ``datetime64[us]`` and ``[ms]`` are exact; only ``[ns]``, which is
    polars' and pandas' default unit, is affected.

    Accepted rather than worked around, because 256 ns is far below any bin edge a capture
    time is cut on — the alternative is a second integral path carrying its own missing
    sentinel, to preserve a distinction no binning can express. It does mean the distinct
    *count* of an ``[ns]`` column can read lower than the data holds, which reaches
    :func:`~dataeval.core.is_continuous` and the near-uniqueness test; cast to ``[us]``
    before adding the factor if a sub-microsecond difference is one you need kept.
    """
    if data.dtype.kind not in "Mm":
        return data
    return np.where(np.isnat(data), np.nan, data.astype(np.int64).astype(np.float64))


def _name_list(names: Sequence[str]) -> str:
    """Render factor names as a reader would say them: ``a``, ``a and b``, ``a, b and c``."""
    quoted = [f"`{name}`" for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _was_were(names: Sequence[str]) -> str:
    """Agree the verb with the list :func:`_name_list` just produced."""
    return "was" if len(names) == 1 else "were"


def _reconcile_encoding(
    continuous_factor_bins: Mapping[str, int | Sequence[float]],
    encoding: str | Path | Mapping[str, FactorEncoding] | None,
    factor_levels: Mapping[str, Sequence[Any]] | None,
) -> dict[str, FactorEncoding]:
    """Merge the three ways a caller declares an encoding, refusing any factor named twice.

    Per factor, not per argument. What has no good resolution is *one factor* described
    twice; arguments covering disjoint factors are only a longhand for one record, and
    refusing the pair outright refused a combination the library itself produces —
    ``load(path, continuous_factor_bins=...)`` restores the archive's record for every
    factor the caller said nothing about, leaving both populated, after which ``new()``
    could not reconfigure the instance it had just built.

    ``continuous_factor_bins`` stays where it is and is applied on its own path; the two
    vocabulary-shaped arguments come back as one mapping.
    """
    records: dict[str, FactorEncoding] = read_encoding(encoding) if encoding is not None else {}
    if overlapping := sorted(set(continuous_factor_bins) & set(records)):
        raise ValueError(
            f"Factors {overlapping} are cut by both `continuous_factor_bins` and `encoding`, and two "
            "sources disagreeing about one factor has no good resolution. `encoding` is the general "
            "form and carries everything the other does; pass one.",
        )
    if not factor_levels:
        return records
    if overlapping := sorted(set(factor_levels) & set(records)):
        raise ValueError(
            f"Factors {overlapping} have a vocabulary in both `factor_levels` and `encoding`; "
            "declare each factor once.",
        )
    if overlapping := sorted(set(factor_levels) & set(continuous_factor_bins)):
        raise ValueError(
            f"Factors {overlapping} have a vocabulary in `factor_levels` and a cut in "
            "`continuous_factor_bins`. A factor is encoded one way or the other, and the vocabulary "
            "would silently win; declare each factor once.",
        )
    records.update(declared_levels(factor_levels))
    return records


def _declared_bins(spec: BinSpec) -> int:
    """Intervals the caller's edges describe — the bins the cut is a claim *about*.

    Not every code a value can land in. ``np.digitize`` has to put an out-of-range value
    somewhere, so a finitely bounded list like ``[0, 10, 20]`` also yields a below-first and
    an above-last catchall. Nobody declared those, and their being **empty is the good
    case**: it says every value fell inside the range the caller described. Counting them
    made a cut that fits its data perfectly report "2 of 4 bins hold rows" on every read,
    which teaches a reader to filter the one warning here worth reading.
    """
    return max(len(spec.edges) - 1, 0)


def _declared_codes(spec: "BinSpec | LevelSpec | None") -> list[int]:
    """Codes a record describes, before any data is consulted.

    The point of asking the record rather than the column: a bin nothing reached still has
    a name, because the cut still declares it. Excludes the out-of-range catchalls and the
    missing-value code, which the record does not declare -- they are named only where the
    data actually put something in them.
    """
    if isinstance(spec, BinSpec):
        return list(range(1, _declared_bins(spec) + 1))
    if isinstance(spec, LevelSpec):
        return list(range(len(spec.levels)))
    return []


def _unused_bins(spec: BinSpec, codes: NDArray[np.int64]) -> tuple[NDArray[np.int64], str | None]:
    """Codes a cut actually placed, and a note naming how many of its intervals hold rows.

    Split out from :meth:`Metadata._measure_fit` because emptiness is a question about a
    cut and fineness is a question about a contingency table: only the first is asked here,
    and the codes it returns are what the second counts.
    """
    # Missing rows are not a bin the cut placed. Counting their reserved code as occupancy
    # inflated every tally by one and, where exactly one bin short of the count was empty,
    # cancelled the shortfall out and said nothing at all.
    present = codes[codes != spec.missing_code]
    declared = _declared_bins(spec)
    # Asked of the declared intervals alone — codes 1 through `declared` — since an empty
    # out-of-range catchall is the cut working, not failing.
    occupied = len(np.unique(present[(present >= 1) & (present <= declared)]))
    note = f"{occupied} of {declared} bins hold rows" if declared and occupied < declared else None
    return present, note


def _is_derived_encoding(info: FactorInfo) -> bool:
    """Say whether a factor's encoding was chosen for it rather than by anyone.

    True of a derived cut and of a vocabulary read off whatever values turned up — both are
    decisions DataEval made on the caller's behalf, and both are what :meth:`Metadata.accept`
    exists to let a person ratify.
    """
    return info.encoding is not None and info.encoding.provenance == "derived"


def _is_auto_binned(info: FactorInfo) -> bool:
    """Say whether a factor's *cuts* were chosen for it rather than declared.

    Narrower than :func:`_is_derived_encoding`: a digitized factor's levels are its own
    values, so it was not binned at all and the auto-binning warning has nothing to say
    about it.
    """
    return isinstance(info.encoding, BinSpec) and _is_derived_encoding(info)


def _caller_stacklevel() -> int:
    """Depth of the first frame outside :mod:`dataeval`, counted from this function's caller.

    A constant will not do here. Binning is reached from :attr:`Metadata.factor_names`, from
    :attr:`~Metadata.factor_data` by way of ``_factor_info``, from :attr:`~Metadata.shape`
    and from :attr:`~Metadata.dropped_factors`, each a different number of frames above the
    warning, so any fixed ``stacklevel`` points at library internals from most of them.

    That is not cosmetic. :func:`warnings.warn` stores its once-per-location bookkeeping in
    the globals of the frame ``stacklevel`` selects, keyed by line number: attributing every
    caller to one line inside this module puts them all in one registry entry, so the second
    place in a program that builds a :class:`Metadata` with the same factor names is told
    nothing. Pointing at the caller gives each site its own entry, which is what "warn once"
    is supposed to mean.

    Falls back to 2 -- the caller's caller -- when the stack cannot be walked, which is the
    behaviour this replaces.
    """
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None
    level = 1
    while frame is not None:
        # The dot matters: a downstream package named ``dataeval_studio`` is a caller, not
        # an internal frame, and a bare prefix test would walk straight past its code.
        module = frame.f_globals.get("__name__", "")
        if module != "dataeval" and not module.startswith("dataeval."):
            return level
        frame = frame.f_back
        level += 1
    return 2


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
    encoding : str, Path, Mapping[str, BinSpec | LevelSpec] or None, default None
        A recorded encoding to apply rather than derive: a path to a descriptor written by
        :meth:`export_encoding`, or the records :meth:`encoding` returns. Named factors are
        reapplied — the same value gets the same code in a dataset the cut was never fitted
        to — and the rest are encoded from their own values. The general form of
        ``continuous_factor_bins``, and mutually exclusive with it *per factor*: naming one
        factor in both is an error.
    factor_levels : Mapping[str, Sequence] or None, default None
        Vocabularies declared ahead of the data, one per factor: code ``i`` means
        ``levels[i]``, so two datasets declared against the same list share an alphabet
        without either having been structured first. The categorical counterpart to
        ``continuous_factor_bins``, and likewise an error for a factor already named in
        ``encoding`` or ``continuous_factor_bins``.
    strict : bool, default False
        Whether a value no declared vocabulary holds is an error. The default appends it,
        which is what extension wants; pass True for a closed taxonomy that should report
        the data leaving it rather than be widened to fit. Bin edges are unaffected — an
        unseen magnitude lands in an end bin either way.
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
        encoding: str | Path | Mapping[str, FactorEncoding] | None = None,
        factor_levels: Mapping[str, Sequence[Any]] | None = None,
        strict: bool = False,
        auto_bin_method: Literal["uniform_width", "uniform_count", "clusters"] = "uniform_width",
        exclude: str | Sequence[str] | None = None,
        include: str | Sequence[str] | None = None,
        view: FactorLevel | Literal["image"] | None = None,
        inherited: bool = True,
    ) -> None:
        self._raw: Sequence[Mapping[str, Any]]

        self._warned_level_rename = False
        self._reset_structure()

        self._dataset = dataset
        self._task: TaskOverride | None = task
        self._count = len(dataset) if dataset is not None and isinstance(dataset, Sized) else 0
        self._continuous_factor_bins = dict(continuous_factor_bins) if continuous_factor_bins else {}
        self._encoding: dict[str, FactorEncoding] = _reconcile_encoding(
            self._continuous_factor_bins,
            encoding,
            factor_levels,
        )
        self._strict = strict
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
        # Emptied here rather than only being rebuilt by ``_structure``: binning records
        # drops of its own, so it has to be readable on an instance that has not structured
        # yet. Was previously a bare annotation, which left it genuinely absent until
        # structuring ran.
        self._dropped_factors: dict[str, list[str]] = {}
        # Whether each column names its rows rather than grouping them. Cached because the
        # answer is a property of the column and `_build_factors` re-runs on every view
        # change, filter and include/exclude set; carried onto derived instances so that a
        # filter cannot re-answer it against the rows it happened to keep. Cleared wherever
        # a column's contents change, which is `add_factors` and a fresh structuring.
        self._identifier_cache: dict[str, bool] = {}
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
        encoding: str | Path | Mapping[str, FactorEncoding] | None = None,
        factor_levels: Mapping[str, Sequence[Any]] | None = None,
        strict: bool = False,
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
        encoding : str, Path, Mapping[str, BinSpec | LevelSpec] or None, default None
            A recorded encoding to apply rather than derive — a descriptor path or the
            records :meth:`encoding` returns. See the class docstring.
        factor_levels : Mapping[str, Sequence] or None, default None
            Vocabularies declared ahead of the data, one per factor. See the class
            docstring.
        strict : bool, default False
            Whether a value no declared vocabulary holds is an error rather than an
            append. See the class docstring.
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
            encoding=encoding,
            factor_levels=factor_levels,
            strict=strict,
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
        encoding: str | Path | Mapping[str, FactorEncoding] | None = None,
        strict: bool = False,
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

        Binned **columns** are not restored — the file holds each factor's values, and the
        cut is reapplied lazily on first read — but the **record** of that cut is. The file
        carries the encoding each factor was written under, so a restored instance
        reproduces its codes rather than re-deriving them, and neither a ratified placement
        nor a grown vocabulary is lost to a save. The record is applied *underneath*
        whatever is passed here: ``continuous_factor_bins`` and ``encoding`` re-cut the
        factors they name, and the archive fills in only the rest. One file still serves
        every set of bins a caller might want from it.

        Parameters
        ----------
        path : Path or str
            File written by :meth:`save`.
        dataset : ImageClassificationDataset, ObjectDetectionDataset or None, default None
            The dataset the metadata was built from, bound to the loaded instance.
            Its item count is checked against the file's.
        continuous_factor_bins : Mapping[str, int or Sequence[float]] or None, default None
            Bin counts or explicit edges per factor, applied when factors are first read.
            Overrides the archive's record for the factors it names.
        encoding : str, Path, Mapping[str, BinSpec | LevelSpec] or None, default None
            A recorded encoding to apply instead of the archive's, for the factors it
            names. Mutually exclusive with ``continuous_factor_bins`` per factor.
        strict : bool, default False
            Whether a value no declared vocabulary holds is an error. Restored from the
            archive when not set here, so a closed taxonomy stays closed across a round
            trip; passing True closes one the archive left open.
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
            encoding=encoding,
            strict=strict,
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
        # Only once binning has happened — a repr must not trigger the expensive pass, and
        # before it there is nothing to count. Reported so the silent default is visible on
        # inspection as well as through the warning, which a caller may have filtered.
        if self._is_binned:
            derived = sum(1 for info in self._factor_cache.values() if _is_auto_binned(info))
            if derived:
                parts.append(f"auto_encoded={derived}")
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
        This property triggers dataset structure analysis and binning on first access.
        """
        if not self._is_fitted:
            raise NotFittedError("No dataset bound. Call bind() first.")
        self._structure()
        # Counted rather than read off factor_names, which would sort the names only to
        # discard the sorted list.
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
        # Both, where both are set, but never the same factor twice. Carried because
        # encoding *this* dataset against a record and the next one against its own draw is
        # the drift the record exists to prevent, and this method exists to configure the
        # next one identically.
        #
        # A factor named in both is dropped from the count: `_classify_factor` consults the
        # record first, so the count says nothing the record does not already say, and the
        # constructor refuses the pair. That pair is not hypothetical — an archive stores
        # the declared count *and* the BinSpec it resolved to, and `load` restores both, so
        # `md.save(); Metadata.load(...).new(...)` raised on every declared cut.
        bins = {name: spec for name, spec in self._continuous_factor_bins.items() if name not in self._encoding}
        return self.__class__(
            dataset,
            task=self._task,
            continuous_factor_bins=bins or None,
            encoding=self._encoding or None,
            strict=self._strict,
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
        view._identifier_cache = dict(self._identifier_cache)
        view._encoding = dict(self._encoding)
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
        filtered._identifier_cache = dict(self._identifier_cache)
        filtered._encoding = dict(self._encoding)
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
        rolled._identifier_cache = dict(self._identifier_cache)
        rolled._encoding = dict(self._encoding)

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

        A cut set here supersedes any record the named factor carried — a restored archive
        brings one for every factor it holds, and :meth:`_classify_factor` consults the
        record *before* this mapping, so leaving it in place made the assignment silently
        do nothing on exactly the instances a caller is most likely to re-cut.

        Parameters
        ----------
        bins : Mapping[str, int | Sequence[float]]
            Mapping of factor names to bin counts or explicit edges.
        """
        if self._continuous_factor_bins != bins:
            self._continuous_factor_bins = dict(bins)
            for name in bins:
                self._encoding.pop(name, None)
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

    def encoding(self, factor: str | None = None) -> Any:
        """How a factor's values became its codes.

        Answers "where did you cut, and who chose it" — a question a reader needs answered
        before a finding about a binned factor can be interpreted, and one that had no
        answer before: bin edges were computed, used once and dropped.

        Parameters
        ----------
        factor : str or None, default None
            The factor to describe, or None for every factor.

        Returns
        -------
        BinSpec or LevelSpec or None, or a Mapping of them
            The record for ``factor``, or a mapping from every factor name to its record
            when ``factor`` is None. None for a factor that reached neither encoding path.

        Raises
        ------
        KeyError
            When ``factor`` is not one of this metadata's factors.

        See Also
        --------
        export_encoding : Write the record out as a reviewable descriptor.

        Examples
        --------
        >>> md = Metadata(dataset)
        >>> spec = md.encoding("weather")
        >>> spec.levels
        ('clear', 'cloudy', 'rainy')

        ``provenance`` is what separates a cut somebody chose from one DataEval derived on
        their behalf, and is the field a reviewer audits:

        >>> spec.provenance
        'derived'
        """
        info = self._factor_info
        if factor is None:
            return {name: entry.encoding for name, entry in info.items()}
        if factor not in info:
            raise KeyError(f"{factor!r} is not among this metadata's factors {sorted(info)}.")
        return info[factor].encoding

    def code_names(self, factor: str | None = None) -> Any:
        """How each of a factor's codes reads, as the record spells it.

        A code is opaque on its own -- ``illum_lux = 3`` says nothing -- and the encoding
        record is what turns it back into ``[0, 12.4)`` or ``rain``. This is that lookup,
        asked directly. It is what :attr:`~dataeval.bias.ParityOutput.insufficient_data`
        reports its levels through and what names the groups of a ``label=`` axis, so a
        caller rendering the same factor themselves gets the same strings those outputs
        use rather than approximating them.

        Names come from the **record**, not from the rows. A bin is named for the interval
        its edges describe, so the same policy reads the same way over a different draw and
        a declared cutoff survives into its own label -- naming a bin after its contents
        rendered ``{"temp_c": [-inf, 0.0, inf]}`` as ``[-40, -0.3]``, with nothing saying
        that zero was where the meaning was.

        Every code the record declares is named, whether or not this sample reached it. An
        empty bin is a finding rather than an absence -- it is how a locked encoding says
        it no longer fits -- and something reporting one still has to name it. Codes the
        record does not declare but the data holds are named too: the out-of-range
        catchalls, and the reserved code for a missing value.

        Parameters
        ----------
        factor : str or None, default None
            The factor to name codes for, or None for every factor.

        Returns
        -------
        dict[int, str], or a Mapping of them
            Code to display name for ``factor``, or a mapping from every factor name to
            its own lookup when ``factor`` is None. Empty for a factor that reached
            neither encoding path, which has codes for nothing to name.

        Raises
        ------
        KeyError
            When ``factor`` is not one of this metadata's factors.

        See Also
        --------
        encoding : The record the names are read from.

        Examples
        --------
        >>> md = Metadata(dataset)
        >>> md.code_names("weather")
        {0: 'clear', 1: 'cloudy', 2: 'rainy'}
        """
        info = self._factor_info
        if factor is not None and factor not in info:
            raise KeyError(f"{factor!r} is not among this metadata's factors {sorted(info)}.")
        wanted = info if factor is None else {factor: info[factor]}
        named = {name: self._names_for(name, entry) for name, entry in wanted.items()}
        return named if factor is None else named[factor]

    def _names_for(self, name: str, info: FactorInfo) -> dict[int, str]:
        """Name every code one factor can produce, declared or merely present."""
        # Imported here rather than at module scope: `_helpers` reaches back into this
        # module, so a module-level import closes the cycle. Same reason `_helpers` defers
        # its own import of `Metadata`.
        from dataeval._helpers import _code_names

        if not (info.is_binned or info.is_digitized):
            # No companion column, so nothing holds codes and there is nothing to name.
            return {}
        present = self._store.frame(info.level)[to_col(name, info)].to_numpy()
        declared = _declared_codes(info.encoding)
        return _code_names(np.concatenate([np.asarray(declared, dtype=present.dtype), present]), info.encoding)

    @property
    def encoding_digest(self) -> str:
        """A fingerprint of every factor's encoding, for attributing a result to it.

        Comparing two passes is only sound if each can say which encoding produced it.
        Without that, a ``Balance`` score that moved between runs is unattributable between
        *my override worked* and *the data changed* — the two readings a caller is trying
        to tell apart. This is cheap enough to carry on every result and stable enough to
        compare across processes.

        Covers what was applied, so it moves when a cutoff is declared, when a vocabulary
        grows, and when a derived encoding is ratified — and not when the rows change under
        an encoding that stayed put.

        Returns
        -------
        str
            Sixteen hex characters over the descriptor, or the digest of an empty encoding
            when no factor carries one.
        """
        payload = json.dumps(encoding_to_mapping(self.encoding()), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()

    def accept(self, *factors: str) -> None:
        """Ratify a derived encoding, so the record says a person looked at it.

        A reviewer who reads the bins DataEval chose and judges them adequate has done
        exactly the semantic work that requiring declared cutoffs is meant to force — and
        has changed no edges. Without somewhere to record that, a descriptor cannot tell
        **nobody looked** from **someone looked and approved**, which is the distinction
        treating binning as policy depends on. Entries still carrying
        ``provenance="derived"`` mark the factors nobody has reviewed yet.

        Accepting also *fixes* the placement. The edges stop being re-derived from each new
        draw and are reapplied instead, which is the point: a cut somebody has approved
        should not move because the next sample was shaped differently. No code changes —
        the same edges over the same values give the same answers — so nothing computed
        before needs recomputing.

        Parameters
        ----------
        *factors : str
            Factors to ratify. With none given, every factor whose encoding DataEval
            derived is accepted, which is the usual move once the whole set has been read.

        Raises
        ------
        KeyError
            When a named factor is not one of this metadata's factors.

        See Also
        --------
        encoding : Read what is about to be accepted.
        export_encoding : Write the ratified record out for review.
        """
        info = self._factor_info
        names = factors or tuple(name for name, entry in info.items() if _is_derived_encoding(entry))
        if unknown := sorted(name for name in names if name not in info):
            raise KeyError(f"{unknown} are not among this metadata's factors {sorted(info)}.")
        for name in names:
            spec = info[name].encoding
            if spec is None or spec.provenance != "derived":
                continue
            accepted = replace(spec, provenance="accepted")
            # Both, deliberately. ``_encoding`` is what survives a re-bin and what gets
            # exported; the cached info is what a reader sees now, without having to
            # trigger a pass that would produce identical codes.
            #
            # Rebound in the cache rather than written into the FactorInfo. That object is
            # shared with every copy `at`, `where`, `agg` and `reencode` make -- they copy
            # the dict, and a dict copy shares its values -- so mutating it ratified the
            # source's encoding as well, from a view the caller was holding precisely to
            # leave the source alone.
            self._encoding[name] = accepted
            self._factor_cache[name] = replace(info[name], encoding=accepted)

    def reencode(self, *, keep_declared: bool = True) -> Self:
        """Re-derive the encodings from the data currently held, on a new instance.

        The escape hatch, and deliberately explicit: re-deriving **moves codes**. A factor
        digitized at five levels over three hundred rows can cross the level budget as rows
        arrive and become binned instead, which changes what every existing code means —
        so classification and placement are reapplied rather than recomputed on every
        ordinary pass, and changing them is something a caller has to ask for.

        Parameters
        ----------
        keep_declared : bool, default True
            Whether cuts a person chose or ratified survive. Re-deriving over a declaration
            discards the semantic work the declaration *is*, so it is opt-out rather than
            the default; pass False to start from the data alone.

        Returns
        -------
        Metadata
            A new instance. This one is untouched, so a result already computed under the
            old codes stays attributable to the encoding that produced it.

        See Also
        --------
        accept : Fix a derived placement instead of re-deriving it.
        """
        fresh = copy.copy(self)
        # The same copy discipline as :meth:`at`: everything mutable this instance reads
        # the store through is owned, or the "this one is untouched" promise above holds
        # only for the two containers that happened to be rebuilt.
        fresh._factors = set(self._factors)
        fresh._factor_cache = dict(self._factor_cache)
        fresh._factors_by_level = {name: set(names) for name, names in self._factors_by_level.items()}
        fresh._dropped_factors = {name: list(reasons) for name, reasons in self._dropped_factors.items()}
        fresh._aggregated_from = dict(self._aggregated_from)
        fresh._identifier_cache = dict(self._identifier_cache)
        fresh._warned_level_rename = False
        # Both spellings of a declaration, or neither. ``continuous_factor_bins`` is as much
        # a cut somebody chose as a ``BinSpec`` is, and it is consulted on the re-derived
        # pass, so leaving it in place made ``keep_declared=False`` keep half the
        # declarations it says it drops.
        fresh._encoding = (
            {name: spec for name, spec in self._encoding.items() if spec.provenance != "derived"}
            if keep_declared
            else {}
        )
        fresh._continuous_factor_bins = dict(self._continuous_factor_bins) if keep_declared else {}
        fresh._reset_bins()
        return fresh

    def export_encoding(self, path: str | Path) -> None:
        """Write every factor's encoding out as a descriptor a person can review.

        The artifact this produces is *policy*: it belongs in a repository next to the code
        that reads the dataset, gets read in a pull request, and is handed back through
        ``Metadata(dataset, encoding=...)`` so a later dataset is encoded against the same
        cuts rather than against its own draw. That is why it is JSON with sorted keys and
        not a member inside the metadata archive — the archive is state, this is the
        decision, and a decision written in a form nobody can read is hard to review.

        Entries still carrying ``provenance="derived"`` mark the factors nobody has
        reviewed yet.

        Parameters
        ----------
        path : str or Path
            Where to write the descriptor.

        See Also
        --------
        encoding : Read the records without writing them.
        """
        Path(path).write_text(encoding_to_json(self.encoding()), encoding="utf-8")

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

        One more is recorded while the factor set is built:

        - ``"cardinality_over_budget"`` — nearly every row of a non-numeric column held a
          different value, so the column identifies rows rather than grouping them, and it
          has no order along which to be cut into groups. A numeric or temporal column in
          the same position is binned instead and kept. Map the values onto a smaller
          vocabulary — a coarser taxonomy, a prefix, a lookup — if the factor is meant to
          be categorical. A merely *wide* vocabulary is kept: a factor whose levels are
          thin for the sample is reported by
          :attr:`~dataeval.bias.ParityOutput.insufficient_data`, not removed.

        Every reason is decided during structuring, so this answers the same whatever has
        been accessed before it.
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
    def factor_values(self) -> NDArray[np.float64]:
        """Factor data as measured, with nothing cut.

        The other representation of the same factors: where :attr:`factor_data` reports
        which interval a temperature fell in, this reports the temperature. Same columns
        in the same order, same rows, read at the same level — so a caller can hold both
        and know that column *j* is the same factor in each.

        Returns
        -------
        NDArray[np.float64]
            Array with shape (n_samples, n_factors). Empty when there are no factors.

        Notes
        -----
        A factor with no numeric reading — a category, whose values are strings — has no
        native form to report, so it contributes its **codes** here, exactly as in
        :attr:`factor_data`. There is nothing lost by that: a category's codes are its own
        alphabet already, and cutting is what this representation exists to avoid.

        Which of the two an evaluator reads is
        :attr:`~dataeval.bias.Balance.factor_source`, and under its default the choice is
        made per factor from :meth:`encoding` — declared cuts keep their codes, because a
        declared cut is a claim about the world and reading past it would discard the
        claim. See :doc:`/concepts/Binning` for what each read costs.

        Examples
        --------
        >>> metadata = Metadata(dataset)
        >>> metadata.factor_values.shape == metadata.factor_data.shape
        True
        """
        info_by_name = self._factor_info
        if not info_by_name:
            return self._empty_projection(np.float64)
        # Read off the store rather than through ``dataframe.schema``, which would build
        # the whole flat frame to answer "is this numeric" about a handful of names.
        schema = {name: self._store.dtype_of(name) for name in info_by_name}
        return self._project([float_col(name, info, schema) for name, info in info_by_name.items()], np.float64)

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

        Structuring is enough: every factor that will not survive to ``factor_data`` is
        removed while the factor set is built, so this and the binned array agree without
        the expensive pass having to run first.
        """
        return self._visible_factors()

    def _visible_factors(self) -> list[str]:
        """Return the filtered factor set, structured but not binned.

        :attr:`factor_names` forces a bin as well and is what callers should use. This
        spelling exists for :meth:`_bin` itself and for the paths it runs through, which
        would otherwise re-enter it.

        Structuring is still forced here: ``_factors`` is empty until it runs, so a
        :meth:`_bin` reached before any other access would otherwise find nothing to do
        and mark itself complete.
        """
        self._structure()
        return sorted(filter(self._filter, self._factors))

    @property
    def _factor_info(self) -> Mapping[str, FactorInfo]:
        """:attr:`factor_info` without the rename warning, for internal callers.

        The warning belongs on the paths that put a :class:`FactorInfo` in *user* hands,
        not on ``factor_data``, which every bias evaluator calls.
        """
        self._structure()
        if not (visible := self._visible_factors()):
            return {}
        self._bin()
        # Read once: `_visible_factors` sorts the whole factor set, `_bin` cannot change
        # which names are visible, and this property is the funnel every array-shaped
        # accessor and `encoding()` route through.
        #
        # Visible *and* processed: a factor is in ``_factors`` from registration but in
        # the cache only once it has a companion column, so the intersection is exactly
        # what factor_data can project.
        return {name: self._factor_cache[name] for name in visible if name in self._factor_cache}

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
        usable = {k for k in visible if not isinstance(self._store.dtype_of(k), pl.List | pl.Struct | pl.Array)}
        identifiers, unannounced = self._identifiers(usable)
        # Assigned before anything is announced. A warning filter turned into an error
        # otherwise raises out of `_structure`, which has already set `_is_structured`, and
        # leaves the instance permanently claiming it has no factors at all.
        self._factors = usable - identifiers
        self._announce_identifier_drops(unannounced)

    def _identifiers(self, names: set[str]) -> tuple[set[str], list[str]]:
        """Record the columns that name their rows instead of grouping them.

        Returns the identifiers among ``names``, and the ones this instance has not yet
        announced. Announced only where the reason is new: `_build_factors` re-runs whenever
        the view or the filters change, and a derived instance inherits the reasons its
        parent recorded, so announcing the whole set would repeat a drop the reader has
        already been told about -- once per `at()`, `where()` or filter set.
        """
        identifiers = {name for name in names if self._is_identifier(name)}
        unannounced = sorted(
            name for name in identifiers if "cardinality_over_budget" not in self._dropped_factors.get(name, ())
        )
        for name in identifiers:
            self._record_over_budget(name)
        return identifiers, unannounced

    def _is_identifier(self, name: str) -> bool:
        """Whether a non-numeric column names its rows rather than grouping them.

        A factor with one distinct value per row is not a category: every contingency table
        over it is a table of ones, every diversity index reads maximal, and its vocabulary
        would be written verbatim into an exported encoding, where a person has to read it.
        A numeric column in the same position is cut into bins instead and kept, which is
        why this asks only about the ones that cannot be.

        The line is **near-uniqueness**, not the level budget. The budget answers "how many
        cells can this sample fill", which is the right question for choosing a *bin count*
        and the wrong one for deciding whether a column is a factor at all — twenty-five
        cities over a hundred images overruns it and is a perfectly good factor.
        :attr:`~dataeval.bias.ParityOutput.insufficient_data` is what reports a thin level;
        deleting the factor forecloses the mechanism that exists to say so.

        Answered once per column and remembered, because it is a question about the column
        and not about which rows are in view. Re-asking it on a derived instance made
        ``where()`` delete a factor its source had kept: the ratio is measured against the
        surviving row count, so twenty-five cities over sixty rows is an ordinary factor and
        the same twenty-two cities over the thirty rows a filter left are an "identifier".
        A filter is a question about rows and must not silently change the factor set.
        """
        if (remembered := self._identifier_cache.get(name)) is not None:
            return remembered
        verdict = self._measure_identifier(name)
        self._identifier_cache[name] = verdict
        return verdict

    def _measure_identifier(self, name: str) -> bool:
        """Read the near-uniqueness of one column off the rows currently held."""
        dtype = self._store.dtype_of(name)
        if dtype is None or dtype.is_numeric() or dtype.is_temporal():
            return False
        level = self._factor_level(name)
        column = self._store.frame(level)[name]
        height = column.len()
        # Floored so a small sample cannot make an ordinary vocabulary look like an
        # identifier: five values over eight rows is near-unique by ratio alone.
        return height > 0 and column.n_unique() > max(MIN_LEVEL_BUDGET, height // 2)

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
        # A fresh set of columns, so no verdict carried from a previous structuring holds.
        self._identifier_cache = {}
        # ``class_labels`` and ``item_indices`` are deliberately not stored: both are
        # columns the store already holds, and the properties read them from there.
        self._dropped_factors = {name: list(reasons) for name, reasons in data.dropped_factors.items()}
        self._is_structured = True

        self._build_factors()

    def _apply_recorded(self, col: str, data: NDArray, spec: FactorEncoding) -> tuple[NDArray[np.int64], FactorInfo]:
        """Encode one factor against the record the caller supplied.

        Nothing is measured here except the values themselves: the cut and the vocabulary
        both come from the record. A :class:`~dataeval.types.LevelSpec` can still grow, and
        the grown spec is what is stored, so the factor's own record always describes the
        codes actually in the column.
        """
        # `factor_type` is a fact about the variable, not about the map: an integer count is
        # discrete however its codes were produced. Read from the values, which is cheap and
        # deterministic, while the placement -- the expensive and unstable half -- comes from
        # the record. Assuming a kind here instead relabelled every restored discrete factor
        # as categorical.
        numeric = bool(np.issubdtype(data.dtype, np.number))
        if isinstance(spec, BinSpec):
            if not numeric:
                # Said here rather than left to the cast. Digitizing a column of strings
                # against float edges raises `could not convert string to float: 'fog'`
                # from inside NumPy, which names a value and neither the factor nor the
                # record that sent it there.
                raise TypeError(
                    f"The encoding recorded for factor {col!r} is a BinSpec, which cuts an ordered "
                    f"quantity, but the factor holds {data.dtype} values. Record a LevelSpec for a "
                    "non-numeric factor, or drop the entry to encode it from its own values.",
                )
            kind = "continuous" if is_continuous(data) else "discrete"
            return apply_bin_spec(data, spec).astype(np.int64), FactorInfo(kind, is_binned=True, encoding=spec)
        codes, grown = apply_level_spec(data, spec, strict=self._strict)
        # Written back, not only cached. `_encoding` is what a re-bin reapplies and what
        # `new()` hands the next dataset, so leaving the pre-growth vocabulary there sends
        # the next dataset an alphabet this one has already outgrown.
        if grown is not spec:
            self._encoding[col] = grown
        return codes, FactorInfo("discrete" if numeric else "categorical", is_digitized=True, encoding=grown)

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
        data = _as_orderable(data)
        # A recorded encoding is reapplied rather than re-derived. That is the whole of
        # what a descriptor buys: the same value gets the same code in a dataset the cut
        # was never fitted to, so two Metadata built from one record share an alphabet.
        if (recorded := self._encoding.get(col)) is not None:
            return self._apply_recorded(col, data, recorded)

        if col in factor_bins:
            binned, spec = digitize_data(data, factor_bins[col])
            return binned.astype(np.int64), FactorInfo("continuous", is_binned=True, encoding=spec)

        distinct, ordinal = np.unique(data, return_inverse=True)
        # No de-duplication argument: one value per entity means no propagated repeats
        # for is_continuous to mistake for discrete support.
        # No factor carries more levels than the sample can fill, whichever path bins it.
        budget = level_budget(data.shape[0])
        # `np.unique` sorts, so the codes and the recorded order agree and both match what
        # a reader sorting the values would expect. The record is what makes that order
        # survive: extending a factor appends to `levels` rather than re-sorting, which is
        # the one thing that keeps a code meaning what it meant.
        if not np.issubdtype(data.dtype, np.number):
            spec = LevelSpec(levels=tuple(distinct.tolist()), provenance="derived")
            return ordinal.astype(np.int64), FactorInfo("categorical", is_digitized=True, encoding=spec)
        return self._classify_numeric(col, data, distinct, ordinal, budget)

    def _classify_numeric(
        self,
        col: str,
        data: NDArray,
        distinct: NDArray,
        ordinal: NDArray,
        budget: int,
    ) -> tuple[NDArray[np.int64], FactorInfo]:
        """Choose the encoding for an ordered factor nobody declared one for."""
        if is_continuous(data):
            _logger.warning(
                f"A user defined binning was not provided for {col}. "
                f"Using the {self.auto_bin_method} method to discretize the data. "
                "It is recommended that the user rerun and supply the desired "
                "bins using the continuous_factor_bins parameter.",
            )
            binned, spec = bin_data(data, self.auto_bin_method, max_bins=budget)
            return binned.astype(np.int64), FactorInfo("continuous", is_binned=True, encoding=spec)
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
            binned, spec = bin_data(data, self.auto_bin_method, max_bins=budget)
            return binned.astype(np.int64), FactorInfo("discrete", is_binned=True, encoding=spec)
        # Digitized so factor_data holds non-negative integers, which np.bincount in the
        # downstream bias evaluators requires.
        spec = LevelSpec(levels=tuple(distinct.tolist()), provenance="derived")
        return ordinal.astype(np.int64), FactorInfo("discrete", is_digitized=True, encoding=spec)

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

    def _warn_unknown_factor_keys(self, factor_bins: Mapping[str, int | Sequence[float]]) -> None:
        """Name any declared encoding that matches no column, whichever way it was declared.

        Ignored rather than rejected, because a caller reusing one configuration across
        several datasets will legitimately name factors a given one does not carry — which
        is the same reason it has to be *said*. A descriptor that names nothing is not a
        no-op with a small cost: every factor it was meant to pin falls back to a cut
        derived from this draw, which is precisely the drift the descriptor exists to
        prevent, and the failure looks identical to never having passed one.

        All three declaration channels, in one check. ``continuous_factor_bins`` had it and
        the two the encoding record arrives through — ``encoding`` and ``factor_levels`` —
        did not, so a renamed or misspelled factor was caught on the older spelling and
        silently dropped on the newer ones.
        """
        columns = set(self._store.columns)
        unknown = sorted((set(factor_bins) | set(self._encoding)) - columns)
        if not unknown:
            return
        _logger.warning(
            f"The keys - {set(unknown)} - were given an encoding but are not columns in the metadata "
            "DataFrame. Unknown keys will be ignored.",
        )
        warnings.warn(
            f"{_name_list(unknown)} {_was_were(unknown)} given an encoding but "
            f"{'is' if len(unknown) == 1 else 'are'} not a factor of this metadata "
            f"{sorted(columns & set(self._factors)) or sorted(columns)[:8]}. The encoding is ignored, so "
            "those factors are cut from this draw instead of from what was declared — check the names "
            "against `factor_names`.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )

    def _announce_identifier_drops(self, dropped: Sequence[str]) -> None:
        """Say out loud that a factor was removed for being an identifier rather than a category.

        Emitted from structuring, where the decision is made, so it fires once per instance
        rather than once per re-bin. Same reasoning as :meth:`_announce_derived_encodings`
        for why it is a warning at all.
        """
        if not dropped:
            return
        names = sorted(dropped)
        warnings.warn(
            f"{_name_list(names)} {_was_were(names)} dropped: nearly every row holds a different "
            "value, so the column identifies rows rather than grouping them, and it is not numeric "
            "so there is no order along which to cut it into groups. Map the values onto a smaller "
            "vocabulary to keep the factor. See Metadata.dropped_factors.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )

    def _measure_fit(self, factor_info: Mapping[str, FactorInfo]) -> tuple[list[str], list[str]]:
        """Count, per declared encoding, the bins nothing reached and the levels the sample cannot fill.

        Both counts are read at the factor's **own** level, which is where its codes were
        assigned and the only place it holds one value per entity. Measuring the occupancy
        there and the budget at the view compared two different samples: a unit-level factor
        over five hundred images was judged against the budget for ten thousand detections,
        which is the number :meth:`_classify_factor` deliberately does not use either.

        The two counts do not cover the same factors. Emptiness is a question about a *cut*
        and is asked of :class:`~dataeval.types.BinSpec` alone; fineness is a question about
        a contingency table, which a declared vocabulary fills exactly as a declared cut
        does, so it is asked of both.
        """
        empty: list[str] = []
        overcut: list[str] = []
        for name, info in sorted(factor_info.items()):
            spec = info.encoding
            # A derived encoding is skipped because nobody claimed anything for this report
            # to hold up against the data. Both automatic *binning* paths cut against
            # `level_budget` and cannot overrun it anyway; a derived *vocabulary* can — a
            # non-numeric column is kept on near-uniqueness rather than on the budget, so
            # sixty cities over two hundred rows is retained by design — and whether that
            # deserves the same notice is a scope question this report does not settle.
            # What it does cover is what somebody *declared*, which is the one path that
            # takes a caller at their word: taking them at their word is the point, so the
            # answer is to say what it cost rather than to quietly cut it back.
            if spec is None or spec.provenance == "derived":
                continue
            codes = self._store.frame(info.level)[to_col(name, info)].to_numpy()
            # A declared vocabulary is not asked the emptiness question at all, so it keeps
            # every code. An unused bin says the cut no longer describes the data; an unused
            # *level* is the ordinary case for a closed taxonomy declared once and applied to
            # every subset of it, so asking would fire the warning on its intended use.
            present, unused = _unused_bins(spec, codes) if isinstance(spec, BinSpec) else (codes, None)
            if unused:
                empty.append(f"{name} ({unused})")
            # Whether the encoding is too fine for the sample is asked of both kinds, over
            # every code that reaches a contingency table — catchalls included, because each
            # one is a cell. A vocabulary reaches a table the same way a cut does, so a
            # `factor_levels` finer than the sample fails in exactly the same way, and went
            # unreported while the identical shape on `continuous_factor_bins` did not.
            levels = len(np.unique(present))
            rows = self._store.height(info.level)
            budget = level_budget(rows) if rows else 0
            if budget and levels > budget:
                overcut.append(f"{name} ({levels} levels over {rows} rows)")
        return empty, overcut

    def _announce_fit(self, factor_info: Mapping[str, FactorInfo]) -> None:
        """Report where an encoding does not suit the data it was just applied to.

        Two failures with one shape, which is why they are one report. A locked descriptor
        **degrades**: bins that were well-occupied when it was written can empty out as the
        data moves under it, and saying so is more useful than re-deriving quietly —
        re-deriving is :meth:`reencode`, and it is the caller's decision because it moves
        codes. And an encoding can **never have fitted**: a declared count finer than the
        sample supports leaves a contingency table with more cells than rows, and a
        chance-corrected association over one of those tends to zero whatever the data says.

        Reported rather than fixed, in both cases. The cut is the caller's policy, so what
        a record gives them is notice that it no longer matches the data, not a new fit.

        **No replacement count is offered, deliberately.** The obvious thing to name is the
        budget this fires against, and it would be read as a recommendation. It is not one:
        measured on a pair whose true dependence is 0.810, the count recovering the most
        association is about 8 at n=200 while the budget is 20, and cutting at the budget
        instead reports 0.507. The budget is a ceiling past which a table stops describing
        the data at all; the count that reads best sits well under it and moves with the
        sample size, and DataEval has no defensible estimate of it. A number that reads as a
        recommendation without being one would mislead more than saying only that the score
        moves with the cut, so that is what is said, with the measurements a page away.
        """
        empty, overcut = self._measure_fit(factor_info)

        if empty:
            warnings.warn(
                f"Declared cuts left bins unused: {', '.join(empty)}. The encoding still applies "
                "and the codes are unchanged; this is the data no longer matching the policy, not "
                "an error. Call reencode() to refit — which moves codes — or leave it and read the "
                "empty groups as the finding they are.",
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
        if overcut:
            warnings.warn(
                f"Declared encodings are finer than the sample supports: {', '.join(overcut)}. "
                "A contingency table with more cells than rows records which cells were hit rather "
                "than anything about the factor, so a chance-corrected association over one reads "
                "near zero whatever the data holds, and a factor encoded this finely will look "
                "uncorrelated with everything. There is no count to recommend in its place: the "
                "resolution that recovers the most association moves with the sample size and sits "
                "well below the ceiling this warning fires against. See the binning concept page "
                "for the measurements, and choose a coarser encoding deliberately.",
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )

    def _announce_derived_encodings(self, factor_info: Mapping[str, FactorInfo]) -> None:
        """Say out loud what structuring decided on the caller's behalf.

        The advice already existed on both auto-binning paths and reached nobody: both are
        ``_logger.warning``, and :mod:`dataeval` attaches a ``NullHandler`` to its root
        logger, which suppresses Python's last-resort stderr handler. Advice that reaches
        only the callers who configured logging reaches very few of them.

        Aggregated rather than per factor: twelve continuous factors would otherwise emit
        twelve near-identical warnings during one construction, which is how a warning
        teaches people to filter it. The per-factor detail stays in the log, where a reader
        who wants it is already looking.

        ``UserWarning``, not ``DeprecatedWarning`` — nothing here is going away. Auto-binning
        stays supported; it is being reported, not retired.
        """
        auto_binned = sorted(name for name, info in factor_info.items() if _is_auto_binned(info))
        if auto_binned:
            warnings.warn(
                f"{_name_list(auto_binned)} {_was_were(auto_binned)} binned automatically "
                f"({self._auto_bin_method!r}) because no bins were declared. The bin "
                "count is derived from the data, so it is not stable across samples and the "
                "same factor measured twice may not be comparable. Declare cutoffs with "
                'continuous_factor_bins={"' + auto_binned[0] + '": [...]} to control this.',
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )

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

        self._warn_unknown_factor_keys(factor_bins)

        column_set = set(self._store.columns)
        factors_to_process = [col for col in self._visible_factors() if not {binned(col), digitized(col)} & column_set]
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
        # After the flag, so a warning filter turned into an error cannot leave the object
        # half-binned and force the whole pass to run again on the next access.
        self._announce_derived_encodings(factor_info)
        self._announce_fit(factor_info)

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

    def _record_over_budget(self, name: str) -> None:
        """Record a non-numeric factor dropped for carrying more levels than the sample supports.

        Reason-only, like its two siblings: :meth:`_build_factors` decides which names are
        in the analysed set and simply does not admit this one. Writing back into
        ``_factors_by_level`` from a later pass would edit the registry structuring owns,
        which made the same object serialize differently depending on what had been read
        from it first.
        """
        reasons = self._dropped_factors.setdefault(name, [])
        if "cardinality_over_budget" not in reasons:
            reasons.append("cardinality_over_budget")

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
        # A replaced column holds different values, so its remembered verdict is stale.
        for factor in resolved:
            self._identifier_cache.pop(factor.name, None)

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

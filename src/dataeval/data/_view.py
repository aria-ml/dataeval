__all__ = []

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from dataeval._experimental import warn_experimental_once
from dataeval.data._reads import Reads, recordable, skeleton
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset, Dataset, DatasetMetadata
from dataeval.types import ReprMixin, SourceIndex
from dataeval.utils._internal import read_dataset
from dataeval.utils.data import DatasetKind, validate_dataset

_TDatum = TypeVar("_TDatum")

Transform = Callable[[Any], Any]

IndexedTransform = Callable[[int, Any], Any]


class _Restricted:
    """A transform registered with `where`, in the ``(index, datum)`` form :meth:`View.read` applies.

    A class rather than a closure so that a view holding it pickles, which a multi-worker DataLoader needs.
    """

    def __init__(self, fn: Transform, where: set[int] | None) -> None:
        self.fn = fn
        self.where = where

    def __call__(self, index: int, datum: Any) -> Any:
        """Apply the transform if `index` is one it targets, and return `datum` unchanged otherwise."""
        if self.where is None or index in self.where:
            return self.fn(datum)
        return datum


class _EachIndex:
    """The transforms :meth:`View.map_each` registers, looked up by source index on each read.

    A class rather than a closure so that a view holding it pickles, which a multi-worker DataLoader needs.
    """

    def __init__(self, chosen: Mapping[int, Transform]) -> None:
        self.chosen = dict(chosen)

    def __call__(self, index: int, datum: Any) -> Any:
        """Apply the transform chosen for `index`, and return `datum` unchanged if none was."""
        return self.chosen[index](datum) if index in self.chosen else datum


def _aggregate_required_kind(kinds: Iterable[DatasetKind | None]) -> DatasetKind | None:
    """Fold per-op MAITE requirements into the single strictest kind to validate.

    ``None`` means no op inspects targets (validation is skipped, keeping image-only
    datasets usable); a specific kind (``classification`` / ``object_detection`` /
    ``segmentation`` / ``multiobject_tracking``) wins over the generic ``any_target``.
    """
    present: list[DatasetKind] = [k for k in kinds if k is not None]
    if not present:
        return None
    specific: list[DatasetKind] = [k for k in present if k != "any_target"]
    return specific[0] if specific else "any_target"


class Operation(ReprMixin, ABC):
    """Abstract base for a single step in a :class:`View` pipeline.

    A :class:`View` is built from an ordered list of operations, each applied in
    turn to the source dataset. An operation may do any combination of three things
    by mutating the view it is handed in :meth:`apply`:

    - **change cardinality / order** — set or reorder :attr:`View.selection`, the list
      of surviving source indices (filter, reorder, limit, resample).
    - **rewrite content** — register a per-datum transform via :meth:`View.map`; a
      transform may target every datum or only a subset of source indices, or,
      through :meth:`View.map_each`, give each source index its own transform.
    - **rewrite metadata** — override :meth:`apply_metadata`, folded once at build.

    Operations run in the order given, and each reads the source *through* the
    transforms registered by earlier operations (see :meth:`View.read`), so a filter
    placed after :class:`Relabel` sees the relabeled targets.

    Attributes
    ----------
    requires : DatasetKind or None, default None
        The MAITE datum shape this operation needs when it reads each datum's
        *target*. :class:`View` aggregates the ``requires`` of all operations and
        validates the source dataset once, upfront, raising
        :class:`~dataeval.exceptions.MaiteShapeError` before any operation runs.
        Leave as ``None`` when the operation ignores targets. An operation whose need
        depends on its arguments sets it on the instance in ``__init__``, as
        :class:`~dataeval.data.Indices` does, before the view reads it.

    Examples
    --------
    A custom operation that keeps only even-indexed items:

    >>> from dataeval.data import View, Operation
    >>>
    >>> class KeepEven(Operation):
    ...     def apply(self, view: View) -> None:
    ...         view.selection = [i for i in view.selection if i % 2 == 0]
    >>>
    >>> view = View(dataset, [KeepEven()])  # doctest: +SKIP
    """

    requires: DatasetKind | None = None

    @property
    def invalidates(self) -> ImageStats:
        """The statistics this operation makes describe *the transform* rather than the data.

        A resize, for example, invalidates ``DIMENSION_WIDTH`` (it now reports the resize
        target) and ``VISUAL_SHARPNESS`` (it now measures the interpolation kernel). Quality
        evaluators intersect this against the statistics they were asked to compute and warn
        on the overlap; it never changes what is computed.

        Unlike :attr:`requires`, which is an attribute an operation may set per instance,
        this is a read-only property, and concrete operations override it as one — what a transform invalidates
        depends on its constructor arguments (``Resize(size, mode="stretch")`` leaves the
        pixel statistics alone; ``mode="pad"`` does not). A wrapper *dataset* may declare it
        directly instead (see :class:`~dataeval.data.DetectionCrops`), as either a class
        attribute or a property.
        """
        return ImageStats.NONE

    def apply_metadata(self, metadata: DatasetMetadata) -> DatasetMetadata:
        """Return possibly-updated dataset-level metadata (default: unchanged)."""
        return metadata

    @abstractmethod
    def apply(self, view: "View[Any]") -> None:
        """Apply this operation by mutating ``view`` in place."""
        ...


class View(AnnotatedDataset[_TDatum]):
    """
    Dataset view built from an ordered pipeline of :class:`Operation`.

    Wraps a source dataset and applies operations that filter, reorder, transform,
    or relabel it, producing a subset/rewritten view without modifying the original.

    Parameters
    ----------
    dataset : Dataset[_TDatum]
        Source dataset to apply operations to. Any object implementing the
        :class:`~dataeval.protocols.Dataset` interface -- indexed access via
        ``__getitem__`` and ``__len__`` -- is accepted, including a bare,
        image-only dataset. A target-bearing (annotated) dataset is only
        required when an applied operation reads per-datum targets, i.e. when an
        operation declares :attr:`Operation.requires` (e.g. :class:`Filter`,
        :class:`ClassBalance`, :class:`Relabel`). Target-free operations such as
        :class:`Limit`, :class:`Shuffle`, :class:`Reverse`, and :class:`Indices`
        operate on a plain dataset. Operations that need targets trigger an
        upfront validation of the source dataset, raising
        :class:`~dataeval.exceptions.MaiteShapeError` if the targets are missing.
    operations : Operation or Sequence[Operation] or None, default None
        Operations to apply, **in order**. When ``None`` the view is an unfiltered,
        untransformed pass-through.
    record_reads : bool, default False
        Record each datum's structure, without its pixels, as the view reads it, so
        :class:`~dataeval.Metadata` built from the view afterward does not read the dataset
        again. Run the pass that reads the pixels first, such as :class:`~dataeval.Embeddings`
        or :class:`~dataeval.quality.Outliers`, and build the metadata after it: in the other
        order nothing is saved, because the pass that follows needs the pixels. The record
        holds each datum's target and metadata for the life of the view, so it costs memory in
        proportion to them. It keeps the first read of each item, so a dataset whose targets
        change between reads gives metadata that differs from a fresh walk. Set it on the
        outermost view that both consumers receive. It has no effect on a tracking dataset, on
        an item that is not an ``(item, target, metadata)`` tuple, or on a pickled copy. When
        the record holds only part of the view, as when no pass ran or one was built but never
        computed, :class:`~dataeval.Metadata` raises a ``UserWarning`` with the counts.
        Setting it raises :class:`~dataeval.exceptions.ExperimentalWarning` once per process.

        .. versionadded:: 1.3

    Notes
    -----
    Operations are applied strictly in the order provided, each one seeing the result
    of the previous. Order is therefore meaningful — ``[Relabel(...), Filter(classes=[0])]``
    filters on the *relabeled* classes, while the reverse filters on the source ones,
    and ``[Limit(1000), Shuffle(), Limit(100)]`` keeps a random 100 of the first 1000.

    A view does not inherit the source dataset's attributes. It exposes only its own
    surface — :attr:`metadata`, :attr:`selection`, :attr:`source`, :attr:`root`,
    :attr:`operation_groups` — so a value an operation rewrote has exactly one reachable
    form. ``Relabel`` folds a new ``index2label`` into :attr:`metadata`; an inherited
    ``view.index2label`` would still hold the source's, and the two would disagree.
    Reach anything source-specific through :attr:`source`.

    Examples
    --------
    >>> from dataeval.data import View, Filter, Limit

    >>> view = View(dataset, [Filter(classes=[0, 2]), Limit(size=5)])
    >>> print(view)  # doctest: +SKIP
    View Dataset
    ------------
        Operations: [Filter(where=None, classes=[0, 2], drop='both'), Limit(size=5)]
        Selected Size: 5
    """

    _dataset: Dataset[_TDatum]
    _operations: Sequence[Operation]
    selection: list[int]
    _reads: dict[int, Any] | None = None
    """Skeletons of this view's datums (see ``dataeval.data._reads``), keyed by source index.

    Created by a :class:`Filter` given `where`, which structures every datum anyway, or, when `record_reads` is set, by
    the first datum the view reads; a :class:`Filter` given `classes` alone keeps one current but never starts one. Read
    by :class:`~dataeval.Metadata` in place of the view. Keyed by source index, so a change of `selection` never
    invalidates it; any new transform does, and `record_reads` then starts it again.
    """
    _record_reads: bool = False

    def __init__(  # noqa: C901
        self,
        dataset: Dataset[_TDatum],
        operations: Operation | Sequence[Operation] | None = None,
        *,
        record_reads: bool = False,
    ) -> None:
        if record_reads:
            warn_experimental_once("View(record_reads=True)", details="The keyword and what it records may change.")
        self._record_reads = record_reads
        self._dataset = dataset
        self._operations = self._normalize(operations)
        self.selection = list(range(len(dataset)))
        self._transforms: list[IndexedTransform] = []

        # Fail fast if any operation requires a target the source dataset cannot provide.
        required_kind = _aggregate_required_kind(op.requires for op in self._operations)
        if required_kind is not None and len(dataset) > 0:
            validate_dataset(dataset, expected=required_kind, caller="View")

        # Fold dataset-level metadata once, in order (e.g. produce a new index2label).
        _metadata = dict(getattr(dataset, "metadata", {}))
        if "id" not in _metadata:
            _metadata["id"] = dataset.__class__.__name__
        metadata = DatasetMetadata(**_metadata)
        for op in self._operations:
            metadata = op.apply_metadata(metadata)
        self._metadata = metadata

        # Run operations in order; each reads through transforms registered by earlier ops.
        for op in self._operations:
            op.apply(self)

    @staticmethod
    def _normalize(operations: Operation | Sequence[Operation] | None) -> list[Operation]:
        if not operations:
            return []
        return [operations] if isinstance(operations, Operation) else list(operations)

    # -- levers an Operation uses ---------------------------------------------
    def map(self, fn: Transform, *, where: set[int] | None = None) -> None:
        """Register a per-datum content transform. ``where=None`` targets all indices."""
        self._reads = None
        self._transforms.append(_Restricted(fn, where))

    def map_each(self, transforms: Mapping[int, Transform]) -> None:
        """
        Register a different content transform for each of several source indices, as one step.

        :meth:`map` gives every datum it targets the same transform. An operation that rewrites each datum
        differently, such as :class:`~dataeval.data.Indices` removing different detections from each image, would
        need one :meth:`map` call per datum, and every read would then check every one of them. This registers
        them together, and a read looks up its own index once.

        Parameters
        ----------
        transforms : Mapping[int, Callable[[Any], Any]]
            The transform for each source index, keyed as :meth:`map`'s `where` is: by index into the dataset this
            view wraps. An index the mapping does not hold is left unchanged.

        Notes
        -----
        The step runs in registration order with the ones :meth:`map` registers. The mapping is copied, so changing
        it afterwards changes nothing. Registering a transform clears the view's read cache, since it can change what
        a read returns.

        Examples
        --------
        >>> from dataeval.data import View

        Upper-case the first datum and title-case the third, leaving the second as it is:

        >>> view = View(["ab", "cd", "ef"])
        >>> view.map_each({0: str.upper, 2: str.title})
        >>> list(view)
        ['AB', 'cd', 'Ef']
        """
        self._reads = None
        self._transforms.append(_EachIndex(transforms))

    def read(self, src_index: int) -> _TDatum:
        """Read one source datum with all currently-registered transforms applied."""
        datum = self._dataset[src_index]
        for transform in self._transforms:
            datum = transform(src_index, datum)
        return datum

    def _skeletons(self) -> list[Any]:
        """Return the datums without their pixels, in selection order, starting or extending the read cache."""
        reads = {} if self._reads is None else self._reads
        wanted = list(dict.fromkeys(self.selection))
        missing = [index for index in wanted if index not in reads]
        for index, datum in zip(missing, self._read_each(missing, skeleton), strict=True):
            reads[index] = datum
        self._reads = reads
        return [reads[index] for index in self.selection]

    def _read_each(self, indices: Iterable[int], use: Callable[[Any], Any]) -> Iterator[Any]:
        """Yield ``use(self.read(index))`` for each index in order, in several threads when ``load_threads`` is set.

        `use` runs in the loading thread, so a result that no longer holds the pixels, a skeleton or a verdict, is all
        that waits for the caller. ``read_dataset`` in ``utils._internal`` says when the reads stay serial, such as for
        a tracking dataset. With no threads set this is the plain loop.
        """
        return read_dataset(self, indices, use, read=self.read)

    def _structuring_source(self) -> Any:
        """Return what :class:`~dataeval.Metadata` structures in place of this view: its cached reads if it has any.

        A view with no cache answers itself, so building metadata never starts one.
        """
        if self._reads is None:
            return self
        if self._record_reads:
            self._warn_unrecorded(self._reads)
        return Reads(self._skeletons(), self.metadata)

    def _warn_unrecorded(self, reads: dict[int, Any]) -> None:
        """Warn that :class:`~dataeval.Metadata` reads from the dataset the selected items the record does not hold.

        Only the metadata path warns: a :class:`Filter` that completes the record while the view is built reads what
        it must anyway, and names no consumer the reader asked to save.
        """
        wanted = dict.fromkeys(self.selection)
        missing = sum(index not in reads for index in wanted)
        if not missing:
            return
        # Imported here: dataeval._metadata imports this module, so a module-level import would be circular.
        from dataeval._metadata._metadata import _caller_stacklevel

        warnings.warn(
            f"View(record_reads=True) held {len(wanted) - missing} of the {len(wanted)} dataset items it selects, so "
            f"Metadata reads the other {missing} from the dataset. Run the pass that reads the pixels on this view, "
            "such as Embeddings(view, ...).compute(), before building Metadata from it.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )

    def _keep_reads(self, reads: dict[int, Any] | None, changed: Mapping[int, Transform]) -> None:
        """Reinstate a read cache an operation's own transform cleared, pruned to the selection.

        ``changed`` holds the transform the operation registered for each source index, applied here to that index's
        skeleton. Only a transform that never reads ``datum[0]`` may be passed, since a skeleton holds no pixels.
        """
        if reads is None:
            return
        kept = {index: reads[index] for index in dict.fromkeys(self.selection) if index in reads}
        for index, transform in changed.items():
            if index in kept:
                kept[index] = transform(kept[index])
        self._reads = kept

    # -- dataset interface ----------------------------------------------------
    @property
    def source(self) -> Dataset[_TDatum]:
        """The dataset this view directly wraps -- one link up the chain.

        A view exposes only its own surface: :attr:`metadata` (folded through the
        operations, so a rewritten ``index2label`` is the one you get), :attr:`selection`,
        :attr:`operation_groups`, and this. Nothing else crosses the boundary -- reach
        anything source-specific explicitly through ``view.source``, which cannot go
        stale the way a copied attribute would.

        See Also
        --------
        root : the dataset at the *bottom* of the chain, however deep.
        """
        return self._dataset

    @property
    def _source_items(self) -> NDArray[np.intp]:
        """The source item behind each view item -- the private hook ``root_items`` walks."""
        return np.asarray(self.selection, dtype=np.intp)

    @property
    def root(self) -> Dataset[_TDatum]:
        """The original dataset at the bottom of any :class:`View` wrapping chain.

        See Also
        --------
        source : the dataset one link up, whether or not it is itself a view.
        """
        current: Dataset[_TDatum] = self
        while isinstance(current, View):
            current = current.source
        return current

    @property
    def operation_groups(self) -> list[list[Operation]]:
        """Operation lists from each construction call, innermost (oldest) first.

        ``View(View(base, [A, B]), [C])`` returns ``[[A, B], [C]]``. Empty wrappers
        contribute nothing. The grouping matches the user's nesting intent and is the
        natural shape for sidecar metadata.
        """
        groups: list[list[Operation]] = []
        current: Dataset[_TDatum] = self
        while isinstance(current, View):
            if current._operations:
                groups.append(list(current._operations))
            current = current.source
        groups.reverse()
        return groups

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata information, including any operation rewrites."""
        return self._metadata

    def __getitem__(self, index: int) -> _TDatum:
        src_index = self.selection[index]
        datum = self.read(src_index)
        if self._record_reads:
            self._record(src_index, datum)
        return datum

    def _record(self, src_index: int, datum: Any) -> None:
        """Keep the skeleton of a datum a pass just read, the first time only, so structuring it later reads nothing."""
        reads = self._reads
        if (reads is None or src_index not in reads) and recordable(datum):
            if reads is None:
                reads = self._reads = {}
            reads[src_index] = skeleton(datum)

    def __iter__(self) -> Iterator[_TDatum]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.selection)

    def __repr__(self) -> str:
        operations = ", ".join(repr(op) for op in self._operations)
        return f"{type(self).__name__}(dataset={self._dataset!r}, operations=[{operations}], len={len(self)})"

    def __str__(self) -> str:
        nt = "\n    "
        title = f"{type(self).__name__} Dataset"
        sep = "-" * len(title)
        operations = f"Operations: [{', '.join(str(op) for op in self._operations)}]"
        return f"{title}\n{sep}{nt}{operations}{nt}Selected Size: {len(self)}\n\n{self._dataset}"

    def __getstate__(self) -> dict[str, Any]:
        """Pickle without the read cache or the recording flag.

        A DataLoader worker is then not sent every target, and does not record into a copy that is thrown away.
        """
        state = self.__dict__.copy()
        state["_reads"] = None
        state["_record_reads"] = False
        return state

    def resolve_indices(self, indices: int | SourceIndex | Sequence[int | SourceIndex] | None = None) -> list[int]:
        """
        Return the list of source dataset indices after all operations have been applied.

        Parameters
        ----------
        indices : int or SourceIndex or Sequence[int | SourceIndex] or None, default None
            Specific indices from the view to resolve to source indices. When None,
            returns all selected indices.

        Returns
        -------
        list[int]
            The list of selected indices from the original dataset.
        """
        if indices is None:
            return self.selection.copy()

        resolved_indices: list[int] = []
        for idx in [indices] if isinstance(indices, int | SourceIndex) else indices:
            idx = idx.item if isinstance(idx, SourceIndex) else idx
            if idx is None or idx < 0 or idx >= len(self.selection):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self._dataset)}")
            resolved_indices.append(self.selection[idx])
        return resolved_indices

__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset, Dataset, DatasetMetadata
from dataeval.types import ReprMixin, SourceIndex
from dataeval.utils._validate import DatasetKind, aggregate_required_kind, validate_dataset

_TDatum = TypeVar("_TDatum")

Transform = Callable[[Any], Any]


class Operation(ReprMixin, ABC):
    """Abstract base for a single step in a :class:`View` pipeline.

    A :class:`View` is built from an ordered list of operations, each applied in
    turn to the source dataset. An operation may do any combination of three things
    by mutating the view it is handed in :meth:`apply`:

    - **change cardinality / order** — set or reorder :attr:`View.selection`, the list
      of surviving source indices (filter, reorder, limit, resample).
    - **rewrite content** — register a per-datum transform via :meth:`View.map`; a
      transform may target every datum or only a subset of source indices.
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
        Leave as ``None`` when the operation ignores targets.

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

        Unlike :attr:`requires`, which is a static class attribute, this is a read-only
        property, and concrete operations override it as one — what a transform invalidates
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
        operation declares :attr:`Operation.requires` (e.g. :class:`ClassFilter`,
        :class:`ClassBalance`, :class:`Relabel`). Target-free operations such as
        :class:`Limit`, :class:`Shuffle`, :class:`Reverse`, and :class:`Indices`
        operate on a plain dataset. Operations that need targets trigger an
        upfront validation of the source dataset, raising
        :class:`~dataeval.exceptions.MaiteShapeError` if the targets are missing.
    operations : Operation or Sequence[Operation] or None, default None
        Operations to apply, **in order**. When ``None`` the view is an unfiltered,
        untransformed pass-through.

    Notes
    -----
    Operations are applied strictly in the order provided, each one seeing the result
    of the previous. Order is therefore meaningful — ``[Relabel(...), ClassFilter([0])]``
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
    >>> from dataeval.data import View, ClassFilter, Limit

    >>> view = View(dataset, [ClassFilter(classes=[0, 2]), Limit(size=5)])
    >>> print(view)  # doctest: +SKIP
    View Dataset
    ------------
        Operations: [ClassFilter(classes=[0, 2], filter_detections=True), Limit(size=5)]
        Selected Size: 5
    """

    _dataset: Dataset[_TDatum]
    _operations: Sequence[Operation]
    selection: list[int]

    def __init__(
        self,
        dataset: Dataset[_TDatum],
        operations: Operation | Sequence[Operation] | None = None,
    ) -> None:
        self._dataset = dataset
        self._operations = self._normalize(operations)
        self.selection = list(range(len(dataset)))
        self._transforms: list[tuple[Transform, set[int] | None]] = []

        # Fail fast if any operation requires a target the source dataset cannot provide.
        required_kind = aggregate_required_kind(op.requires for op in self._operations)
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
        self._transforms.append((fn, where))

    def read(self, src_index: int) -> _TDatum:
        """Read one source datum with all currently-registered transforms applied."""
        datum = self._dataset[src_index]
        for fn, where in self._transforms:
            if where is None or src_index in where:
                datum = fn(datum)
        return datum

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
        return self.read(self.selection[index])

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

__all__ = []

from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

from dataeval.protocols import AnnotatedDataset, Dataset, DatasetMetadata
from dataeval.types import ReprMixin
from dataeval.utils._validate import DatasetKind, aggregate_required_kind, validate_dataset

_TDatum = TypeVar("_TDatum")


class Conformer(ReprMixin, Generic[_TDatum]):
    """Base class for a per-datum conformation applied by :class:`Conform`.

    A conformer rewrites the *content* of a dataset to make it conform to a target
    schema — relabeling to a reference vocabulary now; renaming metadata factors,
    converting metadata values, or mutating image/video later. Subclasses override:

    - :meth:`conform_metadata` — transform dataset-level metadata (e.g. replace
      ``index2label``); called once at construction.
    - :meth:`keeps` — a cheap predicate deciding whether a datum survives; scanned
      once at construction to fix the conformed dataset's length.
    - :meth:`conform_datum` — transform a single datum; applied lazily on access.

    Subclasses that read the *target* of each datum should set :attr:`requires` to
    declare the MAITE datum shape they need; :class:`Conform` aggregates these and
    validates the source dataset once, upfront, raising
    :class:`~dataeval.exceptions.MaiteShapeError` before any datum is conformed.
    """

    requires: DatasetKind | None = None

    def conform_metadata(self, metadata: DatasetMetadata) -> DatasetMetadata:
        """Return possibly-updated dataset-level metadata (default: unchanged)."""
        return metadata

    def keeps(self, datum: _TDatum) -> bool:  # noqa: ARG002
        """Return whether ``datum`` survives this conformer (default: always)."""
        return True

    def conform_datum(self, datum: _TDatum) -> _TDatum:
        """Return the transformed ``datum`` (default: unchanged)."""
        return datum


class Conform(AnnotatedDataset[_TDatum]):
    """
    Dataset view that conforms each datum via one or more :class:`Conformer` ops.

    Wraps a dataset and applies a sequence of conformers that rewrite content
    (labels now; metadata, units, or imagery later) to make it conform to a target
    schema, without modifying the original dataset. Datums a conformer drops are
    excluded; surviving datums are transformed lazily on access. Dataset-level
    metadata (e.g. ``index2label``) is updated once at construction.

    Parameters
    ----------
    dataset : Dataset[_TDatum]
        Source dataset to wrap and conform.
    conformers : Conformer or Sequence[Conformer] or None, default None
        Conformations to apply, in order. ``None`` is an identity view.

    Examples
    --------
    >>> from dataeval.data import Conform, Relabel
    >>> conformed = Conform(dataset, [Relabel(alignment, target)])  # doctest: +SKIP
    """

    _dataset: Dataset[_TDatum]
    _conformers: Sequence[Conformer[_TDatum]]
    _selection: list[int]

    def __init__(
        self,
        dataset: Dataset[_TDatum],
        conformers: Conformer[_TDatum] | Sequence[Conformer[_TDatum]] | None = None,
    ) -> None:
        self.__dict__.update(dataset.__dict__)
        self._dataset = dataset
        self._conformers = (
            [] if conformers is None else [conformers] if isinstance(conformers, Conformer) else list(conformers)
        )

        # Fail fast if any conformer needs a target the source dataset cannot provide.
        required_kind = aggregate_required_kind(c.requires for c in self._conformers)
        if required_kind is not None and len(dataset) > 0:
            validate_dataset(dataset, expected=required_kind, caller="Conform")

        # Fold dataset-level metadata once (e.g. produce a new index2label).
        _metadata = dict(getattr(dataset, "metadata", {}))
        if "id" not in _metadata:
            _metadata["id"] = dataset.__class__.__name__
        metadata = DatasetMetadata(**_metadata)
        for conformer in self._conformers:
            metadata = conformer.conform_metadata(metadata)
        self._metadata = metadata

        # Fix length up front: keep source indices every conformer accepts. Only scan
        # (which fetches each datum, incl. image I/O) if some conformer can actually drop.
        if any(type(c).keeps is not Conformer.keeps for c in self._conformers):
            self._selection = [i for i in range(len(dataset)) if self._keeps(dataset[i])]
        else:
            self._selection = list(range(len(dataset)))

    def _keeps(self, datum: _TDatum) -> bool:
        return all(conformer.keeps(datum) for conformer in self._conformers)

    def _conform(self, datum: _TDatum) -> _TDatum:
        for conformer in self._conformers:
            datum = conformer.conform_datum(datum)
        return datum

    @property
    def metadata(self) -> DatasetMetadata:
        """Conformed dataset-level metadata (e.g. the target ``index2label``)."""
        return self._metadata

    def __getitem__(self, index: int) -> _TDatum:
        return self._conform(self._dataset[self._selection[index]])

    def __iter__(self) -> Iterator[_TDatum]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._selection)

    def __repr__(self) -> str:
        conformers = ", ".join(repr(c) for c in self._conformers)
        return f"Conform(dataset={self._dataset!r}, conformers=[{conformers}], len={len(self)})"

    def __str__(self) -> str:
        nt = "\n    "
        title = f"{self.__class__.__name__} Dataset"
        sep = "-" * len(title)
        conformers = f"Conformers: [{', '.join(str(c) for c in self._conformers)}]"
        return f"{title}\n{sep}{nt}{conformers}{nt}Conformed Size: {len(self)}\n\n{self._dataset}"

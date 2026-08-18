__all__ = []

import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TypeVar

from dataeval.protocols import AnnotatedDataset, DatasetMetadata

_TDatum = TypeVar("_TDatum")


def _datum_signature(datum: Any) -> tuple[Any, ...]:
    """Return a cheap, structural fingerprint of a single datum for shape-consistency checks.

    Only inspects the top-level container (never element contents / image dimensions),
    so it flags obvious mismatches (e.g. 2-tuple vs 3-tuple datums) without false
    positives on legitimately varying image sizes.
    """
    if isinstance(datum, tuple):
        return ("tuple", len(datum))
    if isinstance(datum, Sequence) and not isinstance(datum, (str, bytes)):
        return ("sequence", len(datum))
    return (type(datum).__name__,)


def _peek_datum_signature(dataset: AnnotatedDataset[Any]) -> tuple[Any, ...] | None:
    """Best-effort signature of a dataset's first datum, or None if it can't be read.

    Peeks only ``dataset[0]`` (never iterates), and swallows any error so a merge is
    never failed by an unreadable dataset.
    """
    try:
        if len(dataset) == 0:
            return None
        return _datum_signature(dataset[0])
    except Exception:  # noqa: BLE001 - best-effort peek must never fail a merge
        return None


def _validate_vocabularies(datasets: tuple[AnnotatedDataset[Any], ...]) -> None:
    """Enforce a shared ``index2label``; warn when the guard would be vacuous.

    Raises ValueError if some datasets expose a vocabulary and others don't, or if the
    present vocabularies disagree. Warns when *none* expose a vocabulary, since the
    equality check is then vacuously satisfied and label-space compatibility is unverified.
    """
    present = [dict(v) for v in (getattr(d, "metadata", {}).get("index2label", {}) for d in datasets) if v]
    if not present:
        warnings.warn(
            "merge_datasets: none of the datasets expose an 'index2label' mapping, so their "
            "label spaces cannot be verified to match. Merging assumes the integer labels already "
            "denote the same classes. Conform them to a common vocabulary first "
            "(see dataeval.data.View with dataeval.data.Relabel) to silence this warning.",
            UserWarning,
            stacklevel=3,
        )
        return
    # At least one dataset carries a vocabulary: every dataset must carry it and all must agree.
    if len(present) != len(datasets) or any(vocabulary != present[0] for vocabulary in present[1:]):
        raise ValueError(
            "merge_datasets requires all datasets to share the same 'index2label'. "
            "Conform them to a common vocabulary first (see dataeval.data.View with dataeval.data.Relabel)."
        )


def _warn_on_datum_shape_mismatch(datasets: tuple[AnnotatedDataset[Any], ...]) -> None:
    """Warn on an obvious datum-shape mismatch across datasets (cheap first-item peek)."""
    signatures = [sig for sig in (_peek_datum_signature(d) for d in datasets) if sig is not None]
    if signatures and any(sig != signatures[0] for sig in signatures[1:]):
        warnings.warn(
            "merge_datasets: datasets appear to have inconsistent datum shapes "
            f"(first-item signatures differ: {signatures}). Merging anyway, but ensure all "
            "datasets share a compatible datum structure.",
            UserWarning,
            stacklevel=3,
        )


class _MergedDataset(AnnotatedDataset[_TDatum]):
    """Read-only concatenation of datasets that share a label vocabulary."""

    def __init__(self, datasets: Iterable[AnnotatedDataset[_TDatum]], metadata: DatasetMetadata) -> None:
        self._datasets = list(datasets)
        self._lengths = [len(d) for d in self._datasets]
        self._metadata = metadata

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, index: int) -> _TDatum:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for merged dataset of size {len(self)}")
        for dataset, length in zip(self._datasets, self._lengths, strict=True):
            if index < length:
                return dataset[index]
            index -= length
        raise IndexError(index)  # pragma: no cover - guarded above

    def __iter__(self) -> Iterator[_TDatum]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"merge_datasets({len(self._datasets)} datasets, len={len(self)})"


def merge_datasets(*datasets: AnnotatedDataset[_TDatum]) -> AnnotatedDataset[_TDatum]:
    """
    Concatenate datasets that share a label vocabulary into one dataset view.

    Returns a lazy, read-only :class:`~dataeval.protocols.AnnotatedDataset` that
    indexes through the given datasets in order. All datasets must already share
    the same ``index2label`` so their integer labels denote the same classes —
    use :class:`dataeval.data.View` with :class:`dataeval.data.Relabel`
    to conform datasets to a common reference vocabulary first. Datasets must also
    share a compatible datum shape (e.g. all MAITE ``(input, target, metadata)``
    triples); merging structurally different datums is not supported.

    Parameters
    ----------
    *datasets : AnnotatedDataset
        Two or more datasets to merge. Each should expose an ``index2label`` in its
        metadata; when present, all must be equal.

    Returns
    -------
    AnnotatedDataset
        A concatenated view whose ``metadata`` carries the shared ``index2label``.

    Raises
    ------
    ValueError
        If no datasets are given, or their ``index2label`` mappings differ, or some
        datasets expose an ``index2label`` while others do not.

    Warns
    -----
    UserWarning
        If *none* of the datasets expose an ``index2label`` (the label spaces cannot
        be verified to match, so equality is vacuous), or if the datasets have an
        obviously inconsistent datum shape (e.g. differing tuple arity).

    See Also
    --------
    dataeval.data.View : Build a dataset view (e.g. relabel to a reference vocabulary).
    """
    if not datasets:
        raise ValueError("merge_datasets requires at least one dataset.")

    # Enforce a shared label vocabulary (and surface the vacuous-equality case).
    _validate_vocabularies(datasets)
    # Lightweight datum-shape consistency check (peeks only the first datum of each dataset).
    _warn_on_datum_shape_mismatch(datasets)

    metadata = dict(getattr(datasets[0], "metadata", {}))
    metadata["id"] = "merged"
    return _MergedDataset(datasets, DatasetMetadata(**metadata))

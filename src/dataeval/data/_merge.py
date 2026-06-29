__all__ = []

from collections.abc import Iterable, Iterator
from typing import TypeVar

from dataeval.protocols import AnnotatedDataset, DatasetMetadata

_TDatum = TypeVar("_TDatum")


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
    use :class:`dataeval.data.Conform` with :class:`dataeval.data.Relabel`
    to conform datasets to a common reference vocabulary first.

    Parameters
    ----------
    *datasets : AnnotatedDataset
        Two or more datasets to merge. Each must expose an ``index2label`` in its
        metadata, and all must be equal.

    Returns
    -------
    AnnotatedDataset
        A concatenated view whose ``metadata`` carries the shared ``index2label``.

    Raises
    ------
    ValueError
        If no datasets are given, or their ``index2label`` mappings differ.

    See Also
    --------
    dataeval.data.Conform : Conform a dataset to a reference vocabulary.
    """
    if not datasets:
        raise ValueError("merge_datasets requires at least one dataset.")

    vocabularies = [dict(getattr(d, "metadata", {}).get("index2label", {})) for d in datasets]
    if any(vocabulary != vocabularies[0] for vocabulary in vocabularies[1:]):
        raise ValueError(
            "merge_datasets requires all datasets to share the same 'index2label'. "
            "Conform them to a common vocabulary first (see dataeval.data.Conform / Relabel)."
        )

    metadata = dict(getattr(datasets[0], "metadata", {}))
    metadata["id"] = "merged"
    return _MergedDataset(datasets, DatasetMetadata(**metadata))

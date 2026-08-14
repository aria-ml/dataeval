"""The level model plus an obligation to read a dataset and produce rows from it."""

__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from dataeval._metadata._structurers._base import Structurer
from dataeval._metadata._structurers._data import StructuredData
from dataeval._metadata._structurers._reserved import safe_column_name
from dataeval.protocols import AnnotatedDataset, DatumMetadata, ProgressCallback
from dataeval.utils._internal import merge_metadata


class DatasetStructurer(Structurer, ABC):
    """Strategy for turning a dataset into levelled metadata rows.

    Parameters
    ----------
    first_datum : tuple or None, default None
        The dataset's first ``(item, target, metadata)`` triple, when the caller has
        already read it. :func:`select_structurer` has to read it to detect the task,
        and a dataset that decodes on ``__getitem__`` would otherwise pay for that
        item twice; handing it back here makes :meth:`build` reuse it.
    """

    def __init__(self, first_datum: tuple[Any, Any, DatumMetadata] | None = None) -> None:
        self._first_datum = first_datum

    def _datum(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        index: int,
    ) -> tuple[Any, Any, DatumMetadata]:
        """Read one datum, reusing — and then releasing — the one the task probe read.

        The cached datum holds a decoded item, which for an image or a video frame is
        the largest object in this class by orders of magnitude. A structurer outlives
        :meth:`build` (:class:`~dataeval.Metadata` keeps it for its level model), so
        handing the datum out has to drop the reference with it, or every long-lived
        Metadata pins item 0's pixels for its whole lifetime.
        """
        if index == 0 and self._first_datum is not None:
            datum, self._first_datum = self._first_datum, None
            return datum
        return dataset[index]

    @abstractmethod
    def build(
        self,
        dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> StructuredData:
        """Extract rows, factors and labels from ``dataset``."""
        ...

    def _merge_factors(
        self,
        raw: Sequence[Mapping[str, Any]],
        *,
        ignore_lists: bool,
        targets_per_item: Sequence[int] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Sequence[str]]]:
        """Merge per-item metadata dictionaries into flat factor arrays.

        Parameters
        ----------
        raw : Sequence[Mapping[str, Any]]
            Per-item metadata as provided by the dataset.
        ignore_lists : bool
            When True, list-valued metadata is discarded and one value per item
            is produced. When False, list-valued metadata is expanded across the
            targets of each item.
        targets_per_item : Sequence[int] or None, default None
            Number of target-level rows contributed by each item; required to
            expand list-valued metadata.

        Returns
        -------
        tuple[Mapping[str, Any], Mapping[str, Sequence[str]]]
            The merged factors and the factors that were dropped.
        """
        merged, dropped = merge_metadata(
            raw,
            return_dropped=True,
            ignore_lists=ignore_lists,
            targets_per_image=targets_per_item,
        )
        factors = {safe_column_name(k): v for k, v in merged.items() if k != "_image_index"}
        return factors, dropped

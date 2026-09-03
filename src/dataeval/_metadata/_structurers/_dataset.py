"""The level model plus an obligation to read a dataset and produce rows from it."""

__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval._metadata._structurers._base import Structurer
from dataeval._metadata._structurers._data import StructuredData
from dataeval._metadata._structurers._reserved import ID_KEY, safe_column_name
from dataeval.protocols import AnnotatedDataset, DatumMetadata, ProgressCallback
from dataeval.utils._merge import _merge, sorted_drop_reasons

_logger = get_logger(__name__)


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

    def __init__(
        self,
        first_datum: tuple[Any, Any, DatumMetadata] | None = None,
        *,
        partial_factors: bool = False,
    ) -> None:
        self._first_datum = first_datum
        self._partial_factors = partial_factors

    @property
    def partial_factors(self) -> bool:
        """Whether a factor only some rows declare is kept, with the rest missing.

        One policy, applied wherever this walk meets an incompletely declared value: a
        metadata key some items omit, and — for a tracking dataset — a timing or dimension
        some frames omit. Two opposite answers to that question in one structuring pass
        would be the harder thing to explain.
        """
        return self._partial_factors

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
    ) -> tuple[Mapping[str, Any], Mapping[str, Sequence[str]], Mapping[str, list[Any]]]:
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
        tuple[Mapping[str, Any], Mapping[str, Sequence[str]], Mapping[str, list[Any]]]
            The merged factors, the factors that were dropped, and the values of the
            columns held back because they mix numbers with text — kept as the dataset
            wrote them, since nobody has yet said how they should be read.
        """
        merged, dropped, unusable = _merge(
            list(raw),
            ignore_lists,
            False,
            targets_per_item,
            self._partial_factors,
        )
        # Drop the merged ``id`` only when a datum actually carried one at its top level:
        # that is the identity :meth:`_item_ids` reads, and it is already on ``item_id``.
        # A *nested* key can minimize to the bare ``id`` too — but only when no top-level
        # one exists, since minimization lengthens it to ``sensor_id`` as soon as both do —
        # and that one is a factor the dataset measured. Escaping it keeps it; dropping it
        # here would delete it with no entry in ``dropped`` to say where it went.
        own_id = any(ID_KEY in mapping for mapping in raw)
        factors = {safe_column_name(k): v for k, v in merged.items() if not (own_id and k == ID_KEY)}
        return factors, sorted_drop_reasons(dropped), {safe_column_name(k): v for k, v in unusable.items()}

    def _item_ids(self, raw: Sequence[Mapping[str, Any]]) -> Sequence[Any] | NDArray[Any]:
        """Return the datum's own ``id`` per item: reserved identity, not a factor.

        MAITE requires every datum to carry an ``id``, so this is normally the dataset's
        identifier for each item. A dataset with no ids at all — or a mix of present and
        absent ones — falls back to the positional index, so the ``item_id`` column is
        always present.

        Repeats are kept as they stand. ``item_id`` is a lookup column, not a key: every
        join in the library goes through ``item_index``, and a view that draws one item
        twice — :class:`~dataeval.data.ClassBalance` oversampling, or ``Indices([0, 0, 1])``
        — puts one id on two rows, which is the truth about where those rows came from.
        Refusing the repeat would close those views to every dataset that carries ids at
        all. The datasets DataEval builds keep their own ids distinct where it matters:
        :func:`~dataeval.data.merge_datasets` namespaces by source, and
        :class:`~dataeval.data.DetectionCrops` numbers the crop, keeping the item it came
        from on ``source_id``.
        """
        ids = [mapping.get(ID_KEY) for mapping in raw]
        present = sum(value is not None for value in ids)
        if present == len(ids):
            self._validate_ids(ids)
            self._log_repeats(ids)
            return self._id_column(ids)
        # None at all is ordinary — plenty of datasets predate the requirement. *Some* is a
        # dataset bug, and it costs the ids that were there, so it is said out loud.
        log = _logger.debug if present == 0 else _logger.warning
        log(
            "Only %d of %d items carry a datum %r; item_id falls back to the positional "
            "index for all of them, and the ids that were present are not kept.",
            present,
            len(ids),
            ID_KEY,
        )
        return np.arange(len(ids), dtype=np.intp)

    @staticmethod
    def _log_repeats(ids: Sequence[Any]) -> None:
        """Note repeated ids without refusing them; see :meth:`_item_ids`.

        Logged at debug rather than warned: a view that draws an item more than once is the
        ordinary way to get here, so warning would fire on every
        :class:`~dataeval.data.ClassBalance` oversample. The line is here for the other
        reader — someone whose source dataset repeats an id by mistake and wants to know why
        two rows name one item.
        """
        try:
            distinct = len(set(ids))
        except TypeError:  # unhashable ids (a list handed in as one): nothing to count
            return
        if distinct < len(ids):
            _logger.debug(
                "%d of %d datum %r values repeat; item_id is a lookup column, not a key, so "
                "the repeats are kept. Expected when a view draws an item more than once.",
                len(ids) - distinct,
                len(ids),
                ID_KEY,
            )

    @staticmethod
    def _id_column(ids: list[Any]) -> Sequence[Any] | NDArray[Any]:
        """Array-back the ids when their type allows it, so gathering them is a take.

        ``_gather_ids`` runs once per row block, and the tracking task's instance block is
        the largest one DataEval builds: a list comprehension there materializes a Python
        object per detection before polars sees any data, which is the cost ``_as_column``
        exists to avoid. Only the types MAITE declares for an ``id`` are converted — an id
        of some other type may be ragged or unrepresentable, and stays a list.
        """
        return np.asarray(ids) if ids and isinstance(ids[0], (int, str, np.generic)) else ids

    @staticmethod
    def _validate_ids(ids: Sequence[Any]) -> None:
        """Raise when the ``id``s cannot form a column, since ``item_id`` is one.

        The only requirement is a single type. Values of several types reach polars as an
        unreadable ``TypeError`` naming neither the ``id`` nor the item it came from, so
        the dataset is told what is wrong here instead. Repeats are *not* refused — see
        :meth:`_item_ids`.
        """
        types = sorted({type(value).__name__ for value in ids})
        if len(types) > 1:
            raise ValueError(
                f"Datum 'id' values must all have one type, but got {types}. The id is a "
                "column of the metadata, and a column holds one type; give every item's id "
                "the same type, or omit them so the positional index is used instead.",
            )

    @staticmethod
    def _gather_ids(item_ids: Sequence[Any] | NDArray[Any], item_index: Any) -> Sequence[Any] | NDArray[Any]:
        """Map each row's ``item_id`` off its ``item_index`` (a list or an array)."""
        if isinstance(item_ids, np.ndarray):
            return item_ids[np.asarray(item_index, dtype=np.intp)]
        return [item_ids[index] for index in item_index]

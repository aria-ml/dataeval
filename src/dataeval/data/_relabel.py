__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from dataeval._ontology import Ontology
from dataeval.data._conform import Conformer
from dataeval.exceptions import OntologyError
from dataeval.protocols import Array, DatasetMetadata, ObjectDetectionTarget
from dataeval.utils._internal import MaskedTarget, as_numpy, mask_metadata
from dataeval.utils._validate import DatasetKind

TargetVocabulary: TypeAlias = Ontology | Mapping[int, str] | Sequence[str]
"""A target label vocabulary: an :class:`.Ontology`, an ``index -> label`` mapping,
or an ordered sequence of class names."""


def _resolve_target(target: TargetVocabulary) -> tuple[dict[str, int], dict[int, str]]:
    """Normalize a target vocabulary into ``(key -> index, index -> label)``.

    The *key* is what a ``class_remap`` value must match: a concept id for an
    :class:`.Ontology` (ids equal labels for hand-built ontologies), otherwise the
    label itself.
    """
    if isinstance(target, Ontology):
        ids = target.ids
        return {cid: i for i, cid in enumerate(ids)}, {i: target.concept(cid).label for i, cid in enumerate(ids)}
    if isinstance(target, Mapping):
        index2label = {int(i): str(name) for i, name in target.items()}
        return {name: i for i, name in index2label.items()}, index2label
    if isinstance(target, str):
        raise TypeError("target must be an Ontology, a Mapping[int, str], or a sequence of class names.")
    index2label = {i: str(name) for i, name in enumerate(target)}
    return {name: i for i, name in index2label.items()}, index2label


def _label_remap(
    source_index2label: Mapping[int, str],
    class_remap: Mapping[str, str],
    target: TargetVocabulary | None = None,
) -> tuple[dict[int, int], dict[int, str], dict[int, str]]:
    """Compose a dataset's label indexing with a class remap into an integer remap.

    When ``target`` is ``None`` the target vocabulary is derived from the distinct
    ``class_remap`` values, in first-seen order.
    """
    if target is None:
        target = list(dict.fromkeys(class_remap.values()))
    key2index, index2label = _resolve_target(target)

    mapping: dict[int, int] = {}
    dropped: dict[int, str] = {}
    for index, name in source_index2label.items():
        target_key = class_remap.get(str(name))
        if target_key is not None and target_key in key2index:
            mapping[int(index)] = key2index[target_key]
        else:
            dropped[int(index)] = str(name)

    return (mapping, index2label, dropped)


class Relabel(Conformer[Any]):
    """
    Conform a dataset's class labels to a target vocabulary via a class mapping.

    Rewrites each datum's integer labels from the source vocabulary into the
    ``target`` vocabulary using a ``class_remap`` (source class name to target
    concept), and replaces the dataset's ``index2label`` with the target's. The
    ``class_remap`` is typically the ``remap`` of a
    :func:`~dataeval.core.label_alignment` result, but may be any hand-written
    mapping — equivalences are renamed and coarsenings collapse, so two source
    classes may map to one target class. Source classes with no entry in
    ``class_remap`` (or whose target is absent from ``target``) are
    out-of-vocabulary; by default they are dropped.

    Parameters
    ----------
    class_remap : Mapping[str, str]
        Maps a source class name to its target concept. Target values are concept
        ids when ``target`` is an :class:`.Ontology` (ids equal labels for
        hand-built ontologies), otherwise target labels.
    target : Ontology or Mapping[int, str] or Sequence[str], optional
        The target vocabulary and its integer indexing: an :class:`.Ontology`
        (concepts indexed in order), an ``index -> label`` mapping, or an ordered
        sequence of class names. A plain mapping/sequence needs no ontology, so
        relabeling can be done entirely by hand. If omitted, the vocabulary is
        derived from the distinct ``class_remap`` values (first-seen order) — handy
        for one-off maps. To merge several datasets, pass the *same* explicit
        target so they share an indexing.
    on_unmatched : {"drop", "raise"}, default "drop"
        What to do with out-of-vocabulary source classes. ``"drop"`` removes them
        (an image-classification datum whose class is OOV is dropped; an
        object-detection detection that is OOV is dropped, and an image left with
        no detections is dropped). ``"raise"`` raises if any source class is OOV.

    Raises
    ------
    OntologyError
        If the dataset metadata provides no ``index2label``, or if
        ``on_unmatched="raise"`` and any source class is out-of-vocabulary.
    """

    requires: DatasetKind | None = "any_target"

    def __init__(
        self,
        class_remap: Mapping[str, str],
        target: TargetVocabulary | None = None,
        *,
        on_unmatched: Literal["drop", "raise"] = "drop",
    ) -> None:
        self._class_remap = class_remap
        self.target = target
        self.on_unmatched = on_unmatched
        self._mapping: dict[int, int] | None = None
        self._dropped: dict[int, str] | None = None
        self._index2label: dict[int, str] | None = None

    def _repr_overrides(self) -> dict[str, str]:
        return {"class_remap": f"<{len(self._class_remap)} entries>"}

    @property
    def mapping(self) -> Mapping[int, int]:
        """Source label index to target label index (computed during conform)."""
        if self._mapping is None:
            raise OntologyError("Relabel must be applied through Conform(...) before use.")
        return self._mapping

    @property
    def dropped(self) -> Mapping[int, str]:
        """Source classes dropped as out-of-vocabulary (source index to name)."""
        if self._dropped is None:
            raise OntologyError("Relabel must be applied through Conform(...) before use.")
        return self._dropped

    @property
    def index2label(self) -> Mapping[int, str]:
        if self._index2label is None:
            raise OntologyError("Relabel must be applied through Conform(...) before use.")
        return self._index2label

    def conform_metadata(self, metadata: DatasetMetadata) -> DatasetMetadata:
        source_index2label = metadata.get("index2label")
        if not source_index2label:
            raise OntologyError("Relabel requires the dataset metadata to provide 'index2label'.")
        self._mapping, self._index2label, self._dropped = _label_remap(
            source_index2label, self._class_remap, self.target
        )
        if self.on_unmatched == "raise" and self.dropped:
            names = ", ".join(sorted(self.dropped.values()))
            raise OntologyError(f"Source classes not expressible in target vocabulary: {names}")
        return cast(DatasetMetadata, {**metadata, "index2label": self.index2label})

    def keeps(self, datum: Any) -> bool:
        target = datum[1]
        if isinstance(target, ObjectDetectionTarget):
            return any(int(label) in self.mapping for label in as_numpy(target.labels))
        if isinstance(target, Array):
            return int(np.argmax(as_numpy(target))) in self.mapping
        raise TypeError(f"Relabel does not support targets of type {type(target)}.")

    def conform_datum(self, datum: Any) -> Any:
        image, target, metadata = datum
        if isinstance(target, ObjectDetectionTarget):
            new_target, mask = self._conform_detections(target, self.mapping)
            return image, new_target, mask_metadata(metadata, mask)
        if isinstance(target, Array):
            return image, self._conform_scores(target, self.mapping, len(self.index2label)), metadata
        raise TypeError(f"Relabel does not support targets of type {type(target)}.")

    @staticmethod
    def _conform_scores(target: Array, mapping: Mapping[int, int], n_target: int) -> NDArray[np.float64]:
        """Image classification: fold source scores into the target vocabulary."""
        scores = as_numpy(target)
        out = np.zeros(n_target, dtype=np.float64)
        for source_index, target_index in mapping.items():
            out[target_index] += scores[source_index]
        return out

    @staticmethod
    def _conform_detections(
        target: ObjectDetectionTarget, mapping: Mapping[int, int]
    ) -> tuple[MaskedTarget, NDArray[np.bool_]]:
        """Object detection: drop unmapped detections and remap the rest."""
        labels = as_numpy(target.labels)
        keep = [int(label) in mapping for label in labels]
        mask = np.array(keep, dtype=np.bool_)
        new_labels = np.array([mapping[int(label)] for label, k in zip(labels, keep, strict=True) if k], dtype=np.intp)
        return MaskedTarget(target, mask, {"labels": new_labels}), mask

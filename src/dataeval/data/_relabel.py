__all__ = []

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from dataeval._ontology import Ontology
from dataeval.data._view import Operation, View
from dataeval.exceptions import OntologyError
from dataeval.protocols import Array, DatasetMetadata, ObjectDetectionTarget
from dataeval.types._target import own_class_scores
from dataeval.utils._internal import MaskedTarget, argmax_label, as_numpy, mask_metadata
from dataeval.utils._validate import DatasetKind

TargetVocabulary: TypeAlias = Ontology | Mapping[int, str] | Sequence[str]
"""A target label vocabulary: an :class:`.Ontology`, an ``index -> label`` mapping,
or an ordered sequence of class names."""


def _resolve_target(target: TargetVocabulary) -> tuple[dict[str, int], dict[int, str]]:  # noqa: C901
    """Normalize a target vocabulary into ``(key -> index, index -> label)``.

    The *key* is what a ``class_remap`` value must match: a concept id for an
    :class:`.Ontology` (ids equal labels for hand-built ontologies), otherwise the
    label itself.

    An ontology's *alias* ids are accepted alongside its canonical ids and map to
    the same index. Aliases are transparent everywhere else on
    :class:`.Ontology`, and keying off :attr:`~.Ontology.ids` alone would make a
    ``class_remap`` value naming an alias look out-of-vocabulary — silently
    dropping the data under the default ``on_unmatched="drop"``.
    """
    if isinstance(target, Ontology):
        ids = target.ids
        key2index = {cid: i for i, cid in enumerate(ids)}
        for index, cid in enumerate(ids):
            for alias in target.aliases(cid):
                key2index[alias] = index
        return key2index, {i: target.concept(cid).label for i, cid in enumerate(ids)}
    if isinstance(target, Mapping):
        index2label = {int(i): str(name) for i, name in target.items()}
        # A column position is read as the label value it scores, so a negative index has
        # no column to be: it would wrap onto another class's, silently scoring that one.
        negative = sorted(i for i in index2label if i < 0)
        if negative:
            raise OntologyError(f"Target vocabulary indices must be non-negative; got {negative}.")
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


class Relabel(Operation):
    """
    Conform a dataset's class labels to a target vocabulary via a class mapping.

    Rewrites each datum's integer labels from the source vocabulary into the
    ``target`` vocabulary using a ``class_remap`` (source class name to target
    concept), and replaces the dataset's ``index2label`` with the target's. Scores
    move with the labels — see Notes, which is where the two tasks differ. The
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
        If the dataset metadata provides no ``index2label``, if a ``target`` mapping
        indexes a class negatively, or if ``on_unmatched="raise"`` and any source class
        is out-of-vocabulary.

    Notes
    -----
    A target's scores are indexed by the vocabulary they were measured against, so labels
    cannot be conformed without them. What that takes differs by task:

    - **Object detection.** Each detection's score is read down to one confidence — the
      score of the class it was labelled with, taken against its *source* label, which is
      the last moment the columns still mean what the labels say. A confidence is a
      property of the detection, so the result carries no vocabulary, needs no width, and
      stacks with any other dataset's. A detection whose score cannot be read (a target
      with no column for its class) conforms to ``nan``, not to ``0.0``.
    - **Image classification.** The datum's target *is* its score vector and the label is
      read back out of it by argmax, so the vector is folded into the target vocabulary
      rather than reduced. Two source classes coarsened into one have their scores summed
      there. The fold is not renormalized — dropping an out-of-vocabulary class removes
      its mass, and a coarsening can carry a sum past 1.0 — so treat the result as
      per-class weights whose argmax is the label, not as a distribution.

    .. versionchanged:: 1.2
        Object-detection scores are conformed rather than left in the source vocabulary.
        v1.1 masked dropped detections out of a per-class score array but left its columns
        source-indexed, so a conformed detection's score was read from another class's
        column, and two datasets conformed to one vocabulary kept the two widths they
        arrived with. A classification fold is also sized by the target's largest index
        rather than its class count, so a non-contiguous ``Mapping[int, str]`` target no
        longer raises ``IndexError``.
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
        self._score_width: int = 0

    def _repr_overrides(self) -> dict[str, str]:
        return {"class_remap": f"<{len(self._class_remap)} entries>"}

    @property
    def mapping(self) -> Mapping[int, int]:
        """Source label index to target label index (computed during conform)."""
        if self._mapping is None:
            raise OntologyError("Relabel must be applied through View(...) before use.")
        return self._mapping

    @property
    def dropped(self) -> Mapping[int, str]:
        """Source classes dropped as out-of-vocabulary (source index to name)."""
        if self._dropped is None:
            raise OntologyError("Relabel must be applied through View(...) before use.")
        return self._dropped

    @property
    def index2label(self) -> Mapping[int, str]:
        if self._index2label is None:
            raise OntologyError("Relabel must be applied through View(...) before use.")
        return self._index2label

    def apply_metadata(self, metadata: DatasetMetadata) -> DatasetMetadata:
        source_index2label = metadata.get("index2label")
        if not source_index2label:
            raise OntologyError("Relabel requires the dataset metadata to provide 'index2label'.")
        self._mapping, self._index2label, self._dropped = _label_remap(
            source_index2label, self._class_remap, self.target
        )
        # Computed here, where the vocabulary is settled, rather than per datum: ``_remap``
        # is the lazy transform ``View.map`` registers, so a property would rescan the
        # whole vocabulary on every read, of every pass, forever.
        self._score_width = max(self._index2label) + 1 if self._index2label else 0
        if self.on_unmatched == "raise" and self.dropped:
            names = ", ".join(sorted(self.dropped.values()))
            raise OntologyError(f"Source classes not expressible in target vocabulary: {names}")
        return cast(DatasetMetadata, {**metadata, "index2label": self.index2label})

    def apply(self, view: View[Any]) -> None:
        # Drop out-of-vocabulary datums (cheap keep-check reads through preceding ops),
        # then register the label remap applied lazily on access.
        view.selection = [i for i in view.selection if self._keep(view.read(i))]
        view.map(self._remap)

    def _keep(self, datum: Any) -> bool:
        target = datum[1]
        if isinstance(target, ObjectDetectionTarget):
            return any(int(label) in self.mapping for label in as_numpy(target.labels))
        if isinstance(target, Array):
            return argmax_label(target) in self.mapping
        raise TypeError(f"Relabel does not support targets of type {type(target)}.")

    def _remap(self, datum: Any) -> Any:
        image, target, metadata = datum
        if isinstance(target, ObjectDetectionTarget):
            new_target, mask = self._conform_detections(target, self.mapping)
            return image, new_target, mask_metadata(metadata, mask)
        if isinstance(target, Array):
            return image, self._conform_scores(target, self.mapping, self._score_width), metadata
        raise TypeError(f"Relabel does not support targets of type {type(target)}.")

    @staticmethod
    def _conform_scores(scores: Any, mapping: Mapping[int, int], width: int) -> NDArray[np.float64]:
        """Fold a classification datum's per-class scores into the target vocabulary.

        The target of a classification datum *is* its score vector, so this one has to stay
        a vector over the target vocabulary rather than reduce to the datum's own class:
        the label is read back out of it by argmax. Detection scores need no fold at all --
        see :meth:`_conform_detections`.

        Two source classes coarsened into one target class have their scores summed there.
        The result is not renormalized: dropping an out-of-vocabulary class removes its mass
        and a coarsening can carry a summed score past 1.0, so the vector stays a set of
        per-class weights whose argmax is the label, not a distribution.
        """
        values = as_numpy(scores)
        folded = np.zeros((*values.shape[:-1], width), dtype=np.float64)
        columns = values.shape[-1] if values.ndim else 0
        for source_index, target_index in mapping.items():
            if 0 <= source_index < columns:
                folded[..., target_index] += values[..., source_index]
        return folded

    @staticmethod
    def _conform_detections(
        target: ObjectDetectionTarget, mapping: Mapping[int, int]
    ) -> tuple[MaskedTarget, NDArray[np.bool_]]:
        """Object detection: drop unmapped detections, and remap the rest with their scores.

        Scores are read down to one confidence per detection *here*, against the source
        labels, because that is the last moment the source vocabulary is still the one the
        columns are indexed by. Masking alone — all ``MaskedTarget`` can do — would leave
        source-indexed columns beside target-indexed labels, and folding into the target
        vocabulary would allocate a column per class to read one number back out of it.
        """
        labels = as_numpy(target.labels).reshape(-1).astype(np.intp)
        keep = [int(label) in mapping for label in labels]
        mask = np.array(keep, dtype=np.bool_)
        new_labels = np.array([mapping[int(label)] for label, k in zip(labels, keep, strict=True) if k], dtype=np.intp)
        overrides: dict[str, Any] = {"labels": new_labels}
        scores = getattr(target, "scores", None)
        if scores is not None:
            # own_class_scores answers one value per *label*, so this is len(mask) long
            # whatever shape the target's scores were — including the disagreeing row
            # count it reads as nan, which masking on its own declines to touch at all.
            overrides["scores"] = own_class_scores(scores, labels)[mask]
        return MaskedTarget(target, mask, overrides), mask

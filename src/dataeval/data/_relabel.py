__all__ = []

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from dataeval._ontology import Ontology
from dataeval.data._view import Operation, View
from dataeval.exceptions import DeprecatedWarning, OntologyError
from dataeval.protocols import Array, DatasetMetadata, ObjectDetectionTarget
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
        # A score column's position is the label value it scores, so a negative index has
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


def _own_class_scores(scores: Any, labels: NDArray[np.intp]) -> NDArray[np.float32]:
    """Reduce a target's scores to one confidence per detection, in its own class.

    Handles both layouts MAITE permits — ``(N,)``, one per box, and ``(N, CLASSES)``, where
    a column position is read as the label value it scores. Answers ``nan`` where there is
    nothing to read: a detection past the end of the array, or a label with no column of
    its own. The labels are authoritative on the count.
    """
    read = np.full(len(labels), np.nan, dtype=np.float32)
    if scores is None:
        return read
    values = as_numpy(scores)
    if values.ndim == 1:
        rows = min(len(labels), len(values))
        read[:rows] = values[:rows]
        return read
    if values.ndim != 2:
        return read
    rows = min(len(labels), values.shape[0])
    own = labels[:rows]
    picked = np.flatnonzero((own >= 0) & (own < values.shape[1]))
    read[picked] = values[picked, own[picked]]
    return read


class Relabel(Operation):
    """
    Conform a dataset's class labels to a target vocabulary via a class mapping.

    Rewrites each datum's integer labels from the source vocabulary into the
    ``target`` vocabulary using a ``class_remap`` (source class name to target
    concept), and replaces the dataset's ``index2label`` with the target's. Scores
    move with the labels — see Notes. The
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
    reduce_detection_scores : bool or None, default None
        How a *detection* target's per-class scores are conformed — the one thing v1.2
        changes here. ``False`` folds them into the target vocabulary, keeping a column
        per target class. ``True`` reduces them to one confidence per detection, the
        score of the class the box was labelled with. Leaving it unset takes the release
        default and warns once, on the first target that actually scores every class.

        The default is ``False`` in v1.1.x and becomes ``True`` in v1.2; the parameter
        is removed in v1.3, after which detection scores are always reduced. So passing
        ``True`` now adopts v1.2's behavior early, and passing ``False`` in v1.2 buys one
        release of the old behavior while you migrate. Either silences the warning.

        Reducing is what lets datasets that scored differently be merged: a confidence
        carries no vocabulary, so a dataset scoring every class and one scoring every
        box conform to the same thing, where folding leaves them a per-class array and a
        per-box one that no single ``score`` column can hold.

        Has no effect on classification targets, which keep the fold in every version.

        .. deprecated:: 1.2
            The parameter exists only to span the change of default, and is removed in
            v1.3.

    Raises
    ------
    OntologyError
        If the dataset metadata provides no ``index2label``, if a ``target`` mapping
        indexes a class negatively, or if ``on_unmatched="raise"`` and any source class
        is out-of-vocabulary.

    Notes
    -----
    A target's scores are indexed by the vocabulary they were measured against, so labels
    cannot be conformed without them. A per-class score array — ``(N, CLASSES)`` for a
    detection target, the whole target for a classification datum — is folded into the
    target vocabulary, keeping one column per target index and summing where a coarsening
    collapses two source classes into one. Scores that are one per box carry no vocabulary
    and are only masked alongside the detections they belong to.

    The fold is not renormalized: dropping an out-of-vocabulary class removes its mass, and
    a coarsening can carry a summed score past 1.0. Treat a folded vector as per-class
    weights whose argmax is the label, not as a distribution.

    .. note::
        v1.2 changes what a *detection's* score conforms to. Rather than a vector over the
        target vocabulary, a detection keeps one confidence — the score of the class it was
        labelled with, read against its source label — so the value carries no vocabulary
        at all, and the metadata frame's ``score`` column becomes a single ``Float32`` per
        row rather than a per-class array. Classification targets keep the fold, since the
        label is read back out of the vector by argmax.

        ``reduce_detection_scores`` spans the change: pass ``True`` to adopt it now,
        ``False`` to keep folding through v1.2, and leave it unset to take the release
        default and be warned once where the two differ. The parameter is removed in v1.3.
        Code that reads ``dataset[i][1].scores`` off a relabeled detection dataset, or the
        ``score`` column for one, should expect one number per detection from v1.2 on.

    Examples
    --------
    Conform two datasets that score differently to one vocabulary, and merge them:

    >>> shared = ["car", "truck"]
    >>> conform = {"sedan": "car", "lorry": "truck"}
    >>> op = Relabel(conform, shared, reduce_detection_scores=True)
    >>> op.reduce_detection_scores
    True
    """

    requires: DatasetKind | None = "any_target"

    def __init__(
        self,
        class_remap: Mapping[str, str],
        target: TargetVocabulary | None = None,
        *,
        on_unmatched: Literal["drop", "raise"] = "drop",
        reduce_detection_scores: bool | None = None,
    ) -> None:
        self._class_remap = class_remap
        self.target = target
        self.on_unmatched = on_unmatched
        self.reduce_detection_scores = reduce_detection_scores
        self._mapping: dict[int, int] | None = None
        self._dropped: dict[int, str] | None = None
        self._index2label: dict[int, str] | None = None
        self._score_width: int = 0
        # Warned at most once per operation, and only where the choice changes the answer.
        self._warned_scores: bool = False

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
        # One past the largest target index, not one per class: a ``Mapping[int, str]``
        # target may index its classes however it likes, and a column position is the
        # label value it scores, so a vocabulary indexed 0 and 7 needs eight columns.
        # Settled here, where the vocabulary is fixed, rather than per datum: ``_remap``
        # is the lazy transform ``View.map`` registers.
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
            reduce = self._reduce_choice(target)
            new_target, mask = self._conform_detections(target, self.mapping, self._score_width, reduce)
            return image, new_target, mask_metadata(metadata, mask)
        if isinstance(target, Array):
            # Classification keeps the fold in every version: the label is read back out of
            # the vector by argmax, so there is nothing here to choose and nothing to warn.
            return image, self._conform_scores(target, self.mapping, self._score_width), metadata
        raise TypeError(f"Relabel does not support targets of type {type(target)}.")

    def _reduce_choice(self, target: ObjectDetectionTarget) -> bool:
        """Whether to reduce this target's scores, warning once if the caller never chose.

        Warned here rather than from ``__init__`` so that only the callers it applies to
        hear it. The two layouts conform identically today *and* in v1.2 unless the target
        scores every class — a per-box score carries no vocabulary either way — so
        warning at construction would tell most callers their results are about to change
        when nothing about them will.
        """
        if self.reduce_detection_scores is not None:
            return self.reduce_detection_scores
        scores = getattr(target, "scores", None)
        if not self._warned_scores and scores is not None and as_numpy(scores).ndim == 2:
            self._warned_scores = True
            warnings.warn(
                "Leaving `reduce_detection_scores` unset takes the release default, which "
                "changes in 1.2. Relabel folds a detection target's per-class scores into "
                "the target vocabulary today; from 1.2 it reduces them to one confidence "
                "per detection by default — the score of the class the box was labelled "
                "with — so the value carries no vocabulary and the metadata `score` column "
                "becomes a single number per row rather than a per-class array.\n"
                "Pass reduce_detection_scores=True to adopt that now, which is also what "
                "lets datasets that scored differently be merged, or False to keep folding "
                "through 1.2. Either silences this warning. The parameter is removed in "
                "1.3, after which scores are always reduced.",
                DeprecatedWarning,
                # Reaches the caller's own ``view[i]``; iterating is one frame deeper and
                # lands in ``View`` instead, which is why the message names the parameter
                # rather than relying on where it points.
                stacklevel=5,
            )
        return False

    @staticmethod
    def _conform_scores(scores: Any, mapping: Mapping[int, int], width: int) -> NDArray[np.float64]:
        """Fold per-class scores from the source vocabulary into the target one.

        Takes ``(..., SOURCE_CLASSES)`` and returns ``(..., width)``, so it serves a
        classification datum's single vector and a detection target's whole stack of them
        alike. Two source classes coarsened into one target class have their scores summed
        there. A source class the scores have no column for contributes nothing rather than
        raising, since a dataset may declare a wider vocabulary than its targets score.

        The result is not renormalized: dropping an out-of-vocabulary class removes its
        mass and a coarsening can carry a sum past 1.0, so treat it as per-class weights
        whose argmax is the label, not as a distribution.
        """
        values = as_numpy(scores)
        folded = np.zeros((*values.shape[:-1], width), dtype=np.float64)
        columns = values.shape[-1] if values.ndim else 0
        for source_index, target_index in mapping.items():
            if 0 <= source_index < columns:
                folded[..., target_index] += values[..., source_index]
        return folded

    @staticmethod
    def _row_aligned(values: NDArray[Any], count: int) -> NDArray[Any]:
        """One score row per detection, ``nan`` where the target supplied none.

        The labels are authoritative on how many detections there are, so a ``scores``
        holding a different number of rows is read to its end rather than allowed to
        change that count — and rather than declined, which would leave the array
        untouched for :func:`try_mask_object` to decline in turn on the same length test,
        passing source-vocabulary columns through beside target-vocabulary labels.
        """
        if len(values) == count:
            return values
        aligned = np.full((count, values.shape[1]), np.nan, dtype=np.float64)
        rows = min(count, len(values))
        aligned[:rows] = values[:rows]
        return aligned

    @staticmethod
    def _conform_detections(
        target: ObjectDetectionTarget, mapping: Mapping[int, int], width: int, reduce: bool = False
    ) -> tuple[MaskedTarget, NDArray[np.bool_]]:
        """Object detection: drop unmapped detections, and remap the rest with their scores.

        Per-class scores are indexed by the source vocabulary, so masking rows out of them
        — all ``MaskedTarget`` does on its own — leaves an array whose columns say one
        thing and whose labels now say another: a detection's score would be read from
        whichever class landed at its new index, or from off the end of the array. Scores
        that are one per box carry no vocabulary and need only the masking.
        """
        labels = as_numpy(target.labels)
        keep = [int(label) in mapping for label in labels]
        mask = np.array(keep, dtype=np.bool_)
        new_labels = np.array([mapping[int(label)] for label, k in zip(labels, keep, strict=True) if k], dtype=np.intp)
        overrides: dict[str, Any] = {"labels": new_labels}
        scores = getattr(target, "scores", None)
        values = None if scores is None else as_numpy(scores)
        if reduce and values is not None:
            # Read against the *source* labels, the last moment the columns still mean what
            # the labels say, and mask after: the result is one number per detection that
            # no vocabulary indexes, so it needs no width and stacks with any other
            # dataset's however that one scored.
            overrides["scores"] = _own_class_scores(values, labels.reshape(-1).astype(np.intp))[mask]
        elif values is not None and values.ndim == 2:
            aligned = Relabel._row_aligned(values, len(mask))
            overrides["scores"] = Relabel._conform_scores(aligned[mask], mapping, width)
        return MaskedTarget(target, mask, overrides), mask

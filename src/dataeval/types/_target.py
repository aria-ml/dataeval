"""Reading detections off a MAITE target, one way for every reader.

A target's fields are loosely specified on purpose — ``scores`` may be one per box or one
per box per class, ``track_ids`` is absent on the tasks that have no tracks — and every
reader that guesses differently is a place two parts of the library disagree about the
same detection. These are the one reading, shared by the structuring walk behind
:class:`~dataeval.Metadata` and by anything that follows a
:class:`~dataeval.types.SourceIndex` back to a detection — the two must agree about *which*
detection a ``target_index`` names, and about what its score is.

Both score layouts are read down to **one confidence per detection** here, so that no
caller has to carry the shape. That is not just a convenience: under a per-class layout the
array's width is the *vocabulary's*, so two datasets with different class counts produce
score arrays that cannot be stacked, and a relabeled dataset produces one whose columns no
longer mean what its labels say. A confidence is a property of the detection; the
vocabulary it was measured against is not, and does not survive this reading.
"""

__all__ = []

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import ObjectDetectionTarget, SingleFrameObjectTrackingTarget
from dataeval.utils._array import as_numpy


def detection_count(target: Any) -> int:
    """How many detections one target holds.

    **Labels are the authority.** Every other per-detection array is read against this
    count, which is what makes a detection's position mean the same thing to every reader —
    and a ``target_index`` name the same row whether it is being assigned or followed.
    """
    return len(as_numpy(target.labels).reshape(-1))


def instance_arrays(
    target: "ObjectDetectionTarget | SingleFrameObjectTrackingTarget",
) -> tuple[NDArray[np.intp], NDArray[np.float32], NDArray[np.float32]]:
    """Read one target's per-detection labels, boxes and scores.

    Parameters
    ----------
    target : ObjectDetectionTarget or SingleFrameObjectTrackingTarget
        The target to read. A tracking frame's target is a detection target plus
        ``track_ids``, so both tasks read boxes and labels the same way.

    Returns
    -------
    tuple
        ``(labels, boxes, scores)``, each with one entry per detection. **Labels establish
        the count** — every other per-detection array is read against it, which is what
        makes one detection's position mean the same thing to every reader. ``scores`` is
        one confidence per detection whichever layout the target carried, ``nan`` where it
        carried none this detection can be read from; see :func:`own_class_scores`.
    """
    labels = as_numpy(target.labels).reshape(-1).astype(np.intp)
    count = len(labels)
    boxes = as_numpy(target.boxes).astype(np.float32).reshape(count, 4) if count else np.empty((0, 4), dtype=np.float32)
    return labels, boxes, own_class_scores(getattr(target, "scores", None), labels)


def own_class_scores(scores: Any, labels: NDArray[np.intp]) -> NDArray[np.float32]:
    """Reduce a target's scores to one confidence per detection, in its own class.

    MAITE permits either of two score layouts — ``(N,)``, one confidence per box, or
    ``(N, CLASSES)``, a distribution over the vocabulary — and says nothing about which
    class a column belongs to. DataEval reads a column position as the *label value* it
    scores, the only reading under which a non-contiguous or non-zero-based
    ``index2label`` still addresses its own classes; a target that instead collapsed its
    columns onto the classes it happens to use has no column for the label it states, and
    is read as scoreless rather than as scoring a class it never named.

    A detection's own class is read rather than its highest, so the number answers "how
    confident, in what this box is labelled" — the question a row of the metadata frame is
    about — and not "what did the model most believe", which is a different one and is
    already answered by ``class_label``.

    Parameters
    ----------
    scores : Any
        The target's ``scores``, in either layout, or ``None`` where it carries none.
    labels : NDArray[np.intp]
        The detections' labels, as :func:`instance_arrays` read them. **Authoritative on
        the count**: a ``scores`` holding fewer rows is read to its end and the rest
        answered ``nan``, and a longer one is not read past, for the same reason
        :func:`track_ids_of` pads rather than resizes — a disagreeing companion array must
        not be allowed to change how many detections there are.

    Returns
    -------
    NDArray[np.float32]
        One confidence per label, ``nan`` where there is none to read: no ``scores`` at
        all, a layout that is neither of the two, a detection past the end of the array,
        or a label with no column of its own under the per-class layout. ``nan`` rather
        than an exception because a score is the one field of a detection that is optional
        in practice, and rather than ``0.0`` because absent is not the same as unconfident.
        The metadata frame spells the same absence as null, since ``nan`` is not null to
        polars — it survives a confidence threshold and poisons an aggregate.
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
    # A label outside the array's columns is the collapsed-column case above, and the
    # relabeled-to-a-wider-vocabulary one; both leave the score unreadable, not wrong.
    picked = np.flatnonzero((own >= 0) & (own < values.shape[1]))
    read[picked] = values[picked, own[picked]]
    return read


def track_ids_of(target: Any, count: int) -> NDArray[np.intp]:
    """Read one frame's per-detection track ids, ``-1`` where a detection has no track.

    Parameters
    ----------
    target : Any
        A tracking frame's target.
    count : int
        Detections in this frame, as :func:`instance_arrays` established it.

    Returns
    -------
    NDArray[np.intp]
        One id per detection. ``-1`` marks a detection no tracker linked, and stands in
        wherever the target supplies no id at all: a frame holding no detections need not
        carry ``track_ids``, and one carrying too few is padded rather than read past its
        end. Tolerant here because ``count`` is the authority on how many rows exist, so a
        disagreeing ``track_ids`` must not be allowed to change that count.
    """
    if not count:
        return np.empty(0, dtype=np.intp)
    ids = getattr(target, "track_ids", None)
    if ids is None:
        return np.full(count, -1, dtype=np.intp)
    values = as_numpy(ids).reshape(-1).astype(np.intp)
    if len(values) == count:
        return values
    if len(values) > count:
        return values[:count]
    return np.concatenate([values, np.full(count - len(values), -1, dtype=np.intp)])


def detection_score(target: Any, index: int, label: int) -> float | None:
    """Read one detection's confidence, from either score layout a target may carry.

    Parameters
    ----------
    target : Any
        The target the detection sits in.
    index : int
        The detection's position within `target`.
    label : int
        The detection's class, which picks the column under a per-class layout.

    Returns
    -------
    float or None
        The confidence, or ``None`` where the target carries none this detection can be
        read from. A ground-truth target scores ``1.0``. Where scores are ``(N, CLASSES)``
        this is the score of the box's **own** class rather than its highest, so a
        detection's score is about the class it was labelled with.

    Notes
    -----
    One detection's spelling of the rule :func:`own_class_scores` applies to a whole
    target, stated in three comparisons rather than routed through the array kernel: this
    is the answer :class:`~dataeval.data.SourceLocator` gives for a detection it
    retrieved, and the metadata frame's ``score`` column is the same reading of the same
    target, so the two disagreeing would be two answers about one box.

    Reads the value the target holds, at the target's own precision, and passes a genuine
    ``nan`` through as itself. The frame's column is float32 and spells an unreadable
    score as null; those are the column's types, not this answer's, and rounding a score
    to them here would make ``score == 0.1`` false for a target that holds exactly 0.1.
    """
    scores = getattr(target, "scores", None)
    if scores is None:
        return None
    values = as_numpy(scores)
    # ``ndim`` is the layout test and also what guarantees a length: a 0-d ``scores`` has
    # neither, and must reach None rather than raise on its way there. Negative indices
    # are refused rather than wrapped, since -2 quietly scoring the second-to-last box is
    # the worst way for a detection's position to be wrong.
    if values.ndim not in (1, 2) or not 0 <= index < len(values):
        return None
    return _score_at(values, index, label)


def _score_at(values: NDArray[Any], index: int, label: int) -> float | None:
    """One detection's own-class value, under whichever of the two layouts ``values`` is in."""
    if values.ndim == 1:
        return float(values[index])
    return float(values[index, label]) if 0 <= label < values.shape[1] else None

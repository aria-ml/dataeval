"""Which task a dataset is, and what that task says about the levels its rows sit at.

More than one reader needs this and none of them is the other's caller.
:class:`~dataeval.Metadata` structures a dataset into leveled rows; anything that follows
a :class:`~dataeval.types.SourceIndex` back to the datum it names walks the same graph in
the other direction. All of them have to agree on which task a dataset is, which levels it
therefore has, and which of those one item and one label sit at. Declared once here so they
cannot answer differently: a level model that disagrees between two readers means an
address one of them minted is refused, or followed to the wrong row, by the other.

It sits outside ``dataeval._metadata`` for that reason. That package's own doctrine is
that a module which acquires a second caller belongs back outside rather than imported
across the boundary, and the level model has one.

Dispatch reads the *target* alone. MAITE constrains what a target is but places no
constraint at all on what an item is — a path string, a PIL handle and a lazy loader are
all valid — so predicating on the item type rejects perfectly good datasets.
"""

__all__ = []

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from dataeval.protocols import Array, MultiobjectTrackingTarget, ObjectDetectionTarget, _is_protocol_instance
from dataeval.types._factors import FactorLevel, FactorLevelSchema

if TYPE_CHECKING:
    # Only a type: ``dataeval.utils.data`` imports this module to build its own checks.
    from dataeval.utils.data import DatasetKind

# The tasks a dataset can be *detected* as, which is narrower than the set of structuring
# strategies: ``"factors"`` is reached through :meth:`~dataeval.Metadata.from_factors` and
# has no dataset to dispatch on, and ``"unknown"`` is only the unspecialized base's default.
DatasetTask = Literal["IC", "OD", "MOT"]


@dataclass(frozen=True)
class TaskProfile:
    """One task a dataset can be: how it is recognized, and the level model it structures into.

    Each task, its ``kind``, and the MAITE types it reads:

    - ``"MOT"``, ``multiobject_tracking``: a :obj:`.MultiobjectTrackingTarget` in a
      :obj:`.MultiobjectTrackingDatum` of a :obj:`.MultiobjectTrackingDataset`.
    - ``"OD"``, ``object_detection``: an :obj:`.ObjectDetectionTarget` in an
      :obj:`.ObjectDetectionDatum` of an :obj:`.ObjectDetectionDataset`.
    - ``"IC"``, ``classification``: an :class:`.Array` of class scores in an
      :obj:`.ImageClassificationDatum` of an :obj:`.ImageClassificationDataset`.

    Only the target is checked: the datum and dataset types are generic aliases, which cannot be
    checked at run time.

    Attributes
    ----------
    task : str
        Short task identifier, ``"IC"``, ``"OD"`` or ``"MOT"``.
    kind : DatasetKind
        The same task in :data:`~dataeval.utils.data.DatasetKind`'s vocabulary, the name
        :func:`~dataeval.utils.data.validate_dataset` returns and ``requires=`` takes.
    matches : Callable[[Any], bool]
        Whether a target, ``datum[1]``, belongs to this task. :func:`detect_task` and
        :func:`~dataeval.utils.data.validate_dataset` both run it.
    levels : FactorLevelSchema
        Every level this task's rows sit at, and how they sit relative to each other.
    item_level : FactorLevel
        The level of one dataset item — what the dataset yields. ``unit`` for an
        image-based task, ``sequence`` for tracking.
    label_level : FactorLevel
        The level of one labeled thing. ``instance`` for every task, which is why an
        address with a key means the same thing whatever it was measured over.
    unit_type : str
        What one row at the ``unit`` level holds, in the dataset's own vocabulary.
        Descriptive only — it never affects structuring, and exists so that messages can
        name the medium without the level vocabulary having to.
    """

    task: DatasetTask
    kind: "DatasetKind"
    matches: Callable[[Any], bool]
    levels: FactorLevelSchema
    item_level: FactorLevel
    label_level: FactorLevel
    unit_type: str


# In the order the checks run; the first that matches wins. Most specific first, so the
# tracking check, a *positive* check for its own target type, sits above the detection one
# rather than being carved out of it.
TASK_PROFILES: Mapping[DatasetTask, TaskProfile] = MappingProxyType({
    "MOT": TaskProfile(
        "MOT",
        "multiobject_tracking",
        lambda target: _is_protocol_instance(target, MultiobjectTrackingTarget),
        FactorLevelSchema.of("sequence", "unit", "track", "instance"),
        "sequence",
        "instance",
        "frame",
    ),
    "OD": TaskProfile(
        "OD",
        "object_detection",
        lambda target: _is_protocol_instance(target, ObjectDetectionTarget),
        FactorLevelSchema.of("unit", "instance"),
        "unit",
        "instance",
        "image",
    ),
    "IC": TaskProfile(
        "IC",
        "classification",
        lambda target: isinstance(target, Array),
        FactorLevelSchema.of("unit", "instance"),
        "unit",
        "instance",
        "image",
    ),
})


def detect_task(target: Any) -> DatasetTask | None:
    """Name the task a target belongs to.

    Parameters
    ----------
    target : Any
        A dataset's ``datum[1]`` — the only part of a datum whose shape MAITE constrains.

    Returns
    -------
    str or None
        The task, or ``None`` where `target` matches no predicate. ``None`` rather than a
        raise, because the two callers say different things about it: structuring refuses
        the dataset outright, while a locator can still answer at the item level.
    """
    for profile in TASK_PROFILES.values():
        if profile.matches(target):
            return profile.task
    return None

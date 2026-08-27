"""Which task a dataset is, and what that task says about the levels its rows sit at.

More than one reader needs this and none of them is the other's caller.
:class:`~dataeval.Metadata` structures a dataset into levelled rows; anything that follows
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
from typing import Any, Literal

from dataeval.protocols import Array, ObjectDetectionTarget, is_multiobject_tracking_target
from dataeval.types._factors import FactorLevel, FactorLevelSchema

# The tasks a dataset can be *detected* as, which is narrower than the set of structuring
# strategies: ``"factors"`` is reached through :meth:`~dataeval.Metadata.from_factors` and
# has no dataset to dispatch on, and ``"unknown"`` is only the unspecialized base's default.
DatasetTask = Literal["IC", "OD", "MOT"]


@dataclass(frozen=True)
class TaskProfile:
    """The level model of one task: which levels it has, and what sits at each end.

    Attributes
    ----------
    task : str
        Short task identifier, ``"IC"``, ``"OD"`` or ``"MOT"``.
    levels : FactorLevelSchema
        Every level this task's rows sit at, and how they sit relative to each other.
    item_level : FactorLevel
        The level of one dataset item — what the dataset yields. ``unit`` for an
        image-based task, ``sequence`` for tracking.
    label_level : FactorLevel
        The level of one labelled thing. ``instance`` for every task, which is why an
        address with a key means the same thing whatever it was measured over.
    unit_type : str
        What one row at the ``unit`` level holds, in the dataset's own vocabulary.
        Descriptive only — it never affects structuring, and exists so that messages can
        name the medium without the level vocabulary having to.
    """

    task: DatasetTask
    levels: FactorLevelSchema
    item_level: FactorLevel
    label_level: FactorLevel
    unit_type: str


TASK_PROFILES: Mapping[DatasetTask, TaskProfile] = MappingProxyType({
    "IC": TaskProfile("IC", FactorLevelSchema.of("unit", "instance"), "unit", "instance", "image"),
    "OD": TaskProfile("OD", FactorLevelSchema.of("unit", "instance"), "unit", "instance", "image"),
    "MOT": TaskProfile(
        "MOT",
        FactorLevelSchema.of("sequence", "unit", "track", "instance"),
        "sequence",
        "instance",
        "frame",
    ),
})

# Target predicates in priority order; the first that matches wins. Most specific first, so
# the tracking predicate — a *positive* check for its own target type — sits above the
# detection entry rather than being carved out of it. Tracking is checked structurally:
# MAITE's ``MultiobjectTrackingTarget`` is not ``@runtime_checkable``, so an instance check
# against it raises instead of answering.
DISPATCH: tuple[tuple[Callable[[Any], bool], DatasetTask], ...] = (
    (is_multiobject_tracking_target, "MOT"),
    (lambda target: isinstance(target, ObjectDetectionTarget), "OD"),
    (lambda target: isinstance(target, Array), "IC"),
)


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
    for matches, task in DISPATCH:
        if matches(target):
            return task
    return None

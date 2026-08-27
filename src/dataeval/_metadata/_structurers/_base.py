"""The level model a task declares, independent of how its rows are produced."""

__all__ = []

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from dataeval.types import FactorLevel, FactorLevelSchema
from dataeval.types._task import TaskProfile

# Task identifier of a structuring strategy.
TASK = Literal["IC", "OD", "MOT", "factors", "unknown"]


class Structurer:
    """Level model for a task: which levels exist, and what the items and labels are.

    Subclasses declare which levels they produce, which level a dataset *item*
    corresponds to, and which level the labels sit at. The core engine consumes
    the resulting :class:`StructuredData` identically regardless of task.

    Declaring the level model is deliberately separate from producing rows.
    Most structurers read a dataset and so derive from :class:`DatasetStructurer`,
    but :class:`FactorsStructurer` is fed raw arrays and has no dataset to
    iterate; it declares a level model without acquiring an obligation to
    implement :meth:`DatasetStructurer.build`.

    Attributes
    ----------
    profile : TaskProfile or None
        The task's level model, which supplies the four attributes below. Declaring one is
        how a task subclass declares its levels; ``None`` on the strategies that have no
        dataset to detect a task from.
    task : str
        Short task identifier, e.g. ``"IC"`` or ``"OD"``.
    levels : FactorLevelSchema
        Levels this structurer emits rows for.
    item_level : str
        Level corresponding to one dataset item.
    label_level : str
        Level whose rows carry ``class_label``.
    multi_target : bool
        Whether one dataset item can yield more than one labelled row.
    unit_type : str
        What one row at the unit level holds, e.g. ``"image"`` or ``"frame"``.
        Descriptive only; it never affects structuring.
    """

    # The level model, declared once in ``TASK_PROFILES`` and
    # copied onto the class by ``__init_subclass__``. A task subclass names its profile and
    # nothing else; the four attributes below stay declared here so the unspecialized base
    # and :class:`FactorsStructurer` — which has no dataset and so no profile — still have
    # them, and so a reader sees the shape without following the indirection.
    profile: TaskProfile | None = None
    task: TASK = "unknown"
    levels: FactorLevelSchema = FactorLevelSchema.of("unit")
    item_level: FactorLevel = "unit"
    label_level: FactorLevel = "unit"
    multi_target: bool = False
    # What one ``unit`` row holds, in the dataset's own vocabulary. Descriptive only:
    # it is never consulted by structuring, binning or projection, and exists so that
    # messages and reports can name the medium without the level vocabulary having to.
    # A plain ``str`` on purpose — a new modality adds a value here and edits no type.
    unit_type: str = "item"
    # ``"image"`` was the media-unit level's name through v1.1 and is accepted, with a
    # warning, until v1.2.0. Declared on the base rather than per subclass because every
    # task had it; ``"target"`` stays an object-detection-only entry because only object
    # detection ever reported it.
    legacy_level_aliases: Mapping[str, FactorLevel] = MappingProxyType({"image": "unit"})

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject a subclass whose item or label level sits outside its own schema.

        The three declarations are interdependent, and a mismatch surfaces far
        from its cause: propagation and every level filter would quietly select
        no rows. Checking at class creation puts the error on the declaration.
        """
        super().__init_subclass__(**kwargs)
        # Declared once, in the profile, so that this package and every other reader of the
        # level model cannot answer "which levels does this task have" differently. Read
        # from ``__dict__`` rather than the class, so a subclass that names no profile of
        # its own does not re-copy its parent's.
        profile = cls.__dict__.get("profile")
        if profile is not None:
            cls.task = profile.task
            cls.levels = profile.levels
            cls.item_level = profile.item_level
            cls.label_level = profile.label_level
            cls.unit_type = profile.unit_type
        for attribute in ("item_level", "label_level"):
            level = getattr(cls, attribute)
            if level not in cls.levels:
                raise TypeError(
                    f"{cls.__name__}.{attribute} is {level!r}, which is not one of its "
                    f"declared levels {list(cls.levels)}.",
                )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self.task!r}, levels={list(self.levels)})"

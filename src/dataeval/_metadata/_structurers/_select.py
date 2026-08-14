"""Choosing a structuring strategy for a dataset, by target type or by name.

Selection inspects the *target* alone. MAITE constrains what a target is but
places no constraint at all on what an item is — a path string, a PIL handle
and a lazy loader are all valid — so predicating on the item type rejects
perfectly good datasets. Several of these target protocols do not exist in
MAITE yet, so the predicates duck-type on the attributes they should carry.
"""

__all__ = []

from collections.abc import Callable, Mapping, Sized
from types import MappingProxyType
from typing import Any, Literal

from dataeval._log import get_logger
from dataeval._metadata._structurers._classification import ICStructurer
from dataeval._metadata._structurers._dataset import DatasetStructurer
from dataeval._metadata._structurers._detection import ODImageStructurer
from dataeval._metadata._structurers._tracking import MOTStructurer
from dataeval.protocols import (
    AnnotatedDataset,
    Array,
    DatumMetadata,
    ObjectDetectionTarget,
    is_multiobject_tracking_target,
)

_logger = get_logger(__name__)


# Target predicates in priority order; the first that matches wins. Entries are
# ordered most specific first, so the tracking predicate — a *positive* check for its
# own target type — sits above the detection entry rather than being carved out of it.
DISPATCH: tuple[tuple[Callable[[Any], bool], type[DatasetStructurer]], ...] = (
    (is_multiobject_tracking_target, MOTStructurer),
    (lambda x: isinstance(x, ObjectDetectionTarget), ODImageStructurer),
    (lambda x: isinstance(x, Array), ICStructurer),
)

# Task names accepted as an explicit override. Narrower than :data:`TASK`, which
# also names the strategies no caller can ask for by name: ``"factors"`` is reached
# through :meth:`~dataeval.Metadata.from_factors` and ``"unknown"`` is only the
# unspecialized base class's default.
TaskOverride = Literal["IC", "OD", "MOT"]

# Explicit task overrides, for datasets whose protocols MAITE has not defined yet.
TASK_STRUCTURERS: Mapping[str, type[DatasetStructurer]] = MappingProxyType(
    {
        "IC": ICStructurer,
        "OD": ODImageStructurer,
        "MOT": MOTStructurer,
    },
)


def select_structurer(  # noqa: C901
    dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
    task: TaskOverride | None = None,
) -> DatasetStructurer:
    """Choose a structuring strategy for a dataset.

    Parameters
    ----------
    dataset : AnnotatedDataset
        Dataset to inspect. Only the first datum is read.
    task : {"IC", "OD"} or None, default None
        Explicit task override. Matched case-insensitively, so an untyped caller
        may pass ``"od"``. When None the target of the first datum is matched
        against :data:`DISPATCH`.

    Returns
    -------
    DatasetStructurer
        Strategy instance for the detected or requested task, carrying the datum this
        function read so that :meth:`DatasetStructurer.build` does not read it again.

    Raises
    ------
    ValueError
        When ``task`` is unrecognized.
    TypeError
        When no registered predicate matches the dataset's target.

    Notes
    -----
    An empty dataset carries no datum to inspect, so it falls back to image
    classification. This keeps the historical behavior, where an empty dataset
    structured into an empty unit-level dataframe rather than failing; pass an
    explicit ``task`` to structure an empty dataset any other way.

    The fallback is silent here. :class:`~dataeval.Metadata` warns about it instead,
    from ``__init__``/``bind``, because that is the only frame that can point a
    ``stacklevel`` at the user's line: structuring is lazy, so by the time this
    function runs the triggering call is an arbitrary attribute access.
    """
    if task is not None:
        key = str(task).upper()
        if key not in TASK_STRUCTURERS:
            raise ValueError(f"Unknown task {task!r}. Supported tasks are {sorted(TASK_STRUCTURERS)}.")
        return TASK_STRUCTURERS[key]()

    if not isinstance(dataset, Sized) or len(dataset) == 0:
        _logger.debug("Cannot infer a task from an empty dataset; assuming image classification.")
        return ICStructurer()

    # Handed to the chosen structurer so a dataset that decodes on __getitem__ pays
    # for item 0 once, not once here and again on the first iteration of build().
    first_datum = dataset[0]
    _, target, _ = first_datum
    for target_check, structurer in DISPATCH:
        if target_check(target):
            _logger.debug("Selected %s for target %s", structurer.__name__, type(target))
            return structurer(first_datum)

    raise TypeError(
        f"Unable to infer a task from target type {type(target).__name__}. "
        f"Pass an explicit task, one of {sorted(TASK_STRUCTURERS)}.",
    )

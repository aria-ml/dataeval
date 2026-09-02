"""Choosing a structuring strategy for a dataset, by target type or by name.

Selection inspects the *target* alone. MAITE constrains what a target is but
places no constraint at all on what an item is — a path string, a PIL handle
and a lazy loader are all valid — so predicating on the item type rejects
perfectly good datasets. Several of these target protocols do not exist in
MAITE yet, so the predicates duck-type on the attributes they should carry.
"""

__all__ = []

from collections.abc import Mapping, Sized
from types import MappingProxyType
from typing import Any

from dataeval._log import get_logger
from dataeval._metadata._structurers._classification import ICStructurer
from dataeval._metadata._structurers._dataset import DatasetStructurer
from dataeval._metadata._structurers._detection import ODImageStructurer
from dataeval._metadata._structurers._tracking import MOTStructurer
from dataeval.protocols import AnnotatedDataset, DatumMetadata
from dataeval.types._task import DatasetTask, detect_task

_logger = get_logger(__name__)


# Task names accepted as an explicit override. Narrower than :data:`TASK`, which also
# names the strategies no caller can ask for by name: ``"factors"`` is reached through
# :meth:`~dataeval.Metadata.from_factors` and ``"unknown"`` is only the unspecialized base
# class's default. The same set ``TASK_PROFILES`` is keyed on,
# since a task a dataset can be detected as is a task that can be asked for.
TaskOverride = DatasetTask

# Explicit task overrides, for datasets whose protocols MAITE has not defined yet.
TASK_STRUCTURERS: Mapping[str, type[DatasetStructurer]] = MappingProxyType(
    {
        "IC": ICStructurer,
        "OD": ODImageStructurer,
        "MOT": MOTStructurer,
    },
)


def select_structurer(
    dataset: AnnotatedDataset[tuple[Any, Any, DatumMetadata]],
    task: TaskOverride | None = None,
    *,
    partial_factors: bool = False,
) -> DatasetStructurer:
    """Choose a structuring strategy for a dataset.

    Parameters
    ----------
    dataset : AnnotatedDataset
        Dataset to inspect. Only the first datum is read.
    task : {"IC", "OD"} or None, default None
        Explicit task override. Matched case-insensitively, so an untyped caller
        may pass ``"od"``. When None the target of the first datum is matched
        against ``dataeval.types._task.DISPATCH``.
    partial_factors : bool, default False
        Keep a factor only some rows declare, with the rest missing. Carried on the
        structurer rather than passed to :meth:`DatasetStructurer.build`, so that every
        place the walk meets an incompletely declared value reads one answer.

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
        return TASK_STRUCTURERS[key](partial_factors=partial_factors)

    if not isinstance(dataset, Sized) or len(dataset) == 0:
        _logger.debug("Cannot infer a task from an empty dataset; assuming image classification.")
        return ICStructurer(partial_factors=partial_factors)

    # Handed to the chosen structurer so a dataset that decodes on __getitem__ pays
    # for item 0 once, not once here and again on the first iteration of build().
    first_datum = dataset[0]
    _, target, _ = first_datum
    # Which task a target is belongs to the shared level model rather than to this package:
    # anything that resolves an address against a dataset asks the same question of the same
    # target and must get the same answer. What a task *structures into* stays here.
    task_name = detect_task(target)
    if task_name is not None:
        structurer = TASK_STRUCTURERS[task_name]
        _logger.debug("Selected %s for target %s", structurer.__name__, type(target))
        return structurer(first_datum, partial_factors=partial_factors)

    raise TypeError(
        f"Unable to infer a task from target type {type(target).__name__}. "
        f"Pass an explicit task, one of {sorted(TASK_STRUCTURERS)}.",
    )

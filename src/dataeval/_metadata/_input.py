"""Normalizing what a caller passed before anything trusts it.

:meth:`~dataeval.Metadata.from_factors` and :meth:`~dataeval.Metadata.add_factors` accept
the same shapes of input — a factor mapping, a whole stats result, a retired level
spelling — and the two must agree on what each one means. These functions are that
agreement, held once rather than restated on each path, and they are pure: they translate
and reject, and never touch the metadata they are about to be used on.
"""

__all__ = []

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from dataeval.exceptions import ShapeMismatchError
from dataeval.types import Array1D, FactorLevel, SourceIndex


def _is_stats_result(candidate: Any) -> bool:
    """Whether a mapping is a :class:`~dataeval.core.StatsResult` rather than a factor mapping.

    ``StatsResult`` is a :class:`~typing.TypedDict`, so at runtime it is an ordinary dict
    and there is nothing to check with ``isinstance``. Both keys are required and both are
    checked structurally: a caller's factor mapping could plausibly hold a factor named
    ``stats`` or ``source_index``, but not one whose ``stats`` is itself a mapping *and*
    whose ``source_index`` is a sequence of :class:`~dataeval.types.SourceIndex`. Being
    strict here matters more than being permissive — a false positive silently discards
    every factor the caller passed.

    The first entry stands for the sequence. A stats result's index is homogeneous by
    construction, and this runs on every ``add_factors`` call, including the common one
    where the argument really is a factor mapping holding one entry per detection.
    """
    if not isinstance(candidate, Mapping) or not candidate.keys() >= {"stats", "source_index"}:
        return False
    source_index = candidate["source_index"]
    return (
        isinstance(candidate["stats"], Mapping)
        and isinstance(source_index, Sequence)
        and (not source_index or isinstance(source_index[0], SourceIndex))
    )


def unpack_stats_result(
    factors: Any,
    source_index: Sequence[SourceIndex] | None,
    *,
    level: Any = None,
) -> tuple[Mapping[str, Array1D[Any]], Sequence[SourceIndex] | None]:
    """Accept a whole stats result wherever a factor mapping is accepted.

    :func:`~dataeval.core.compute_stats` and :func:`~dataeval.core.compute_ratios` return
    the statistics and the labels that place them in one object, and separating them again
    at every call site is busywork that also invites passing one without the other. When
    the result is recognised, its ``stats`` become the factors and its ``source_index``
    the placement — unless the caller passed an explicit one, which wins so that a
    hand-corrected index remains usable.

    The bookkeeping keys — ``object_count``, ``invalid_box_count``, ``image_count`` —
    describe the run rather than the images and are not factors, so they are dropped.

    Raises
    ------
    ValueError
        When a level is named as well. The result already says what each value describes,
        and honouring one of the two silently would discard a real contradiction.
    """
    if not _is_stats_result(factors):
        return factors, source_index
    if level is not None and level != "auto":
        raise ValueError(
            f"`level` and the source_index carried by this stats result are mutually exclusive; "
            f"the result already labels each value with what it describes. Pass the result's "
            f"['stats'] mapping instead to place its values at level={level!r}.",
        )
    return factors["stats"], source_index if source_index is not None else factors["source_index"]


def reject_length_mismatch(factors: Mapping[str, Any], source_index: Sequence[SourceIndex]) -> None:
    """Reject factors that do not hold exactly one value per source-index entry.

    Shared by both constructors: the source index is the placement, so a factor that is
    not as long as it names rows the caller never described, whichever spelling was used
    to get here.
    """
    mismatched = {name: len(values) for name, values in factors.items() if len(values) != len(source_index)}
    if mismatched:
        raise ShapeMismatchError(
            f"All factors must have one value per source_index entry ({len(source_index)}); got {mismatched}.",
        )


def build_index2label(
    provided: Mapping[int, str] | None,
    observed_labels: Iterable[Any],
) -> dict[int, str]:
    """Map each class index to a name, backfilling observed labels missing from ``provided``.

    When ``provided`` is given it is the source of truth; any observed label without an
    entry gets an ``UNDEFINED_CLASS_<i>`` placeholder. Otherwise labels name themselves.
    """
    if provided is not None:
        index2label = {int(k): str(v) for k, v in provided.items()}
        for lbl in observed_labels:
            index2label.setdefault(int(lbl), f"UNDEFINED_CLASS_{int(lbl)}")
        return index2label
    return {int(lbl): str(int(lbl)) for lbl in observed_labels}


def resolve_legacy_level(level: str, aliases: Mapping[str, FactorLevel], stacklevel: int, unit_type: str) -> str:
    """Translate a retired level spelling, warning at the caller's line.

    Shared by the two paths that can be handed one: :meth:`Metadata._resolve_level`,
    which has a structurer and so knows the task's full alias map, and the factors-only
    loading path, which is choosing the level a :class:`FactorsStructurer` will be built
    with and so has only the base map. One function so the two cannot word the
    deprecation differently.
    """
    alias = aliases.get(level)
    if alias is None:
        return level
    # Naive "+ s" pluralization: correct for every current unit_type ("image",
    # "frame", "item"). A future unit_type needing an irregular plural should be
    # given one at its declaration site rather than inflected here.
    warnings.warn(
        f"Level {level!r} is deprecated and will stop resolving in a future "
        f"release. It is no longer a level name; pass {alias!r} instead "
        f"(this dataset's units are {unit_type}s).",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return alias

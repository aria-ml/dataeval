"""What structuring did not carry into the rows, said out loud.

Both cases here are legitimate shapes rather than errors — a partially labelled dataset,
and a metadata key spelling the same name as a factor the structurer derives itself — and
both change what a caller sees without changing what they asked for. They are logged at
info level rather than warned about, and collected in one module so that the two reads
the same way.
"""

__all__ = []

from collections.abc import Container, Mapping, Sequence
from typing import Any

from dataeval._log import get_logger
from dataeval.types import FactorLevel

_logger = get_logger(__name__)


def log_items_without_targets(without: Sequence[int], level: FactorLevel, items: int) -> None:
    """Note dataset items that carried no target.

    These items keep their item-level row and every factor on it; they contribute no
    row at ``level``, so label-aware analysis covers fewer items than the dataset has.
    Informational rather than a warning: a partially labelled dataset is a legitimate
    shape, and it costs no data now that the item level is separate from the target
    level.
    """
    if not without:
        return
    _logger.info(
        "%d of %d dataset item(s) %s carried no target and contribute no %r rows. Their item-level "
        "rows and factors are unaffected; Metadata.item_indices lists the items that do have targets.",
        len(without),
        items,
        list(without) if len(without) <= 10 else [*without[:10], "..."],
        level,
    )


def without_displaced(factors: Mapping[str, Any], displaced: Container[str], level: FactorLevel) -> dict[str, Any]:
    """Drop factor names a structurer derives itself, logging each one it removes.

    A factor belongs to exactly one level, so a metadata key spelling the same name as a
    derived factor cannot simply coexist with it. The derived value wins — it is read off
    the dataset's own frames and targets — and the displacement is logged rather than
    silent, because the value a caller sees is not the one their metadata supplied.
    """
    kept = {name: values for name, values in factors.items() if name not in displaced}
    for name in factors:
        if name not in kept:
            _logger.info(
                "Metadata key %r at the %r level is displaced by the derived factor of the same "
                "name, which is read from the dataset's own frames and targets.",
                name,
                level,
            )
    return kept

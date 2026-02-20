"""
Check for skewed or imbalanced datasets and incomplete feature representation.

This may impact model performance.
"""

__all__ = [
    "Balance",
    "BalanceOutput",
    "Diversity",
    "DiversityOutput",
    "Parity",  # type: ignore - experimental
    "ParityOutput",  # type: ignore - experimental
]

from typing import Any

from ._balance import Balance, BalanceOutput
from ._diversity import Diversity, DiversityOutput

_EXPERIMENTAL: dict[str, tuple[str, str]] = {
    "Parity": ("dataeval.bias._parity", "Parity"),
    "ParityOutput": ("dataeval.bias._parity", "ParityOutput"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPERIMENTAL:
        from dataeval._experimental import _lazy_import_with_warning

        module_path, attr_name = _EXPERIMENTAL[name]
        return _lazy_import_with_warning(module_path, attr_name, f"dataeval.bias.{name}", "experimental")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

__all__ = []

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from dataeval.core._calculators._base import Calculator
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core.flags import ImageStats

if TYPE_CHECKING:
    from dataeval.core._calculate import CalculatorCache


@CalculatorRegistry.register(ImageStats)
class HashStatCalculator(Calculator):
    """Calculator for hash-based statistics."""

    def __init__(self, datum: NDArray[Any], cache: CalculatorCache, per_channel: bool = False) -> None:
        self.datum = datum
        self.cache = cache

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.HASH

    def _xxhash(self) -> list[str]:
        from dataeval.core._hash import xxhash

        return [xxhash(self.cache.raw)]

    def _pchash(self) -> list[str]:
        from dataeval.core._hash import pchash

        return [pchash(self.cache.raw)]

    def get_handlers(self) -> dict[ImageStats, tuple[str, Callable[[], list[Any]]]]:
        """Return mapping of flags to (stat_name, handler_function)."""
        return {
            ImageStats.HASH_XXHASH: ("xxhash", self._xxhash),
            ImageStats.HASH_PCHASH: ("pchash", self._pchash),
        }

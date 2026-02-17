"""Calculator for hash-based image statistics."""

__all__ = []

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from dataeval.core._calculators._base import Calculator
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats

if TYPE_CHECKING:
    from dataeval.core._calculate import CalculatorCache


@CalculatorRegistry.register(ImageStats)
class HashStatCalculator(Calculator):
    """Calculator for hash-based statistics.

    Computes various hash values for duplicate detection:
    - xxhash: Fast non-cryptographic hash for exact duplicates
    - phash: DCT-based perceptual hash for near duplicates
    - dhash: Gradient-based perceptual hash for near duplicates
    """

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache", per_channel: bool = False) -> None:  # noqa: ARG002
        self.datum = datum
        self.cache = cache
        self.warnings: list[str] = []

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.HASH

    def _collect(self, result: tuple[str, str | None]) -> list[str]:
        hash_value, warning = result
        if warning:
            self.warnings.append(warning)
        return [hash_value]

    def _compute_xxhash(self) -> list[str]:
        from dataeval.core._hash import _xxhash

        return self._collect(_xxhash(self.cache.image))

    def _compute_phash(self) -> list[str]:
        from dataeval.core._hash import _phash

        return self._collect(_phash(self.cache.image))

    def _compute_phash_d4(self) -> list[str]:
        from dataeval.core._hash import _phash_d4

        return self._collect(_phash_d4(self.cache.image))

    def _compute_dhash(self) -> list[str]:
        from dataeval.core._hash import _dhash

        return self._collect(_dhash(self.cache.image))

    def _compute_dhash_d4(self) -> list[str]:
        from dataeval.core._hash import _dhash_d4

        return self._collect(_dhash_d4(self.cache.image))

    def get_empty_values(self) -> dict[str, Any]:
        """Return empty values for hash statistics."""
        return {
            "xxhash": "",
            "phash": "",
            "dhash": "",
            "phash_d4": "",
            "dhash_d4": "",
        }

    def get_handlers(self) -> dict[ImageStats, tuple[str, Callable[[], list[Any]]]]:
        """Return mapping of flags to (stat_name, handler_function)."""
        return {
            ImageStats.HASH_XXHASH: ("xxhash", self._compute_xxhash),
            ImageStats.HASH_PHASH: ("phash", self._compute_phash),
            ImageStats.HASH_DHASH: ("dhash", self._compute_dhash),
            ImageStats.HASH_PHASH_D4: ("phash_d4", self._compute_phash_d4),
            ImageStats.HASH_DHASH_D4: ("dhash_d4", self._compute_dhash_d4),
        }

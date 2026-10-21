"""Calculator for hash-based image statistics."""

__all__ = []

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats


@CalculatorRegistry.register(ImageStats)
class HashStatCalculator(Calculator):
    """Calculator for hash-based statistics.

    Computes various hash values for duplicate detection:
    - xxhash: Fast non-cryptographic hash for exact duplicates
    - phash: DCT-based perceptual hash for near duplicates
    - dhash: Gradient-based perceptual hash for near duplicates
    """

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache
        self.warnings: list[str] = []

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.HASH

    def _collect(self, digest: Callable[[NDArray[Any]], tuple[str, str | None]], *, anchor: bool) -> list[str]:
        """Hash the view, or report its absence.

        Data that is entirely NaN was not measured — an out-of-bounds box, or a band group
        the datum cannot supply. Hashing it anyway digests whatever the grayscale
        conversion substitutes for NaN, which is the *same* substitution every time: a
        real-looking digest that makes every unmeasured region a duplicate of every other.
        The empty string is what this calculator already uses for an absent hash.
        """
        if self.cache.is_all_nan:
            return [""]
        # `xxhash` digests the raw bytes, so an 8-bit view and a float64 one of the same
        # pixels hash differently, and which of the two a row gets would otherwise depend
        # on whether its window needed padding. It therefore takes a float64 anchor, which
        # keeps its digest comparable across rows, runs and releases. The perceptual hashes
        # must not be given one: `to_canonical_grayscale` reads float data with a maximum
        # of at most 1 as [0, 1] and stretches it to full contrast, so a near-black integer
        # frame would hash as if it were full contrast.
        image = np.asarray(self.cache.image, dtype=np.float64) if anchor else self.cache.image
        hash_value, warning = digest(image)
        if warning:
            self.warnings.append(warning)
        return [hash_value]

    def _compute_xxhash(self) -> list[str]:
        from dataeval.core._hash import _xxhash

        return self._collect(_xxhash, anchor=True)

    def _compute_phash(self) -> list[str]:
        from dataeval.core._hash import _phash

        return self._collect(_phash, anchor=False)

    def _compute_phash_d4(self) -> list[str]:
        from dataeval.core._hash import _phash_d4

        return self._collect(_phash_d4, anchor=False)

    def _compute_dhash(self) -> list[str]:
        from dataeval.core._hash import _dhash

        return self._collect(_dhash, anchor=False)

    def _compute_dhash_d4(self) -> list[str]:
        from dataeval.core._hash import _dhash_d4

        return self._collect(_dhash_d4, anchor=False)

    def get_empty_values(self) -> dict[str, Any]:
        """Return empty values for hash statistics."""
        return {
            "xxhash": "",
            "phash": "",
            "dhash": "",
            "phash_d4": "",
            "dhash_d4": "",
        }

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        Hashes digest the pixel buffer, which NaN destabilizes and which resize routines
        turn into noise — so no masked region has a hash. A band group does: every hash
        differs across band subsets, and the hash of a named group is in fact the one a
        multispectral caller needs, since hashing the whole cube runs a grayscale
        conversion whose four-channel branch guesses between CMYK and RGBA.
        """
        banded = ViewKind.WHOLE | ViewKind.BAND
        return {
            ImageStats.HASH_XXHASH: Handler("xxhash", self._compute_xxhash, banded),
            ImageStats.HASH_PHASH: Handler("phash", self._compute_phash, banded),
            ImageStats.HASH_DHASH: Handler("dhash", self._compute_dhash, banded),
            ImageStats.HASH_PHASH_D4: Handler("phash_d4", self._compute_phash_d4, banded),
            ImageStats.HASH_DHASH_D4: Handler("dhash_d4", self._compute_dhash_d4, banded),
        }

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dataeval.core._hash import pchash, xxhash
from dataeval.core._processor import BaseProcessor, process
from dataeval.typing import ArrayLike
from dataeval.utils._boundingbox import BoxLike


class HashStatsProcessor(BaseProcessor):
    def process(self) -> dict[str, list[Any]]:
        return {
            "xxhash": [xxhash(self.raw)],
            "pchash": [pchash(self.raw)],
        }


def hashstats(
    images: Iterable[ArrayLike],
    boxes: Iterable[Iterable[BoxLike] | None] | None,
) -> dict[str, Any]:
    return process(images, boxes, [HashStatsProcessor])

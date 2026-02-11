"""Shared test helpers for verification tests.

Provides lightweight mock objects that satisfy DataEval protocols without
depending on the unit test suite or heavyweight dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimpleMetadata:
    """Minimal implementation of the Metadata protocol for verification tests."""

    class_labels: NDArray[np.intp]
    factor_data: NDArray[np.int64]
    factor_names: list[str]
    is_discrete: list[bool]
    index2label: dict[int, str] = field(default_factory=dict)


def make_metadata(
    n_samples: int = 60,
    n_factors: int = 3,
    n_classes: int = 3,
    seed: int = 42,
) -> SimpleMetadata:
    """Create a simple metadata object for bias evaluation tests."""
    rng = np.random.default_rng(seed)
    class_labels = np.tile(np.arange(n_classes, dtype=np.intp), n_samples // n_classes + 1)[:n_samples]
    factor_data = rng.integers(0, 4, size=(n_samples, n_factors), dtype=np.int64)
    factor_names = [f"factor_{i}" for i in range(n_factors)]
    is_discrete = [True] * n_factors
    index2label = {i: f"class_{i}" for i in range(n_classes)}

    return SimpleMetadata(
        class_labels=class_labels,
        factor_data=factor_data,
        factor_names=factor_names,
        is_discrete=is_discrete,
        index2label=index2label,
    )


@dataclass
class SimpleImageDataset:
    """Minimal dataset satisfying the Dataset protocol for image data."""

    images: NDArray[np.floating]

    def __getitem__(self, idx: int):
        return self.images[idx]

    def __len__(self) -> int:
        return len(self.images)


@dataclass
class SimpleAnnotatedDataset:
    """Minimal dataset satisfying the AnnotatedDataset protocol."""

    images: NDArray[np.floating]
    labels: NDArray[np.integer]
    _metadata: dict = field(default_factory=lambda: {"id": "verification-test"})

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx]), {}

    def __len__(self) -> int:
        return len(self.images)

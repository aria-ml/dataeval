"""A drift detector given an extractor reads its reference set once.

The reference set can be a whole dataset whose items are expensive to load, so each extra pass of the extractor over it
is a full read of that dataset.
"""

from typing import Any

import numpy as np
import pytest

from dataeval.shift import DriftDomainClassifier, DriftKNeighbors, DriftMMD, DriftUnivariate, DriftWasserstein


class _CountingDataset:
    """Indexable items that count how often they are read. No ``__iter__``, so iteration goes through indexing."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.reads = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> np.ndarray:
        if index >= len(self._values):
            raise IndexError(index)
        self.reads += 1
        return self._values[index]


class _Stack:
    """An extractor that reads every item of what it is given."""

    def __call__(self, data: Any) -> np.ndarray:
        return np.stack([np.asarray(item, dtype=np.float32) for item in data])


def _data(seed: int, n: int = 40) -> _CountingDataset:
    return _CountingDataset(np.random.default_rng(seed).normal(size=(n, 4)).astype(np.float32))


@pytest.mark.required
class TestReferenceReadOnce:
    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda: DriftUnivariate(extractor=_Stack()), id="univariate"),
            pytest.param(lambda: DriftMMD(extractor=_Stack(), sigma=np.array([1.0]), n_permutations=10), id="mmd"),
            pytest.param(lambda: DriftKNeighbors(extractor=_Stack()), id="kneighbors"),
            pytest.param(lambda: DriftDomainClassifier(extractor=_Stack(), n_folds=2), id="domain_classifier"),
        ],
    )
    def test_fit_then_predict_reads_the_reference_once(self, make):
        reference = _data(0)
        detector = make().fit(reference)
        detector.predict(_data(1))
        assert reference.reads == len(reference)
        assert detector.n == len(reference)

    def test_wasserstein_reads_the_reference_once(self):
        reference = _data(0)
        detector = DriftWasserstein(extractor=_Stack()).fit(reference, _data(2))
        detector.predict(_data(1))
        assert reference.reads == len(reference)
        assert detector.n == len(reference)

    def test_reference_data_is_the_extracted_reference(self):
        reference = _data(0)
        detector = DriftUnivariate(extractor=_Stack()).fit(reference)
        np.testing.assert_array_equal(detector.reference_data, reference._values)

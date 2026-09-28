"""from_embeddings fits on the reference and tests the data in one call: the same as fit, then predict."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from dataeval.shift import (
    DriftDomainClassifier,
    DriftKNeighbors,
    DriftMMD,
    DriftReconstruction,
    DriftUnivariate,
    DriftWasserstein,
    OODDomainClassifier,
    OODKNeighbors,
    OODReconstruction,
)

_RNG = np.random.default_rng(0)
_REFERENCE = _RNG.random((120, 6)).astype(np.float32)
_VALIDATION = _RNG.random((120, 6)).astype(np.float32)
_DATA = (_RNG.random((80, 6)) + 0.3).astype(np.float32)

_DRIFT: list[Callable[[], Any]] = [
    lambda: DriftUnivariate(),
    lambda: DriftMMD(n_permutations=10, device="cpu"),
    lambda: DriftKNeighbors(k=5),
    lambda: DriftDomainClassifier(n_folds=2),
]
_DRIFT_IDS = ["univariate", "mmd", "kneighbors", "domain_classifier"]


def _seeded(call: Callable[[], Any]) -> Any:
    """Run `call` from a fixed NumPy and torch state, then put both back as they were.

    MMD draws its permutations from torch's global generator, so two calls agree only from the same state.
    The suite-wide DataEval seed is left alone.
    """
    numpy_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    try:
        np.random.seed(0)
        torch.manual_seed(0)
        return call()
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


@pytest.mark.required
class TestDriftFromEmbeddings:
    @pytest.mark.parametrize("make", _DRIFT, ids=_DRIFT_IDS)
    def test_it_is_fit_then_predict(self, make):
        one_call = _seeded(lambda: make().from_embeddings(_REFERENCE, _DATA))
        two_calls = _seeded(lambda: make().fit(_REFERENCE).predict(_DATA))
        assert (one_call.drifted, one_call.distance, one_call.threshold) == (
            two_calls.drifted,
            two_calls.distance,
            two_calls.threshold,
        )

    @pytest.mark.parametrize("make", _DRIFT, ids=_DRIFT_IDS)
    def test_it_records_its_own_name(self, make):
        assert make().from_embeddings(_REFERENCE, _DATA).meta().name.endswith("from_embeddings")

    def test_wasserstein_fits_on_the_reference_and_validation(self):
        one_call = DriftWasserstein().from_embeddings(_REFERENCE, _VALIDATION, _DATA)
        two_calls = DriftWasserstein().fit(_REFERENCE, _VALIDATION).predict(_DATA)
        assert (one_call.drifted, one_call.distance) == (two_calls.drifted, two_calls.distance)


@pytest.mark.required
class TestChunkedFromEmbeddings:
    def test_it_is_fit_then_predict(self):
        one_call = DriftUnivariate().chunked(chunk_count=4).from_embeddings(_REFERENCE, _DATA)
        two_calls = DriftUnivariate().chunked(chunk_count=4).fit(_REFERENCE).predict(_DATA)
        assert one_call.details.equals(two_calls.details)

    def test_wasserstein_takes_its_validation_set_between(self):
        one_call = DriftWasserstein().chunked(chunk_count=4).from_embeddings(_REFERENCE, _VALIDATION, _DATA)
        two_calls = DriftWasserstein().chunked(chunk_count=4).fit(_REFERENCE, _VALIDATION).predict(_DATA)
        assert one_call.details.equals(two_calls.details)

    def test_it_records_its_own_name(self):
        output = DriftUnivariate().chunked(chunk_count=4).from_embeddings(_REFERENCE, _DATA)
        assert output.meta().name.endswith("ChunkedDrift.from_embeddings")

    def test_one_array_is_refused(self):
        with pytest.raises(ValueError, match="at least two"):
            DriftUnivariate().chunked(chunk_count=4).from_embeddings(_REFERENCE)


@pytest.mark.required
class TestOODFromEmbeddings:
    @pytest.mark.parametrize(
        "make", [lambda: OODKNeighbors(k=5), lambda: OODDomainClassifier(n_folds=2, n_repeats=1)], ids=["kn", "dc"]
    )
    def test_it_is_fit_then_predict(self, make):
        one_call = _seeded(lambda: make().from_embeddings(_REFERENCE, _DATA))
        two_calls = _seeded(lambda: make().fit(_REFERENCE).predict(_DATA))
        assert np.array_equal(one_call.is_ood, two_calls.is_ood)
        assert np.allclose(one_call.instance_score, two_calls.instance_score)

    def test_the_predict_keywords_are_passed_on(self):
        one_call = OODKNeighbors(k=5).from_embeddings(_REFERENCE, _DATA, batch_size=16, ood_type="instance")
        assert one_call.meta().name.endswith("from_embeddings")
        assert len(one_call.is_ood) == len(_DATA)


@pytest.mark.required
@pytest.mark.parametrize("detector", [DriftReconstruction, OODReconstruction], ids=lambda c: c.__name__)
def test_reconstruction_evaluators_take_no_embeddings(detector):
    """Their model reconstructs whatever array it was trained on, usually images, not embeddings."""
    assert not hasattr(detector, "from_embeddings")

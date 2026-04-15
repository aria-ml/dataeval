"""Verify that dataset prioritization evaluators produce correct output types.

Maps to meta repo test cases:
  - TC-11.2: Dataset prioritization (Prioritize)
"""

import numpy as np
import pytest

import dataeval.config as config


@pytest.fixture(autouse=True)
def set_batch_size():
    config.set_batch_size(16)
    yield
    config.set_batch_size(None)


@pytest.mark.test_case("11-2")
class TestPrioritization:
    """Verify Prioritize evaluator."""

    def test_prioritize_ranks_samples_knn(self):
        from dataeval._embeddings import Embeddings
        from dataeval.extractors import FlattenExtractor
        from dataeval.scope import Prioritize, PrioritizeOutput

        rng = np.random.default_rng(42)
        images = rng.standard_normal((50, 3, 16, 16)).astype(np.float32)
        embeddings = Embeddings(images, FlattenExtractor())

        # Prioritize needs an extractor, even if embeddings is passed.
        detector = Prioritize(extractor=FlattenExtractor(), method="knn")
        result = detector.evaluate(embeddings)

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 50
        assert len(result.scores) == 50

    def test_prioritize_output_lazy_evaluation(self):
        from dataeval._embeddings import Embeddings
        from dataeval.extractors import FlattenExtractor
        from dataeval.scope import Prioritize

        rng = np.random.default_rng(42)
        images = rng.standard_normal((20, 3, 16, 16)).astype(np.float32)
        embeddings = Embeddings(images, FlattenExtractor())

        result = Prioritize(extractor=FlattenExtractor(), method="knn").evaluate(embeddings)
        # Accessing indices should trigger computation if lazy
        indices = result.indices
        assert len(indices) == 20
        assert indices.dtype == np.intp

    def test_prioritize_output_order_transformation(self):
        from dataeval._embeddings import Embeddings
        from dataeval.extractors import FlattenExtractor
        from dataeval.scope import Prioritize

        rng = np.random.default_rng(42)
        images = rng.standard_normal((20, 3, 16, 16)).astype(np.float32)
        embeddings = Embeddings(images, FlattenExtractor())

        result = Prioritize(extractor=FlattenExtractor(), method="knn").evaluate(embeddings)

        # Test order transformations
        hard_first = result.hard_first()
        assert hard_first.order == "hard_first"
        assert len(hard_first.indices) == 20

        easy_first = hard_first.easy_first()
        assert easy_first.order == "easy_first"
        assert len(easy_first.indices) == 20

    def test_prioritize_stratified_policy(self):
        from dataeval._embeddings import Embeddings
        from dataeval.extractors import FlattenExtractor
        from dataeval.scope import Prioritize

        rng = np.random.default_rng(42)
        images = rng.standard_normal((100, 3, 16, 16)).astype(np.float32)
        embeddings = Embeddings(images, FlattenExtractor())

        # Use stratified() method on the output
        result = Prioritize(extractor=FlattenExtractor(), method="knn").evaluate(embeddings)
        strat_result = result.stratified(num_bins=5)
        assert strat_result.policy == "stratified"
        assert len(strat_result.indices) == 100

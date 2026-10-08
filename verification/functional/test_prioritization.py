"""Verify that dataset prioritization evaluators produce correct output types.

Maps to meta repo test cases:
  - TC-11.2: Dataset prioritization (Prioritize)
"""

import numpy as np
import pytest

import dataeval.config as config
from dataeval.scope._prioritize import MethodType


@pytest.fixture(autouse=True)
def set_batch_size():
    config.set_batch_size(16)
    yield
    config.set_batch_size(None)


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
        assert result.scores is not None
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

    @pytest.mark.parametrize(
        "method", ["knn", "kmeans_distance", "hdbscan_distance", "kmeans_complexity", "hdbscan_complexity"]
    )
    def test_prioritize_ranks_samples_with_each_method(self, method: MethodType):
        from dataeval.scope import Prioritize

        rng = np.random.default_rng(0)
        embeddings = np.concatenate([
            rng.normal(0, 1, (60, 8)),
            rng.normal(8, 1, (30, 8)),
            rng.normal(-8, 1, (10, 8)),
        ]).astype(np.float32)
        embeddings[0] = 30  # one far outlier

        result = Prioritize(method=method).evaluate(embeddings)

        assert result.method == method
        assert sorted(result.indices.tolist()) == list(range(100))  # a full ranking of every sample
        if method.endswith("_distance") or method == "knn":
            assert result.scores is not None
            assert len(result.scores) == 100
        else:
            assert result.scores is None  # complexity methods rank without per-sample scores
        if method == "knn":
            assert result.indices[-1] == 0  # the outlier is the hardest sample under easy_first

    def test_prioritize_stratified_and_class_balanced_policies(self):
        from dataeval.scope import Prioritize

        rng = np.random.default_rng(0)
        embeddings = np.concatenate([
            rng.normal(0, 1, (70, 8)),
            rng.normal(8, 1, (20, 8)),
            rng.normal(-8, 1, (10, 8)),
        ]).astype(np.float32)
        labels = np.array([0] * 70 + [1] * 20 + [2] * 10)
        result = Prioritize(method="knn").evaluate(embeddings)
        assert result.scores is not None

        # Stratified: early picks span the score range instead of clustering at the easy end.
        stratified = result.stratified(num_bins=5)
        assert stratified.policy == "stratified"
        assert sorted(stratified.indices.tolist()) == list(range(100))
        top = 10
        spread_stratified = np.ptp(result.scores[stratified.indices[:top]])
        spread_difficulty = np.ptp(result.scores[result.indices[:top]])
        assert spread_stratified > spread_difficulty

        # Class balanced: classes alternate even though class 0 holds 70% of the data.
        balanced = result.class_balanced(labels)
        assert balanced.policy == "class_balanced"
        assert sorted(balanced.indices.tolist()) == list(range(100))
        assert labels[balanced.indices[:30]].tolist() == [0, 1, 2] * 10
        assert np.bincount(labels[result.indices[:30]], minlength=3).tolist() != [10, 10, 10]

    def test_prioritize_on_object_detection_dataset(self):
        try:
            from dataeval.data import DetectionCrops  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("Prioritize reads object-detection datasets through DetectionCrops on this branch")
        from dataeval.extractors import FlattenExtractor
        from dataeval.scope import Prioritize
        from verification.helpers import SimpleODDataset

        rng = np.random.default_rng(0)
        images = rng.random((20, 3, 16, 16)).astype(np.float32)
        labels = [np.array([0, 1]) if i % 2 else np.array([2]) for i in range(20)]
        dataset = SimpleODDataset(images, labels)

        result = Prioritize(extractor=FlattenExtractor(), method="knn").evaluate(dataset)
        assert sorted(result.indices.tolist()) == list(range(20))  # one rank per image
        assert result.scores is not None

        balanced = Prioritize(extractor=FlattenExtractor(), method="knn", policy="class_balanced").evaluate(dataset)
        assert sorted(balanced.indices.tolist()) == list(range(20))

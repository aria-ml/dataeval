import numpy as np
import polars as pl
import pytest
import torch

from dataeval.exceptions import ShapeMismatchError
from dataeval.extractors import TorchExtractor
from dataeval.protocols import DatasetMetadata
from dataeval.scope._coverage import _PER_CLASS_SCHEMA, Coverage, CoverageOutput


class ClassificationDataset:
    """A minimal image-classification dataset with one-hot targets and index2label."""

    def __init__(self, class_names: list[str], per_class: int, shape: tuple[int, ...] = (3, 4, 4)) -> None:
        targets = []
        for i in range(len(class_names)):
            for _ in range(per_class):
                onehot = np.zeros(len(class_names))
                onehot[i] = 1
                targets.append(onehot.copy())
        self._targets = targets
        self._data = np.random.default_rng(0).random((len(targets), *shape)).astype(np.float32)
        self.metadata = DatasetMetadata(id="d", index2label=dict(enumerate(class_names)))

    def __getitem__(self, i):
        return self._data[i], self._targets[i], {"id": i}

    def __len__(self) -> int:
        return len(self._targets)


def _two_class_embeddings(rng, class_a, class_b):
    """Stack two per-class point clouds into one embedding/label array."""
    embeddings = np.vstack([class_a, class_b]).astype(np.float64)
    class_labels = np.array([0] * len(class_a) + [1] * len(class_b), dtype=np.intp)
    index2label = {0: "a", 1: "b"}
    return embeddings, class_labels, index2label


def _rows_by_class(cov, embeddings, class_labels, index2label):
    """Per-class rows keyed by class label (no globally-uncovered samples)."""
    rows = cov._per_class(embeddings, class_labels, index2label, np.array([], dtype=np.intp))
    return {row["class"]: row for row in rows}


@pytest.mark.required
class TestCoverageIsotropy:
    def test_isotropic_class_scores_higher_than_anisotropic(self):
        """An isotropic class fills its directions; a near-collinear one does not."""
        rng = np.random.default_rng(0)
        d = 3
        isotropic = rng.normal(size=(120, d))
        # Anisotropic: variance concentrated on one axis (shot from one angle).
        anisotropic = rng.normal(size=(120, d)) * np.array([5.0, 0.02, 0.02])
        embeddings, class_labels, index2label = _two_class_embeddings(rng, isotropic, anisotropic)

        cov = Coverage(min_class_samples=10)
        rows = _rows_by_class(cov, embeddings, class_labels, index2label)

        assert rows["a"]["isotropy"] is not None
        assert rows["b"]["isotropy"] is not None
        assert rows["a"]["isotropy"] > rows["b"]["isotropy"]

    def test_isotropy_null_below_floor(self):
        """A class with fewer samples than the isotropy floor reports null isotropy."""
        rng = np.random.default_rng(1)
        d = 3
        embeddings, class_labels, index2label = _two_class_embeddings(
            rng, rng.normal(size=(30, d)), rng.normal(size=(30, d))
        )

        cov = Coverage(min_class_samples=10, isotropy_min_samples=1000)
        rows = _rows_by_class(cov, embeddings, class_labels, index2label)

        assert rows["a"]["isotropy"] is None
        assert rows["b"]["isotropy"] is None


@pytest.mark.required
class TestCoverageNearDuplicates:
    def test_duplicated_class_scores_higher_than_varied(self):
        """A class of twinned points is mostly near-duplicates; a varied class is not."""
        rng = np.random.default_rng(0)
        d = 4
        varied = rng.random((60, d))
        # Each point repeated once: every point's nearest neighbor is its exact twin.
        twins = np.repeat(rng.random((30, d)), 2, axis=0)
        embeddings, class_labels, index2label = _two_class_embeddings(rng, varied, twins)

        cov = Coverage(min_class_samples=10)
        rows = _rows_by_class(cov, embeddings, class_labels, index2label)

        assert rows["b"]["near_duplicate_fraction"] >= 0.9
        assert rows["a"]["near_duplicate_fraction"] < rows["b"]["near_duplicate_fraction"]

    def test_factor_widens_what_counts_as_duplicate(self):
        """A larger near_duplicate_factor flags more pairs as duplicates."""
        rng = np.random.default_rng(3)
        d = 4
        points = rng.random((80, d))
        embeddings, class_labels, index2label = _two_class_embeddings(rng, points, rng.random((40, d)))

        strict = Coverage(min_class_samples=10, near_duplicate_factor=0.25)
        loose = Coverage(min_class_samples=10, near_duplicate_factor=0.9)
        strict_rows = _rows_by_class(strict, embeddings, class_labels, index2label)
        loose_rows = _rows_by_class(loose, embeddings, class_labels, index2label)

        assert loose_rows["a"]["near_duplicate_fraction"] > strict_rows["a"]["near_duplicate_fraction"]

    def test_near_duplicate_fraction_null_below_min_class_samples(self):
        """A class too small to assess reports null near_duplicate_fraction."""
        rng = np.random.default_rng(4)
        d = 4
        embeddings, class_labels, index2label = _two_class_embeddings(rng, rng.random((40, d)), rng.random((5, d)))

        cov = Coverage(min_class_samples=10)
        rows = _rows_by_class(cov, embeddings, class_labels, index2label)

        assert rows["b"]["near_duplicate_fraction"] is None


@pytest.mark.required
class TestCoverageSchema:
    def test_rows_with_null_signals_build_dataframe(self):
        """The per-class schema (used by evaluate) accepts null isotropy / near_duplicate_fraction."""
        rng = np.random.default_rng(5)
        d = 4
        embeddings, class_labels, index2label = _two_class_embeddings(rng, rng.random((40, d)), rng.random((5, d)))
        cov = Coverage(min_class_samples=10)
        rows = cov._per_class(embeddings, class_labels, index2label, np.array([], dtype=np.intp))

        df = pl.DataFrame(rows, schema=_PER_CLASS_SCHEMA)
        assert df.schema["isotropy"] == pl.Float64
        assert df.schema["near_duplicate_fraction"] == pl.Float64
        # The 5-sample class is below min_class_samples, so both signals are null for it.
        assert df["near_duplicate_fraction"].null_count() >= 1


@pytest.mark.required
class TestEvaluate:
    def test_evaluate_with_precomputed_embeddings(self):
        ds = ClassificationDataset(["a", "b"], per_class=25)
        emb = np.random.default_rng(1).random((len(ds), 4))
        res = Coverage(num_observations=5, min_class_samples=10).evaluate(ds, embeddings=emb)

        assert isinstance(res, CoverageOutput)
        assert set(res.data()["class"].to_list()) == {"a", "b"}
        assert res.coverage_radius > 0
        assert len(res.critical_value_radii) == len(ds)
        # uncovered_indices index into the sample set
        assert all(0 <= i < len(ds) for i in res.uncovered_indices)

    def test_evaluate_naive_method(self):
        ds = ClassificationDataset(["a", "b"], per_class=25)
        emb = np.random.default_rng(2).random((len(ds), 4))
        res = Coverage(num_observations=5, min_class_samples=10, method="naive").evaluate(ds, embeddings=emb)
        assert res.coverage_radius > 0

    def test_evaluate_computes_embeddings_from_extractor(self):
        ds = ClassificationDataset(["a", "b"], per_class=25)
        extractor = TorchExtractor(torch.nn.Flatten(), device="cpu")
        res = Coverage(extractor=extractor, batch_size=16, num_observations=5, min_class_samples=10).evaluate(ds)
        assert set(res.data()["class"].to_list()) == {"a", "b"}

    def test_evaluate_shape_mismatch_raises(self):
        ds = ClassificationDataset(["a", "b"], per_class=25)
        emb = np.random.default_rng(3).random((10, 4))  # far fewer embeddings than labels
        with pytest.raises(ShapeMismatchError, match="one embedding per"):
            Coverage(num_observations=5, min_class_samples=10).evaluate(ds, embeddings=emb)


@pytest.mark.required
class TestEmbeddingsResolution:
    def test_precomputed_embeddings_returned_as_float64(self):
        cov = Coverage()
        emb = cov._embeddings(dataset=None, embeddings=np.ones((4, 3), dtype=np.float32))  # type: ignore
        assert emb.dtype == np.float64
        assert emb.shape == (4, 3)

    def test_missing_extractor_and_embeddings_raises(self):
        with pytest.raises(ValueError, match="Provide pre-computed embeddings"):
            Coverage()._embeddings(dataset=None, embeddings=None)  # type: ignore


@pytest.mark.required
class TestNearDuplicateFraction:
    def test_empty_distances_return_none(self):
        assert Coverage()._near_duplicate_fraction([]) is None

import numpy as np
import polars as pl
import pytest

from dataeval.shift._drift._base import DriftOutput
from dataeval.shift._drift._kneighbors import DriftKNeighbors


@pytest.fixture
def ref_data():
    """Reference data: standard normal."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 8)).astype(np.float32)


@pytest.fixture
def shifted_data():
    """Clearly shifted data: offset by 10."""
    rng = np.random.default_rng(99)
    return (rng.standard_normal((100, 8)) + 10).astype(np.float32)


@pytest.fixture
def similar_data():
    """Data drawn from the same distribution as ref."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((100, 8)).astype(np.float32)


@pytest.mark.required
class TestDriftKNeighborsInit:
    def test_default_init(self):
        det = DriftKNeighbors()
        assert det._scorer.k == 10
        assert det._scorer.distance_metric == "euclidean"
        assert det._p_val == 0.05

    def test_custom_init(self):
        det = DriftKNeighbors(k=5, distance_metric="cosine", p_val=0.01)
        assert det._scorer.k == 5
        assert det._scorer.distance_metric == "cosine"
        assert det._p_val == 0.01

    def test_config_init(self):
        cfg = DriftKNeighbors.Config(k=3, distance_metric="cosine", p_val=0.1)
        det = DriftKNeighbors(config=cfg)
        assert det._scorer.k == 3
        assert det._scorer.distance_metric == "cosine"
        assert det._p_val == 0.1

    def test_param_overrides_config(self):
        cfg = DriftKNeighbors.Config(k=3, p_val=0.1)
        det = DriftKNeighbors(k=7, config=cfg)
        assert det._scorer.k == 7
        assert det._p_val == 0.1


@pytest.mark.required
class TestDriftKNeighborsFit:
    def test_fit_stores_ref(self, ref_data):
        det = DriftKNeighbors(k=5).fit(ref_data)
        assert det.x_ref.shape == (200, 8)
        assert det._fitted

    def test_predict_before_fit_raises(self):
        det = DriftKNeighbors()
        with pytest.raises(RuntimeError, match="Must call fit"):
            det.predict(np.zeros((10, 8)))

    def test_x_ref_before_fit_raises(self):
        det = DriftKNeighbors()
        with pytest.raises(RuntimeError, match="Must call fit"):
            _ = det.x_ref


@pytest.mark.optional
class TestDriftKNeighborsNonChunked:
    def test_detects_drift(self, ref_data, shifted_data):
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref_data)
        result = det.predict(shifted_data)
        assert isinstance(result, DriftOutput)
        assert result.drifted is True
        assert result.metric_name == "knn_distance"
        assert result.details["p_val"] < 0.05

    def test_no_drift_similar(self, ref_data, similar_data):
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref_data)
        result = det.predict(similar_data)
        assert isinstance(result, DriftOutput)
        assert result.drifted is False
        assert result.details["p_val"] > 0.05

    def test_details_populated(self, ref_data, shifted_data):
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref_data)
        result = det.predict(shifted_data)
        assert isinstance(result, DriftOutput)
        assert "mean_ref_distance" in result.details
        assert "mean_test_distance" in result.details
        assert result.details["mean_test_distance"] > result.details["mean_ref_distance"]

    def test_feature_mismatch_raises(self, ref_data):
        det = DriftKNeighbors(k=5).fit(ref_data)
        with pytest.raises(ValueError, match="different number of features"):
            det.predict(np.zeros((10, 4)))

    def test_detects_subtle_drift(self):
        """Subtle additive noise (small effect size) should still be detected."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((500, 32)).astype(np.float32)
        test = rng.standard_normal((500, 32)).astype(np.float32) + rng.normal(0, 0.2, (500, 32)).astype(np.float32)
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref)
        result = det.predict(test)
        assert result.drifted is True
        assert result.details["p_val"] < 0.05

    def test_x_required(self, ref_data):
        det = DriftKNeighbors(k=5).fit(ref_data)
        with pytest.raises(ValueError, match="x is required"):
            det.predict(None)


@pytest.mark.optional
class TestDriftKNeighborsChunked:
    def test_chunked_fit_predict(self, ref_data, shifted_data):
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref_data, chunk_size=50)
        assert det._chunker is not None
        result = det.predict(shifted_data)
        assert isinstance(result.details, pl.DataFrame)
        assert result.metric_name == "knn_distance"
        assert result.drifted

    def test_chunked_count(self, ref_data, shifted_data):
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref_data, chunk_count=4)
        result = det.predict(shifted_data)
        assert isinstance(result.details, pl.DataFrame)

    def test_prebuilt_chunks_fit(self, ref_data):
        chunks = [ref_data[:50], ref_data[50:100], ref_data[100:150], ref_data[150:]]
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref_data, chunks=chunks)
        assert det._baseline_values is not None
        assert len(det._baseline_values) == 4

    def test_prebuilt_chunks_predict(self, ref_data, shifted_data):
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref_data, chunk_count=4)
        test_chunks = [shifted_data[:50], shifted_data[50:]]
        result = det.predict(chunks=test_chunks)
        assert isinstance(result.details, pl.DataFrame)
        assert len(result.details) == 2

    def test_chunk_indices_predict(self, ref_data, shifted_data):
        det = DriftKNeighbors(k=5, distance_metric="euclidean").fit(ref_data, chunk_count=4)
        indices = [list(range(0, 50)), list(range(50, 100))]
        result = det.predict(shifted_data, chunk_indices=indices)
        assert isinstance(result.details, pl.DataFrame)
        assert len(result.details) == 2

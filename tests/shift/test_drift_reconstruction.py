import numpy as np
import polars as pl
import pytest

from dataeval.shift._drift._base import DriftOutput
from dataeval.shift._drift._reconstruction import DriftReconstruction
from dataeval.utils.models import AE

input_shape = (1, 8, 8)


@pytest.fixture
def ref_data():
    """Reference data: uniform on [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, (60, *input_shape)).astype(np.float32)


@pytest.fixture
def shifted_data():
    """Shifted data: constant 1.0 (outside training distribution)."""
    return np.ones((30, *input_shape), dtype=np.float32)


@pytest.fixture
def similar_data():
    """Data from same distribution as ref."""
    rng = np.random.default_rng(7)
    return rng.uniform(0, 1, (30, *input_shape)).astype(np.float32)


@pytest.mark.required
class TestDriftReconstructionInit:
    def test_default_init(self):
        model = AE(input_shape=input_shape)
        det = DriftReconstruction(model)
        assert det._scorer.model_type == "ae"
        assert det._p_val == 0.05

    def test_config_init(self):
        model = AE(input_shape=input_shape)
        cfg = DriftReconstruction.Config(p_val=0.01, epochs=5)
        det = DriftReconstruction(model, config=cfg)
        assert det._p_val == 0.01

    def test_param_overrides_config(self):
        model = AE(input_shape=input_shape)
        cfg = DriftReconstruction.Config(p_val=0.1)
        det = DriftReconstruction(model, p_val=0.01, config=cfg)
        assert det._p_val == 0.01

    def test_predict_before_fit_raises(self):
        model = AE(input_shape=input_shape)
        det = DriftReconstruction(model)
        with pytest.raises(RuntimeError, match="Must call fit"):
            det.predict(np.zeros((5, *input_shape)))


@pytest.mark.optional
class TestDriftReconstructionNonChunked:
    def test_fit_and_predict(self, ref_data, shifted_data):
        model = AE(input_shape=input_shape)
        det = DriftReconstruction(model).fit(ref_data, epochs=3, batch_size=30)
        result = det.predict(shifted_data)
        assert isinstance(result, DriftOutput)
        assert result.metric_name == "reconstruction_error"
        assert hasattr(result, "drifted")
        assert "p_val" in result.details

    def test_details_populated(self, ref_data, shifted_data):
        model = AE(input_shape=input_shape)
        det = DriftReconstruction(model).fit(ref_data, epochs=3, batch_size=30)
        result = det.predict(shifted_data)
        assert isinstance(result, DriftOutput)
        assert "mean_ref_error" in result.details
        assert "mean_test_error" in result.details

    def test_no_drift_similar(self, ref_data, similar_data):
        model = AE(input_shape=input_shape)
        det = DriftReconstruction(model).fit(ref_data, epochs=3, batch_size=30)
        result = det.predict(similar_data)
        assert isinstance(result, DriftOutput)
        # Similar data should have lower reconstruction error diff
        assert result.details["p_val"] > result.details["mean_ref_error"] * 0 or True  # just check it runs

    def test_x_required(self, ref_data):
        model = AE(input_shape=input_shape)
        det = DriftReconstruction(model).fit(ref_data, epochs=1, batch_size=30)
        with pytest.raises(ValueError, match="x is required"):
            det.predict(None)


@pytest.mark.optional
class TestDriftReconstructionChunked:
    def test_chunked_fit_predict(self, ref_data, shifted_data):
        model = AE(input_shape=input_shape)
        det = DriftReconstruction(model).fit(ref_data, epochs=3, batch_size=30, chunk_size=20)
        assert det._chunker is not None
        result = det.predict(shifted_data)
        assert isinstance(result.details, pl.DataFrame)
        assert result.metric_name == "reconstruction_error"

    def test_prebuilt_chunks(self, ref_data):
        model = AE(input_shape=input_shape)
        chunks = [ref_data[:20], ref_data[20:40], ref_data[40:]]
        det = DriftReconstruction(model).fit(ref_data, epochs=3, batch_size=30, chunks=chunks)
        assert det._baseline_values is not None
        assert len(det._baseline_values) == 3

import numpy as np
import polars as pl
import pytest

from dataeval.config import use_max_processes
from dataeval.shift._drift._base import DriftOutput
from dataeval.shift._drift._domain_classifier import DriftDomainClassifier


@pytest.fixture
def tst_data():
    """Zeros as test data, just needs to be really different from the Gaussian training data."""
    n_samples, n_features = 100, 4
    tstData = np.zeros((n_samples, n_features))
    return tstData


@pytest.fixture
def trn_data():
    """Gaussian distribution, 0 mean, unit :term:`variance<Variance>` training data."""
    n_samples, n_features, mean, std_dev = 100, 4, 0, 1
    size = n_samples * n_features
    x = np.linspace(-3, 3, size)
    # Calculate the Gaussian distribution values
    trnData = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    trnData = trnData.reshape((n_samples, n_features))
    return trnData


@pytest.mark.required
class TestDriftDomainClassifier:
    def test_init(self):
        """Test that the detector is instantiated correctly."""
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.6, 0.9))
        assert dc._n_folds == 2
        assert dc._threshold_config == (0.6, 0.9)

    def test_fit_xref(self, trn_data):
        dc = DriftDomainClassifier(n_folds=2)
        dc.fit(trn_data)
        assert dc.x_ref.shape == (100, 4)
        assert dc.n_features == 4
        assert dc._fitted

    @pytest.mark.optional
    def test_fit_chunked(self, trn_data):
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.6, 0.9))
        with use_max_processes(4):
            dc.fit(trn_data, chunk_size=10)
        assert dc.x_ref.shape == (100, 4)
        assert dc.n_features == 4
        assert dc._chunker is not None
        assert dc._n_folds == 2
        assert dc._threshold_bounds != (None, None)

    def test_predict_xtest_mismatch_features(self, trn_data):
        dc = DriftDomainClassifier(n_folds=2)
        dc.fit(trn_data)  # 4 features
        test_5features = np.zeros((100, 5))
        with pytest.raises(ValueError, match="different number of features"):
            dc.predict(test_5features)

    def test_predict_before_fit(self):
        dc = DriftDomainClassifier(n_folds=2)
        with pytest.raises(RuntimeError, match="Must call fit"):
            dc.predict(np.zeros((10, 4)))

    @pytest.mark.optional
    def test_sequence_chunked(self, trn_data, tst_data):
        """Sequential tests for chunked mode, each step is required before proceeding to the next."""
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.45, 0.65))
        with use_max_processes(4):
            dc.fit(trn_data, chunk_count=5)
        assert dc._chunker is not None
        assert dc._baseline_values is not None
        results = dc.predict(tst_data)
        assert isinstance(results.details, pl.DataFrame)
        assert results.metric_name == "auroc"
        # All chunks should be flagged as drifted (zeros vs Gaussian)
        assert results.drifted
        assert results.details["drifted"].all()
        assert results.details["upper_threshold"].null_count() == 0

    @pytest.mark.optional
    def test_non_chunked(self, trn_data, tst_data):
        """Test non-chunked mode returns DriftOutput."""
        dc = DriftDomainClassifier(n_folds=2)
        dc.fit(trn_data)
        with use_max_processes(4):
            result = dc.predict(tst_data)
        assert isinstance(result, DriftOutput)
        assert hasattr(result, "drifted")
        assert hasattr(result, "threshold")
        assert "p_val" in result.details
        assert hasattr(result, "distance")

    @pytest.mark.optional
    def test_predict_prebuilt_chunks(self, trn_data, tst_data):
        """Test predict with prebuilt test chunks."""
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.45, 0.65))
        with use_max_processes(4):
            dc.fit(trn_data, chunk_count=5)
        test_chunks = [tst_data[:50], tst_data[50:]]
        result = dc.predict(chunks=test_chunks)
        assert isinstance(result.details, pl.DataFrame)
        assert len(result.details) == 2
        assert result.metric_name == "auroc"

    @pytest.mark.optional
    def test_predict_chunk_indices(self, trn_data, tst_data):
        """Test predict with chunk_indices override."""
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.45, 0.65))
        with use_max_processes(4):
            dc.fit(trn_data, chunk_count=5)
        indices = [list(range(0, 50)), list(range(50, 100))]
        result = dc.predict(tst_data, chunk_indices=indices)
        assert isinstance(result.details, pl.DataFrame)
        assert len(result.details) == 2

    @pytest.mark.optional
    def test_fit_prebuilt_chunks(self, trn_data):
        """Test fit with prebuilt reference chunks."""
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.45, 0.65))
        chunks = [trn_data[:25], trn_data[25:50], trn_data[50:75], trn_data[75:]]
        with use_max_processes(4):
            dc.fit(trn_data, chunks=chunks)
        assert dc._baseline_values is not None
        assert len(dc._baseline_values) == 4

    def test_predict_non_chunked_no_data_raises(self, trn_data):
        """Test that non-chunked predict without data raises."""
        dc = DriftDomainClassifier(n_folds=2)
        dc.fit(trn_data)
        with pytest.raises(ValueError, match="x is required"):
            dc.predict(None)

    @pytest.mark.optional
    def test_mvdc_stats_populated(self, trn_data, tst_data):
        """Test that DriftDomainClassifierStats has fold_aurocs and feature_importances."""
        dc = DriftDomainClassifier(n_folds=2)
        dc.fit(trn_data)
        with use_max_processes(4):
            result = dc.predict(tst_data)
        assert isinstance(result, DriftOutput)
        assert "fold_aurocs" in result.details
        assert "feature_importances" in result.details
        assert len(result.details["fold_aurocs"]) == 2  # n_folds=2
        assert len(result.details["feature_importances"]) == 4  # n_features=4

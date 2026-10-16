import numpy as np
import polars as pl
import pytest

from dataeval.config import use_max_processes
from dataeval.exceptions import NotFittedError, ShapeMismatchError
from dataeval.shift._drift._base import DriftOutput
from dataeval.shift._drift._domain_classifier import DriftDomainClassifier
from dataeval.shift._shared._domain_classifier import compute_auroc


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
        assert dc.reference_data.shape == (100, 4)
        assert dc.n_features == 4
        assert dc._fitted

    @pytest.mark.optional
    def test_fit_chunked(self, trn_data):
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.6, 0.9))
        with use_max_processes(4):
            chunked = dc.chunked(chunk_size=10).fit(trn_data)
        assert dc.reference_data.shape == (100, 4)
        assert dc.n_features == 4
        assert chunked._chunker is not None
        assert dc._n_folds == 2
        assert chunked._threshold_bounds != (None, None)

    def test_predict_xtest_mismatch_features(self, trn_data):
        dc = DriftDomainClassifier(n_folds=2)
        dc.fit(trn_data)  # 4 features
        test_5features = np.zeros((100, 5))
        with pytest.raises(ShapeMismatchError, match="different number of features"):
            dc.predict(test_5features)

    def test_predict_before_fit(self):
        dc = DriftDomainClassifier(n_folds=2)
        with pytest.raises(NotFittedError, match="Must call fit"):
            dc.predict(np.zeros((10, 4)))

    @pytest.mark.optional
    def test_sequence_chunked(self, trn_data, tst_data):
        """Sequential tests for chunked mode, each step is required before proceeding to the next."""
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.45, 0.65))
        with use_max_processes(4):
            chunked = dc.chunked(chunk_count=5).fit(trn_data)
        assert chunked._chunker is not None
        assert chunked._baseline_values is not None
        results = chunked.predict(tst_data)
        assert isinstance(results, DriftOutput)
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
            chunked = dc.chunked(chunk_count=5).fit(trn_data)
        test_chunks = [tst_data[:50], tst_data[50:]]
        result = chunked.predict(chunks=test_chunks)
        assert isinstance(result.details, pl.DataFrame)
        assert len(result.details) == 2
        assert result.metric_name == "auroc"

    @pytest.mark.optional
    def test_predict_chunk_indices(self, trn_data, tst_data):
        """Test predict with chunk_indices override."""
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.45, 0.65))
        with use_max_processes(4):
            chunked = dc.chunked(chunk_count=5).fit(trn_data)
        indices = [list(range(0, 50)), list(range(50, 100))]
        result = chunked.predict(tst_data, chunk_indices=indices)
        assert isinstance(result.details, pl.DataFrame)
        assert len(result.details) == 2

    @pytest.mark.optional
    def test_mvdc_stats_populated(self, trn_data, tst_data):
        """Test that DriftDomainClassifier.Stats has fold_aurocs and feature_importances."""
        dc = DriftDomainClassifier(n_folds=2)
        dc.fit(trn_data)
        with use_max_processes(4):
            result = dc.predict(tst_data)
        assert isinstance(result, DriftOutput)
        assert "fold_aurocs" in result.details
        assert "feature_importances" in result.details
        assert len(result.details["fold_aurocs"]) == 2  # n_folds=2
        assert len(result.details["feature_importances"]) == 4  # n_features=4


@pytest.mark.required
class TestDriftDomainClassifierBugs:
    def test_int_threshold_is_honored(self, trn_data, tst_data):
        """An integer threshold is used as given instead of silently falling back to 0.55."""
        dc = DriftDomainClassifier(n_folds=2, threshold=1).fit(trn_data)
        result = dc.predict(tst_data)
        assert result.threshold == 1.0
        assert not result.drifted  # AUROC cannot exceed 1.0

    def test_tuple_threshold_uses_upper_bound_when_not_chunked(self, trn_data, tst_data):
        """A (lower, upper) threshold is not silently replaced by 0.55 in non-chunked mode."""
        dc = DriftDomainClassifier(n_folds=2, threshold=(0.3, 1.0)).fit(trn_data)
        result = dc.predict(tst_data)
        assert result.threshold == 1.0
        assert not result.drifted

    @pytest.mark.filterwarnings("ignore:The least populated class")
    def test_fold_without_both_classes_scores_neutral(self):
        """A CV fold missing a class has an undefined AUROC and is reported as 0.5, not NaN."""
        x = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
        y = np.zeros(20, dtype=np.intp)
        y[0] = 1  # single positive: one of the two test folds has no positive
        _, fold_aurocs, _ = compute_auroc(x, y, n_folds=2)
        assert not np.isnan(fold_aurocs).any()
        assert 0.5 in fold_aurocs


@pytest.mark.required
class TestDriftDomainClassifierHyperparameters:
    STUMP = {"n_estimators": 1, "max_depth": 1, "verbosity": -1}

    def test_hyperparameters_reach_the_classifier(self, trn_data, tst_data):
        """A single stump makes one split per fold, so split importances sum to 1."""
        dc = DriftDomainClassifier(n_folds=2, hyperparameters=self.STUMP).fit(trn_data)
        result = dc.predict(tst_data)
        assert result.details["feature_importances"].sum() == 1

    def test_hyperparameters_default_is_unchanged(self, trn_data, tst_data):
        result = DriftDomainClassifier(n_folds=2).fit(trn_data).predict(tst_data)
        assert result.details["feature_importances"].sum() > 1

    def test_hyperparameters_from_config_and_override(self, trn_data, tst_data):
        config = DriftDomainClassifier.Config(n_folds=2, hyperparameters=self.STUMP)
        assert DriftDomainClassifier(config=config).config.hyperparameters == self.STUMP
        assert DriftDomainClassifier(config=config, hyperparameters={"n_estimators": 2}).config.hyperparameters == {
            "n_estimators": 2
        }

    def test_hyperparameters_reach_chunked_baselines(self, trn_data, tst_data):
        chunked = DriftDomainClassifier(n_folds=2, threshold=(0.45, 0.65), hyperparameters=self.STUMP).chunked(
            chunk_size=50
        )
        result = chunked.fit(trn_data).predict(tst_data)
        assert len(result.details) == 2

from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest

from dataeval.detectors.drift._mvdc import DriftMVDC
from dataeval.detectors.drift._nml._thresholds import ConstantThreshold


@pytest.fixture
def tst_data():
    """Zeros as test data, just needs to be really different from the Gaussian training data"""

    n_samples, n_features = 100, 4
    tstData = np.zeros((n_samples, n_features))
    return tstData


@pytest.fixture
def trn_data():
    """Gaussian distribution, 0 mean, unit :term:`variance<Variance>` training data"""

    n_samples, n_features, mean, std_dev = 100, 4, 0, 1
    size = n_samples * n_features
    x = np.linspace(-3, 3, size)
    # Calculate the Gaussian distribution values
    trnData = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    trnData = trnData.reshape((n_samples, n_features))
    return trnData


@pytest.mark.requires_all
class TestMVDC:
    def test_init(self):
        """Test that the detector is instantiated correctly"""

        dc = DriftMVDC(n_folds=5, chunk_size=10, threshold=(0.6, 0.9))
        assert dc._calc.cv_folds_num == 5
        threshold = dc._calc.threshold
        assert isinstance(threshold, ConstantThreshold)
        assert (threshold.lower, threshold.upper) == (0.6, 0.9)  # threshold specific to this example data

    def test_fit_xref(self, trn_data):
        dc = DriftMVDC(n_folds=5, chunk_size=10, threshold=(0.6, 0.9))
        dc._calc = MagicMock()
        dc.fit(trn_data)
        assert dc.x_ref.shape == (100, 4)
        assert dc.n_features == 4
        dc._calc.fit.assert_called()

    def test_predict_xtest(self, tst_data):
        dc = DriftMVDC(n_folds=5, chunk_size=10, threshold=(0.6, 0.9))
        dc._calc = MagicMock()
        dc.n_features = 4
        dc.predict(tst_data)
        assert dc.x_test.shape == (100, 4)
        assert dc.n_features == 4
        dc._calc.calculate.assert_called()

    def test_predict_xtest_mismatch_features(self, tst_data):
        dc = DriftMVDC(n_folds=5, chunk_size=10, threshold=(0.6, 0.9))
        dc.n_features = 5
        with pytest.raises(ValueError):
            dc.predict(tst_data)

    @pytest.mark.optional
    @pytest.mark.requires_all
    def test_sequence(self, trn_data, tst_data):
        """Sequential tests, each step is required before proceeding to the next"""

        dc = DriftMVDC(n_folds=5, chunk_size=10)
        dc.fit(trn_data)
        assert dc._calc.result is not None

        results = dc.predict(tst_data)
        resdf = results.to_df()
        tstdf = resdf[resdf["chunk"]["period"] == "analysis"]
        tst_auc_vals = tstdf["domain_classifier_auroc"]["value"].values  # type: ignore
        assert np.all(tst_auc_vals > dc.threshold[0])  # type: ignore
        isdrift = tstdf["domain_classifier_auroc"]["alert"].values  # type: ignore
        assert np.all(isdrift)  # type: ignore

        # Verify plot generates the figure and it saves correctly, then remove it
        fig = results.plot(showme=False)
        x_data = fig.axes[0].lines[0].get_xdata()
        x_values = np.arange(0, 20, dtype=int)
        npt.assert_array_equal(x_data, x_values)
        assert fig._dpi == 300  # type: ignore


if __name__ == "__main__":
    # Demo code (uses more features than the pytest, but has the same result)

    # Data defined params
    n_samples, n_features = 100, 1024
    mean, std_dev = 0.0, 1.0

    # User defined params
    chunksz = 10
    bounds = (0.6, 0.9)
    cvfold = 5

    # Create train/test sample data just for the demo
    size = n_samples * n_features
    x = np.linspace(-3, 3, size)
    trnData = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    trnData = trnData.reshape((n_samples, n_features))
    tstData = np.zeros((n_samples, n_features))

    # Domain classifier training (fit) and inference (predict)
    dc = DriftMVDC(n_folds=cvfold, chunk_size=chunksz, threshold=bounds)
    dc.fit(trnData)
    results = dc.predict(tstData)
    results.plot(showme=True)  # fig: DomainClassification.png will be to cwd

    # Test domain data frame and classification
    resdf = results.to_df()
    tstdf = resdf[resdf["chunk"]["period"] == "analysis"]
    isdrift = tstdf["domain_classifier_auroc"]["alert"].values  # type: ignore

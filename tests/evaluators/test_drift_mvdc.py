from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pandas as pd
import polars as pl
import pytest

from dataeval.config import use_max_processes
from dataeval.evaluators.drift._mvdc import DriftMVDC, DriftMVDCOutput
from dataeval.evaluators.drift._nml._base import _validate
from dataeval.evaluators.drift._nml._thresholds import ConstantThreshold


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


@pytest.fixture
def result_df():
    # Convert from MultiIndex to flat column names with polars DataFrame
    return pl.DataFrame(
        {
            "chunk_key": [f"[{i % 5 * 20}:{(i % 5 + 1) * 20 - 1}]" for i in range(10)],
            "chunk_chunk_index": [i % 5 for i in range(10)],
            "chunk_start_index": [i % 5 * 20 for i in range(10)],
            "chunk_end_index": [(i % 5 + 1) * 20 - 1 for i in range(10)],
            "chunk_period": ["reference" if i < 5 else "analysis" for i in range(10)],
            "domain_classifier_auroc_value": [0.5 if i < 5 else 1.0 for i in range(10)],
            "domain_classifier_auroc_upper_threshold": [0.65 for _ in range(10)],
            "domain_classifier_auroc_lower_threshold": [0.45 for _ in range(10)],
            "domain_classifier_auroc_alert": [i >= 5 for i in range(10)],
        }
    )


@pytest.mark.required
class TestMVDC:
    def test_init(self):
        """Test that the detector is instantiated correctly"""

        dc = DriftMVDC(n_folds=2, chunk_size=10, threshold=(0.6, 0.9))
        assert dc._calc.cv_folds_num == 2
        threshold = dc._calc.threshold
        assert isinstance(threshold, ConstantThreshold)
        assert (threshold.lower, threshold.upper) == (0.6, 0.9)  # threshold specific to this example data

    def test_fit_xref(self, trn_data):
        dc = DriftMVDC(n_folds=2, chunk_size=10, threshold=(0.6, 0.9))
        dc._calc = MagicMock()
        dc.fit(trn_data)
        assert dc.x_ref.shape == (100, 4)
        assert dc.n_features == 4
        dc._calc.fit.assert_called()

    def test_predict_xtest(self, tst_data):
        dc = DriftMVDC(n_folds=2, chunk_size=10, threshold=(0.6, 0.9))
        dc._calc = MagicMock()
        dc.n_features = 4
        dc.predict(tst_data)
        assert dc.x_test.shape == (100, 4)
        assert dc.n_features == 4
        dc._calc.calculate.assert_called()

    def test_predict_xtest_mismatch_features(self, tst_data):
        dc = DriftMVDC(n_folds=2, chunk_size=10, threshold=(0.6, 0.9))
        dc.n_features = 5
        with pytest.raises(ValueError):
            dc.predict(tst_data)

    def test_validate_empty(self):
        df = pd.DataFrame([])
        with pytest.raises(ValueError):
            _validate(df)

    def test_validate_feature_mismatch(self):
        df = pd.DataFrame([[1], [2]])
        with pytest.raises(ValueError):
            _validate(df, expected_features=2)

    def test_calculate_before_fit(self):
        dc = DriftMVDC(n_folds=2, chunk_size=10, threshold=(0.6, 0.9))
        with pytest.raises(RuntimeError):
            dc._calc.calculate(None)  # type: ignore

    @pytest.mark.optional
    def test_sequence(self, trn_data, tst_data):
        """Sequential tests, each step is required before proceeding to the next"""

        dc = DriftMVDC(n_folds=2, chunk_count=5)
        with use_max_processes(4):
            dc.fit(trn_data)
        assert dc._calc.result is not None
        results = dc.predict(tst_data)
        resdf = results.data()
        tstdf = resdf.filter(pl.col("chunk_period") == "analysis")
        tst_auc_vals = tstdf["domain_classifier_auroc_value"].to_numpy()
        assert np.all(tst_auc_vals > dc.threshold[0])
        isdrift = tstdf["domain_classifier_auroc_alert"].to_numpy()
        assert np.all(isdrift)


@pytest.mark.required
class TestDriftMVDCOutput:
    def test_output_data(self, result_df):
        output = DriftMVDCOutput(result_df)
        df = output.data()
        assert not output.empty
        assert len(df) == len(output.data())
        assert len(output) == len(df)
        np.testing.assert_equal(df.to_numpy(), output.data().to_numpy())

    def test_output_empty(self):
        output = DriftMVDCOutput(pl.DataFrame([]))
        df = output.data()
        assert output.empty
        assert len(df) == len(output.data())
        assert len(output) == len(df)

    def test_output_filter(self, result_df):
        output = DriftMVDCOutput(result_df)
        o_all = output.filter("all")
        o_ref = output.filter("reference")
        o_anl = output.filter("analysis")
        assert len(o_all) == len(o_ref) + len(o_anl)
        assert len(o_all.data().columns) == len(o_ref.data().columns) == len(o_anl.data().columns)

    def test_output_filter_invalid_metric_raises(self, result_df):
        output = DriftMVDCOutput(result_df)
        with pytest.raises(ValueError):
            output.filter(metrics=1)  # type: ignore

    def test_output_filter_no_metric(self, result_df):
        output = DriftMVDCOutput(result_df)
        with pytest.raises(KeyError):
            output.filter(metrics="foo")

    @pytest.mark.requires_all
    def test_plot(self, result_df):
        output = DriftMVDCOutput(result_df)

        fig = output.plot()
        x_data = fig.axes[0].lines[0].get_xdata()
        x_values = np.arange(0, 10, dtype=int)
        npt.assert_array_equal(x_data, x_values)
        assert fig._dpi == 300  # type: ignore

    @pytest.mark.requires_all
    def test_plot_driftx_not_gt2(self, result_df):
        modified = result_df.head(2)
        output = DriftMVDCOutput(modified)
        fig = output.plot()
        assert fig


@pytest.mark.skip
def driftmvdc_demo():
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
    results.plot().show()  # fig: DomainClassification.png will be to cwd

    # Test domain data using Polars backend
    resdf = results.data()  # Returns pl.DataFrame directly
    tstdf = resdf.filter(pl.col("chunk_period") == "analysis")
    isdrift = tstdf["domain_classifier_auroc_alert"].to_numpy()

    # Display the Polars DataFrame
    print("Results DataFrame (Polars):")
    print(resdf)
    print("\nAnalysis period data:")
    print(tstdf)
    print(f"\nDrift detected: {isdrift}")

import numpy as np
import pytest

from dataeval._internal.metrics.metadata_ks_compare import meta_distribution_compare


@pytest.fixture(scope="class")
def mock_mdc():
    md0 = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, 211101]}
    return md0, md1


class TestMetadataCompare:
    # test conditions, as many as you can think of
    def test_type(self, mock_mdc):
        mdc = meta_distribution_compare(*mock_mdc)
        assert isinstance(mdc, dict)

    def test_shifts(self, mock_mdc):
        mdc = meta_distribution_compare(*mock_mdc)
        print(mdc)
        assert mdc["time"]["shift_magnitude"] == 2.7
        assert np.isclose(mdc["altitude"]["shift_magnitude"], 0.7492490855199898)

    def test_pvalue(self, mock_mdc):
        mdc = meta_distribution_compare(*mock_mdc)
        assert np.isclose(mdc["time"]["pvalue"], 0.0)
        assert np.isclose(mdc["altitude"]["pvalue"], 0.9444444444444444)

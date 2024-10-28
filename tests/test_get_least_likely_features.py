import numpy as np
import pytest

from dataeval._internal.metrics.metadata_least_likely import get_least_likely_features


@pytest.fixture(scope="class")
def mock_llf():
    md0 = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, 211101]}

    is_ood = np.array([False, False, True])

    return md0, md1, is_ood


class TestGetLeastLikelyFeatures:
    def test_llf(self, mock_llf):
        llf = get_least_likely_features(*mock_llf)
        assert llf[0] == "time"

    def test_llf_type(self, mock_llf):
        llf = get_least_likely_features(*mock_llf)
        assert isinstance(llf, np.ndarray)

    def test_llf_types(self, mock_llf):
        llf = get_least_likely_features(*mock_llf)
        assert isinstance(llf, np.ndarray)

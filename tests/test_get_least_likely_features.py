import numpy as np
import pytest

from dataeval._internal.metrics.metadata_least_likely import get_least_likely_features


@pytest.fixture
def mock_llf():
    md0 = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, 211101]}

    is_ood = np.array([False, False, True])

    llf = get_least_likely_features(md0, md1, is_ood)
    return llf


class TestGetLeastLikelyFeatures:
    def test_nothing(self, mock_llf):
        _ = mock_llf
        assert True

    def test_llf(self, mock_llf):
        llf = mock_llf
        assert llf[0] == "time"

    def test_llf_type(self, mock_llf):
        ml = mock_llf
        assert isinstance(ml, np.ndarray)

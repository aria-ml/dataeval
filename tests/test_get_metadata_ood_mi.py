import numpy as np

import pytest

from dataeval._internal.metrics.metadata_ood_mi import get_metadata_ood_mi

pytest.fixture
def mock_md_ood():
    rng = np.random.default_rng(20241022)
    nsamp, nfeatures = 100, 5
    x = rng.normal(size=(nsamp, nfeatures))
    # features 0, 1, and 2 should have some MI, but 3 and 4 will not
    is_ood = np.abs(x[:, 0]) > np.abs(x[:,1]) + np.abs(x[:,2])

    md = {}
    for i in range(nfeatures):
        md.update({'feature_'+str(i): x[:,i]})

    discrete_features = False

    mi_dict = get_metadata_ood_mi(md, is_ood, discrete_features=discrete_features)

    return mi_dict

class TestGetMetadataOodMi():
    def test_type(self, mock_md_ood):
        assert isinstance(mock_md_ood, dict)

    def test_mi_values(self, mock_md_ood):
        mi_dict = mock_md_ood
        assert np.allclose([v for v in mi_dict.values()], [0.17562533, 0.18743624, 0.15217528, 0.        , 0.        ])


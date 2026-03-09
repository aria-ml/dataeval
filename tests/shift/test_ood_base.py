import numpy as np
import pytest
import torch

from dataeval.shift._ood._base import OODOutput, OODScoreOutput
from dataeval.shift._ood._domain_classifier import OODDomainClassifier
from dataeval.shift._ood._kneighbors import OODKNeighbors
from dataeval.shift._ood._reconstruction import OODReconstruction


@pytest.mark.required
def test_ood_score_output():
    """Test OODScoreOutput dataclass."""
    instance_score = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    feature_score = np.array([0.2, 0.6, 1.0], dtype=np.float32)

    # Test with both scores
    output = OODScoreOutput(instance_score=instance_score, feature_score=feature_score)
    assert np.array_equal(output.get("instance"), instance_score)
    assert np.array_equal(output.get("feature"), feature_score)

    # Test with only instance score
    output = OODScoreOutput(instance_score=instance_score)
    assert np.array_equal(output.get("instance"), instance_score)
    assert np.array_equal(output.get("feature"), instance_score)  # Falls back to instance


@pytest.mark.required
def test_ood_output():
    """Test OODOutput dataclass."""
    is_ood = np.array([False, True, True], dtype=np.bool_)
    instance_score = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    feature_score = np.array([0.2, 0.6, 1.0], dtype=np.float32)

    output = OODOutput(is_ood=is_ood, instance_score=instance_score, feature_score=feature_score)
    assert np.array_equal(output.is_ood, is_ood)
    assert np.array_equal(output.instance_score, instance_score)
    assert output.feature_score is not None
    assert np.array_equal(output.feature_score, feature_score)

    # Test with None feature_score
    output = OODOutput(is_ood=is_ood, instance_score=instance_score, feature_score=None)
    assert output.feature_score is None


@pytest.mark.required
class TestOODConfigRepr:
    """Tests that constructor params override config defaults and are reflected in repr."""

    def test_kneighbors_params_override_config(self):
        det = OODKNeighbors(k=3, distance_metric="euclidean", threshold_perc=90.0)
        assert det.config.k == 3
        assert det.config.distance_metric == "euclidean"
        assert det.config.threshold_perc == 90.0
        assert "k=3" in repr(det)
        assert "distance_metric='euclidean'" in repr(det)
        assert "threshold_perc=90.0" in repr(det)

    def test_kneighbors_default_config(self):
        det = OODKNeighbors()
        assert det.config.k == 10
        assert det.config.distance_metric == "cosine"
        assert det.config.threshold_perc == 95.0

    def test_domain_classifier_params_override_config(self):
        det = OODDomainClassifier(n_folds=3, n_std=3.0, threshold_perc=99.0)
        assert det.config.n_folds == 3
        assert det.config.n_std == 3.0
        assert det.config.threshold_perc == 99.0
        assert "n_folds=3" in repr(det)
        assert "n_std=3.0" in repr(det)

    def test_reconstruction_param_override_config(self):
        det = OODReconstruction(model=torch.nn.Identity(), threshold_perc=80.0)
        assert det.config.threshold_perc == 80.0
        assert "threshold_perc=80.0" in repr(det)

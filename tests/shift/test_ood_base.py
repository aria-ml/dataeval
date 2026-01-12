import numpy as np
import pytest

from dataeval.shift._ood._base import OODOutput, OODScoreOutput


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

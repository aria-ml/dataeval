import numpy as np
import pytest

from dataeval.shift._ood._base import OODOutput, OODScoreOutput
from dataeval.shift._ood._domain_classifier import OODDomainClassifier


@pytest.fixture
def ref_data():
    """Reference (in-distribution) data."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 8)).astype(np.float32)


@pytest.fixture
def ood_data():
    """Clearly OOD data: large offset."""
    rng = np.random.default_rng(99)
    return (rng.standard_normal((50, 8)) + 10).astype(np.float32)


@pytest.fixture
def id_data():
    """In-distribution test data."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((50, 8)).astype(np.float32)


@pytest.mark.required
class TestOODDomainClassifierInit:
    def test_default_init(self):
        det = OODDomainClassifier()
        assert det._n_folds == 5
        assert det._n_repeats == 5
        assert det._n_std == 2.0

    def test_custom_init(self):
        det = OODDomainClassifier(n_folds=3, n_repeats=3, n_std=3.0)
        assert det._n_folds == 3
        assert det._n_repeats == 3
        assert det._n_std == 3.0

    def test_config_init(self):
        cfg = OODDomainClassifier.Config(n_folds=2, n_repeats=2)
        det = OODDomainClassifier(config=cfg)
        assert det._n_folds == 2
        assert det._n_repeats == 2

    def test_param_overrides_config(self):
        cfg = OODDomainClassifier.Config(n_folds=2)
        det = OODDomainClassifier(n_folds=4, config=cfg)
        assert det._n_folds == 4


@pytest.mark.required
class TestOODDomainClassifierValidation:
    def test_predict_before_fit_raises(self):
        det = OODDomainClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(np.zeros((10, 8)))

    def test_score_validates_shape(self, ref_data):
        det = OODDomainClassifier(n_folds=2, n_repeats=2)
        det.fit(ref_data)
        with pytest.raises(RuntimeError, match="shape"):
            det.score(np.zeros((10, 4), dtype=np.float32))


@pytest.mark.optional
class TestOODDomainClassifierFitPredict:
    def test_fit_sets_threshold(self, ref_data):
        det = OODDomainClassifier(n_folds=2, n_repeats=2)
        det.fit(ref_data)
        assert det._threshold > 0
        assert det._null_mean >= 0
        assert det._null_std >= 0

    def test_score_returns_output(self, ref_data, ood_data):
        det = OODDomainClassifier(n_folds=2, n_repeats=2)
        det.fit(ref_data)
        result = det.score(ood_data)
        assert isinstance(result, OODScoreOutput)
        assert len(result.instance_score) == len(ood_data)

    def test_predict_returns_output(self, ref_data, ood_data):
        det = OODDomainClassifier(n_folds=2, n_repeats=2)
        det.fit(ref_data)
        result = det.predict(ood_data)
        assert isinstance(result, OODOutput)
        assert len(result.is_ood) == len(ood_data)

    def test_ood_scores_higher_than_id(self, ref_data, ood_data, id_data):
        det = OODDomainClassifier(n_folds=2, n_repeats=2)
        det.fit(ref_data)
        ood_scores = det.score(ood_data)
        id_scores = det.score(id_data)
        # OOD data should have higher average class-1 rates
        assert np.mean(ood_scores.instance_score) > np.mean(id_scores.instance_score)

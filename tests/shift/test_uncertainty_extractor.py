"""Model-agnostic UncertaintyExtractor."""

import numpy as np
import pytest

from dataeval.extractors._uncertainty import ClasswiseUncertaintyExtractor, UncertaintyExtractor


class FakeScores:
    """A FeatureExtractor that returns fixed class scores (no model)."""

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=np.float32)

    def __call__(self, data):
        return self._scores


@pytest.mark.required
class TestUncertaintyExtractor:
    def test_agnostic_call_returns_n_by_1(self):
        scores = FakeScores([[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        ex = UncertaintyExtractor(scores, preds_type="logits")
        out = ex([0, 1])
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 1)
        # uniform row has higher entropy
        assert out[1, 0] > out[0, 0]

    def test_empty_returns_0_by_1(self):
        ex = UncertaintyExtractor(FakeScores(np.empty((0, 3))), preds_type="logits")
        assert ex([]).shape == (0, 1)

    def test_no_threshold_attribute(self):
        ex = UncertaintyExtractor(FakeScores([[1.0, 0.0]]))
        assert not hasattr(ex, "threshold")


@pytest.mark.required
class TestClasswiseUncertaintyExtractor:
    def test_call_returns_dict_grouped_by_class(self):
        # 4 detections confident in class 0, 4 in class 1 (3 classes total)
        scores = FakeScores([[10.0, -10.0, -10.0]] * 4 + [[-10.0, 10.0, -10.0]] * 4)
        ex = ClasswiseUncertaintyExtractor(scores, preds_type="logits")
        out = ex(list(range(8)))
        assert isinstance(out, dict)
        assert set(out) == {0, 1}
        assert all(isinstance(v, np.ndarray) for v in out.values())

    def test_empty_returns_empty_dict(self):
        ex = ClasswiseUncertaintyExtractor(FakeScores(np.empty((0, 3))))
        assert ex([]) == {}

    def test_has_threshold(self):
        ex = ClasswiseUncertaintyExtractor(FakeScores([[1.0, 0.0]]), threshold=0.5)
        assert ex.threshold == 0.5

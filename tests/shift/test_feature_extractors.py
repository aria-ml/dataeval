"""Tests for drift feature extractors."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataeval.extractors import ClasswiseUncertaintyExtractor, TorchExtractor, UncertaintyExtractor
from dataeval.extractors._uncertainty import _classwise_prediction_uncertainty, _prediction_uncertainty


class SimpleModel(nn.Module):
    def __init__(self, n_features, n_output):
        super().__init__()
        self.fc = nn.Linear(n_features, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.mark.required
class TestClassifierUncertainty:
    """Test _prediction_uncertainty function."""

    def test_probs_input(self):
        """Test uncertainty calculation with probability input."""
        # Create mock probabilities (3 samples, 4 classes)
        probs = np.array([[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])

        result = _prediction_uncertainty(probs, preds_type="probs")

        assert result.shape == (3, 1)
        assert isinstance(result, np.ndarray)
        # Uniform distribution has highest entropy
        assert result[1, 0] > result[0, 0]
        assert result[1, 0] > result[2, 0]

    def test_logits_input(self):
        """Test uncertainty calculation with logits input."""
        # Create mock logits
        logits = np.array([[2.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [10.0, 0.0, 0.0, 0.0]])

        result = _prediction_uncertainty(logits, preds_type="logits")

        assert result.shape == (3, 1)
        assert isinstance(result, np.ndarray)
        # All values should be non-negative
        assert (result >= 0).all()

    def test_invalid_probs_sum(self):
        """Test that invalid probabilities raise ValueError."""
        # Probabilities that don't sum to 1
        bad_probs = np.array([[0.5, 0.2, 0.1]])

        with pytest.raises(ValueError, match="Probabilities across labels should sum to 1"):
            _prediction_uncertainty(bad_probs, preds_type="probs")

    def test_invalid_preds_type(self):
        """Test that invalid preds_type raises NotImplementedError."""
        probs = np.array([[0.5, 0.5]])

        with pytest.raises(NotImplementedError, match="Only prediction types 'probs' and 'logits' supported"):
            _prediction_uncertainty(probs, preds_type="invalid")  # type: ignore


@pytest.mark.required
class TestUncertaintyExtractorComposition:
    """Per-instance uncertainty over a TorchExtractor."""

    def test_call_returns_n_by_1(self):
        model = SimpleModel(10, 4)
        scores = TorchExtractor(model, device="cpu", batch_size=8)
        data = np.random.randn(15, 10).astype(np.float32)
        out = UncertaintyExtractor(scores, preds_type="logits")(data)
        assert isinstance(out, np.ndarray)
        assert out.shape == (15, 1)

    def test_empty_returns_0_by_1(self):
        scores = TorchExtractor(SimpleModel(10, 4), device="cpu", batch_size=8)
        assert UncertaintyExtractor(scores)([]).shape == (0, 1)

    def test_repr_names_class(self):
        scores = TorchExtractor(SimpleModel(10, 4), device="cpu", batch_size=8)
        assert "UncertaintyExtractor" in repr(UncertaintyExtractor(scores))


@pytest.mark.required
class TestClasswiseUncertaintyComposition:
    """Per-class uncertainty over a TorchExtractor with a detection postprocess."""

    def test_call_returns_dict(self):
        model = SimpleModel(10, 3)
        data = np.random.randn(8, 10).astype(np.float32)

        def sharpen(out):  # map to confident class-0/class-1 logits
            n = out.shape[0]
            half = n // 2
            sharp = torch.full((n, 3), -10.0)
            sharp[:half, 0] = 10.0
            sharp[half:, 1] = 10.0
            return sharp

        scores = TorchExtractor(model, device="cpu", batch_size=8, postprocess_fn=sharpen)
        result = ClasswiseUncertaintyExtractor(scores, preds_type="logits")(data)
        assert isinstance(result, dict)
        assert set(result) == {0, 1}

    def test_empty_returns_empty_dict(self):
        scores = TorchExtractor(SimpleModel(10, 4), device="cpu", batch_size=8)
        assert ClasswiseUncertaintyExtractor(scores)([]) == {}

    def test_default_preds_type_is_logits(self):
        scores = TorchExtractor(SimpleModel(10, 4), device="cpu", batch_size=8)
        assert ClasswiseUncertaintyExtractor(scores).preds_type == "logits"


@pytest.mark.required
class TestNormalize:
    """Test entropy normalization in _prediction_uncertainty."""

    def test_normalize_divides_by_log_n_classes(self):
        """Normalized entropy of a uniform distribution is 1.0."""
        uniform = np.full((1, 4), 0.25)

        raw = _prediction_uncertainty(uniform, preds_type="probs", normalize=False)
        norm = _prediction_uncertainty(uniform, preds_type="probs", normalize=True)

        # Uniform over 4 classes has entropy log(4); normalized -> 1.0
        np.testing.assert_allclose(raw, np.log(4), rtol=1e-5)
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5)

    def test_normalize_bounds(self):
        """Normalized uncertainty stays within [0, 1]."""
        logits = np.random.randn(20, 5)
        norm = _prediction_uncertainty(logits, preds_type="logits", normalize=True)
        assert (norm >= 0).all()
        assert (norm <= 1 + 1e-6).all()


@pytest.mark.required
class TestClasswisePredictionUncertainty:
    """Test _classwise_prediction_uncertainty grouping function."""

    def test_returns_dict_keyed_by_class(self):
        """Each confidently-predicted class gets its own uncertainty array."""
        # Detection 0 peaks on class 0, detection 1 peaks on class 2
        logits = np.array([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]], dtype=np.float32)

        result = _classwise_prediction_uncertainty(logits, preds_type="logits")

        assert isinstance(result, dict)
        assert set(result) == {0, 2}
        for arr in result.values():
            assert arr.shape[1] == 1

    def test_empty_input_returns_empty_dict(self):
        """Empty predictions yield an empty mapping."""
        assert _classwise_prediction_uncertainty(np.empty((0, 4)), preds_type="logits") == {}

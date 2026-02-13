"""Tests for drift feature extractors."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataeval.extractors._uncertainty import ClassifierUncertaintyExtractor, _classifier_uncertainty


class SimpleModel(nn.Module):
    def __init__(self, n_features, n_output):
        super().__init__()
        self.fc = nn.Linear(n_features, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.mark.required
class TestClassifierUncertainty:
    """Test _classifier_uncertainty function."""

    def test_probs_input(self):
        """Test uncertainty calculation with probability input."""
        # Create mock probabilities (3 samples, 4 classes)
        probs = np.array([[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])

        result = _classifier_uncertainty(probs, preds_type="probs")

        assert result.shape == (3, 1)
        assert isinstance(result, torch.Tensor)
        # Uniform distribution has highest entropy
        assert result[1, 0] > result[0, 0]
        assert result[1, 0] > result[2, 0]

    def test_logits_input(self):
        """Test uncertainty calculation with logits input."""
        # Create mock logits
        logits = np.array([[2.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [10.0, 0.0, 0.0, 0.0]])

        result = _classifier_uncertainty(logits, preds_type="logits")

        assert result.shape == (3, 1)
        assert isinstance(result, torch.Tensor)
        # All values should be non-negative
        assert (result >= 0).all()

    def test_invalid_probs_sum(self):
        """Test that invalid probabilities raise ValueError."""
        # Probabilities that don't sum to 1
        bad_probs = np.array([[0.5, 0.2, 0.1]])

        with pytest.raises(ValueError, match="Probabilities across labels should sum to 1"):
            _classifier_uncertainty(bad_probs, preds_type="probs")

    def test_invalid_preds_type(self):
        """Test that invalid preds_type raises NotImplementedError."""
        probs = np.array([[0.5, 0.5]])

        with pytest.raises(NotImplementedError, match="Only prediction types 'probs' and 'logits' supported"):
            _classifier_uncertainty(probs, preds_type="invalid")  # type: ignore


@pytest.mark.required
class TestUncertaintyFeatureExtractor:
    """Test ClassifierUncertaintyExtractor."""

    def test_basic_extraction_with_mock(self):
        """Test basic uncertainty extraction using mock."""
        model = SimpleModel(10, 4)
        data = np.random.randn(50, 10).astype(np.float32)

        # Mock the predict function to return probabilities
        mock_probs = np.array([[0.7, 0.2, 0.05, 0.05]] * 50)

        with patch("dataeval.extractors._uncertainty.predict", return_value=mock_probs):
            extractor = ClassifierUncertaintyExtractor(model=model, preds_type="probs", batch_size=16)
            result = extractor(data)

        assert result.shape == (50, 1)
        assert isinstance(result, np.ndarray)

    def test_with_logits(self):
        """Test extraction with logits."""
        model = SimpleModel(10, 4)
        data = np.random.randn(20, 10).astype(np.float32)

        mock_logits = np.random.randn(20, 4)

        with patch("dataeval.extractors._uncertainty.predict", return_value=mock_logits):
            extractor = ClassifierUncertaintyExtractor(model=model, preds_type="logits", batch_size=8)
            result = extractor(data)

        assert result.shape == (20, 1)

    def test_with_transforms(self):
        """Test extraction with transforms."""
        model = SimpleModel(10, 4)
        data = np.random.randn(30, 10).astype(np.float32)

        def transform_fn(x):
            return x * 2.0

        mock_probs = np.array([[0.25, 0.25, 0.25, 0.25]] * 30)

        with patch("dataeval.extractors._uncertainty.predict", return_value=mock_probs):
            extractor = ClassifierUncertaintyExtractor(
                model=model,
                preds_type="probs",
                transforms=transform_fn,
                device="cpu",
            )
            result = extractor(data)

        assert result.shape == (30, 1)

    def test_apply_transforms(self):
        """Test _apply_transforms method."""
        model = SimpleModel(10, 4)

        def transform1(x):
            return x * 2.0

        def transform2(x):
            return x + 1.0

        extractor = ClassifierUncertaintyExtractor(model=model, transforms=[transform1, transform2])

        x = torch.tensor([1.0, 2.0, 3.0])
        result = extractor._apply_transforms(x)

        # Should apply both transforms: (x * 2) + 1
        expected = torch.tensor([3.0, 5.0, 7.0])
        assert torch.allclose(result, expected)

    def test_repr(self):
        """Test __repr__ method."""
        model = SimpleModel(10, 4)
        extractor = ClassifierUncertaintyExtractor(model=model, preds_type="logits", batch_size=64)

        repr_str = repr(extractor)

        assert "ClassifierUncertaintyExtractor" in repr_str
        assert "SimpleModel" in repr_str
        assert "preds_type='logits'" in repr_str
        assert "batch_size=64" in repr_str

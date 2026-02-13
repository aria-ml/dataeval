from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_model():
    """
    Generic mock model for tests.

    Not specific to any ML framework (PyTorch, TF, etc.).
    """
    model = MagicMock()
    model.__class__.__name__ = "MockModel"
    return model

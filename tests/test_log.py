import logging
from unittest.mock import MagicMock, patch

import pytest

import dataeval


@pytest.mark.required
@patch.object(logging.StreamHandler, "emit")
def test_dateval_log_default(mock_emit):
    dataeval.log()
    assert mock_emit.called


@pytest.mark.required
def test_dataeval_log_custom():
    mock_handler = logging.StreamHandler()
    mock_handler.emit = MagicMock()
    dataeval.log(logging.DEBUG, mock_handler)
    assert mock_handler.emit.called


@pytest.mark.required
def test_dataeval_log_idempotent():
    """Calling log twice with the same handler does not attach it twice (68->70)."""
    logger = logging.getLogger("dataeval")
    handler = logging.StreamHandler()
    try:
        dataeval.log(logging.DEBUG, handler)
        dataeval.log(logging.DEBUG, handler)
        assert logger.handlers.count(handler) == 1
    finally:
        logger.removeHandler(handler)

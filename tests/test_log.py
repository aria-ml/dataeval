import logging
from unittest.mock import MagicMock, patch

import dataeval


@patch.object(logging.StreamHandler, "emit")
def test_dateval_log_default(mock_emit):
    dataeval.log()
    assert mock_emit.called


def test_dataeval_log_custom():
    mock_handler = logging.StreamHandler()
    mock_handler.emit = MagicMock()
    dataeval.log(logging.DEBUG, mock_handler)
    assert mock_handler.emit.called
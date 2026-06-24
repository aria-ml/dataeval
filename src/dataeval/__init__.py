"""
DataEval provides a simple interface to characterize visual data and its impact on model performance.

It works across classification and object-detection tasks. It also provides capabilities to select and curate
datasets to test and train performant, robust, unbiased and reliable AI models and monitor for data
shifts that impact performance of deployed models.
"""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"

# Strongly type for pyright
__version__ = str(__version__)

__all__ = [
    "__version__",
    "config",
    "exceptions",
    "flags",
    "log",
    "models",
    "protocols",
    "types",
    "Embeddings",
    "Metadata",
]

import logging

from . import config, exceptions, flags, models, protocols, types
from ._embeddings import Embeddings
from ._metadata import Metadata

logging.getLogger(__name__).addHandler(logging.NullHandler())


def log(level: int = logging.DEBUG, handler: logging.Handler | None = None) -> None:
    """
    Add a handler to the logger quickly for debugging.

    Calling this more than once is idempotent: a handler equal to one already
    attached to the logger is not added again, so log lines are not duplicated.

    Parameters
    ----------
    level : int, default logging.DEBUG(10)
        Set the logging level for the logger.
    handler : logging.Handler, optional
        Sets the logging handler for the logger if provided, otherwise logger will be
        provided with a StreamHandler. When a custom handler is supplied its formatter
        is left untouched; the default StreamHandler is given a verbose debugging
        formatter.
    """
    import logging

    _logger = logging.getLogger(__name__)
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s.%(filename)s:%(lineno)s - %(funcName)10s() | %(message)s",
            ),
        )
    if handler not in _logger.handlers:
        _logger.addHandler(handler)
    _logger.setLevel(level)
    _logger.debug("Added logging handler %s to logger: %s", handler, __name__)

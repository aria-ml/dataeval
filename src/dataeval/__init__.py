"""
DataEval provides a simple interface to characterize image data and its impact on model performance
across classification and object-detection tasks. It also provides capabilities to select and curate
datasets to test and train performant, robust, unbiased and reliable AI models and monitor for data
shifts that impact performance of deployed models.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"

# Strongly type for pyright
__version__ = str(__version__)

__all__ = ["__version__", "config", "detectors", "log", "metrics", "typing", "utils", "workflows"]

import logging

from . import config, detectors, metrics, typing, utils, workflows

logging.getLogger(__name__).addHandler(logging.NullHandler())


def log(level: int = logging.DEBUG, handler: logging.Handler | None = None) -> None:
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for debugging.

    Parameters
    ----------
    level : int, default logging.DEBUG(10)
        Set the logging level for the logger.
    handler : logging.Handler, optional
        Sets the logging handler for the logger if provided, otherwise logger will be
        provided with a StreamHandler.
    """
    import logging

    logger = logging.getLogger(__name__)
    if handler is None:
        handler = logging.StreamHandler() if handler is None else handler
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s.%(filename)s:%(lineno)s - %(funcName)10s() | %(message)s"
            )
        )
    logger.addHandler(handler)
    logger.setLevel(level)
    logging.DEBUG
    logger.debug(f"Added logging handler {handler} to logger: {__name__}")

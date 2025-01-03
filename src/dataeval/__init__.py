"""
DataEval provides a simple interface to characterize image data and its impact on model performance
across classification and object-detection tasks. It also provides capabilities to select and curate
datasets to test and train performant, robust, unbiased and reliable AI models and monitor for data
shifts that impact performance of deployed models.
"""

from __future__ import annotations

__all__ = ["detectors", "log", "metrics", "utils", "workflows"]
__version__ = "0.0.0"

import logging

from dataeval import detectors, metrics, utils, workflows

logging.getLogger(__name__).addHandler(logging.NullHandler())


def log(level: int = logging.DEBUG, handler: logging.Handler | None = None) -> None:
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for debugging.
    """
    import logging

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler() if handler is None else handler
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug(f"Added logging handler {handler} to logger: {__name__}")

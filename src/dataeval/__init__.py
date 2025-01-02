__version__ = "0.0.0"

__all__ = ["log_stderr", "detectors", "metrics", "utils", "workflows"]

import logging

from dataeval import detectors, metrics, utils, workflows

logging.getLogger(__name__).addHandler(logging.NullHandler())


def log_stderr(level: int = logging.DEBUG) -> None:
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging.
    """
    import logging

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)

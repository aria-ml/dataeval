__version__ = "0.0.0"

import logging
from importlib.util import find_spec

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


_IS_TORCH_AVAILABLE = find_spec("torch") is not None
_IS_TORCHVISION_AVAILABLE = find_spec("torchvision") is not None

del find_spec

from dataeval import detectors, metrics  # noqa: E402

__all__ = ["log_stderr", "detectors", "metrics"]

if _IS_TORCH_AVAILABLE:
    from dataeval import utils, workflows

    __all__ += ["utils", "workflows"]

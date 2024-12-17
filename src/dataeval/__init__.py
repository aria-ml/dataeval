__version__ = "0.0.0"

from importlib.util import find_spec

_IS_TORCH_AVAILABLE = find_spec("torch") is not None
_IS_TORCHVISION_AVAILABLE = find_spec("torchvision") is not None

del find_spec

from dataeval import detectors, metrics  # noqa: E402

__all__ = ["detectors", "metrics"]

if _IS_TORCH_AVAILABLE:
    from dataeval import utils, workflows

    __all__ += ["utils", "workflows"]

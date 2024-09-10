__version__ = "0.0.0"

from importlib.util import find_spec

_IS_TORCH_AVAILABLE = find_spec("torch") is not None
_IS_TENSORFLOW_AVAILABLE = find_spec("tensorflow") is not None and find_spec("tensorflow_probability") is not None

del find_spec

from . import detectors, flags, metrics  # noqa: E402

__all__ = ["detectors", "flags", "metrics"]

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from . import torch, utils, workflows

    __all__ += ["torch", "utils", "workflows"]

if _IS_TENSORFLOW_AVAILABLE:  # pragma: no cover
    from . import tensorflow

    __all__ += ["tensorflow"]

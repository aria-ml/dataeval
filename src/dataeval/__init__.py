__version__ = "0.0.0"

from importlib.util import find_spec

_IS_TORCH_AVAILABLE = find_spec("torch") is not None
_IS_TENSORFLOW_AVAILABLE = find_spec("tensorflow") is not None and find_spec("tensorflow_probability") is not None

del find_spec

from . import detectors, metrics  # noqa: E402

__all__ = ["detectors", "metrics"]

if _IS_TORCH_AVAILABLE:  # pragma: no cover
    from . import workflows

    __all__ += ["workflows"]

if _IS_TENSORFLOW_AVAILABLE or _IS_TORCH_AVAILABLE:  # pragma: no cover
    from . import utils

    __all__ += ["utils"]

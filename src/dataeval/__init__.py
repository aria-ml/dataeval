from importlib.util import find_spec

from . import detectors, flags, metrics

__version__ = "0.0.0"

__all__ = ["detectors", "flags", "metrics"]

if find_spec("torch") is not None:  # pragma: no cover
    from . import models, utils, workflows

    __all__ += ["models", "utils", "workflows"]
elif find_spec("tensorflow") is not None:  # pragma: no cover
    from . import models

    __all__ += ["models"]

del find_spec

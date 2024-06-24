from importlib.util import find_spec

from . import flags, metrics

__version__ = "0.0.0"

__all__ = ["metrics", "flags"]

if find_spec("torch") is not None:  # pragma: no cover
    from . import models, workflows

    __all__ += ["models", "workflows"]
elif find_spec("tensorflow") is not None:  # pragma: no cover
    from . import models

    __all__ += ["models"]

del find_spec

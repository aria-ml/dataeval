from importlib.util import find_spec

from . import metrics

__version__ = "0.0.0"

__all__ = ["metrics"]

if find_spec("torch") is not None:  # pragma: no cover
    from . import models

    __all__ += ["models"]

del find_spec

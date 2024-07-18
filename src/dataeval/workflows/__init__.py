from importlib.util import find_spec

if find_spec("torch") is not None:  # pragma: no cover
    from dataeval._internal.workflows.sufficiency import Sufficiency

    __all__ = ["Sufficiency"]

del find_spec

from importlib.util import find_spec

if find_spec("torch") is not None:  # pragma: no cover
    from . import ae

    __all__ = ["ae"]

del find_spec

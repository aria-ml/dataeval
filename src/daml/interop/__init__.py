from daml._internal.deps import is_maite_available

if is_maite_available():  # pragma: no cover
    from . import wrappers  # noqa F401

    __all__ = ["wrappers"]

del is_maite_available

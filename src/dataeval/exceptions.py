"""Exception and warning classes for DataEval."""

__all__ = [
    "DeprecatedWarning",
    "ExperimentalWarning",
    "NotFittedError",
    "ShapeMismatchError",
]


class NotFittedError(RuntimeError):
    """Raised when a method is called before the object has been fitted or bound.

    This error indicates that a prerequisite initialization step (such as
    :meth:`fit` or :meth:`bind`) has not been performed before calling a
    method that requires it.
    """


class ShapeMismatchError(ValueError):
    """Raised when array shapes or dimensions are incompatible.

    This error indicates that input arrays have incorrect dimensions,
    incompatible shapes, or mismatched lengths where they are expected
    to be consistent.
    """


class ExperimentalWarning(FutureWarning):
    """Issued when an experimental feature is used.

    Experimental features may change signature, behavior, or be removed
    in any future release without following the normal deprecation cycle.
    """


class DeprecatedWarning(FutureWarning):
    """Issued when a deprecated feature is used.

    Deprecated features will be removed in a future release.
    """

"""Exception and warning classes for DataEval."""

__all__ = [
    "DeprecatedWarning",
    "ExperimentalWarning",
    "MaiteShapeError",
    "NotFittedError",
    "OntologyCycleError",
    "OntologyError",
    "ShapeMismatchError",
]


class MaiteShapeError(TypeError):
    """Raised when a dataset does not match the expected MAITE datum shape.

    Public entry points that consume a MAITE-protocol dataset probe
    ``dataset[0]`` and raise this error when the datum does not have the
    expected ``(image, target, metadata)`` 3-tuple shape, or when the
    target does not match the protocol the consumer requires (e.g.
    :obj:`~dataeval.protocols.ObjectDetectionTarget` for an object
    detection consumer).

    Inherits from :class:`TypeError` so callers that previously caught the
    silent ``IndexError``/``TypeError`` from downstream destructuring keep
    working.
    """


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


class OntologyError(ValueError):
    """Raised when an :class:`dataeval.Ontology` fails validation.

    Covers structural problems detected at construction time, such as duplicate
    concept ids or malformed hierarchy input. Inherits from :class:`ValueError`
    so callers that previously caught the bare ``ValueError`` keep working.
    """


class OntologyCycleError(OntologyError):
    """Raised when an :class:`dataeval.Ontology`'s is-a graph contains a cycle.

    A taxonomy must be acyclic; a cycle (a concept that is its own ancestor)
    makes ancestor/descendant queries ill-defined.
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

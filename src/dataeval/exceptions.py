"""Exception and warning classes for DataEval."""

__all__ = [
    "DeprecatedWarning",
    "ExperimentalWarning",
    "MaiteShapeError",
    "MetadataFormatError",
    "NotFittedError",
    "OntologyCycleError",
    "OntologyError",
    "ShapeMismatchError",
    "StatsInvalidatedWarning",
]


class MaiteShapeError(TypeError, ValueError):
    """Raised when a dataset does not match the expected MAITE datum shape.

    Public entry points that consume a MAITE-protocol dataset probe
    ``dataset[0]`` and raise this error when the datum does not have the
    expected ``(image, target, metadata)`` 3-tuple shape, or when the
    target does not match the protocol the consumer requires (e.g.
    :obj:`~dataeval.protocols.ObjectDetectionTarget` for an object
    detection consumer).

    Every defect this names is one that used to surface later, from whichever
    reader first tripped over it; raising it at entry only moves *where* it is
    reported, so it inherits from both of the types those readers raised and
    stays catchable either way. ``TypeError`` covers the ``IndexError``/
    ``TypeError`` from downstream destructuring; ``ValueError`` covers the
    per-detection array readers, whose reshape against the label count raises
    ``ValueError`` in the structuring walk.
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


class MetadataFormatError(ValueError):
    """Raised when :meth:`dataeval.Metadata.load` cannot read a saved file.

    Covers every way a file can fail to be a metadata archive this version can
    restore: it is not the archive format at all, a member is missing or corrupt,
    or it was written by a version whose layout has since changed.

    The last of those is the ordinary case rather than an error condition. A saved
    :class:`~dataeval.Metadata` is a **cache, not an interchange format**: it stores
    the library's internal per-level layout, and that layout is free to change
    between releases. This error is what a stale file is *supposed* to produce, so a
    caching layer should catch it and recompute rather than treat it as a bug::

        try:
            metadata = Metadata.load(path, dataset)
        except MetadataFormatError:
            metadata = Metadata(dataset)
            metadata.save(path)

    Inherits from :class:`ValueError`, so a caller that catches that keeps working.
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


class StatsInvalidatedWarning(UserWarning):
    """Issued when a view operation invalidates a statistic that was requested.

    An operation that rewrites image content makes some statistics describe the
    transform rather than the source data — a resize makes ``width``/``height``/
    ``aspect_ratio`` report the resize target and makes ``sharpness`` measure the
    interpolation kernel. Findings over those statistics are about the pipeline,
    not the dataset.

    Inherits from :class:`UserWarning` rather than :class:`FutureWarning`: this is a
    statement about the meaning of the data, not about API lifecycle. Nothing is
    deprecated and nothing changes in a future release — the computation proceeds
    exactly as requested.
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

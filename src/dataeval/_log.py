"""Logging plumbing: the curated namespace and deferred message helpers."""

__all__ = []

import logging
from collections.abc import Callable

_ROOT = "dataeval"

# The curated logging namespace, keyed by module-path prefix; the longest match wins.
#
# Logger names are a public configuration surface -- users filter and route on them --
# so they are declared here rather than derived from ``__name__``. Deriving them ties
# that surface to the private module layout, where a rename such as ``_metadata.py``
# becoming ``_metadata/`` silently renames a logger someone has configured.
#
# Prefixes map whole subtrees, so moving a file within a subsystem needs no change
# here. Entries are deliberately coarse: widening a namespace later is backwards
# compatible, because a handler on ``dataeval.core`` still receives records from a
# ``dataeval.core.ber`` added beneath it. Narrowing one is not.
_NAMESPACES: dict[str, str] = {
    "dataeval._embeddings": "dataeval.embeddings",
    "dataeval._metadata": "dataeval.metadata",
    "dataeval._ontology": "dataeval.ontology",
    "dataeval.bias": "dataeval.bias",
    "dataeval.core": "dataeval.core",
    "dataeval.data": "dataeval.data",
    "dataeval.extractors": "dataeval.extractors",
    "dataeval.models": "dataeval.models",
    "dataeval.performance": "dataeval.performance",
    "dataeval.quality": "dataeval.quality",
    "dataeval.scope": "dataeval.scope",
    "dataeval.selection": "dataeval.selection",
    "dataeval.shift": "dataeval.shift",
    "dataeval.types": "dataeval.types",
    "dataeval.utils": "dataeval.utils",
}


def get_logger(module: str) -> logging.Logger:
    """
    Return the logger for a module's curated namespace.

    Call as ``get_logger(__name__)``. The returned logger is a descendant of the
    ``dataeval`` root logger, so configuring that one name still captures every
    subsystem.

    Parameters
    ----------
    module : str
        A module path, typically ``__name__``. A leading ``src.`` is stripped, as
        ``__module__`` carries it when running against the source tree.

    Returns
    -------
    logging.Logger
        The logger for the longest registered prefix of `module`, or the ``dataeval``
        root logger when no prefix is registered. An unregistered module never raises
        -- logging must not break an import -- so ``test_log.py`` carries the coverage
        test that catches a subsystem missing from `_NAMESPACES`.
    """
    parts = module.removeprefix("src.").split(".")
    while parts:
        namespace = _NAMESPACES.get(".".join(parts))
        if namespace is not None:
            return logging.getLogger(namespace)
        parts.pop()
    return logging.getLogger(_ROOT)


class LogMessage:
    """Deferred message callback for logging expensive messages."""

    def __init__(self, fn: Callable[..., str]) -> None:
        self._fn = fn
        self._str = None

    def __str__(self) -> str:
        if self._str is None:
            self._str = self._fn()
        return self._str

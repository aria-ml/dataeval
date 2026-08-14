"""Metadata: structuring, linking, per-level storage, binning and projection.

The whole metadata pipeline, and nothing else. A dataset enters through
``_structurers``, which lays its rows out one block per level; ``_links`` holds the
positional edges between those levels; ``_store`` keeps one frame per level and gathers
along the edges on demand; and ``_metadata`` is the class that reads all of it. Every
module here is reachable only from another module here, which is what makes the package
boundary the same thing as the class's.

The package boundary is the class. :class:`~dataeval.Metadata` is the only name
exported, and the supported import is ``from dataeval import Metadata``. Nothing outside
this package should name a module inside it: ``dataeval._structurers`` and
``dataeval._links`` moved in here because metadata was their only caller, and a module
that acquires a second one belongs back outside rather than imported across the
boundary.
"""

__all__ = ["Metadata"]

from dataeval._metadata._metadata import Metadata

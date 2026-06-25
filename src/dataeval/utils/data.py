"""Deprecated location. Dataset operations moved to :mod:`dataeval.data`.

Importing :func:`split_dataset`, :func:`unzip_dataset`, :class:`TrainValSplit`, or
:class:`DatasetSplits` from here is deprecated; import them from :mod:`dataeval.data`.
The validation helpers (:data:`DatasetKind`, :func:`validate_dataset`,
:func:`requires_maite_dataset`) remain available here.
"""

__all__ = [
    "DatasetKind",
    "requires_maite_dataset",
    "validate_dataset",
]

import warnings
from typing import Any

from dataeval.utils._validate import DatasetKind, requires_maite_dataset, validate_dataset

_MOVED = ("DatasetSplits", "TrainValSplit", "split_dataset", "unzip_dataset")


def __getattr__(name: str) -> Any:
    if name in _MOVED:
        warnings.warn(
            f"dataeval.utils.data.{name} has moved to dataeval.data.{name}; importing it from "
            "dataeval.utils.data is deprecated and will be removed in v1.2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        import dataeval.data

        return getattr(dataeval.data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

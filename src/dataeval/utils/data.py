"""Dataset validation helpers: what shape a dataset is, and refusing the wrong one.

The dataset *operations* that once lived here — :func:`split_dataset`,
:func:`unzip_dataset`, :class:`TrainValSplit`, :class:`DatasetSplits` — moved to
:mod:`dataeval.data` in v1.1 and stopped being importable from here in v1.2.0. The
validation helpers below never moved and were never deprecated, so this module remains
their public home.
"""

__all__ = [
    "DatasetKind",
    "requires_maite_dataset",
    "validate_dataset",
]

from dataeval.utils._validate import DatasetKind, requires_maite_dataset, validate_dataset

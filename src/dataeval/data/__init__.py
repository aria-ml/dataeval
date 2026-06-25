"""Dataset organization tools: filter, split, and reshape dataset views.

This package is the home for DataEval's data-organization operations — lazy,
read-only dataset views that subset (:class:`Select`), split
(:func:`split_dataset`), and reshape (:func:`unzip_dataset`) datasets without
mutating the source.
"""

__all__ = [
    "ClassBalance",
    "ClassFilter",
    "DatasetSplits",
    "Indices",
    "Limit",
    "Reverse",
    "Select",
    "Selection",
    "Shuffle",
    "TrainValSplit",
    "split_dataset",
    "unzip_dataset",
]

from dataeval.data._classbalance import ClassBalance
from dataeval.data._classfilter import ClassFilter
from dataeval.data._indices import Indices
from dataeval.data._limit import Limit
from dataeval.data._reverse import Reverse
from dataeval.data._select import Select, Selection
from dataeval.data._shuffle import Shuffle
from dataeval.data._split import DatasetSplits, TrainValSplit, split_dataset
from dataeval.data._unzip import unzip_dataset

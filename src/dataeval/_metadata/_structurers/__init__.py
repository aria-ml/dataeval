"""Pluggable structuring strategies that turn a dataset into levelled metadata rows.

The core :class:`~dataeval.Metadata` engine is task agnostic: it consumes a
:class:`StructuredData` bundle and never inspects the dataset itself. Everything
that depends on what a dataset item is (e.g. one image, or a video sequence) and
where the labels sit (e.g. ``unit``, or ``instance``) lives in a :class:`Structurer`.

One module per class, grouped by what each is responsible for:

- ``_reserved`` names the non-factor columns and builds a block's worth of them;
  ``_ordering`` numbers rows within their parent; ``_gather`` reads a level's values
  from a descendant's rows; ``_reporting`` says what structuring left out.
- ``_block``, ``_layout`` and ``_data`` are the shapes a structurer produces: rows at
  one level, the positional map across all of them, and the bundle that carries both
  plus the factors.
- ``_base`` declares a task's level model and ``_dataset`` adds the obligation to walk
  a dataset; ``_propagation`` and ``_instances`` are the mixins the tasks share.
- ``_classification``, ``_detection`` and ``_tracking`` are the tasks themselves —
  tracking splitting its per-frame row shape into ``_frames`` and its three-level walk
  into ``_accumulator``.
- ``_source_index`` and ``_factors`` are the dataset-free path behind
  :meth:`~dataeval.Metadata.from_factors`, and ``_select`` picks a strategy for a
  dataset by target type or by name.

The package is the import surface. Everything below is reached through this
``__init__``, so a caller writes ``from dataeval._metadata._structurers import
select_structurer`` and never names the module a class happens to live in — which is
what lets these files be regrouped without touching a caller.
"""

__all__ = [
    "DISPATCH",
    "IDENTIFIER_COLUMNS",
    "LEGACY_COLUMNS",
    "LEVEL_COLUMNS",
    "RESERVED_COLUMNS",
    "TASK",
    "TASK_STRUCTURERS",
    "DatasetStructurer",
    "FactorsStructurer",
    "FrameRows",
    "ICStructurer",
    "InstanceBuildingMixin",
    "MOTAccumulator",
    "MOTStructurer",
    "ODImageStructurer",
    "PropagationMixin",
    "RowBlock",
    "RowLayout",
    "SourceIndexRows",
    "StructuredData",
    "Structurer",
    "TaskOverride",
    "reserved_block_columns",
    "safe_column_name",
    "select_structurer",
]

from dataeval._metadata._structurers._accumulator import MOTAccumulator
from dataeval._metadata._structurers._base import TASK, Structurer
from dataeval._metadata._structurers._block import RowBlock
from dataeval._metadata._structurers._classification import ICStructurer
from dataeval._metadata._structurers._data import StructuredData
from dataeval._metadata._structurers._dataset import DatasetStructurer
from dataeval._metadata._structurers._detection import ODImageStructurer
from dataeval._metadata._structurers._factors import FactorsStructurer
from dataeval._metadata._structurers._frames import FrameRows
from dataeval._metadata._structurers._instances import InstanceBuildingMixin
from dataeval._metadata._structurers._layout import RowLayout
from dataeval._metadata._structurers._propagation import PropagationMixin
from dataeval._metadata._structurers._reserved import (
    IDENTIFIER_COLUMNS,
    LEGACY_COLUMNS,
    LEVEL_COLUMNS,
    RESERVED_COLUMNS,
    reserved_block_columns,
    safe_column_name,
)
from dataeval._metadata._structurers._select import DISPATCH, TASK_STRUCTURERS, TaskOverride, select_structurer
from dataeval._metadata._structurers._source_index import SourceIndexRows
from dataeval._metadata._structurers._tracking import MOTStructurer

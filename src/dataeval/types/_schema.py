"""Portable schema for ``metadata.json`` sidecar files.

DatasetInfo / ModelInfo / ExtractorInfo / MetadataJson define the portable
schema for sidecar metadata describing DataEval-consumable artifacts. The
library itself does not emit these to disk — file I/O is the consumer's
concern (e.g. DataEval-Flow). Use Pydantic's built-in ``model_dump_json()``
/ ``model_validate_json()`` for serialization round trips.
"""

__all__ = [
    "SCHEMA_VERSION",
    "DatasetInfo",
    "ExtractorInfo",
    "MetadataJson",
    "ModelInfo",
    "SelectionInfo",
    "TransformInfo",
]

from datetime import datetime, timezone
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dataeval.types._execution import ExecutionMetadata, __version__

SCHEMA_VERSION = "1.0"


class SelectionInfo(BaseModel):
    """
    Descriptor for a single :mod:`dataeval.data` selection step.

    Mirrors the shape of DataEval-Flow's ``SelectionStep`` config so a single
    selector can be described identically on input (config) and output (sidecar).

    Attributes
    ----------
    type : str
        Operation class name (e.g. ``"ClassFilter"``, ``"Limit"``, ``"Indices"``).
    params : dict
        Keyword arguments passed to the operation's constructor.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    """
    Descriptive metadata for a dataset artifact.

    All fields except ``name`` are optional so the schema accommodates
    partially-known datasets (e.g. when class names are unavailable).

    Attributes
    ----------
    name : str
        Human-readable identifier for the dataset.
    version : str or None
        Dataset version string (semantic versioning recommended).
    description : str or None
        Free-text description.
    source : str or None
        Origin URL or provenance string.
    format : str or None
        Storage/exchange format (e.g. ``"COCO"``, ``"HuggingFace"``, ``"MAITE"``).
    n_samples : int or None
        Total number of samples across all splits.
    n_classes : int or None
        Number of distinct classes/categories.
    class_names : list of str or None
        Ordered list of class names (length should equal ``n_classes`` when both set).
    splits : dict of str to int or None
        Per-split sample counts (e.g. ``{"train": 800, "val": 100, "test": 100}``).
    checksum : str or None
        Content checksum (recommended: ``sha256:<hex>``).
    license : str or None
        SPDX license identifier or license name.
    selections : list of list of SelectionInfo or None
        Applied selection lineage, grouped by construction call to mirror
        :attr:`dataeval.data.View.operation_groups`. The outer list is
        ordered innermost (oldest) first; each inner list is one
        ``View(...)`` call's worth of operations.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str
    version: str | None = None
    description: str | None = None
    source: str | None = None
    format: str | None = None
    n_samples: int | None = None
    n_classes: int | None = None
    class_names: list[str] | None = None
    splits: dict[str, int] | None = None
    checksum: str | None = None
    license: str | None = None
    selections: list[list[SelectionInfo]] | None = None


class TransformInfo(BaseModel):
    """
    Descriptor for a single image transform (typically a torchvision transform).

    Attributes
    ----------
    type : str
        Transform class name (e.g. ``"Resize"``, ``"Normalize"``).
    params : dict
        Keyword arguments passed to the transform's constructor.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """
    Descriptive metadata for a model artifact.

    Attributes
    ----------
    name : str
        Human-readable identifier for the model.
    version : str or None
        Model version string.
    description : str or None
        Free-text description.
    framework : str or None
        Serialization framework (e.g. ``"onnx"``, ``"torch"``, ``"tensorflow"``).
    framework_version : str or None
        Version of the framework the model was produced with.
    task : str or None
        Task category (e.g. ``"classification"``, ``"object_detection"``, ``"embedding"``).
    input_spec : dict or None
        Free-form specification of inputs. Recommended keys: ``shape``, ``dtype``, ``name``.
    output_spec : dict or None
        Free-form specification of outputs (same recommended keys as ``input_spec``).
    checksum : str or None
        Content checksum (recommended: ``sha256:<hex>``).
    license : str or None
        SPDX license identifier or license name.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str
    version: str | None = None
    description: str | None = None
    framework: str | None = None
    framework_version: str | None = None
    task: str | None = None
    input_spec: dict[str, Any] | None = None
    output_spec: dict[str, Any] | None = None
    checksum: str | None = None
    license: str | None = None


class ExtractorInfo(BaseModel):
    """
    Descriptor for a feature extractor as used to produce embeddings.

    Denormalized form of DataEval-Flow's per-model extractor config: the
    underlying model and the preprocessor pipeline are inlined rather than
    referenced by name, so the sidecar is self-contained.

    Attributes
    ----------
    type : str
        Extractor class name (e.g. ``"OnnxExtractor"``, ``"TorchExtractor"``,
        ``"FlattenExtractor"``, ``"BoVWExtractor"``, ``"UncertaintyExtractor"``).
    model : ModelInfo or None
        Underlying model artifact. ``None`` for extractors with no model
        (e.g. ``FlattenExtractor``).
    transforms : list of TransformInfo
        Ordered preprocessing pipeline applied before model inference.
    params : dict
        Extractor-specific runtime parameters (e.g. ``output_name``,
        ``layer_name``, ``flatten``, ``device``, ``batch_size``).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    type: str
    model: ModelInfo | None = None
    transforms: list[TransformInfo] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


def _default_provenance() -> ExecutionMetadata:
    return ExecutionMetadata(
        name="dataeval.types.MetadataJson",
        execution_time=datetime.now(timezone.utc),
        execution_duration=0.0,
        arguments={},
        state={},
        version=__version__,
    )


class MetadataJson(BaseModel):
    """
    Top-level schema for a ``metadata.json`` sidecar file.

    A single file may carry a ``dataset`` block, a ``model`` block, or both —
    e.g. a model fine-tuned on a specific dataset can be described in one file.
    The ``provenance`` field reuses :class:`ExecutionMetadata` so an evaluator
    can stamp the file with its own ``output.meta()`` record.

    Attributes
    ----------
    schema_version : str
        Schema version this document conforms to. Defaults to :data:`SCHEMA_VERSION`.
    provenance : ExecutionMetadata
        Provenance record for the producing entity. Defaults to a record built
        at construction time for ``dataeval.types.MetadataJson``.
    dataset : DatasetInfo or None
        Dataset description, when applicable.
    model : ModelInfo or None
        Model description, when applicable.
    extractor : ExtractorInfo or None
        Feature extractor description, when applicable. An extractor may or
        may not be tied to a model (``FlattenExtractor`` has none), so this
        is a sibling of ``model`` rather than nested under it.
    extra : dict
        Free-form bag for tool-specific fields outside the core schema.

    Examples
    --------
    >>> from dataeval.types import DatasetInfo, MetadataJson
    >>> doc = MetadataJson(dataset=DatasetInfo(name="my-ds"))
    >>> doc.model_dump_json()  # doctest: +SKIP
    >>> restored = MetadataJson.model_validate_json(payload)  # doctest: +SKIP
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    provenance: ExecutionMetadata = Field(default_factory=_default_provenance)
    dataset: DatasetInfo | None = None
    model: ModelInfo | None = None
    extractor: ExtractorInfo | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_at_least_one_section(self) -> Self:
        if self.dataset is None and self.model is None and self.extractor is None:
            raise ValueError("MetadataJson requires at least one of `dataset`, `model`, or `extractor`.")
        return self

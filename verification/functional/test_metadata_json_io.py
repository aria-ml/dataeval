"""Verify that DataEval produces a validated metadata.json schema for datasets and models.

Maps to meta repo test cases:
  - TC-13.1: Dataset/Model Metadata Export (IR-3-S-12)

DataEval owns the portable schema; serialization to disk is the consumer's
concern (Pydantic's built-in ``model_dump_json`` / ``model_validate_json``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from dataeval.types import (
    SCHEMA_VERSION,
    DatasetInfo,
    ExecutionMetadata,
    ExtractorInfo,
    MetadataJson,
    ModelInfo,
    SelectionInfo,
    TransformInfo,
)


@pytest.mark.test_case("13-1")
class TestMetadataJsonSchema:
    """End-to-end verification of the metadata.json schema."""

    def test_dataset_info_round_trip(self):
        ds = DatasetInfo(
            name="example-dataset",
            version="1.0.0",
            format="HuggingFace",
            n_samples=1000,
            n_classes=10,
            splits={"train": 800, "val": 100, "test": 100},
            checksum="sha256:" + "0" * 64,
        )
        restored = MetadataJson.model_validate_json(MetadataJson(dataset=ds).model_dump_json())
        assert restored.dataset is not None
        assert restored.dataset.name == "example-dataset"
        assert restored.dataset.splits == {"train": 800, "val": 100, "test": 100}
        assert restored.schema_version == SCHEMA_VERSION

    def test_model_info_round_trip(self):
        m = ModelInfo(
            name="example-encoder",
            framework="onnx",
            task="embedding",
            input_spec={"shape": [None, 3, 16, 16], "dtype": "float32"},
            output_spec={"shape": [None, 128], "dtype": "float32"},
        )
        restored = MetadataJson.model_validate_json(MetadataJson(model=m).model_dump_json())
        assert restored.model is not None
        assert restored.model.framework == "onnx"
        assert restored.model.output_spec is not None
        assert restored.model.output_spec["shape"][-1] == 128

    def test_combined_dataset_and_model(self):
        doc = MetadataJson(dataset=DatasetInfo(name="ds"), model=ModelInfo(name="m"))
        restored = MetadataJson.model_validate_json(doc.model_dump_json())
        assert restored.dataset is not None
        assert restored.dataset.name == "ds"
        assert restored.model is not None
        assert restored.model.name == "m"

    def test_individual_info_types_serialize_independently(self):
        """DatasetInfo and ModelInfo are independently serializable for consumers that need only one."""
        ds = DatasetInfo(name="standalone-ds", n_samples=10)
        m = ModelInfo(name="standalone-m", framework="onnx")
        assert DatasetInfo.model_validate_json(ds.model_dump_json()) == ds
        assert ModelInfo.model_validate_json(m.model_dump_json()) == m

    def test_emitted_json_is_human_readable(self):
        payload = MetadataJson(dataset=DatasetInfo(name="x")).model_dump_json(indent=2)
        parsed = json.loads(payload)
        assert "\n  " in payload
        assert parsed["schema_version"] == SCHEMA_VERSION
        assert parsed["provenance"]["name"] == "dataeval.types.MetadataJson"

    def test_execution_metadata_serializes_as_provenance(self):
        meta = ExecutionMetadata(
            name="dataeval.bias.Balance.evaluate",
            execution_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
            execution_duration=0.42,
            arguments={"metadata": "Metadata: len=100"},
            state={"factors": "list: len=3"},
            version="1.2.3",
        )
        doc = MetadataJson(dataset=DatasetInfo(name="example"), provenance=meta)
        restored = MetadataJson.model_validate_json(doc.model_dump_json())
        assert isinstance(restored.provenance, ExecutionMetadata)
        assert restored.provenance.name == "dataeval.bias.Balance.evaluate"
        assert restored.provenance.execution_duration == 0.42
        assert restored.provenance.arguments == {"metadata": "Metadata: len=100"}
        assert restored.provenance.version == "1.2.3"

    def test_schema_rejects_empty_payload(self):
        with pytest.raises(ValueError, match="at least one"):
            MetadataJson()

    def test_dataset_selections_and_extractor_round_trip(self):
        """A realistic full sidecar: filtered dataset + ONNX extractor with transforms."""
        ds = DatasetInfo(
            name="filtered-cifar10",
            source="cifar10",
            n_samples=100,
            selections=[
                [SelectionInfo(type="ClassFilter", params={"classes": [0, 1]})],
                [SelectionInfo(type="Limit", params={"size": 100})],
            ],
        )
        ext = ExtractorInfo(
            type="OnnxExtractor",
            model=ModelInfo(name="resnet50", framework="onnx"),
            transforms=[
                TransformInfo(type="Resize", params={"size": [256, 256]}),
                TransformInfo(type="CenterCrop", params={"size": [224, 224]}),
            ],
            params={"output_name": "flatten0", "batch_size": 64},
        )
        doc = MetadataJson(dataset=ds, extractor=ext)
        restored = MetadataJson.model_validate_json(doc.model_dump_json())

        assert restored.dataset is not None
        assert restored.dataset.selections is not None
        assert len(restored.dataset.selections) == 2
        assert restored.dataset.selections[0][0].type == "ClassFilter"

        assert restored.extractor is not None
        assert restored.extractor.type == "OnnxExtractor"
        assert restored.extractor.model is not None
        assert restored.extractor.model.framework == "onnx"
        assert len(restored.extractor.transforms) == 2
        assert restored.extractor.params["batch_size"] == 64

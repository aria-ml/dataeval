"""Unit tests for the metadata.json schema in dataeval.types."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from dataeval import __version__
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


def _sample_execution_metadata() -> ExecutionMetadata:
    return ExecutionMetadata(
        name="dataeval.bias.Balance.evaluate",
        execution_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
        execution_duration=1.234,
        arguments={"metadata": "Metadata: len=100"},
        state={"factors": "list: len=3"},
        version="1.2.3",
    )


class TestDatasetInfo:
    def test_minimal_requires_only_name(self):
        ds = DatasetInfo(name="cifar10")
        assert ds.name == "cifar10"
        assert ds.version is None
        assert ds.splits is None

    def test_full_round_trip(self):
        ds = DatasetInfo(
            name="cifar10",
            version="1.0.0",
            description="60k 32x32 color images in 10 classes",
            source="https://www.cs.toronto.edu/~kriz/cifar.html",
            format="HuggingFace",
            n_samples=60000,
            n_classes=10,
            class_names=["airplane", "automobile"],
            splits={"train": 50000, "test": 10000},
            checksum="sha256:abc123",
            license="CC-BY-4.0",
        )
        assert DatasetInfo.model_validate_json(ds.model_dump_json()) == ds

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            DatasetInfo(name="cifar10", bogus_field=True)  # type: ignore

    def test_selections_round_trip_grouped(self):
        ds = DatasetInfo(
            name="filtered-cifar10",
            source="cifar10",
            n_samples=100,
            selections=[
                [SelectionInfo(type="ClassFilter", params={"classes": [0, 1]})],
                [SelectionInfo(type="Limit", params={"size": 100}), SelectionInfo(type="Shuffle", params={"seed": 42})],
            ],
        )
        restored = DatasetInfo.model_validate_json(ds.model_dump_json())
        assert restored.selections is not None
        assert len(restored.selections) == 2
        assert restored.selections[0][0].type == "ClassFilter"
        assert restored.selections[1][1].params == {"seed": 42}


class TestSelectionInfo:
    def test_minimal(self):
        s = SelectionInfo(type="Limit", params={"size": 100})
        assert s.type == "Limit"
        assert s.params == {"size": 100}

    def test_params_default_empty(self):
        s = SelectionInfo(type="Reverse")
        assert s.params == {}

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            SelectionInfo(type="Limit", bogus=1)  # type: ignore


class TestTransformInfo:
    def test_minimal(self):
        t = TransformInfo(type="Resize", params={"size": [256, 256], "antialias": True})
        assert t.type == "Resize"
        assert t.params["antialias"] is True

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            TransformInfo(type="Resize", bogus=1)  # type: ignore


class TestExtractorInfo:
    def test_minimal_flatten_style(self):
        e = ExtractorInfo(type="FlattenExtractor")
        assert e.type == "FlattenExtractor"
        assert e.model is None
        assert e.transforms == []
        assert e.params == {}

    def test_full_onnx_style_round_trip(self):
        e = ExtractorInfo(
            type="OnnxExtractor",
            model=ModelInfo(name="resnet50", framework="onnx", framework_version="1.16.0"),
            transforms=[
                TransformInfo(type="Resize", params={"size": [256, 256]}),
                TransformInfo(type="CenterCrop", params={"size": [224, 224]}),
                TransformInfo(type="Normalize", params={"mean": [0.485], "std": [0.229]}),
            ],
            params={"output_name": "flatten0", "flatten": True, "batch_size": 64},
        )
        restored = ExtractorInfo.model_validate_json(e.model_dump_json())
        assert restored == e
        assert restored.model is not None
        assert restored.model.framework == "onnx"
        assert len(restored.transforms) == 3
        assert restored.params["batch_size"] == 64

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            ExtractorInfo(type="OnnxExtractor", bogus=1)  # type: ignore


class TestModelInfo:
    def test_minimal_requires_only_name(self):
        m = ModelInfo(name="resnet50")
        assert m.name == "resnet50"
        assert m.framework is None

    def test_full_round_trip(self):
        m = ModelInfo(
            name="resnet50",
            version="2.1",
            framework="onnx",
            framework_version="1.16.0",
            task="embedding",
            input_spec={"shape": [None, 3, 224, 224], "dtype": "float32", "name": "input"},
            output_spec={"shape": [None, 2048], "dtype": "float32", "name": "embeddings"},
            checksum="sha256:deadbeef",
        )
        assert ModelInfo.model_validate_json(m.model_dump_json()) == m

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            ModelInfo(name="resnet50", bogus=1)  # type: ignore


class TestMetadataJson:
    def test_requires_at_least_one_section(self):
        with pytest.raises(ValidationError):
            MetadataJson()

    def test_extractor_alone_satisfies_validator(self):
        doc = MetadataJson(extractor=ExtractorInfo(type="FlattenExtractor"))
        assert doc.dataset is None
        assert doc.model is None
        assert doc.extractor is not None

    def test_extractor_round_trip(self):
        ext = ExtractorInfo(
            type="OnnxExtractor",
            model=ModelInfo(name="resnet50", framework="onnx"),
            transforms=[TransformInfo(type="Resize", params={"size": [224, 224]})],
            params={"output_name": "flatten0"},
        )
        doc = MetadataJson(dataset=DatasetInfo(name="ds"), extractor=ext)
        restored = MetadataJson.model_validate_json(doc.model_dump_json())
        assert restored.extractor is not None
        assert restored.extractor.type == "OnnxExtractor"
        assert restored.extractor.model is not None
        assert restored.extractor.model.name == "resnet50"
        assert restored.extractor.transforms[0].params == {"size": [224, 224]}

    def test_default_provenance_is_populated(self):
        doc = MetadataJson(dataset=DatasetInfo(name="x"))
        assert doc.provenance.name == "dataeval.types.MetadataJson"
        assert doc.provenance.version == __version__

    def test_dataset_only_round_trip(self):
        doc = MetadataJson(
            dataset=DatasetInfo(name="cifar10", n_samples=60000, splits={"train": 50000, "test": 10000}),
        )
        restored = MetadataJson.model_validate_json(doc.model_dump_json())
        assert restored.dataset is not None
        assert restored.dataset.name == "cifar10"
        assert restored.dataset.splits == {"train": 50000, "test": 10000}
        assert restored.model is None
        assert restored.schema_version == SCHEMA_VERSION

    def test_model_only_round_trip(self):
        doc = MetadataJson(model=ModelInfo(name="resnet50", framework="onnx"))
        restored = MetadataJson.model_validate_json(doc.model_dump_json())
        assert restored.model is not None
        assert restored.model.framework == "onnx"
        assert restored.dataset is None

    def test_combined_dataset_and_model(self):
        doc = MetadataJson(dataset=DatasetInfo(name="cifar10"), model=ModelInfo(name="resnet50"))
        restored = MetadataJson.model_validate_json(doc.model_dump_json())
        assert restored.dataset is not None
        assert restored.dataset.name == "cifar10"
        assert restored.model is not None
        assert restored.model.name == "resnet50"

    def test_extra_fields_round_trip(self):
        doc = MetadataJson(
            dataset=DatasetInfo(name="cifar10"),
            extra={"pipeline_id": "abc-123", "notes": "smoke test"},
        )
        restored = MetadataJson.model_validate_json(doc.model_dump_json())
        assert restored.extra == {"pipeline_id": "abc-123", "notes": "smoke test"}

    def test_execution_metadata_round_trips_as_provenance(self):
        meta = _sample_execution_metadata()
        doc = MetadataJson(dataset=DatasetInfo(name="x"), provenance=meta)
        restored = MetadataJson.model_validate_json(doc.model_dump_json())
        assert restored.provenance.name == "dataeval.bias.Balance.evaluate"
        assert restored.provenance.execution_duration == 1.234
        assert restored.provenance.arguments == {"metadata": "Metadata: len=100"}
        assert restored.provenance.version == "1.2.3"

    def test_validate_rejects_unknown_top_level_fields(self):
        with pytest.raises(ValidationError):
            MetadataJson.model_validate({"schema_version": "1.0", "unknown_top_level": True})

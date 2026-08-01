"""Unit tests for structurer selection and the reserved column schema."""

from unittest.mock import Mock

import numpy as np
import pytest

from dataeval._metadata import Metadata
from dataeval._structurers import (
    LEGACY_COLUMNS,
    ICStructurer,
    ODImageStructurer,
    RowBlock,
    StructuredData,
    reserved_block_columns,
    select_structurer,
)
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget


def _od_target(count: int = 2) -> ObjectDetectionTarget:
    return ObjectDetectionTarget(
        np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (count, 1)),
        np.arange(count),
        np.full(count, 0.5),
    )


@pytest.mark.required
class TestStructurerSelection:
    """Dispatch reads the target only; MAITE places no constraint on the item."""

    def test_str_item_with_od_target(self):
        """An item that is a path string still dispatches to object detection."""
        dataset = MockDataset(["/data/0.png", "/data/1.png"], [_od_target(), _od_target(1)])
        assert isinstance(select_structurer(dataset), ODImageStructurer)

    def test_str_item_with_od_target_structures_end_to_end(self):
        """The whole pipeline runs without ever loading the item."""
        dataset = MockDataset(["/data/0.png", "/data/1.png"], [_od_target(), _od_target(1)])
        md = Metadata(dataset)
        assert "image" == "image"
        assert "instance" == "instance"
        assert md.target_data.height == 3

    def test_arbitrary_item_object_with_od_target(self):
        """A lazy loader, a PIL handle — anything that is not an Array is fine."""
        dataset = MockDataset([object(), object()], [_od_target(), _od_target()])
        assert isinstance(select_structurer(dataset), ODImageStructurer)

    def test_mock_backed_od_dataset(self):
        """Mock auto-vivifies every attribute, including ``track_ids``."""
        target = Mock(spec=ObjectDetectionTarget)
        target.boxes = np.array([[1.0, 1.0, 2.0, 2.0]])
        target.labels = np.array([0])
        target.scores = np.array([0.5])
        dataset = MockDataset([np.zeros((3, 4, 4))], [target])

        assert isinstance(target, ObjectDetectionTarget)
        assert isinstance(select_structurer(dataset), ODImageStructurer)

    def test_arraylike_target_dispatches_to_classification(self):
        dataset = MockDataset(np.zeros((2, 3, 4, 4)), np.eye(2))
        assert isinstance(select_structurer(dataset), ICStructurer)

    def test_unrecognized_target_names_only_the_target(self):
        dataset = MockDataset([np.zeros((3, 4, 4))], ["not a target"])
        with pytest.raises(TypeError, match="Unable to infer a task from target type str"):
            select_structurer(dataset)

    def test_explicit_task_overrides_inference(self):
        dataset = MockDataset([np.zeros((3, 4, 4))], ["not a target"])
        assert isinstance(select_structurer(dataset, task="OD"), ODImageStructurer)

    def test_unknown_task_raises(self):
        dataset = MockDataset([np.zeros((3, 4, 4))], [_od_target()])
        with pytest.raises(ValueError, match="Unknown task"):
            select_structurer(dataset, task="segmentation")  # type: ignore

    def test_empty_dataset_falls_back_to_classification(self):
        """Silently here; Metadata is what warns, from the frame that knows the caller."""
        assert isinstance(select_structurer(MockDataset([], [])), ICStructurer)


@pytest.mark.required
class TestReservedBlockColumns:
    """One producer of the reserved column layout, so it cannot drift."""

    def test_unsupplied_legacy_columns_are_null(self):
        columns = reserved_block_columns("image", 2, item_index=[0, 1])
        assert set(columns) == {"level", *LEGACY_COLUMNS}
        assert columns["level"] == ["image", "image"]
        assert columns["item_index"] == [0, 1]
        assert columns["box"] == [None, None]

    def test_level_key_columns_are_omitted_unless_supplied(self):
        assert "instance_index" not in reserved_block_columns("image", 1, item_index=[0])
        assert reserved_block_columns("instance", 1, instance_index=[0])["instance_index"] == [0]

    def test_ndarrays_are_normalized_to_python_scalars(self):
        columns = reserved_block_columns("image", 2, class_label=np.array([3, 4], dtype=np.intp))
        assert columns["class_label"] == [3, 4]
        assert all(type(value) is int for value in columns["class_label"])

    def test_non_reserved_name_raises(self):
        with pytest.raises(ValueError, match="are not reserved columns"):
            reserved_block_columns("image", 1, brightness=[0.5])


@pytest.mark.required
class TestReservedColumnParity:
    """from_factors and the dataset path must agree on the reserved schema."""

    @staticmethod
    def _reserved(md: Metadata) -> list[str]:
        reserved = set(LEGACY_COLUMNS) | {"level"}
        return [name for name in md.dataframe.columns if name in reserved]

    def test_from_factors_matches_ic_dataset_schema(self):
        factors = {"weather": np.array(["sun", "rain", "sun"])}
        labels = np.array([0, 1, 0])

        from_factors = Metadata.from_factors(factors, labels)
        from_dataset = Metadata(
            MockDataset(
                np.zeros((3, 3, 4, 4)),
                np.eye(3, 2)[labels],
                [{"weather": value} for value in factors["weather"].tolist()],
            ),
        )

        # Same reserved columns, in the same order, from both constructors.
        assert self._reserved(from_factors) == self._reserved(from_dataset)
        assert set(self._reserved(from_factors)) == set(LEGACY_COLUMNS) | {"level"}

        # Target-row content agrees too, except ``score``, which raw factors never carry.
        # The two differ in structure by design: a dataset knows which items exist and so
        # carries an item level as well, while raw factor arrays are only the target rows.
        for name in ("item_index", "target_index", "class_label", "box"):
            assert from_factors.target_data[name].to_list() == from_dataset.target_data[name].to_list()

        # Each tags its rows with its own target level, which is what target_data filters on.
        assert from_factors.target_data["level"].to_list() == ["image"] * 3
        assert from_dataset.target_data["level"].to_list() == ["instance"] * 3

    def test_from_factors_at_instance_level_keeps_the_same_reserved_columns(self):
        md = Metadata.from_factors({"iou": np.array([0.1, 0.2])}, np.array([0, 1]), level="instance")
        assert self._reserved(md) == self._reserved(Metadata.from_factors({"a": np.array([1, 2])}))
        assert "instance" == "instance"


def _two_level_blocks() -> list[RowBlock]:
    """An image block of 2 and an instance block of 3, wired for propagation."""
    parents = np.array([0, 0, 1], dtype=np.intp)
    return [
        RowBlock("image", 2, reserved_block_columns("image", 2, item_index=[0, 1]), {"image": np.arange(2)}),
        RowBlock(
            "instance",
            3,
            reserved_block_columns("instance", 3, item_index=parents),
            {"image": parents, "instance": np.arange(3)},
        ),
    ]


@pytest.mark.required
class TestOneNameOneLevel:
    """A factor is one column, and a column belongs to exactly one level."""

    def test_a_name_at_two_levels_is_rejected(self):
        with pytest.raises(ValueError, match="declared at both the 'image' and 'instance' levels"):
            StructuredData(
                _two_level_blocks(),
                {"image": {"timestamp": [0.0, 1.0]}, "instance": {"timestamp": [0.0, 0.0, 1.0]}},
            )

    def test_the_error_suggests_qualified_names(self):
        with pytest.raises(ValueError, match="'image_timestamp' and 'instance_timestamp'"):
            StructuredData(
                _two_level_blocks(),
                {"image": {"timestamp": [0.0, 1.0]}, "instance": {"timestamp": [0.0, 0.0, 1.0]}},
            )

    def test_distinct_names_per_level_are_fine(self):
        data = StructuredData(
            _two_level_blocks(),
            {"image": {"weather": ["sun", "rain"]}, "instance": {"iou": [0.1, 0.2, 0.3]}},
        )
        rows = data.to_rows()
        # The image factor propagates onto instance rows; the instance factor is null above.
        assert rows["weather"] == ["sun", "rain", "sun", "sun", "rain"]
        assert rows["iou"] == [None, None, 0.1, 0.2, 0.3]

    def test_the_real_structurers_do_not_trip_it(self):
        """IC and OD both run two metadata merges that can produce the same name."""
        shared = [{"weather": "sun"}, {"weather": "rain"}, {"weather": "sun"}]
        Metadata(MockDataset(np.zeros((3, 3, 16, 16)), np.eye(3)[[0, 1, 0]], shared))._structure()
        Metadata(MockDataset(np.zeros((3, 3, 16, 16)), [_od_target(2)] * 3, shared))._structure()

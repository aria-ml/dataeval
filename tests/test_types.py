"""Tests for dataeval.types module."""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from dataeval.types import (
    DataFrameOutput,
    DictOutput,
    Evaluator,
    EvaluatorConfig,
    ExecutionMetadata,
    MappingOutput,
    ReprMixin,
    SequenceOutput,
    SourceIndex,
    Track,
)


class TestSourceIndex:
    """Tests for SourceIndex class."""

    def test_source_index_repr_item_only(self):
        """Test __repr__ with only item."""
        si = SourceIndex(item=5)
        assert repr(si) == "SourceIndex(5)"

    def test_source_index_repr_with_target(self):
        """Test __repr__ with item and target."""
        si = SourceIndex(item=5, target=2)
        assert repr(si) == "SourceIndex(5, 2)"

    def test_source_index_repr_with_channel_no_target(self):
        """Test __repr__ with item, channel but no target (line 86-87)."""
        si = SourceIndex(item=5, channel=1)
        assert repr(si) == "SourceIndex(5, None, 1)"

    def test_source_index_repr_with_target_and_channel(self):
        """Test __repr__ with item, target, and channel."""
        si = SourceIndex(item=5, target=2, channel=1)
        assert repr(si) == "SourceIndex(5, 2, 1)"

    def test_source_index_str_item_only(self):
        """Test __str__ with only item."""
        si = SourceIndex(item=5)
        assert str(si) == "5"

    def test_source_index_str_with_target(self):
        """Test __str__ with item and target."""
        si = SourceIndex(item=5, target=2)
        assert str(si) == "5/2"

    def test_source_index_str_with_channel_no_target(self):
        """Test __str__ with item, channel but no target (line 97-98)."""
        si = SourceIndex(item=5, channel=1)
        assert str(si) == "5/-/1"

    def test_source_index_str_with_target_and_channel(self):
        """Test __str__ with item, target, and channel."""
        si = SourceIndex(item=5, target=2, channel=1)
        assert str(si) == "5/2/1"

    def test_source_index_equality(self):
        """Test equality comparison (line 129-140)."""
        si1 = SourceIndex(item=5, target=2, channel=1)
        si2 = SourceIndex(item=5, target=2, channel=1)
        si3 = SourceIndex(item=5, target=2)
        si4 = SourceIndex(item=6, target=2, channel=1)

        # Test equality
        assert si1 == si2
        assert si1 != si3  # Different channel
        assert si1 != si4  # Different item

        # Test with non-SourceIndex
        assert si1 != "5/2/1"
        assert si1 != 5

    def test_from_string_item_only(self):
        """Test from_string with only item."""
        si = SourceIndex.from_string("0")
        assert si == SourceIndex(0)

    def test_from_string_with_target(self):
        """Test from_string with item and target."""
        si = SourceIndex.from_string("0/3")
        assert si == SourceIndex(0, 3)

    def test_from_string_with_none_target(self):
        """Test from_string with item, None target, and channel (line 134)."""
        si = SourceIndex.from_string("0/-/1")
        assert si == SourceIndex(0, None, 1)

    def test_from_string_with_all_fields(self):
        """Test from_string with item, target, and channel."""
        si = SourceIndex.from_string("0/3/1")
        assert si == SourceIndex(0, 3, 1)

    def test_from_string_with_none_channel(self):
        """Test from_string with item, target, and None channel (line 136)."""
        si = SourceIndex.from_string("0/3/-")
        assert si == SourceIndex(0, 3, None)

    def test_from_string_invalid_too_many_parts(self):
        """Test from_string with too many parts (line 137-138)."""
        with pytest.raises(ValueError, match="Invalid SourceIndex string format"):
            SourceIndex.from_string("0/1/2/3")


class TestMappingOutput:
    """Tests for MappingOutput class."""

    def test_getitem(self):
        """Test __getitem__ method (line 247)."""
        data = {"a": 1, "b": 2, "c": 3}
        output = MappingOutput(data)
        assert output["a"] == 1
        assert output["b"] == 2
        assert output["c"] == 3

    def test_iter(self):
        """Test __iter__ method (line 250)."""
        data = {"a": 1, "b": 2, "c": 3}
        output = MappingOutput(data)
        keys = list(output)
        assert keys == ["a", "b", "c"]


class TestSequenceOutput:
    """Tests for SequenceOutput class."""

    def test_getitem_int(self):
        """Test __getitem__ with int index (line 263)."""
        data = [10, 20, 30, 40]
        output = SequenceOutput(data)
        assert output[0] == 10
        assert output[2] == 30
        assert output[-1] == 40

    def test_getitem_slice(self):
        """Test __getitem__ with slice (line 263)."""
        data = [10, 20, 30, 40]
        output = SequenceOutput(data)
        assert output[1:3] == [20, 30]
        assert output[:2] == [10, 20]

    def test_iter(self):
        """Test __iter__ method (line 266)."""
        data = [10, 20, 30, 40]
        output = SequenceOutput(data)
        result = list(output)
        assert result == [10, 20, 30, 40]


def _make_track(track_id=1, length=3):
    """Build a contiguous Track of the given length, mirroring build_tracks."""
    return Track(
        track_id=track_id,
        boxes=np.array([[i, 0, i + 10, 10] for i in range(length)], dtype=np.float32),
        frame_indices=np.arange(length, dtype=np.int64),
        scores=np.ones(length, dtype=np.float32),
        labels=np.zeros(length, dtype=np.int64),
    )


class TestTrack:
    """Tests for the ``Track`` dataclass."""

    def test_is_a_dataclass_with_expected_fields(self):
        assert dataclasses.is_dataclass(Track)
        field_names = {f.name for f in dataclasses.fields(Track)}
        assert field_names == {"track_id", "boxes", "frame_indices", "scores", "labels"}

    def test_construction_stores_track_id(self):
        track = _make_track(track_id=42)
        assert track.track_id == 42

    def test_construction_stores_arrays(self):
        boxes = np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=np.float32)
        frames = np.array([0, 1], dtype=np.int64)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        labels = np.array([2, 2], dtype=np.int64)
        track = Track(track_id=1, boxes=boxes, frame_indices=frames, scores=scores, labels=labels)

        np.testing.assert_array_equal(track.boxes, boxes)
        np.testing.assert_array_equal(track.frame_indices, frames)
        np.testing.assert_allclose(track.scores, scores, rtol=1e-6)
        np.testing.assert_array_equal(track.labels, labels)

    def test_boxes_shape(self):
        track = _make_track(length=4)
        assert track.boxes.shape == (4, 4)
        assert track.frame_indices.shape == (4,)

    def test_single_observation_track(self):
        track = Track(
            track_id=7,
            boxes=np.array([[1, 1, 2, 2]], dtype=np.float32),
            frame_indices=np.array([5], dtype=np.int64),
            scores=np.array([1.0], dtype=np.float32),
            labels=np.array([0], dtype=np.int64),
        )
        assert track.boxes.shape == (1, 4)
        np.testing.assert_array_equal(track.frame_indices, [5])

    def test_track_with_gap_preserves_frame_indices(self):
        track = Track(
            track_id=3,
            boxes=np.array([[0, 0, 10, 10], [30, 0, 40, 10]], dtype=np.float32),
            frame_indices=np.array([0, 3], dtype=np.int64),
            scores=np.array([0.9, 0.9], dtype=np.float32),
            labels=np.array([1, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(track.frame_indices, [0, 3])
        assert track.boxes.shape == (2, 4)

    def test_field_dtypes_preserved(self):
        track = _make_track()
        assert track.boxes.dtype == np.float32
        assert track.frame_indices.dtype == np.int64
        assert track.scores.dtype == np.float32
        assert track.labels.dtype == np.int64

    def test_repr_includes_class_name(self):
        track = _make_track()
        assert "Track" in repr(track)


class TestReprMixin:
    """Tests for ReprMixin.__repr__ rendering of init params, overrides, and extras."""

    def test_repr_renders_params_overrides_and_extras(self):
        class Widget(ReprMixin):
            def __init__(self, shown, hidden, model):
                self.shown = shown
                self._hidden = hidden  # rendered via the "_name" fallback
                self.model = model

            def _repr_overrides(self):
                return {"model": "ResNet"}

            def _repr_extras(self):
                return {"fitted": True}

        assert repr(Widget(1, 2, object())) == "Widget(shown=1, hidden=2, model=ResNet, fitted=True)"

    def test_repr_defaults_are_empty(self):
        class Bare(ReprMixin):
            def __init__(self, value):
                self.value = value

        assert repr(Bare(7)) == "Bare(value=7)"


class TestEvaluatorRepr:
    """Tests for Evaluator._repr across config kinds."""

    def test_repr_with_pydantic_config(self):
        class MyEval(Evaluator):
            class Config(EvaluatorConfig):
                alpha: float = 0.5
                beta: int = 3

            alpha: float
            beta: int

            def __init__(self, alpha=None, beta=None, config=None):
                super().__init__(locals())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pydantic model_fields instance-access deprecation
            assert repr(MyEval(alpha=0.9)) == "MyEval(alpha=0.9, beta=3)"

    def test_repr_with_dataclass_config(self):
        @dataclass
        class DataclassConfig:
            x: int = 1

        class DriftLike(Evaluator):
            pass

        evaluator = DriftLike()
        evaluator._config = DataclassConfig()
        assert evaluator._repr() == "DriftLike(x=1)"
        # extras=False takes the branch that skips the extras loop
        assert evaluator._repr(extras=False) == "DriftLike(x=1)"

    def test_repr_without_config_has_no_fields(self):
        class NoConfig(Evaluator):
            pass

        assert repr(NoConfig()) == "NoConfig()"


class TestExecutionMetadata:
    """Tests for ExecutionMetadata.__repr__ and __str__."""

    def _make(self):
        return ExecutionMetadata(
            name="my_fn",
            execution_time=datetime(2020, 1, 2, 3, 4, 5),
            execution_duration=1.2345,
            arguments={"a": 1},
            state={},
            version="1.0",
        )

    def test_repr(self):
        assert repr(self._make()) == (
            "ExecutionMetadata(name='my_fn', execution_time=2020-01-02T03:04:05, "
            "execution_duration=1.2345s, version='1.0')"
        )

    def test_str(self):
        assert str(self._make()) == "my_fn (1.2345s)"


class TestDataFrameOutput:
    """Tests for the DataFrame-proxying dunder methods."""

    def _output(self):
        return DataFrameOutput(pl.DataFrame({"a": [1, 2], "b": [3, 4]}))

    def test_repr_and_str_delegate(self):
        out = self._output()
        assert repr(out) == repr(out.data())
        assert str(out) == str(out.data())

    def test_len_iter_contains_getitem(self):
        out = self._output()
        assert len(out) == 2
        assert [s.name for s in out] == ["a", "b"]
        assert "a" in out
        assert "z" not in out
        assert out["a"].to_list() == [1, 2]

    def test_getattr_delegates_public_attributes(self):
        out = self._output()
        assert out.columns == ["a", "b"]

    def test_getattr_rejects_private_names(self):
        out = self._output()
        with pytest.raises(AttributeError, match="has no attribute '_missing'"):
            out._missing  # noqa: B018


class TestDictOutput:
    """Tests for DictOutput.__repr__ and value formatting."""

    def test_repr_formats_dataframe_ndarray_and_scalars(self):
        class MyDict(DictOutput):
            def __init__(self, scalar, arr, frame):
                self.scalar = scalar
                self.arr = arr
                self.frame = frame

        out = MyDict(5, np.array([1, 2, 3]), pl.DataFrame({"a": [1]}))
        assert repr(out) == "MyDict(scalar=5, arr=ndarray(shape=(3,), dtype=int64), frame=DataFrame(shape=(1, 1)))"

    def test_str_returns_data_dict(self):
        class MyDict(DictOutput):
            def __init__(self, value):
                self.value = value

        out = MyDict(42)
        assert str(out) == str({"value": 42})

"""Tests for dataeval.types module."""

from __future__ import annotations

import copy
import dataclasses
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, get_args

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
    _factors,
)
from dataeval.types._factors import (
    _FACTOR_LEVEL_HIERARCHY,
    FactorLevel,
    FactorLevelSchema,
    _closure,
    _relink,
    _validate_acyclic,
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


@pytest.mark.required
class TestLevelSchema:
    def test_of_relinks_parents_around_omitted_levels(self):
        schema = FactorLevelSchema.of("instance")
        assert schema.levels == ("instance",)
        assert schema.parents_of("instance") == ()

    def test_parents_view_is_read_only(self):
        schema = FactorLevelSchema.of("image", "instance")
        assert schema.parents == {"image": (), "instance": ("image",)}
        with pytest.raises(TypeError):
            schema.parents["instance"] = ("instance",)  # type: ignore[index]

    def test_rejects_unknown_level(self):
        with pytest.raises(ValueError, match="Unknown level"):
            FactorLevelSchema.of("sequence")  # type: ignore[arg-type]

    def test_rejects_repeated_level(self):
        with pytest.raises(ValueError, match="appear more than once"):
            FactorLevelSchema(("image", "image"), {"image": ()})

    def test_rejects_dangling_parent(self):
        with pytest.raises(ValueError, match="not part of this schema"):
            FactorLevelSchema(("instance",), {"instance": ("image",)})

    def test_rejects_a_bare_string_of_parents(self):
        """``str`` is a ``Sequence[str]``, so this would otherwise become 5 parents."""
        with pytest.raises(TypeError, match="not the bare string"):
            FactorLevelSchema(("image", "instance"), {"instance": "image"})  # type: ignore[dict-item]

    def test_rejects_a_repeated_parent(self):
        with pytest.raises(ValueError, match="same parent more than once"):
            FactorLevelSchema(("image", "instance"), {"instance": ("image", "image")})

    def test_deepcopy_round_trips(self):
        """The schema travels inside every Metadata, so it has to be copyable."""
        schema = FactorLevelSchema.of("image", "instance")
        clone = copy.deepcopy(schema)
        assert clone == schema
        assert clone.parents == schema.parents


# A level graph with a genuine diamond, of the shape multi-object tracking needs: a
# per-frame detection sits inside both a frame and a track, which are themselves
# siblings under a sequence. The shipped vocabulary has only two levels and so cannot
# express one, but the machinery has to handle it before anything declares it.
#
# Typed ``Any`` throughout this section: the names deliberately sit outside the
# ``FactorLevel`` literal, which is exactly what makes the shape untypable today.
DIAMOND: Any = {
    "sequence": (),
    "image": ("sequence",),
    "track": ("sequence",),
    "instance": ("image", "track"),
}

# Re-bound through Any so the checker does not reject every synthetic level name below.
# Annotating each call instead would put a type-ignore on nearly every line of these
# three classes, which buries what they are actually asserting.
closure: Any = _closure
relink: Any = _relink
validate_acyclic: Any = _validate_acyclic
schema_of: Any = FactorLevelSchema.of


@pytest.mark.required
class TestGraphTraversal:
    """The graph walks, exercised on a shape the current vocabulary cannot declare."""

    def test_closure_reports_both_branches_nearest_first(self):
        assert closure("instance", DIAMOND) == ("image", "track", "sequence")

    def test_closure_reports_a_shared_ancestor_once(self):
        """`sequence` is reachable by two paths and must not appear twice."""
        assert closure("instance", DIAMOND).count("sequence") == 1

    def test_closure_of_a_root_is_empty(self):
        assert closure("sequence", DIAMOND) == ()

    def test_relink_keeps_both_branches(self):
        assert relink("instance", {"image", "track", "instance"}, DIAMOND) == ("image", "track")

    def test_relink_collapses_a_diamond_to_its_meet(self):
        """Dropping both middle levels splices the edges rather than severing them."""
        assert relink("instance", {"sequence", "instance"}, DIAMOND) == ("sequence",)

    def test_relink_drops_a_branch_that_is_entirely_absent(self):
        """Projecting the tracking graph onto plain object detection."""
        assert relink("instance", {"image", "instance"}, DIAMOND) == ("image",)

    def test_relink_of_a_root_is_empty(self):
        assert relink("sequence", {"sequence", "instance"}, DIAMOND) == ()


@pytest.mark.required
class TestAcyclicValidation:
    def test_a_diamond_is_not_a_cycle(self):
        validate_acyclic(tuple(DIAMOND), DIAMOND)

    def test_a_cycle_is_rejected(self):
        cyclic = {"image": ("instance",), "instance": ("image",)}
        with pytest.raises(ValueError, match="form a cycle"):
            validate_acyclic(("image", "instance"), cyclic)

    def test_a_level_parented_to_itself_is_rejected(self):
        with pytest.raises(ValueError, match="form a cycle"):
            validate_acyclic(("image",), {"image": ("image",)})


@pytest.mark.required
class TestMultiParentSchema:
    """A whole schema over the diamond, with the vocabulary patched to allow it."""

    @pytest.fixture
    def schema(self, monkeypatch) -> Any:
        monkeypatch.setattr(_factors, "_FACTOR_LEVEL_HIERARCHY", DIAMOND)
        return schema_of("sequence", "image", "track", "instance")

    def test_instance_reports_both_parents(self, schema: Any):
        assert schema.parents_of("instance") == ("image", "track")

    def test_ancestors_span_both_branches(self, schema: Any):
        assert schema.ancestors("instance") == ("image", "track", "sequence")

    def test_factors_propagate_down_every_edge(self, schema: Any):
        """The predicate _build_factors uses: a track factor must reach instance rows."""
        for source in ("image", "track", "sequence"):
            assert schema.propagates_to(source, "instance")

    def test_siblings_do_not_propagate_to_each_other(self, schema: Any):
        assert not schema.propagates_to("track", "image")
        assert not schema.propagates_to("image", "track")

    def test_descendants_follow_both_branches(self, schema: Any):
        assert schema.descendants("sequence") == ("image", "track", "instance")
        assert schema.descendants("track") == ("instance",)

    def test_highest_of_incomparable_levels_is_schema_order(self, schema: Any):
        """No graph answer exists for two siblings, so declaration order decides."""
        assert schema.highest(["track", "image"]) == "image"
        assert schema.highest(["instance", "track"]) == "track"


@pytest.mark.required
class TestLevelSchemaValidate:
    """A schema knows only levels that exist; retired spellings never reach it."""

    def test_real_level_resolves_without_warning(self, recwarn):
        schema = FactorLevelSchema.of("image", "instance")
        assert schema.validate("instance") == "instance"
        assert not recwarn.list

    def test_unknown_level_raises(self):
        schema = FactorLevelSchema.of("image", "instance")
        with pytest.raises(ValueError, match="Unknown level 'nope'"):
            schema.validate("nope")  # type: ignore[arg-type]

    def test_retired_target_is_not_a_level(self):
        """``"target"`` is translated by Metadata, not by the schema."""
        schema = FactorLevelSchema.of("image", "instance")
        with pytest.raises(ValueError, match="Unknown level 'target'"):
            schema.validate("target")  # type: ignore[arg-type]


@pytest.mark.required
class TestLevelHierarchy:
    """LEVEL_HIERARCHY is the sole declaration of the vocabulary and its edges."""

    def test_ordered_coarsest_first(self):
        assert tuple(_FACTOR_LEVEL_HIERARCHY) == ("image", "instance")

    def test_instance_hangs_off_image(self):
        """Both tasks put their targets here, so it has exactly one parent."""
        assert _FACTOR_LEVEL_HIERARCHY["instance"] == ("image",)

    def test_level_literal_matches_the_hierarchy(self):
        """The one declaration that cannot be derived, so it is asserted instead.

        A Literal alias is static and cannot be computed from a runtime mapping, so
        adding a level means editing both. This is what catches forgetting one.
        """
        assert set(get_args(FactorLevel)) == set(_FACTOR_LEVEL_HIERARCHY)

    def test_every_parent_is_itself_a_level(self):
        named = {parent for edges in _FACTOR_LEVEL_HIERARCHY.values() for parent in edges}
        assert named <= set(_FACTOR_LEVEL_HIERARCHY)

    def test_parents_are_declared_before_their_children(self):
        """Canonical order must be a topological order of the edges, not just coarse-first.

        Schema order is what :meth:`FactorLevelSchema.highest` resolves ties on, and the graph
        is a partial order, so the declaration has to keep every level after its parents
        for that tie-break to never contradict the ancestry.
        """
        ordered = list(_FACTOR_LEVEL_HIERARCHY)
        for index, level in enumerate(ordered):
            for parent in _FACTOR_LEVEL_HIERARCHY[level]:
                assert ordered.index(parent) < index

    def test_no_level_is_its_own_parent(self):
        for level, edges in _FACTOR_LEVEL_HIERARCHY.items():
            assert level not in edges

    def test_hierarchy_is_read_only(self):
        with pytest.raises(TypeError):
            _FACTOR_LEVEL_HIERARCHY["frame"] = ()  # type: ignore[index]

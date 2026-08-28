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
from dataeval.types._target import detection_score, own_class_scores
from dataeval.types._track import frame_size


class TestSourceIndex:
    """Tests for SourceIndex class."""

    def test_source_index_repr_item_only(self):
        """Test __repr__ with only item."""
        si = SourceIndex(item=5)
        assert repr(si) == "SourceIndex(5)"

    def test_source_index_repr_with_target(self):
        """Test __repr__ with item and key."""
        si = SourceIndex(item=5, key=2)
        assert repr(si) == "SourceIndex(5, 2)"

    def test_source_index_repr_with_explicit_none_target(self):
        """An explicit null target renders the same as an omitted one."""
        si = SourceIndex(item=5, target=None)
        assert repr(si) == "SourceIndex(5)"

    def test_source_index_str_item_only(self):
        """Test __str__ with only item."""
        si = SourceIndex(item=5)
        assert str(si) == "5"

    def test_source_index_str_with_target(self):
        """Test __str__ with item and key."""
        si = SourceIndex(item=5, key=2)
        assert str(si) == "5/2"

    def test_source_index_equality(self):
        """Test equality comparison."""
        si1 = SourceIndex(item=5, key=2)
        si2 = SourceIndex(item=5, key=2)
        si3 = SourceIndex(item=5)
        si4 = SourceIndex(item=6, key=2)

        # Test equality
        assert si1 == si2
        assert si1 != si3  # Different target
        assert si1 != si4  # Different item

        # Test with non-SourceIndex
        assert si1 != "5/2"
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
        """Test from_string with an explicit null target."""
        si = SourceIndex.from_string("0/-")
        assert si == SourceIndex(0, None)

    def test_from_string_invalid_too_many_parts(self):
        """Test from_string with too many parts."""
        with pytest.raises(ValueError, match="Invalid SourceIndex string format"):
            SourceIndex.from_string("0/1/2/3")


@pytest.mark.required
class TestSourceIndexIsAnAddress:
    """`SourceIndex` names one row at one level, and `target` is `key`'s retired spelling."""

    def test_the_target_property_reads_the_key(self):
        with pytest.warns(DeprecationWarning, match="retired spelling of SourceIndex.key"):
            assert SourceIndex(3, 7).target == 7

    def test_reading_target_off_an_unkeyed_address_warns_too(self):
        """It is the spelling that is retired, not the value it happens to return."""
        with pytest.warns(DeprecationWarning, match="removed in v1.3.0"):
            assert SourceIndex(3).target is None

    def test_target_constructs_what_key_constructs(self):
        with pytest.warns(DeprecationWarning, match=r"SourceIndex\(target=\.\.\.\)"):
            aliased = SourceIndex(item=3, target=7)
        assert aliased == SourceIndex(item=3, key=7)

    def test_an_explicit_null_target_does_not_warn(self):
        """Indistinguishable from not passing it, so warning would be noise."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert SourceIndex(item=3, target=None) == SourceIndex(3)

    def test_both_spellings_together_are_rejected(self):
        with pytest.raises(TypeError, match="pass one of key= or target="):
            SourceIndex(3, key=7, target=7)

    def test_the_named_tuple_machinery_still_works(self):
        si = SourceIndex(3, 7)
        assert SourceIndex._fields == ("item", "key", "level")
        assert si._asdict() == {"item": 3, "key": 7, "level": None}
        assert si._replace(level="instance") == SourceIndex(3, 7, "instance")
        assert SourceIndex._make((3, 7, None)) == si
        assert list(si) == [3, 7, None]
        assert copy.deepcopy(si) == si
        assert {si: "x"}[SourceIndex(3, 7)] == "x"

    def test_it_survives_a_pickle_round_trip(self):
        import pickle

        si = SourceIndex(3, 12, "unit")
        assert pickle.loads(pickle.dumps(si)) == si

    @pytest.mark.parametrize(
        ("address", "item_level", "expected"),
        [
            (SourceIndex(3), "unit", "unit"),
            (SourceIndex(3), "sequence", "sequence"),
            (SourceIndex(3, 7), "unit", "instance"),
            (SourceIndex(3, 7), "sequence", "instance"),
            (SourceIndex(3, 12, "unit"), "sequence", "unit"),
            (SourceIndex(3, 5, "track"), "sequence", "track"),
        ],
    )
    def test_an_unstated_level_resolves_against_the_task(self, address, item_level, expected):
        """`None` is the task-generic level, not an unknown one."""
        assert address.resolve(item_level, "instance") == expected

    def test_a_stated_level_wins_over_the_default(self):
        assert SourceIndex(3, None, "sequence").resolve("unit", "instance") == "sequence"

    def test_two_spellings_of_one_address_are_not_equal(self):
        """Why producers emit the minimal spelling: these hash apart.

        `resolve` says they name the same row, but a mapping keyed on addresses cannot
        know that, and `Outliers.issues` is such a mapping.
        """
        implicit, explicit = SourceIndex(3, 7), SourceIndex(3, 7, "instance")
        assert implicit != explicit
        assert implicit.resolve("unit", "instance") == explicit.resolve("unit", "instance")


@pytest.mark.required
class TestSourceIndexWireFormat:
    """The string form carries the level and round-trips through `from_string`."""

    @pytest.mark.parametrize("spelling", ["3", "3/7", "3/12/unit", "3/5/track", "3/-/sequence"])
    def test_round_trip(self, spelling):
        assert str(SourceIndex.from_string(spelling)) == spelling

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("3", SourceIndex(3)),
            ("3/7", SourceIndex(3, 7)),
            ("3/-", SourceIndex(3)),
            ("3/12/unit", SourceIndex(3, 12, "unit")),
            ("3/-/sequence", SourceIndex(3, None, "sequence")),
        ],
    )
    def test_what_each_spelling_names(self, spelling, expected):
        assert SourceIndex.from_string(spelling) == expected

    def test_an_unstated_level_is_omitted_rather_than_marked(self):
        """`-` is needed in the key slot only, which is the one that has to be held open."""
        assert str(SourceIndex(3, 7)) == "3/7"
        assert str(SourceIndex(3)) == "3"

    def test_an_empty_string_names_this_type_in_its_rejection(self):
        """`str.split` never returns nothing, so emptiness arrives as an empty first part."""
        with pytest.raises(ValueError, match="Invalid SourceIndex string format"):
            SourceIndex.from_string("")

    def test_a_level_that_is_not_one_is_rejected_on_construction(self):
        """The v1.1 third positional argument was a channel index, and is silent otherwise.

        `type: ignore` because a checked caller is told statically, which is the other half
        of the answer; the runtime check is for the callers pyright never sees.
        """
        with pytest.raises(ValueError, match="level=2 is not a level"):
            SourceIndex(0, 1, 2)  # type: ignore[arg-type]

    def test_the_construction_rejection_says_what_the_slot_is_now(self):
        with pytest.raises(ValueError, match="was a channel index before v1.2"):
            SourceIndex(0, 1, 2)  # type: ignore[arg-type]

    def test_a_level_that_is_not_one_is_rejected(self):
        with pytest.raises(ValueError, match="'frame' is not a level"):
            SourceIndex.from_string("3/1/frame")

    def test_the_rejection_lists_the_levels(self):
        with pytest.raises(ValueError, match="sequence, unit, track, instance"):
            SourceIndex.from_string("3/1/frame")

    @pytest.mark.parametrize(
        ("address", "expected"),
        [
            (SourceIndex(3), "SourceIndex(3)"),
            (SourceIndex(3, 7), "SourceIndex(3, 7)"),
            (SourceIndex(3, 12, "unit"), "SourceIndex(3, 12, 'unit')"),
            (SourceIndex(3, None, "sequence"), "SourceIndex(3, None, 'sequence')"),
        ],
    )
    def test_repr_omits_trailing_unstated_fields(self, address, expected):
        assert repr(address) == expected


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


@pytest.mark.required
class TestFrameSize:
    """Tests for ``frame_size``, the single reader of a video frame's dimensions."""

    class _Frame:
        def __init__(self, pixels):
            self.pixels = pixels

    def test_reads_width_and_height_from_chw_pixels(self):
        # MAITE frames are (C, H, W), so the trailing axes are height then width.
        frame = self._Frame(np.zeros((3, 24, 32), dtype=np.uint8))
        assert frame_size(frame) == (32, 24)

    def test_accepts_a_bare_two_dimensional_frame(self):
        frame = self._Frame(np.zeros((24, 32), dtype=np.uint8))
        assert frame_size(frame) == (32, 24)

    def test_missing_frame_answers_none(self):
        # `track_stats` passes `next(iter(stream), None)` straight through, so an
        # empty video stream reaches this as None rather than a frame.
        assert frame_size(None) == (None, None)

    def test_frame_without_pixels_answers_none(self):
        # Dispatch does not require the full VideoFrame protocol, so a duck-typed
        # stream is free to carry no pixels at all.
        assert frame_size(object()) == (None, None)

    def test_pixels_without_two_axes_answer_none(self):
        assert frame_size(self._Frame(np.zeros(32, dtype=np.uint8))) == (None, None)


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
        schema = FactorLevelSchema.of("unit", "instance")
        assert schema.parents == {"unit": (), "instance": ("unit",)}
        with pytest.raises(TypeError):
            schema.parents["instance"] = ("instance",)  # type: ignore[index]

    def test_rejects_unknown_level(self):
        # A plausible misspelling of a real level: the frame level is called "unit".
        with pytest.raises(ValueError, match="Unknown level"):
            FactorLevelSchema.of("frame")  # type: ignore[arg-type]

    def test_rejects_repeated_level(self):
        with pytest.raises(ValueError, match="appear more than once"):
            FactorLevelSchema(("unit", "unit"), {"unit": ()})

    def test_rejects_dangling_parent(self):
        with pytest.raises(ValueError, match="not part of this schema"):
            FactorLevelSchema(("instance",), {"instance": ("unit",)})

    def test_rejects_a_bare_string_of_parents(self):
        """``str`` is a ``Sequence[str]``, so this would otherwise become 4 parents."""
        with pytest.raises(TypeError, match="not the bare string"):
            FactorLevelSchema(("unit", "instance"), {"instance": "unit"})  # type: ignore[dict-item]

    def test_rejects_a_repeated_parent(self):
        with pytest.raises(ValueError, match="same parent more than once"):
            FactorLevelSchema(("unit", "instance"), {"instance": ("unit", "unit")})

    def test_deepcopy_round_trips(self):
        """The schema travels inside every Metadata, so it has to be copyable."""
        schema = FactorLevelSchema.of("unit", "instance")
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
    "unit": ("sequence",),
    "track": ("sequence",),
    "instance": ("unit", "track"),
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
        assert closure("instance", DIAMOND) == ("unit", "track", "sequence")

    def test_closure_reports_a_shared_ancestor_once(self):
        """`sequence` is reachable by two paths and must not appear twice."""
        assert closure("instance", DIAMOND).count("sequence") == 1

    def test_closure_of_a_root_is_empty(self):
        assert closure("sequence", DIAMOND) == ()

    def test_relink_keeps_both_branches(self):
        assert relink("instance", {"unit", "track", "instance"}, DIAMOND) == ("unit", "track")

    def test_relink_collapses_a_diamond_to_its_meet(self):
        """Dropping both middle levels splices the edges rather than severing them."""
        assert relink("instance", {"sequence", "instance"}, DIAMOND) == ("sequence",)

    def test_relink_drops_a_branch_that_is_entirely_absent(self):
        """Projecting the tracking graph onto plain object detection."""
        assert relink("instance", {"unit", "instance"}, DIAMOND) == ("unit",)

    def test_relink_of_a_root_is_empty(self):
        assert relink("sequence", {"sequence", "instance"}, DIAMOND) == ()


@pytest.mark.required
class TestAcyclicValidation:
    def test_a_diamond_is_not_a_cycle(self):
        validate_acyclic(tuple(DIAMOND), DIAMOND)

    def test_a_cycle_is_rejected(self):
        cyclic = {"unit": ("instance",), "instance": ("unit",)}
        with pytest.raises(ValueError, match="form a cycle"):
            validate_acyclic(("unit", "instance"), cyclic)

    def test_a_level_parented_to_itself_is_rejected(self):
        with pytest.raises(ValueError, match="form a cycle"):
            validate_acyclic(("unit",), {"unit": ("unit",)})


@pytest.mark.required
class TestMultiParentSchema:
    """A whole schema over the diamond, with the vocabulary patched to allow it."""

    @pytest.fixture
    def schema(self, monkeypatch) -> Any:
        monkeypatch.setattr(_factors, "_FACTOR_LEVEL_HIERARCHY", DIAMOND)
        return schema_of("sequence", "unit", "track", "instance")

    def test_instance_reports_both_parents(self, schema: Any):
        assert schema.parents_of("instance") == ("unit", "track")

    def test_ancestors_span_both_branches(self, schema: Any):
        assert schema.ancestors("instance") == ("unit", "track", "sequence")

    def test_factors_propagate_down_every_edge(self, schema: Any):
        """The predicate _build_factors uses: a track factor must reach instance rows."""
        for source in ("unit", "track", "sequence"):
            assert schema.propagates_to(source, "instance")

    def test_siblings_do_not_propagate_to_each_other(self, schema: Any):
        assert not schema.propagates_to("track", "unit")
        assert not schema.propagates_to("unit", "track")

    def test_descendants_follow_both_branches(self, schema: Any):
        assert schema.descendants("sequence") == ("unit", "track", "instance")
        assert schema.descendants("track") == ("instance",)

    def test_highest_of_incomparable_levels_is_schema_order(self, schema: Any):
        """No graph answer exists for two siblings, so declaration order decides."""
        assert schema.highest(["track", "unit"]) == "unit"
        assert schema.highest(["instance", "track"]) == "track"


@pytest.mark.required
class TestLevelSchemaValidate:
    """A schema knows only levels that exist; retired spellings never reach it."""

    def test_real_level_resolves_without_warning(self, recwarn):
        schema = FactorLevelSchema.of("unit", "instance")
        assert schema.validate("instance") == "instance"
        assert not recwarn.list

    def test_unknown_level_raises(self):
        schema = FactorLevelSchema.of("unit", "instance")
        with pytest.raises(ValueError, match="Unknown level 'nope'"):
            schema.validate("nope")  # type: ignore[arg-type]

    def test_retired_target_is_not_a_level(self):
        """``"target"`` is translated by Metadata, not by the schema."""
        schema = FactorLevelSchema.of("unit", "instance")
        with pytest.raises(ValueError, match="Unknown level 'target'"):
            schema.validate("target")  # type: ignore[arg-type]


@pytest.mark.required
class TestLevelHierarchy:
    """LEVEL_HIERARCHY is the sole declaration of the vocabulary and its edges."""

    def test_ordered_coarsest_first(self):
        assert tuple(_FACTOR_LEVEL_HIERARCHY) == ("sequence", "unit", "track", "instance")

    def test_instance_hangs_off_both_unit_and_track(self):
        """A detection is one observation: of a track, in a frame. Hence the diamond."""
        assert _FACTOR_LEVEL_HIERARCHY["instance"] == ("unit", "track")

    def test_unit_and_track_hang_off_sequence(self):
        """Both sit inside a video; an image-item task omits sequence and re-roots."""
        assert _FACTOR_LEVEL_HIERARCHY["unit"] == ("sequence",)
        assert _FACTOR_LEVEL_HIERARCHY["track"] == ("sequence",)
        assert FactorLevelSchema.of("unit", "instance").parents_of("unit") == ()

    def test_sequence_is_the_only_root(self):
        roots = [level for level, parents in _FACTOR_LEVEL_HIERARCHY.items() if not parents]
        assert roots == ["sequence"]


@pytest.mark.required
class TestLevelDiamond:
    """``instance`` has two parents, which is the case the DAG machinery exists for."""

    def setup_method(self):
        self.schema = FactorLevelSchema.of("sequence", "unit", "track", "instance")

    def test_ancestors_report_the_meeting_level_once(self):
        """Breadth-first, so the branches come before the level where they part."""
        assert self.schema.ancestors("instance") == ("unit", "track", "sequence")

    def test_siblings_do_not_propagate_to_each_other(self):
        assert self.schema.propagates_to("unit", "track") is False
        assert self.schema.propagates_to("track", "unit") is False

    def test_both_branches_reach_the_label_level(self):
        assert self.schema.propagates_to("unit", "instance") is True
        assert self.schema.propagates_to("track", "instance") is True

    def test_the_root_reaches_everything(self):
        assert all(self.schema.propagates_to("sequence", level) for level in self.schema.levels)

    def test_omitting_one_branch_leaves_the_other_intact(self):
        """An image-based task keeps neither sequence nor track, and still sees one parent."""
        assert FactorLevelSchema.of("unit", "instance").parents_of("instance") == ("unit",)

    def test_omitting_the_level_where_branches_part_keeps_both(self):
        """Dropping sequence re-roots unit and track separately rather than severing them."""
        schema = FactorLevelSchema.of("unit", "track", "instance")
        assert schema.parents_of("instance") == ("unit", "track")
        assert schema.parents_of("unit") == ()
        assert schema.parents_of("track") == ()

    def test_highest_tie_breaks_on_schema_order_for_incomparable_levels(self):
        """unit and track are genuinely incomparable, so declaration order decides."""
        assert self.schema.highest(["track", "unit"]) == "unit"
        assert self.schema.highest(["instance", "track"]) == "track"

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


@pytest.mark.required
class TestLevelPaths:
    """Which route reaches an ancestor is a property of the graph, not of a structurer.

    A chain has one route and never needs asking. A diamond has two, and they disagree
    exactly where one branch stops short — an untracked detection reaches its sequence
    through its frame but not through a track. Before this the answer lived in the order
    a structurer merged two dictionaries.
    """

    def setup_method(self):
        self.schema = FactorLevelSchema.of("sequence", "unit", "track", "instance")

    def test_one_step_to_a_parent(self):
        assert self.schema.paths("instance", "unit") == (("unit",),)

    def test_a_chain_has_a_single_route(self):
        assert self.schema.paths("unit", "sequence") == (("sequence",),)

    def test_a_diamond_reports_both_branches(self):
        assert self.schema.paths("instance", "sequence") == (("unit", "sequence"), ("track", "sequence"))

    def test_routes_follow_canonical_parent_order(self):
        """The unit branch comes first because parents_of does, which is the precedence
        the structurers already had by merging the unit branch last."""
        routes = self.schema.paths("instance", "sequence")
        assert routes[0][0] == self.schema.parents_of("instance")[0] == "unit"

    def test_a_level_that_is_not_above_has_no_route(self):
        assert self.schema.paths("unit", "track") == ()
        assert self.schema.paths("sequence", "instance") == ()

    def test_a_level_has_no_route_to_itself(self):
        assert self.schema.paths("unit", "unit") == ()

    def test_an_image_task_collapses_the_diamond_to_one_route(self):
        assert FactorLevelSchema.of("unit", "instance").paths("instance", "unit") == (("unit",),)

    def test_unknown_levels_are_rejected(self):
        with pytest.raises(ValueError, match="Unknown level"):
            self.schema.paths("instance", "nope")  # type: ignore[arg-type]

    def test_every_ancestor_is_reachable_by_at_least_one_route(self):
        for level in self.schema.levels:
            for ancestor in self.schema.ancestors(level):
                assert self.schema.paths(level, ancestor), f"{level} -> {ancestor}"


@pytest.mark.required
class TestFactorLevelSchemaIdentity:
    """Equality and hashing are by (levels, parents), so schemas work as dict keys."""

    @staticmethod
    def _schema() -> FactorLevelSchema:
        return FactorLevelSchema(("unit", "instance"), {"instance": ("unit",)})

    def test_equal_schemas_hash_alike(self):
        assert self._schema() == self._schema()
        assert hash(self._schema()) == hash(self._schema())

    def test_usable_as_a_mapping_key(self):
        assert {self._schema(): "value"}[self._schema()] == "value"

    def test_comparison_with_a_foreign_type_defers(self):
        """Returning NotImplemented lets Python fall back rather than claiming inequality."""
        assert self._schema().__eq__(object()) is NotImplemented
        assert self._schema() != object()

    def test_highest_of_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="empty collection"):
            self._schema().highest([])


@pytest.mark.required
class TestOwnClassScores:
    """Reducing either MAITE score layout to one confidence per detection."""

    @staticmethod
    def _target(scores: Any) -> Any:
        return type("T", (), {"scores": scores})()

    def test_per_box_scores_pass_through(self):
        read = own_class_scores(np.array([0.4, 0.6], dtype=np.float32), np.array([1, 0]))
        np.testing.assert_allclose(read, [0.4, 0.6])

    def test_per_class_scores_read_the_own_class_column(self):
        # deliberately not the row's maximum: the question a score answers is "how
        # confident in what this box is labelled", which class_label already names
        read = own_class_scores(np.array([[0.2, 0.7], [0.9, 0.1]], dtype=np.float32), np.array([0, 1]))
        np.testing.assert_allclose(read, [0.2, 0.1])

    def test_label_with_no_column_of_its_own_is_unreadable(self):
        # the collapsed-column construction, and a target relabeled into a wider
        # vocabulary: neither can say what this detection scored
        read = own_class_scores(np.array([[0.2, 0.7]], dtype=np.float32), np.array([5]))
        assert np.isnan(read[0])

    def test_labels_are_authoritative_on_the_count(self):
        short = own_class_scores(np.array([0.5], dtype=np.float32), np.array([0, 1, 2]))
        assert len(short) == 3
        np.testing.assert_allclose(short[:1], [0.5])
        assert np.isnan(short[1:]).all()

        long = own_class_scores(np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0]))
        np.testing.assert_allclose(long, [0.1])

    def test_a_target_carrying_no_scores_reads_as_unconfident_not_zero(self):
        read = own_class_scores(None, np.array([0, 1]))
        assert np.isnan(read).all()

    def test_no_detections_reads_empty(self):
        assert len(own_class_scores(np.array([[0.5]]), np.array([], dtype=np.intp))) == 0

    def test_an_unrecognized_layout_is_unreadable(self):
        read = own_class_scores(np.zeros((1, 2, 2), dtype=np.float32), np.array([0]))
        assert np.isnan(read).all()

    def test_detection_score_reads_one_the_same_way(self):
        target = self._target(np.array([[0.2, 0.7], [0.9, 0.1]], dtype=np.float32))
        assert detection_score(target, 0, 0) == pytest.approx(0.2)
        assert detection_score(target, 1, 1) == pytest.approx(0.1)

    def test_detection_score_answers_none_where_there_is_nothing_to_read(self):
        assert detection_score(self._target(None), 0, 0) is None
        # a label the score array has no column for, rather than an IndexError
        assert detection_score(self._target(np.array([[0.2, 0.7]])), 0, 5) is None
        # a detection past the end of the array
        assert detection_score(self._target(np.array([[0.2, 0.7]])), 3, 0) is None
        assert detection_score(self._target(np.zeros((1, 2, 2))), 0, 0) is None
        # a layout with no length at all, which must not raise on its way to None
        assert detection_score(self._target(np.float32(0.9)), 0, 0) is None
        # a negative index names no detection; wrapping onto a real box would be worse
        assert detection_score(self._target(np.array([0.1, 0.2, 0.3])), -2, 0) is None

    def test_detection_score_reads_the_value_the_target_holds(self):
        """The frame's column is float32 and nullable; this answer is neither."""
        # not rounded to the column's dtype: a target holding exactly 0.1 answers 0.1
        assert detection_score(self._target(np.array([[0.1, 0.2]], dtype=np.float64)), 0, 0) == 0.1
        # a score the target genuinely recorded as nan is not "carries no score"
        recorded = detection_score(self._target(np.array([[np.nan, 0.5]])), 0, 0)
        assert recorded is not None
        assert np.isnan(recorded)

"""Tests for the MAITE dataset shape validator.

Covers :func:`dataeval.utils.data.validate_dataset`, the
``@requires_maite_dataset`` decorator, and integration with the
public entry points it guards (:class:`Embeddings`, :class:`Metadata`,
:class:`View` + :class:`ClassFilter`, ``split_dataset``, ``unzip_dataset``).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from dataeval import Metadata
from dataeval._embeddings import Embeddings
from dataeval.data import ClassFilter, Limit, Operation, Shuffle, View, split_dataset, unzip_dataset
from dataeval.exceptions import MaiteShapeError
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.utils import data
from dataeval.utils.data import requires_maite_dataset, validate_dataset

# ---------- fixtures ----------


class _ImageOnly:
    """Bare-image dataset: dataset[i] is a (3, 8, 8) array."""

    def __init__(self, n: int = 4) -> None:
        self.data = np.zeros((n, 3, 8, 8), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.data[i]


class _ICDataset:
    """Image classification MAITE dataset."""

    metadata = DatasetMetadata(id="ic_test")

    def __init__(self, n: int = 4, k: int = 3) -> None:
        self.data = np.zeros((n, 3, 8, 8), dtype=np.float32)
        self.y = np.eye(k)[np.arange(n) % k].astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
        return self.data[i], self.y[i], DatumMetadata(id=i)


class _ODTarget:
    def __init__(self) -> None:
        self.boxes = np.zeros((1, 4), dtype=np.float32)
        self.labels = np.array([0], dtype=np.intp)
        self.scores = np.array([[1.0]], dtype=np.float32)


class _ODDataset:
    metadata: DatasetMetadata = DatasetMetadata(id="od_test")

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[np.ndarray, _ODTarget, DatumMetadata]:
        return np.zeros((3, 8, 8), dtype=np.float32), _ODTarget(), DatumMetadata(id=i)


class _SegTarget:
    def __init__(self) -> None:
        self.mask = np.zeros((1, 8, 8), dtype=np.float32)
        self.labels = np.array([0], dtype=np.intp)
        self.scores = np.array([[1.0]], dtype=np.float32)


class _SegDataset:
    metadata: DatasetMetadata = DatasetMetadata(id="seg_test")

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[np.ndarray, _SegTarget, DatumMetadata]:
        return np.zeros((3, 8, 8), dtype=np.float32), _SegTarget(), DatumMetadata(id=i)


class _FrameTarget:
    def __init__(self) -> None:
        self.boxes = np.zeros((1, 4), dtype=np.float32)
        self.labels = np.array([0], dtype=np.intp)
        self.scores = np.array([1.0], dtype=np.float32)
        self.track_ids = np.array([0], dtype=np.intp)


class _MOTTarget:
    def __init__(self, frames: int = 2) -> None:
        self.frame_tracks = [_FrameTarget() for _ in range(frames)]


class _MOTDataset:
    metadata: DatasetMetadata = DatasetMetadata(id="mot_test")

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[list, _MOTTarget, DatumMetadata]:
        return [np.zeros((3, 8, 8), dtype=np.float32)] * 2, _MOTTarget(), DatumMetadata(id=i)


# ---------- validate_dataset: happy paths ----------


class TestValidateDatasetHappy:
    def test_image_only_accepts_bare(self) -> None:
        assert validate_dataset(_ImageOnly(), expected="image_only") == "image_only"

    def test_image_only_accepts_tuple(self) -> None:
        assert validate_dataset(_ICDataset(), expected="image_only") == "image_only"

    def test_classification(self) -> None:
        assert validate_dataset(_ICDataset(), expected="classification") == "classification"

    def test_object_detection(self) -> None:
        assert validate_dataset(_ODDataset(), expected="object_detection") == "object_detection"

    def test_segmentation(self) -> None:
        assert validate_dataset(_SegDataset(), expected="segmentation") == "segmentation"

    def test_multiobject_tracking(self) -> None:
        assert validate_dataset(_MOTDataset(), expected="multiobject_tracking") == "multiobject_tracking"

    def test_any_target_resolves_concrete_kind(self) -> None:
        assert validate_dataset(_ICDataset(), expected="any_target") == "classification"
        assert validate_dataset(_ODDataset(), expected="any_target") == "object_detection"
        assert validate_dataset(_SegDataset(), expected="any_target") == "segmentation"
        assert validate_dataset(_MOTDataset(), expected="any_target") == "multiobject_tracking"

    def test_detection_target_is_not_taken_for_tracking(self) -> None:
        """Tracking is probed first, so it has to answer False for every other kind."""
        assert validate_dataset(_ODDataset(), expected="any_target") == "object_detection"
        with pytest.raises(MaiteShapeError, match="MultiobjectTrackingTarget"):
            validate_dataset(_ODDataset(), expected="multiobject_tracking")


# ---------- validate_dataset: failure modes ----------


class TestValidateDatasetFailures:
    def test_bare_image_fails_target_kind(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            validate_dataset(_ImageOnly(), expected="classification")

    def test_bare_image_fails_any_target(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            validate_dataset(_ImageOnly(), expected="any_target")

    def test_ic_target_rejected_as_od(self) -> None:
        with pytest.raises(MaiteShapeError, match="ObjectDetectionTarget"):
            validate_dataset(_ICDataset(), expected="object_detection")

    def test_od_target_rejected_as_ic(self) -> None:
        with pytest.raises(MaiteShapeError, match="Array of class scores"):
            validate_dataset(_ODDataset(), expected="classification")

    def test_empty_dataset_is_allowed(self) -> None:
        # Empty datasets are legal (e.g. after filtering) — no probe, no rejection.
        class _Empty:
            def __len__(self) -> int:
                return 0

            def __getitem__(self, i: int) -> Any:
                raise IndexError

        assert validate_dataset(_Empty(), expected="any_target") == "image_only"
        assert validate_dataset(_Empty(), expected="object_detection") == "object_detection"

    def test_unsized(self) -> None:
        class _NoLen:
            def __getitem__(self, i: int) -> int:
                return i

        with pytest.raises(MaiteShapeError, match="not Sized"):
            validate_dataset(_NoLen(), expected="image_only")

    def test_wrong_tuple_arity(self) -> None:
        class _TwoTuple:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, i: int) -> tuple[Any, Any]:
                return np.zeros((3, 8, 8)), {"id": i}

        with pytest.raises(MaiteShapeError, match="3-tuple"):
            validate_dataset(_TwoTuple(), expected="any_target")

    def test_unknown_kind_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="unknown expected"):
            validate_dataset(_ICDataset(), expected="bogus")  # type: ignore[arg-type]

    def test_sized_but_not_indexable_dataset(self) -> None:
        # has __len__ (Sized) but no __getitem__, so it is not a Dataset
        class _NoGetItem:
            def __len__(self) -> int:
                return 3

        with pytest.raises(MaiteShapeError, match="not a Dataset"):
            validate_dataset(_NoGetItem(), expected="image_only")

    def test_image_only_wrong_tuple_arity(self) -> None:
        class _TwoTuple:
            metadata = DatasetMetadata(id="two")

            def __len__(self) -> int:
                return 1

            def __getitem__(self, i: int) -> tuple[Any, Any]:
                return np.zeros((3, 8, 8)), {"id": i}

        with pytest.raises(MaiteShapeError, match="tuple of length 2"):
            validate_dataset(_TwoTuple(), expected="image_only")

    def test_non_tuple_datum_described_by_type(self) -> None:
        # a scalar datum has no shape, so _describe falls back to the bare type name
        class _ScalarDatum:
            metadata = DatasetMetadata(id="scalar")

            def __len__(self) -> int:
                return 1

            def __getitem__(self, i: int) -> int:
                return 42

        with pytest.raises(MaiteShapeError, match="got int"):
            validate_dataset(_ScalarDatum(), expected="classification")


class TestValidateHelpers:
    def test_target_matches_image_only_is_false(self) -> None:
        # _target_matches is never called with "image_only" through validate_dataset
        # (it is short-circuited earlier), so exercise the fallback directly
        assert data._target_matches(object(), "image_only") is False  # type: ignore[arg-type]

    def test_infer_caller_without_frame_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # when no frame is available, the caller name defaults to "validate_dataset"
        monkeypatch.setattr(data.inspect, "currentframe", lambda: None)
        assert data._infer_caller() == "validate_dataset"


# ---------- validate_dataset: MOT coverage (per-frame arrays + stream count) ----------


@dataclass
class _FrameTracks:
    frame_tracks: Sequence[Any]


class _MOTLike:
    """Single-datum MOT dataset with a configurable stream and frame targets."""

    metadata = DatasetMetadata(id="mot_like")

    def __init__(self, stream: Any, frame_targets: Any) -> None:
        self._stream = stream
        self._target = _FrameTracks(frame_targets)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> tuple[Any, _FrameTracks, DatumMetadata]:
        return self._stream, self._target, DatumMetadata(id=i)


def _frame(boxes: Any = (1, 4), labels: Any = (1,), scores: Any = (1,), track_ids: Any = (1,)) -> _FrameTarget:
    # A None argument deletes the attribute (the default constructor sets all four).
    t = _FrameTarget()
    for name, value, dtype in (
        ("boxes", boxes, np.float32),
        ("labels", labels, np.intp),
        ("scores", scores, np.float32),
        ("track_ids", track_ids, np.intp),
    ):
        if value is None:
            delattr(t, name)
        else:
            setattr(t, name, np.zeros(value, dtype=dtype))
    return t


def _frames(n: int) -> list[np.ndarray]:
    return [np.zeros((3, 8, 8), dtype=np.float32) for _ in range(n)]


class TestMOTCoverage:
    def test_matching_counts_pass(self) -> None:
        ds = _MOTLike(_frames(2), [_frame(), _frame()])
        assert validate_dataset(ds, expected="multiobject_tracking") == "multiobject_tracking"
        # the same checks run when the kind is resolved through any_target
        assert validate_dataset(ds, expected="any_target") == "multiobject_tracking"

    def test_validation_never_touches_the_stream(self) -> None:
        # validation is targets-only: a one-shot iterator is left untouched, so a caller's
        # stream is never drained (or decoded) by the check
        class _ConsumedStream:
            def __iter__(self) -> Any:
                if getattr(self, "used", False):
                    raise RuntimeError("stream consumed during validation")
                self.used = True
                return iter([])

        ds = _MOTLike(_ConsumedStream(), [_frame(), _frame()])
        assert validate_dataset(ds, expected="multiobject_tracking") == "multiobject_tracking"

    def test_cardinality_mismatch(self) -> None:
        ds = _MOTLike(_frames(2), [_frame(), _frame(boxes=(2, 4), labels=(1,), scores=(1,), track_ids=(1,))])
        with pytest.raises(MaiteShapeError, match=r"has boxes with 8 value\(s\) for 1 label\(s\)"):
            validate_dataset(ds, expected="multiobject_tracking")

    def test_boxes_too_few_values(self) -> None:
        ds = _MOTLike(_frames(1), [_frame(boxes=(1, 3))])
        with pytest.raises(MaiteShapeError, match=r"has boxes with 3 value\(s\) for 1 label\(s\)"):
            validate_dataset(ds, expected="multiobject_tracking")

    def test_transposed_boxes_rejected(self) -> None:
        # (4, N) has the right value count but the wrong axis order: instance_arrays'
        # reshape(count, 4) would read every coordinate into the wrong slot, silently
        ds = _MOTLike(_frames(1), [_frame(boxes=(4, 2), labels=(2,), scores=(2,), track_ids=(2,))])
        with pytest.raises(MaiteShapeError, match=r"has boxes of shape \(4, 2\) for 2 label\(s\)"):
            validate_dataset(ds, expected="multiobject_tracking")

    def test_flat_boxes_accepted(self) -> None:
        # a flat (4N,) buffer reshapes to (N, 4) row-major with coordinates intact
        ds = _MOTLike(_frames(1), [_frame(boxes=(8,), labels=(2,), scores=(2,), track_ids=(2,))])
        assert validate_dataset(ds, expected="multiobject_tracking") == "multiobject_tracking"

    def test_short_track_ids_is_not_a_defect(self) -> None:
        # dataeval reads track_ids against labels, padding -1 where absent -- a frame whose
        # detections are untracked (short or empty track_ids) is legitimate, not malformed
        ds = _MOTLike(
            _frames(1),
            [_frame(boxes=(2, 4), labels=(2,), scores=(2,), track_ids=(0,))],
        )
        assert validate_dataset(ds, expected="multiobject_tracking") == "multiobject_tracking"

    def test_missing_scores_is_not_a_defect(self) -> None:
        class _NoScores:
            boxes = np.zeros((1, 4), dtype=np.float32)
            labels = np.array([0], dtype=np.intp)
            track_ids = np.array([0], dtype=np.intp)

        ds = _MOTLike(_frames(1), [_NoScores()])
        assert validate_dataset(ds, expected="multiobject_tracking") == "multiobject_tracking"

    def test_frame_missing_labels(self) -> None:
        class _NoLabels:
            boxes = np.zeros((1, 4), dtype=np.float32)
            scores = np.array([1.0], dtype=np.float32)
            track_ids = np.array([0], dtype=np.intp)

        ds = _MOTLike(_frames(1), [_NoLabels()])
        with pytest.raises(MaiteShapeError, match="lacks a required per-frame array"):
            validate_dataset(ds, expected="multiobject_tracking")

    def test_detection_free_frame_needs_no_boxes(self) -> None:
        ds = _MOTLike(_frames(1), [_frame(boxes=(0, 4), labels=(0,), scores=(0,), track_ids=(0,))])
        assert validate_dataset(ds, expected="multiobject_tracking") == "multiobject_tracking"

    def test_error_names_frame_index(self) -> None:
        ds = _MOTLike(_frames(2), [_frame(), _frame(), _frame(boxes=(1, 3))])
        with pytest.raises(MaiteShapeError, match=r"whose frame 2 has boxes"):
            validate_dataset(ds, expected="multiobject_tracking")

    def test_detection_free_frame_may_omit_boxes(self) -> None:
        # instance_arrays never reads boxes when the label count is 0, so a frame with no
        # detections is free to omit them entirely -- the validator must agree
        class _EmptyNoBoxes:
            labels = np.array([], dtype=np.intp)

        ds = _MOTLike(_frames(1), [_EmptyNoBoxes()])
        assert validate_dataset(ds, expected="multiobject_tracking") == "multiobject_tracking"

    def test_torch_backed_frame_target_accepted(self) -> None:
        # as_numpy detaches; np.asarray raises. The validator must read a target exactly
        # the way the consumer that will read it next does.
        import torch

        class _TorchFrame:
            boxes = torch.zeros((1, 4), requires_grad=True)
            labels = torch.zeros(1, dtype=torch.int64)

        ds = _MOTLike(_frames(1), [_TorchFrame()])
        assert validate_dataset(ds, expected="multiobject_tracking") == "multiobject_tracking"

    def test_raising_boxes_getter_is_reported(self) -> None:
        # dispatch sees the member without calling it; this is where a getter that raises
        # is turned into a shape error rather than escaping as itself
        class _Guarded:
            labels = np.array([0], dtype=np.intp)

            @property
            def boxes(self) -> Any:
                raise RuntimeError("no boxes above the image level")

        ds = _MOTLike(_frames(1), [_Guarded()])
        with pytest.raises(MaiteShapeError, match="cannot be read as an array"):
            validate_dataset(ds, expected="multiobject_tracking")

    def test_shape_error_stays_catchable_as_value_error(self) -> None:
        # the per-frame checks intercept, at entry, a defect that instance_arrays' reshape
        # used to raise as ValueError deep in the walk; a caller that caught it there must
        # keep working now that it is caught earlier
        ds = _MOTLike(_frames(1), [_frame(boxes=(2, 4), labels=(1,))])
        with pytest.raises(ValueError, match=r"has boxes with 8 value\(s\) for 1 label\(s\)"):
            validate_dataset(ds, expected="multiobject_tracking")

    def test_non_sequence_frame_tracks_rejected(self) -> None:
        ds = _MOTLike(_frames(1), None)
        with pytest.raises(MaiteShapeError, match="frame_tracks is NoneType"):
            validate_dataset(ds, expected="multiobject_tracking")


# ---------- @requires_maite_dataset ----------


class TestRequiresMaiteDataset:
    def test_param_name_passed_through(self) -> None:
        @requires_maite_dataset("ds", expected="image_only")
        def f(ds: Any) -> int:
            return len(ds)

        assert f(_ImageOnly()) == 4

    def test_failure_message_includes_qualname(self) -> None:
        @requires_maite_dataset(expected="object_detection")
        def f(dataset: Any) -> None: ...

        with pytest.raises(MaiteShapeError, match="f:"):
            f(_ImageOnly())

    def test_none_is_skipped(self) -> None:
        @requires_maite_dataset(expected="any_target")
        def f(dataset: Any = None) -> str:
            return "ok"

        assert f() == "ok"
        assert f(dataset=None) == "ok"

    def test_keyword_passing_works(self) -> None:
        @requires_maite_dataset(expected="classification")
        def f(dataset: Any) -> int:
            return len(dataset)

        assert f(dataset=_ICDataset()) == 4

    def test_decorator_rejects_missing_param(self) -> None:
        with pytest.raises(TypeError, match="no parameter named 'dataset'"):

            @requires_maite_dataset(expected="any_target")
            def f(x: Any) -> None: ...

    def test_unbindable_args_pass_through_to_real_call(self) -> None:
        # too many positional args make sig.bind_partial raise; validation is skipped
        # and the underlying call runs to surface the real TypeError
        @requires_maite_dataset(expected="any_target")
        def f(dataset: Any) -> str:
            return "ok"

        with pytest.raises(TypeError):
            f(1, 2, 3)  # type: ignore


# ---------- integration: public entry points ----------


class TestIntegrationEmbeddings:
    def test_image_only_dataset_accepted(self) -> None:
        # Embeddings explicitly supports Dataset[ArrayLike]
        Embeddings(_ImageOnly())

    def test_maite_dataset_accepted(self) -> None:
        Embeddings(_ICDataset())

    def test_none_dataset_accepted(self) -> None:
        # Unbound construction is supported
        Embeddings()


class TestIntegrationMetadata:
    def test_maite_dataset_accepted(self) -> None:
        Metadata(_ICDataset())
        Metadata(_ODDataset())

    def test_bare_image_rejected(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            Metadata(_ImageOnly())  # pyright: ignore[reportArgumentType]

    def test_none_dataset_accepted(self) -> None:
        Metadata(None)


class TestIntegrationView:
    def test_target_agnostic_operations_skip_validation(self) -> None:
        # Limit/Shuffle don't read targets; image-only datasets must keep working.
        assert len(View(_ImageOnly(10), operations=[Limit(size=3)])) == 3
        assert len(View(_ImageOnly(10), operations=[Shuffle()])) == 10

    def test_classfilter_on_bare_image_fails_fast(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            View(_ImageOnly(), operations=[ClassFilter(classes=[0])])

    def test_classfilter_on_maite_dataset_works(self) -> None:
        # Should not raise; whether anything is selected depends on labels.
        View(_ICDataset(), operations=[ClassFilter(classes=[0, 1, 2])])

    def test_strictest_required_kind_wins_across_operations(self) -> None:
        # A specific kind beats the generic "any_target" declared by ClassFilter.
        class _NeedsOD(Operation):
            requires = "object_detection"

            def apply(self, view: View) -> None: ...

        with pytest.raises(MaiteShapeError, match="ObjectDetectionTarget"):
            View(_ICDataset(), operations=[ClassFilter(classes=[0]), _NeedsOD()])

    def test_empty_dataset_skips_validation(self) -> None:
        # Nothing to probe — an empty source is legal even for target-reading operations.
        assert len(View(_ImageOnly(0), operations=[ClassFilter(classes=[0])])) == 0


class TestIntegrationUnzip:
    def test_per_target_requires_od(self) -> None:
        with pytest.raises(MaiteShapeError, match="ObjectDetectionTarget"):
            unzip_dataset(_ICDataset(), per_target=True)

    def test_per_target_false_accepts_bare_image(self) -> None:
        images, targets = unzip_dataset(_ImageOnly(), per_target=False)
        assert targets is None
        assert len(list(images)) == 4

    def test_per_target_true_on_od_works(self) -> None:
        images, targets = unzip_dataset(_ODDataset(), per_target=True)
        assert targets is not None
        assert len(list(images)) == 4


class TestIntegrationSplit:
    def test_bare_image_dataset_rejected(self) -> None:
        with pytest.raises(MaiteShapeError, match="3-tuple"):
            split_dataset(_ImageOnly(20), num_folds=2, val_frac=0.0)  # pyright: ignore[reportArgumentType]

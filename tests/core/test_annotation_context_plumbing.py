from enum import Flag, auto
from typing import Any

import numpy as np
import pytest

from dataeval.core._calculators._base import ALL_VIEWS, Calculator, ContextNeed, Handler, ViewKind
from dataeval.core._image_stats import BandPlan, BatchPlan, _compute_batch, _enumerate_datum
from dataeval.utils._annotations import AnnotationContext, Annotations, build_annotations

pytestmark = pytest.mark.required

SHAPE = (3, 20, 40)
BOXES = [(1, 1, 10, 10), (5, 5, 15, 15), (2, 2, 8, 8)]
SEEN: list[tuple[Any, int | None, ViewKind, str | None]] = []


class _StubFlag(Flag):
    A = auto()
    B = auto()


class _Stub(Calculator[_StubFlag]):
    """Records what the cache handed it on every pass, and declares a need through its handlers."""

    needs = ContextNeed.CONTEXT

    def __init__(self, datum: Any, cache: Any) -> None:
        self.cache = cache

    def get_applicable_flags(self) -> _StubFlag:
        return _StubFlag.A | _StubFlag.B

    def get_handlers(self) -> dict[_StubFlag, Handler]:
        return {
            _StubFlag.A: Handler("stub_a", lambda: [0.0], ALL_VIEWS, self.needs),
            _StubFlag.B: Handler("stub_b", lambda: [0.0], ALL_VIEWS),
        }

    def compute(self, flags: _StubFlag, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:
        context, focal = self.cache.context, self.cache.focal
        status = context.status(focal) if context is not None and focal is not None else None
        SEEN.append((context, focal, view, status))
        return {"stub": [0.0]}


class _LabelsStub(_Stub):
    needs = ContextNeed.LABELS


class _NothingStub(_Stub):
    needs = ContextNeed.NONE


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()


def plan(**overrides: Any) -> BatchPlan:
    fields: dict[str, Any] = {
        "calculators": [],
        "per_image": True,
        "per_target": True,
        "normalize_pixel_values": False,
        "per_background": False,
        "background_calculators": [],
        "declared_range": None,
        "band_plans": {},
        "wide_band_names": "",
        "unmeasurable_names": "",
        "needs": ContextNeed.NONE,
    }
    return BatchPlan(**{**fields, **overrides})


def datum(shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    return np.random.default_rng(0).random(shape)


def test_context_is_built_once_per_datum_and_every_pass_receives_it():
    stub = [(_Stub, _StubFlag.A)]
    band = BandPlan(
        indices=(0,), value_range=None, calculators=stub, background_calculators=stub, unmeasurable_names=""
    )
    batch = plan(
        calculators=stub,
        background_calculators=stub,
        band_plans={"b": band},
        per_background=True,
        needs=ContextNeed.CONTEXT,
    )

    _compute_batch((0, datum(), build_annotations(BOXES, SHAPE, 0)), batch)

    contexts = {id(context) for context, *_ in SEEN}
    assert len(contexts) == 1
    assert isinstance(SEEN[0][0], AnnotationContext)
    assert sorted({focal for _, focal, _, _ in SEEN if focal is not None}) == [0, 1, 2]
    # The image row has no focal box: whole, background, band and band-background passes.
    assert [view for _, focal, view, _ in SEEN if focal is None] == [
        ViewKind.WHOLE,
        ViewKind.BAND,
        ViewKind.MASK,
        ViewKind.MASK | ViewKind.BAND,
    ]
    assert {view for _, _, view, _ in SEEN} == {
        ViewKind.WHOLE,
        ViewKind.BAND,
        ViewKind.MASK,
        ViewKind.MASK | ViewKind.BAND,
    }


def test_nothing_declared_builds_nothing(monkeypatch):
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an AnnotationContext was built although no statistic declared a need")

    monkeypatch.setattr(AnnotationContext, "__init__", refuse)
    _compute_batch(
        (0, datum(), build_annotations(BOXES, SHAPE, 0)),
        plan(calculators=[(_NothingStub, _StubFlag.A)], needs=ContextNeed.NONE),
    )

    assert SEEN
    assert all(context is None for context, *_ in SEEN)


def test_an_unannotated_datum_gets_an_empty_context():
    _compute_batch((0, datum(), None), plan(calculators=[(_Stub, _StubFlag.A)], needs=ContextNeed.CONTEXT))

    ((context, focal, _, _),) = SEEN
    assert focal is None
    assert context.raw.shape == (0, 4)


def test_a_datum_with_no_spatial_extent_gets_an_empty_extent():
    _compute_batch(
        (0, datum((40,)), None),
        plan(calculators=[(_Stub, _StubFlag.A)], per_target=False, needs=ContextNeed.CONTEXT),
    )

    ((context, *_),) = SEEN
    assert context.extent == (0, 0)


def test_a_two_dimensional_datum_passes_its_extent_and_statuses():
    _compute_batch(
        (0, datum((20, 40)), build_annotations([(5, 5, 15, 15), (10, 0, 2, 10)], (20, 40), 0)),
        plan(calculators=[(_Stub, _StubFlag.A)], needs=ContextNeed.CONTEXT),
    )

    assert SEEN[0][0].extent == (20, 40)
    assert [(focal, status) for _, focal, _, status in SEEN] == [
        (None, None),
        (0, "incomplete_geometry"),
        (1, "invalid_box"),
    ]


def test_context_needs_ors_the_selected_handlers():
    both = _StubFlag.A | _StubFlag.B

    assert _LabelsStub.context_needs(both) == ContextNeed.LABELS
    assert _LabelsStub.context_needs(_StubFlag.B) == ContextNeed.NONE
    assert _Stub.context_needs(both) == ContextNeed.CONTEXT
    assert _NothingStub.context_needs(both) == ContextNeed.NONE


def test_enumerate_datum_yields_annotations_that_keep_the_raw_boxes():
    images = [np.zeros(SHAPE), np.zeros(SHAPE), np.zeros(SHAPE)]
    carried = build_annotations([(1, 1, 5, 5)], SHAPE, 2)
    boxes = [[(30, 0, 10, 10), None], None, carried]

    out = list(_enumerate_datum(images, boxes))

    assert [i for i, _, _ in out] == [0, 1, 2]
    assert isinstance(out[0][2], Annotations)
    assert out[0][2].raw[0].tolist() == [30, 0, 10, 10]
    assert out[0][2].raw[1].tolist() == [0, 0, 40, 20]
    assert out[1][2] is not None
    assert out[1][2].raw.shape == (0, 4)
    assert out[2][2] is carried


def test_enumerate_datum_without_boxes_yields_no_annotations():
    assert [a for _, _, a in _enumerate_datum([np.zeros(SHAPE)], None)] == [None]

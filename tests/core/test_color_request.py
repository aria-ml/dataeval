"""Color is dropped from a request with no declared channels, and warns experimental only where it is measured."""

import logging
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from dataeval import _experimental as experimental_module
from dataeval.core import _image_stats as image_stats_module
from dataeval.core import image_stats
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required

IMAGES = [np.random.default_rng(i).integers(0, 256, (3, 16, 16)).astype(np.uint8) for i in range(3)]


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with no experimental warning and no dropped-color notice given yet, as in a new process."""
    monkeypatch.setattr(experimental_module, "_WARNED", set())
    monkeypatch.setattr(image_stats_module, "_NOTICES_GIVEN", set())


BOXES = [[(1, 1, 8, 8)] for _ in IMAGES]
LABELS = [[0] for _ in IMAGES]


def run(stats: Any, **kw: Any) -> tuple[Any, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = image_stats(IMAGES, stats=stats, **kw)
    return result, caught


def texts(caught: list[warnings.WarningMessage], kind: type[Warning] = Warning) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, kind)]


def columns(result: Any) -> Iterator[str]:
    yield from result["stats"]
    yield from result.get("details", {})


def test_an_explicit_color_request_without_channels_warns_and_drops():
    result, caught = run(ImageStats.PIXEL_MEAN | ImageStats.COLOR)
    (message,) = [t for t in texts(caught, UserWarning) if "color_rgb_channels" in t]
    for needle in ("color_rgb_channels=(0, 1, 2)", "sRGB", "(2, 1, 0)", "image_stats", "from_stats"):
        assert needle in message
    assert not [name for name in columns(result) if name.startswith("color_")]
    assert "mean" in result["stats"]


def test_all_without_channels_equals_all_without_color_and_says_it_in_the_log(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="dataeval"):
        everything, caught = run(ImageStats.ALL)
    without, _ = run(ImageStats.ALL & ~ImageStats.COLOR)
    assert not [t for t in texts(caught, UserWarning) if "color_rgb_channels" in t]
    assert [r for r in caplog.records if "color_rgb_channels" in r.getMessage()]
    assert set(columns(everything)) == set(columns(without))
    for name, values in everything["stats"].items():
        np.testing.assert_array_equal(values, without["stats"][name], err_msg=name)


def test_all_per_target_false_warns_what_it_did_before():
    _, with_color = run(ImageStats.ALL, per_target=False)
    experimental_module._WARNED.clear()  # once per process: give the second call the same fresh start
    _, without = run(ImageStats.ALL & ~ImageStats.COLOR, per_target=False)
    assert sorted(texts(with_color)) == sorted(texts(without))


def test_a_request_of_only_color_flags_is_the_no_statistics_case():
    _, caught = run(ImageStats.COLOR)
    messages = texts(caught, UserWarning)
    assert [t for t in messages if "requests no statistics" in t]
    assert [t for t in messages if "color_rgb_channels" in t]


def test_with_channels_color_warns_experimental_once():
    _, first = run(ImageStats.PIXEL_MEAN | ImageStats.COLOR, color_rgb_channels=(0, 1, 2))
    named = [t for t in texts(first, ExperimentalWarning) if "ImageStats.COLOR" in t]
    assert len(named) == 1
    _, second = run(ImageStats.PIXEL_MEAN | ImageStats.COLOR, color_rgb_channels=(0, 1, 2))
    assert not [t for t in texts(second, ExperimentalWarning) if "ImageStats.COLOR" in t]


def test_all_without_channels_does_not_warn_experimental_about_color():
    _, caught = run(ImageStats.ALL)
    assert not [t for t in texts(caught, ExperimentalWarning) if "ImageStats.COLOR" in t]


@pytest.mark.parametrize("group", ["BASIC", "STANDARD", "NO_HASH"])
def test_the_smaller_groups_hold_no_color_and_say_nothing_about_it(group: str):
    _, caught = run(getattr(ImageStats, group))
    assert not [t for t in texts(caught) if "color" in t.lower()]


def dropped(caught: list[warnings.WarningMessage]) -> list[str]:
    return [t for t in texts(caught, UserWarning) if "color_rgb_channels" in t]


def test_an_explicit_request_warns_once_per_process_and_logs_every_time(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="dataeval"):
        _, first = run(ImageStats.PIXEL_MEAN | ImageStats.COLOR)
        _, second = run(ImageStats.PIXEL_MEAN | ImageStats.COLOR)
    assert len(dropped(first)) == 1
    assert not dropped(second)
    assert len([r for r in caplog.records if "color_rgb_channels" in r.getMessage()]) == 2


@pytest.mark.parametrize(
    "stats",
    [ImageStats.ALL & ~ImageStats.BLUR, ~ImageStats.HASH, ImageStats.ALL & ~ImageStats.HASH],
    ids=["all-minus-blur", "not-hash", "all-minus-hash"],
)
def test_a_complement_of_all_warns_once_per_process_and_not_on_every_call(stats: Any):
    # The library itself suggests these spellings, so a warning on every call would be noise for a caller who never
    # asked for color.
    _, first = run(stats, boxes=BOXES, labels=LABELS)
    _, second = run(stats, boxes=BOXES, labels=LABELS)
    assert len(dropped(first)) == 1
    assert not dropped(second)


def test_a_request_that_asks_for_everything_never_warns_about_the_drop():
    for _ in range(2):
        _, caught = run(ImageStats.ALL, boxes=BOXES, labels=LABELS)
        assert not dropped(caught)

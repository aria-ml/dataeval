"""The ``VideoStats.BLUR`` family's experimental warning and the requests that need OpenCV to get past its check."""

import warnings
from typing import Any, cast

import pytest

pytest.importorskip("cv2", reason="VideoStats.BLUR requires opencv")

import dataeval._experimental as experimental_module
from dataeval.core import video_stats
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import VideoStats
from tests.core.test_video_stats import _stream, _Unreadable

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    """Start each test with no experimental warning given yet."""
    monkeypatch.setattr(experimental_module, "_WARNED", set())


def test_a_refused_blur_samples_does_not_spend_the_warning():
    with pytest.raises(ValueError, match="blur_samples"):
        video_stats(cast(Any, _Unreadable()), stats=VideoStats.BLUR, blur_samples=0)
    with pytest.warns(ExperimentalWarning, match="VideoStats.BLUR' is experimental"):
        video_stats(cast(Any, [_stream(3)]), stats=VideoStats.BLUR)


@pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("stats", [VideoStats.PERSISTENCE, VideoStats.EGO])
def test_blur_samples_is_ignored_without_a_blur_flag(stats):
    video_stats(cast(Any, [_stream(3)]), stats=stats, blur_samples=0)


@pytest.mark.parametrize("stats", [VideoStats.BLUR, VideoStats.BLUR_LEVEL, VideoStats.ALL])
def test_any_blur_flag_or_all_warns_experimental(stats):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        video_stats(cast(Any, [_stream(3)]), stats=stats)
    messages = [str(w.message) for w in caught if issubclass(w.category, ExperimentalWarning)]
    (blur,) = [m for m in messages if "'VideoStats.BLUR' is experimental" in m]
    assert "blur_samples" in blur
    assert "VideoStats.ALL & ~VideoStats.BLUR" in blur


def test_the_ego_group_does_not_warn_about_blur():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        video_stats(cast(Any, [_stream(3)]), stats=VideoStats.NO_EGO | VideoStats.EGO)


def test_a_persistence_request_does_not_warn_about_blur():
    with pytest.warns(ExperimentalWarning, match="VideoStats.PERSISTENCE") as caught:
        video_stats(cast(Any, [_stream(40)]), stats=VideoStats.PERSISTENCE)
    assert not [w for w in caught if "VideoStats.BLUR" in str(w.message)]


def test_the_blur_warning_is_given_once_per_process():
    with pytest.warns(ExperimentalWarning, match="VideoStats.BLUR"):
        video_stats(cast(Any, [_stream(3)]), stats=VideoStats.BLUR_LEVEL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ExperimentalWarning)
        video_stats(cast(Any, [_stream(3)]), stats=VideoStats.BLUR)
    assert not [w for w in caught if issubclass(w.category, ExperimentalWarning)]

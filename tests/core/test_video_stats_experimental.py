"""``VideoStats.PERSISTENCE`` is experimental: selecting it warns once per process, the default stays silent."""

import warnings
from typing import Any, cast

import pytest

pytest.importorskip("cv2", reason="VideoStats.PERSISTENCE requires opencv")

import dataeval._experimental as experimental_module
from dataeval._experimental import quiet_experimental
from dataeval.core import video_stats
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import VideoStats
from tests.core.test_video_stats import _stream

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    """Start each test with no experimental warning given yet."""
    monkeypatch.setattr(experimental_module, "_WARNED", set())


def run(stats: VideoStats) -> None:
    video_stats(cast(Any, [_stream(40)]), stats=stats)


def test_persistence_flag_warns_experimental():
    with pytest.warns(ExperimentalWarning, match="VideoStats.PERSISTENCE' is experimental"):
        run(VideoStats.PERSISTENCE)


@pytest.mark.parametrize(
    "stats",
    [
        VideoStats.PERSISTENCE_EGO,
        VideoStats.PERSISTENCE_BLUR,
        VideoStats.PERSISTENCE_BLUR_RAW,
        VideoStats.ALL,
    ],
)
def test_any_persistence_flag_or_all_warns(stats):
    with pytest.warns(ExperimentalWarning):
        run(stats)


def test_default_stats_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        video_stats(cast(Any, [_stream(40)]))
        run(VideoStats.NO_EGO | VideoStats.EGO)


def test_warns_once_per_process():
    with pytest.warns(ExperimentalWarning):
        run(VideoStats.PERSISTENCE_EGO)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ExperimentalWarning)
        run(VideoStats.PERSISTENCE)
        run(VideoStats.ALL)
    assert not [x for x in w if issubclass(x.category, ExperimentalWarning)]


def test_quiet_experimental_silences_it_and_leaves_it_unspent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        with quiet_experimental():
            run(VideoStats.PERSISTENCE)
    with pytest.warns(ExperimentalWarning):
        run(VideoStats.PERSISTENCE)


def test_refused_request_does_not_spend_the_warning():
    with pytest.raises(ValueError, match="blur_sigma"):
        video_stats(cast(Any, [_stream(40)]), stats=VideoStats.PERSISTENCE, blur_sigma=0.0)
    with pytest.warns(ExperimentalWarning):
        run(VideoStats.PERSISTENCE)

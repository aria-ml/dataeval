"""The ``VideoStats.BLUR`` family: its flags, the ``blur_samples`` option and the option checks.

The tests here cover the flags, the option checks and what ``video_stats`` does before any blur measurement, and none
of them needs OpenCV. The warning and the requests that run past the OpenCV check are in
``test_video_stats_blur_opencv.py``, and the calculator that measures the frames is in
``test_video_stats_blur_calculator.py``.
"""

import inspect
import sys
import warnings
from typing import Any, cast

import numpy as np
import pytest

import dataeval.core._video_stats as video_stats_module
from dataeval.core import video_stats
from dataeval.core._video_runner import VideoOptions
from dataeval.core._video_stats import _check_blur
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import VideoStats
from tests.core.test_video_stats import _stream, _Unreadable

pytestmark = pytest.mark.required


def test_blur_is_the_level_flag_and_ends_the_enum():
    # The group name is an alias of the one flag, so it can grow without the flag changing.
    assert VideoStats.BLUR is VideoStats.BLUR_LEVEL
    assert not hasattr(VideoStats, "BLUR_DIRECTIONALITY")
    members = [flag for flag in VideoStats if flag.value and flag.value & (flag.value - 1) == 0]
    # New members come last, so no existing flag value moves.
    assert members[-2:] == [VideoStats.PERSISTENCE_BLUR_RAW, VideoStats.BLUR_LEVEL]
    assert VideoStats.BLUR_LEVEL.value == 2 * VideoStats.PERSISTENCE_BLUR_RAW.value


def test_basic_is_the_core_set_under_its_v1_2_name_no_ego():
    assert VideoStats.BASIC is VideoStats.NO_EGO
    assert VideoStats.BASIC == VideoStats.DESCRIPTIVE | VideoStats.REDUNDANCY
    # The first name of the two is the one Python prints.
    assert "BASIC" in repr(VideoStats.BASIC)
    assert "NO_EGO" not in repr(VideoStats.BASIC)


def test_all_is_basic_plus_everything_this_release_adds():
    assert VideoStats.ALL == VideoStats.BASIC | VideoStats.EGO | VideoStats.PERSISTENCE | VideoStats.BLUR
    assert VideoStats.BLUR in VideoStats.ALL
    assert VideoStats.BLUR not in VideoStats.BASIC
    assert video_stats_module._BLUR == VideoStats.PERSISTENCE_BLUR | VideoStats.PERSISTENCE_BLUR_RAW


def test_the_default_is_basic():
    assert inspect.signature(video_stats).parameters["stats"].default is VideoStats.BASIC


def test_the_default_returns_the_basic_columns_and_needs_no_opencv(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    default = video_stats(cast(Any, [_stream(10)]))
    named = video_stats(cast(Any, [_stream(10)]), stats=VideoStats.NO_EGO)
    assert list(default["stats"]) == list(named["stats"])
    for name, column in named["stats"].items():
        np.testing.assert_array_equal(default["stats"][name], column, err_msg=name)


def test_blur_samples_is_the_last_option_and_defaults_to_64():
    assert VideoOptions._fields[-1] == "blur_samples"
    assert VideoOptions().blur_samples == 64


@pytest.mark.parametrize("blur_samples", [0, -1])
@pytest.mark.parametrize("stats", [VideoStats.BLUR, VideoStats.BLUR_LEVEL])
def test_a_non_positive_blur_samples_is_refused_before_reading(stats, blur_samples):
    with pytest.raises(ValueError, match=f"blur_samples must be positive; got {blur_samples}"):
        video_stats(cast(Any, _Unreadable()), stats=stats, blur_samples=blur_samples)


@pytest.mark.parametrize("blur_samples", [0, 2.5, None])
def test_blur_samples_is_ignored_without_a_blur_flag(blur_samples):
    video_stats(cast(Any, [_stream(3)]), stats=VideoStats.NO_EGO, blur_samples=cast(Any, blur_samples))


@pytest.mark.parametrize("blur_samples", [True, 2.5, 64.0, float("nan"), float("inf"), None, "64"])
@pytest.mark.parametrize("stats", [VideoStats.BLUR, VideoStats.BLUR_LEVEL])
def test_a_non_integer_blur_samples_is_refused_before_reading(stats, blur_samples):
    with pytest.raises(ValueError, match="blur_samples must be an integer; got"):
        video_stats(cast(Any, _Unreadable()), stats=stats, blur_samples=cast(Any, blur_samples))


@pytest.mark.parametrize("blur_samples", [1, 64, np.int64(8)])
def test_a_positive_integer_blur_samples_passes_the_check(blur_samples):
    _check_blur(VideoOptions(blur_samples=blur_samples))


def test_missing_opencv_names_the_extra_before_reading(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match=r"dataeval\[opencv\]"):
        video_stats(cast(Any, _Unreadable()), stats=VideoStats.BLUR_LEVEL)


def test_a_non_blur_request_does_not_need_opencv(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    video_stats(cast(Any, [_stream(3)]), stats=VideoStats.DESCRIPTIVE)


def test_the_default_does_not_warn_about_blur():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        video_stats(cast(Any, [_stream(3)]))


def test_the_flag_docstring_names_the_new_members():
    doc = VideoStats.__doc__ or ""
    for name in ("BLUR_LEVEL", "blur_level", "blur_samples", "BASIC"):
        assert name in doc


def test_default_columns_unchanged():
    # The columns of `video_stats(NO_EGO)` before BLUR existed.
    result = video_stats(cast(Any, [_stream(10)]), stats=VideoStats.NO_EGO)
    assert [*result["stats"], "source_index"] == [
        "frame_rate",
        "longest_run",
        "n_frames",
        "redundant_fraction",
        "video_duration",
        "source_index",
    ]
    assert VideoStats.BLUR not in VideoStats.NO_EGO

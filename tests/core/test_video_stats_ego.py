"""``VideoStats.EGO``: one row per video of the camera-motion roll-ups ``ego_stats`` declares."""

import gc
import weakref
from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.config import use_max_processes
from dataeval.core import ego_stats, video_stats
from dataeval.flags import VideoStats

pytest.importorskip("cv2", reason="VideoStats.EGO requires opencv")

# Imported as a module so pytest does not collect its test classes a second time here.
from tests.core import test_ego_stats as ego_tests
from tests.core.test_ego_overlay import DENSE, hud_pan
from tests.core.test_ego_stats import FakeDataset, FakeFrame, ambiguous_stream, panning_stream, texture

pytestmark = pytest.mark.optional

EXPECTED = ego_tests.TestDeclaredRollUps.EXPECTED


def test_ego_equals_the_metadata_roll_up():
    ds = ego_tests.TestMetadataIntegration._dataset([3, 0])
    md = Metadata(ds)
    md.add_factors(cast("Any", ego_stats(ds)), level="unit", key="unit_index")
    ours = video_stats(ds, stats=VideoStats.EGO)["stats"]
    rows = md.rows_at("sequence")
    assert set(ours) == EXPECTED
    for name in EXPECTED - {"motion_mode"}:
        np.testing.assert_array_equal(ours[name], np.asarray(rows[name].to_list(), dtype=float), err_msg=name)
    assert ours["motion_mode"].tolist() == rows["motion_mode"].to_list()


@pytest.mark.filterwarnings(
    "ignore:.*binned automatically:UserWarning"
)  # the pan speeds vary; only roll-ups are compared
def test_ego_mask_matches_ego_stats():
    from tests.embeddings.test_embeddings import MockDataset
    from tests.metadata.test_structurers import _FrameTracks, _MOTTarget

    streams = [hud_pan(8, 8, seed=s) for s in range(3)]
    ds = MockDataset(streams, [_MOTTarget([_FrameTracks([0]) for _ in s]) for s in streams], None)
    md = Metadata(ds)
    md.add_factors(cast("Any", ego_stats(ds, mask=DENSE)), level="unit", key="unit_index")
    with use_max_processes(2):  # the mask goes through the threaded runner
        ours = video_stats(ds, stats=VideoStats.EGO, ego_mask=DENSE)["stats"]
    rows = md.rows_at("sequence")
    assert set(ours) == EXPECTED
    for name in EXPECTED - {"motion_mode"}:
        np.testing.assert_array_equal(ours[name], np.asarray(rows[name].to_list(), dtype=float), err_msg=name)
    assert ours["motion_mode"].tolist() == rows["motion_mode"].to_list()


def test_ego_mask_is_ignored_without_ego_flags():
    video_stats(cast(Any, [panning_stream(n_frames=3)]), stats=VideoStats.DESCRIPTIVE, ego_mask=np.ones((3, 4, 4)))


def test_ego_equals_the_metadata_roll_up_across_trusted_ambiguous_and_empty_videos():
    from tests.embeddings.test_embeddings import MockDataset
    from tests.metadata.test_structurers import _FrameTracks, _MOTTarget

    streams = [panning_stream(n_frames=8), ambiguous_stream(n_frames=8), []]
    ds = MockDataset(streams, [_MOTTarget([_FrameTracks([0]) for _ in s]) for s in streams], None)
    md = Metadata(ds)
    md.add_factors(cast("Any", ego_stats(ds)), level="unit", key="unit_index")
    ours = video_stats(ds, stats=VideoStats.EGO)["stats"]
    rows = md.rows_at("sequence")
    assert set(ours) == EXPECTED
    for name in EXPECTED - {"motion_mode"}:
        np.testing.assert_array_equal(ours[name], np.asarray(rows[name].to_list(), dtype=float), err_msg=name)
    assert ours["motion_mode"].tolist() == rows["motion_mode"].to_list()
    # The ambiguous video's refused pairs leave too little trust coverage for a kinematic summary.
    assert np.isnan(ours["pan_speed_median"][1])
    assert np.isnan(ours["pan_x_variability"][1])
    assert ours["motion_mode"][1] is None


def test_an_empty_video_takes_each_reductions_identity():
    s = video_stats(cast(Any, FakeDataset([panning_stream(), []])), stats=VideoStats.EGO)["stats"]
    assert s["zoom_rate_excursion"][1] == 0.0
    assert s["roll_rate_excursion"][1] == 0.0
    assert np.isnan(s["pan_speed_median"][1])
    assert s["motion_mode"][1] is None


def test_a_stitched_stream_is_read_in_stream_order():
    frames = panning_stream(n_frames=8)
    restarted = [FakeFrame(f.pixels[0], f.frame_index) for f in frames]
    for f in restarted[4:]:
        f.time_s -= 4 / 30  # time restarts at the midpoint, as VideoStitch "source" does
    want = video_stats(cast(Any, frames), stats=VideoStats.EGO_PAN_X_VARIABILITY)["stats"]["pan_x_variability"]
    got = video_stats(cast(Any, restarted), stats=VideoStats.EGO_PAN_X_VARIABILITY)["stats"]["pan_x_variability"]
    np.testing.assert_allclose(got, want)


def test_an_untimed_frame_nulls_only_the_variabilities():
    frames = panning_stream(n_frames=8)
    frames[3].time_s = cast(Any, None)
    s = video_stats(cast(Any, frames), stats=VideoStats.EGO)["stats"]
    assert np.isnan(s["pan_x_variability"][0])
    assert np.isfinite(s["pan_speed_median"][0])


class _Unreadable:
    def __iter__(self):
        raise AssertionError("the stream was read before the dependency check")


def test_missing_opencv_fails_before_reading(monkeypatch):
    import dataeval.core._ego_stats as ego

    monkeypatch.setattr(ego, "_require_cv2", lambda: (_ for _ in ()).throw(ImportError("opencv")))
    with pytest.raises(ImportError):
        video_stats(cast(Any, _Unreadable()), stats=VideoStats.EGO)


def test_shared_views_are_computed_once_per_frame(monkeypatch):
    import dataeval.core._video_runner as runner

    calls = []
    real = runner.to_canonical_grayscale
    monkeypatch.setattr(runner, "to_canonical_grayscale", lambda p: calls.append(1) or real(p))
    video_stats(cast(Any, panning_stream(n_frames=5)), stats=VideoStats.REDUNDANCY | VideoStats.EGO)
    assert len(calls) == 5


def test_a_walk_holds_a_bounded_number_of_frames():
    alive: list[weakref.ref[Any]] = []
    peak = 0

    def frames():
        nonlocal peak
        base = texture(0)
        for k in range(12):
            gc.collect()
            peak = max(peak, sum(ref() is not None for ref in alive))
            frame = FakeFrame(np.roll(base, 3 * k, axis=1), k)  # built lazily, never listed
            alive.append(weakref.ref(frame.pixels))
            yield frame

    video_stats(cast(Any, frames()), stats=VideoStats.ALL)
    assert peak <= 2  # ego_gap + 1


def test_the_default_skips_ego():
    assert not set(video_stats(cast(Any, panning_stream(n_frames=3)))["stats"]) & EXPECTED


def test_no_frame_anywhere_still_takes_each_reductions_identity():
    s = video_stats(cast(Any, FakeDataset([[]])), stats=VideoStats.EGO)["stats"]
    assert s["zoom_rate_excursion"].tolist() == [0.0]
    assert s["motion_mode"].tolist() == [None]
    empty = video_stats(ego_tests.TestMetadataIntegration._dataset([]), stats=VideoStats.EGO)["stats"]
    assert set(empty) == EXPECTED
    assert empty["pan_speed_median"].dtype == np.float64
    assert empty["motion_mode"].dtype == object


def test_bad_ego_parameters_are_refused_before_reading():
    with pytest.raises(ValueError, match="positive"):
        video_stats(cast(Any, _Unreadable()), stats=VideoStats.EGO, ego_gap=0)

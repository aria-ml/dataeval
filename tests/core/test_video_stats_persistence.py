"""``VideoStats.PERSISTENCE``: per-video maps of what is fixed to the camera, from one walk of each video."""

import inspect
import itertools
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("cv2", reason="VideoStats.PERSISTENCE requires opencv")

import cv2

from dataeval import Metadata
from dataeval.config import use_max_processes
from dataeval.core import combine_stats_results, video_stats
from dataeval.core._persistence import StartSampler
from dataeval.flags import VideoStats
from tests.core.test_ego_stats import FakeFrame, texture
from tests.core.test_persistence import (
    PATCH,
    cells,
    iter_scene,
    maps,
    scene,
    with_blur,
    with_patch,
    with_patch_iter,
)
from tests.core.test_video_stats import _CountingDataset, _OneShot, _stream, _tracking

pytestmark = [
    pytest.mark.optional,
    pytest.mark.filterwarnings("ignore:.*PERSISTENCE.* experimental:dataeval.exceptions.ExperimentalWarning"),
]

MAPS = {"ego_map", "blur_map", "blur_map_raw"}


def stream(frames) -> list[FakeFrame]:
    """Wrap 2D frames as a stream of decoded frames."""
    return [FakeFrame(f, k) for k, f in enumerate(frames)]


def run(videos, **kwargs) -> Mapping[str, Any]:
    return video_stats(cast(Any, videos), **kwargs)["stats"]


def test_streaming_matches_the_in_memory_maps():
    frames = with_patch(with_blur(scene(60)))
    got = run([stream(frames)], stats=VideoStats.PERSISTENCE)
    want = maps(frames)
    assert set(got) == MAPS
    for name in got:
        np.testing.assert_allclose(got[name][0], getattr(want, name), atol=0.02, equal_nan=True, err_msg=name)


@pytest.mark.parametrize(
    ("flag", "name"),
    [
        (VideoStats.PERSISTENCE_EGO, "ego_map"),
        (VideoStats.PERSISTENCE_BLUR, "blur_map"),
        (VideoStats.PERSISTENCE_BLUR_RAW, "blur_map_raw"),
    ],
)
def test_each_flag_returns_only_its_map(flag, name):
    assert set(run([stream(scene(20))], stats=flag)) == {name}


def test_a_still_camera_logs_why_its_maps_are_nan(caplog):
    caplog.set_level("WARNING", logger="dataeval.core")
    s = run([stream(with_patch(scene(40, shift=0)))], stats=VideoStats.PERSISTENCE_EGO)
    assert np.isnan(s["ego_map"]).all()
    assert sum("ego_map" in r.getMessage() and "camera motion" in r.getMessage() for r in caplog.records) == 1


def test_a_measure_breaking_its_contract_raises_by_name():
    def bad_shape(gray, grid, /):
        return np.ones((2, 2), np.float32)

    with pytest.raises(ValueError, match="bad_shape"):
        run([stream(scene(20))], stats=VideoStats.PERSISTENCE_BLUR, blur_measure=bad_shape)


def test_a_measure_with_zeros_on_flat_cells_runs():
    frames = scene(60)
    for f in frames:
        f[:45] = 0  # a letterbox bar: the default measure returns 0 there
    s = run([stream(frames)], stats=VideoStats.PERSISTENCE_BLUR)
    assert np.isfinite(s["blur_map"]).any()


def test_a_long_video_stays_under_the_ceiling_and_reaches_its_end(monkeypatch):
    from dataeval.core._calculators._persistencestats import PersistenceStatCalculator as Calc

    peak = {"starts": 0, "samples": 0}
    real = Calc.update

    def spy(self, view):
        real(self, view)
        peak["starts"] = max(peak["starts"], len(self.sampler.kept))
        peak["samples"] = max(peak["samples"], len(self.samples))

    monkeypatch.setattr(Calc, "update", spy)
    # Still for 1000 frames, then panning: only the last third gives evidence.
    frames = (FakeFrame(f, k) for k, f in enumerate(with_patch_iter(iter_scene(1500, still_until=1000))))
    ego = run([frames], stats=VideoStats.PERSISTENCE_EGO, persistence_samples=30)["ego_map"][0]
    assert peak["starts"] <= 30
    assert peak["samples"] <= 60  # a start and its +10 partner each; no +30 frames without a blur flag
    assert ego[cells(PATCH, inset=1)].min() >= 0.9


def test_ego_and_persistence_read_each_frame_once(monkeypatch):
    import dataeval.core._video_runner as runner

    calls = []
    real = runner.to_canonical_grayscale
    monkeypatch.setattr(runner, "to_canonical_grayscale", lambda p: calls.append(1) or real(p))
    run([_OneShot(cast(Any, stream(scene(20))))], stats=VideoStats.EGO | VideoStats.PERSISTENCE)
    assert len(calls) == 20


def test_maps_share_one_grid_across_sizes():
    wide = stream(with_patch(scene(60)))
    four_three = stream(with_patch(scene(60, size=(360, 480)), (slice(90, 180), slice(160, 320))))
    s = run([wide, four_three], stats=VideoStats.PERSISTENCE)
    for name in MAPS:
        assert s[name].shape == (2, 48, 85)
        assert s[name].dtype == np.float64
    assert np.nanmax(s["ego_map"][1]) >= 0.9


def test_add_factors_skips_the_maps():
    ds = _tracking([stream(scene(20)), stream(scene(20, seed=1))])
    md = Metadata(ds)
    md.add_factors(video_stats(ds, stats=VideoStats.DESCRIPTIVE_FRAME_COUNT | VideoStats.PERSISTENCE_EGO))
    assert md.dropped_factors["ego_map"] == ["multi_dimensional"]
    assert md.rows_at("sequence")["n_frames"].to_list() == [20, 20]


# Review Focus: inputs the acceptance tests do not reach


def test_tiny_frames_read_nan_and_warn(caplog):
    caplog.set_level("WARNING", logger="dataeval.core")
    s = run([_stream(30, shape=(3, 8, 8))], stats=VideoStats.PERSISTENCE)
    assert s["ego_map"].shape == (1, 48, 85)
    assert np.isnan(s["ego_map"]).all()
    assert any("ego_map" in r.getMessage() for r in caplog.records)


def test_a_size_change_keeps_the_first_size(caplog):
    caplog.set_level("WARNING", logger="dataeval.core")
    first = with_patch(scene(40))
    later = [FakeFrame(f, 40 + k) for k, f in enumerate(scene(20, size=(360, 480)))]
    got = run([stream(first) + later], stats=VideoStats.PERSISTENCE)
    want = run([stream(first)], stats=VideoStats.PERSISTENCE)
    for name in got:
        np.testing.assert_allclose(got[name], want[name], atol=0.02, equal_nan=True, err_msg=name)
    assert sum("size" in r.getMessage() for r in caplog.records) == 1


def test_a_portrait_video_is_measured():
    frames = with_patch(scene(60, size=(640, 360)), (slice(200, 290), slice(100, 260)))
    ego = run([stream(frames)], stats=VideoStats.PERSISTENCE_EGO)["ego_map"][0]
    assert ego.shape == (48, 85)
    assert np.nanmax(ego) >= 0.9
    assert np.nanmedian(ego) <= 0.1


def test_parallel_matches_serial_and_results_combine():
    videos = [with_patch(scene(40)), with_blur(scene(40, seed=1))]
    serial = video_stats(cast(Any, [stream(f) for f in videos]), stats=VideoStats.PERSISTENCE)
    with use_max_processes(2):
        threaded = video_stats(cast(Any, [stream(f) for f in videos]), stats=VideoStats.PERSISTENCE)
    for name in serial["stats"]:
        np.testing.assert_allclose(
            threaded["stats"][name], serial["stats"][name], atol=0.02, equal_nan=True, err_msg=name
        )
    stats, _, _ = combine_stats_results([serial, serial])
    assert stats["ego_map"].shape == (4, 48, 85)


def test_an_empty_video_and_an_empty_dataset(caplog):
    caplog.set_level("WARNING", logger="dataeval.core")
    s = run([[]], stats=VideoStats.PERSISTENCE)
    assert s["ego_map"].shape == (1, 48, 85)
    assert np.isnan(s["ego_map"]).all()
    assert any("too few frames" in r.getMessage() for r in caplog.records)
    empty = run(_CountingDataset([]), stats=VideoStats.PERSISTENCE)
    assert empty["blur_map"].shape == (0, 48, 85)


def test_a_long_second_clip_of_another_size_keeps_the_first_clips_evidence(caplog):
    caplog.set_level("WARNING", logger="dataeval.core")
    first = with_patch(scene(60))
    later = (FakeFrame(f, 60 + k) for k, f in enumerate(iter_scene(1000, size=(360, 480))))
    got = run([itertools.chain(stream(first), later)], stats=VideoStats.PERSISTENCE)
    want = run([stream(first)], stats=VideoStats.PERSISTENCE)
    for name in got:
        np.testing.assert_allclose(got[name], want[name], atol=0.02, equal_nan=True, err_msg=name)
    assert any("640x360" in r.getMessage() and "480x360" in r.getMessage() for r in caplog.records)


def test_the_default_ceiling_keeps_the_spikes_evidence_at_any_length():
    # The spike read 28 frame pairs 10 apart over 150 frames; no longer video may keep fewer.
    default = inspect.signature(video_stats).parameters["persistence_samples"].default
    for n in (150, 161, 320, 321, 1000, 5000):
        starts = StartSampler(default)
        for i in range(n):
            starts.offer(i)
        assert sum(k + 10 < n for k in starts.kept) >= 28, n


def _too_little_motion(caplog) -> int:
    return sum("ego_map" in r.getMessage() and "too little camera motion" in r.getMessage() for r in caplog.records)


def test_a_still_camera_over_large_movers_reads_nan(caplog):
    # The band changes over 30% of the textured pixels, but the motion fitted to it misaligns the still background,
    # so "nothing moved" explains more of the frame and no pair counts.
    caplog.set_level("WARNING", logger="dataeval.core")
    background = scene(1)[0]
    band = scene(60, shift=4, size=(180, 640), seed=3)
    frames = []
    for b in band:
        frame = background.copy()
        frame[150:330] = b
        frames.append(frame)
    ego = run([stream(frames)], stats=VideoStats.PERSISTENCE_EGO)["ego_map"][0]
    assert np.isnan(ego).all()
    assert _too_little_motion(caplog) == 1


def test_a_slow_pan_reads_nan_rather_than_its_scenery_as_fixed(caplog):
    # 0.12 px a frame is 1.2 px a pair: weak-edged scenery barely changes, and counted, 7.5% of its cells read as fixed.
    caplog.set_level("WARNING", logger="dataeval.core")
    big = texture(0, (360, 680), blur=1.5)
    frames = [cv2.warpAffine(big, np.array([[1, 0, -0.12 * k], [0, 1, 0]], np.float32), (640, 360)) for k in range(60)]
    ego = run([stream(frames)], stats=VideoStats.PERSISTENCE_EGO)["ego_map"][0]
    assert np.isnan(ego).all()
    assert _too_little_motion(caplog) == 1

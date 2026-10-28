"""The camera-map building blocks and the two maps, on synthetic video with a known truth."""

import math
from collections.abc import Iterable, Iterator

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="the PERSISTENCE maps require opencv")

from dataeval.core._persistence import (  # noqa: E402
    STEP,
    StartSampler,
    camera_maps,
    detrend,
    edge_sharpness,
    report,
    sample,
    solved_grid,
)
from tests.core.test_ego_stats import HEIGHT as H  # noqa: E402
from tests.core.test_ego_stats import WIDTH as W  # noqa: E402
from tests.core.test_ego_stats import texture  # noqa: E402

pytestmark = pytest.mark.optional

PATCH = (slice(90, 180), slice(320, 480))  # a burned-in patch: cells rows 12-23, columns 43-62
SOFT = (slice(135, 225), slice(380, 470))  # a blurred region of about 12 x 12 cells


def iter_scene(n, shift=3, *, still_until=0, size=(H, W), seed=0) -> Iterator[np.ndarray]:
    """Yield frames of a textured scene, still until `still_until`, then panning left `shift` px a frame, unwrapped."""
    h, w = size
    big = texture(seed, (h, w + shift * max(n - still_until, 0)), blur=1.0)
    for k in range(n):
        x = shift * max(k - still_until, 0)
        yield big[:, x : x + w].copy()


def scene(n, shift=3, **kwargs) -> list[np.ndarray]:
    """Return :func:`iter_scene` as a list."""
    return list(iter_scene(n, shift, **kwargs))


def with_patch_iter(frames: Iterable[np.ndarray], region=PATCH) -> Iterator[np.ndarray]:
    """Yield `frames` with one fixed textured patch burned in at `region`."""
    patch = None
    for frame in frames:
        out = frame.copy()
        if patch is None:
            patch = texture(9, out[region].shape, blur=1.0)
        out[region] = patch
        yield out


def with_patch(frames, region=PATCH) -> list[np.ndarray]:
    """Return :func:`with_patch_iter` as a list."""
    return list(with_patch_iter(frames, region))


def with_blur(frames, region=SOFT, sigma=1.5) -> list[np.ndarray]:
    """Return `frames` with `region` Gaussian-blurred in every frame, as a defect fixed on the lens."""
    out = []
    for frame in frames:
        soft = frame.copy()
        soft[region] = cv2.GaussianBlur(frame, (0, 0), sigma)[region]
        out.append(soft)
    return out


def soft_right(frames, sigma=1.0) -> list[np.ndarray]:
    """Return `frames` sharp on the left and blurred on the right, through a smoothstep over columns 280-400."""
    alpha = np.clip((np.arange(frames[0].shape[1]) - 280) / 120, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    return [np.round((1 - alpha) * f + alpha * cv2.GaussianBlur(f, (0, 0), sigma)).astype(np.uint8) for f in frames]


def cells(region, inset=0, size=(H, W), grid=(48, 85)) -> tuple[slice, slice]:
    """Return the grid cells wholly inside a pixel region, shrunk by `inset` cells."""
    (h, w), (rows, cols) = size, grid
    return (
        slice(math.ceil(region[0].start * rows / h) + inset, math.floor(region[0].stop * rows / h) - inset),
        slice(math.ceil(region[1].start * cols / w) + inset, math.floor(region[1].stop * cols / w) - inset),
    )


def away(region, by, size=(H, W), grid=(48, 85)) -> np.ndarray:
    """Return a mask of the cells at least `by` cells from every cell a pixel region touches."""
    (h, w), (rows, cols) = size, grid
    out = np.ones(grid, bool)
    r0, r1 = math.floor(region[0].start * rows / h) - by, math.ceil(region[0].stop * rows / h) + by
    c0, c1 = math.floor(region[1].start * cols / w) - by, math.ceil(region[1].stop * cols / w) + by
    out[max(r0, 0) : r1, max(c0, 0) : c1] = False
    return out


def maps(frames, measure=edge_sharpness, *, blur=True, sigma=6.0):
    """Return the maps of `frames`, every frame sampled and a start every `STEP` frames."""
    grid = solved_grid(*frames[0].shape)
    samples = {k: sample(f, grid, measure if blur else None) for k, f in enumerate(frames)}
    return camera_maps(samples, range(0, len(frames), STEP), grid, blur=blur, sigma=sigma)


def wrong_shape(gray, grid, /):
    return np.ones((2, 2), np.float32)


def not_finite(gray, grid, /):
    return np.full(grid, np.nan, np.float32)


def negative(gray, grid, /):
    return -np.ones(grid, np.float32)


def test_solved_grid_follows_the_aspect():
    assert solved_grid(1080, 1920) == (48, 85)
    assert solved_grid(480, 640) == (48, 64)
    assert solved_grid(1920, 1080) == (48, 27)


def test_edge_sharpness_is_zero_on_flat_cells_and_positive_on_texture():
    flat = edge_sharpness(np.full((H, W), 128, np.float32), (48, 85))
    assert flat.shape == (48, 85)
    assert flat.dtype == np.float32
    assert np.allclose(flat, 0, atol=1e-3)
    assert (edge_sharpness(texture(0).astype(np.float32), (48, 85)) > 0).all()


@pytest.mark.parametrize("measure", [wrong_shape, not_finite, negative])
def test_a_measure_breaking_its_contract_raises_by_name(measure):
    with pytest.raises(ValueError, match=measure.__name__):
        sample(texture(0), (48, 85), measure)


def test_a_measure_may_return_zeros():
    s = sample(texture(0), (48, 85), lambda gray, grid, /: np.zeros(grid, np.float32))
    np.testing.assert_array_equal(s.log_s, np.log(1e-6))
    assert s.log_median == np.log(1e-6)


def test_a_sample_keeps_its_own_copy_at_360_px():
    gray = texture(0)
    s = sample(gray, (48, 85), None)
    assert s.proc.shape == (360, 640)
    assert not np.shares_memory(s.proc, gray)
    assert s.log_s is None
    assert np.isnan(s.log_median)
    assert sample(np.zeros((720, 1280), np.uint8), (48, 85), None).proc.shape == (360, 640)


def test_starts_never_exceed_the_ceiling_and_cover_the_video():
    starts = StartSampler(8)
    for i in range(400):
        starts.offer(i)
        assert len(starts.kept) <= 8
    assert starts.stride % 5 == 0
    assert starts.kept == list(range(0, max(starts.kept) + 1, starts.stride))
    assert max(starts.kept) + starts.stride >= 400


def test_a_short_video_keeps_every_fifth_frame():
    starts = StartSampler(32)
    for i in range(150):
        starts.offer(i)
    assert starts.kept == list(range(0, 150, 5))
    assert starts.stride == 5


def test_a_ceiling_of_one_keeps_frame_zero():
    starts = StartSampler(1)
    for i in range(100):
        starts.offer(i)
    assert starts.kept == [0]


def test_a_frame_is_needed_as_a_start_or_a_partner():
    starts = StartSampler(32)
    for i in range(15):
        starts.offer(i)  # kept: 0, 5, 10
    assert starts.needs(20, blur=False)  # 10 + 10
    assert not starts.needs(35, blur=False)
    assert starts.needs(35, blur=True)  # 5 + 30
    assert not starts.needs(31, blur=True)


def test_detrend_removes_a_ramp_and_keeps_a_bump_and_nan():
    cells = np.tile(np.linspace(-1.0, 1.0, 85), (48, 1))
    cells[22:25, 40:43] -= 1.0
    cells[0:5, 0:5] = np.nan
    out = detrend(cells, 6.0)
    assert np.isnan(out[0:5, 0:5]).all()
    assert np.isfinite(out[5:, 5:]).all()
    assert np.abs(out[30:48, 20:65]).max() < 0.06
    assert out[23, 41] < -0.8


def test_report_leaves_a_16_9_map_unchanged():
    m = np.random.default_rng(0).normal(size=(48, 85))
    m[3:6, 10:20] = np.nan
    np.testing.assert_array_equal(report(m), m)


@pytest.mark.parametrize("cols", [27, 64, 112])
def test_report_fits_any_aspect_without_losing_known_cells(cols):
    m = np.random.default_rng(0).normal(size=(48, cols))
    out = report(m)
    assert out.shape == (48, 85)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    m[:, cols // 2] = np.nan
    assert np.isnan(report(m)).sum() <= 48


def test_report_keeps_a_thin_defect_when_shrinking():
    m = np.zeros((48, 112))
    m[:, 50] = -1.0
    assert report(m).min() <= -0.5


def test_report_of_an_unknown_map_is_unknown():
    assert np.isnan(report(np.full((48, 64), np.nan))).all()


def test_a_burned_in_patch_persists_and_gives_no_blur_equations():
    m = maps(with_patch(scene(40)))
    inside = cells(PATCH, inset=1)
    assert np.isfinite(m.ego_map[inside]).all()
    assert m.ego_map[inside].min() >= 0.9
    out = away(PATCH, 2)
    assert np.nanmax(m.ego_map[out]) <= 0.1
    assert np.isfinite(m.ego_map[out]).mean() >= 0.9
    assert np.isnan(m.blur_map_raw[inside]).all()


def test_a_still_camera_reads_nan_and_says_why():
    m = maps(with_patch(scene(40, shift=0)))
    assert np.isnan(m.ego_map).all()
    assert np.isnan(m.blur_map).all()
    assert any("ego_map" in w and "camera motion" in w for w in m.warnings)


def test_local_blur_is_found():
    m = maps(with_blur(scene(60)))
    flagged = m.blur_map <= np.log(0.5)
    assert flagged[cells(SOFT, inset=1)].mean() >= 0.6
    out = away(SOFT, 3) & np.isfinite(m.blur_map)
    assert flagged[out].mean() <= 0.05


def test_a_clean_video_is_clean():
    m = maps(scene(60))
    known = np.isfinite(m.blur_map)
    assert known.mean() >= 0.8
    assert (m.blur_map[known] <= np.log(0.5)).mean() < 0.05


def test_wide_blur_shows_in_the_raw_map_only():
    m = maps(soft_right(scene(60)))

    def gap(a):
        return np.nanmean(a[:, 57:]) - np.nanmean(a[:, :28])

    assert gap(m.blur_map_raw) < np.log(0.7)
    assert abs(gap(m.blur_map)) < 0.5 * abs(gap(m.blur_map_raw))


def test_uniform_softness_is_invisible():
    soft = [cv2.GaussianBlur(f, (0, 0), 1.5) for f in scene(60)]
    assert np.nanpercentile(np.abs(maps(soft).blur_map_raw), 95) < 0.25


def test_the_measure_scale_does_not_matter():
    frames = with_blur(scene(60))
    a = maps(frames)
    b = maps(frames, lambda gray, grid, /: 3 * edge_sharpness(gray, grid))
    np.testing.assert_allclose(b.blur_map, a.blur_map, atol=0.02, equal_nan=True)
    np.testing.assert_allclose(b.blur_map_raw, a.blur_map_raw, atol=0.02, equal_nan=True)


def test_a_custom_measure_shapes_the_map():
    weight = np.ones((48, 85), np.float32)
    weight[cells(SOFT)] = 0.25
    m = maps(scene(60), lambda gray, grid, /: edge_sharpness(gray, grid) * weight)
    assert (m.blur_map[cells(SOFT, inset=1)] <= np.log(0.5)).mean() >= 0.6


def test_no_frames_read_nan():
    m = camera_maps({}, [], None, blur=True, sigma=6.0)
    assert m.ego_map.shape == (48, 85)
    assert np.isnan(m.ego_map).all()
    assert np.isnan(m.blur_map).all()
    assert any("too few frames" in w for w in m.warnings)

"""Tests for the overlay mask the ego path takes: validating it, fitting it to each frame size, reading around it."""

import shutil
from functools import partial

import numpy as np
import pytest

import dataeval.core._ego_stats as ego_stats_module
from dataeval.core._ego_stats import (
    _MASK_MARGIN,
    _TWO_MOTION_FLOOR,
    FlowMethod,
    _as_mask,
    _biweight_weights,
    _clear_of_overlay,
    _fit_irls,
    _flow_estimator,
    _FlowFrame,
    _grid_flow_dis,
    _measure_pair,
    _OverlayMask,
    _pair_ego,
    _photometric_agreement,
    _quadratic_residual,
    _robust_cutoff,
    _two_motion,
    _two_motion_residual,
    ego_stats,
)
from dataeval.core._video_stats import video_stats
from dataeval.flags import VideoStats
from tests.core.test_ego_stats import HEIGHT, WIDTH, FakeFrame, ambiguous_stream, panning_stream, texture
from tests.fixtures.video_generation_scripts.video_corruptions import make_canonical_mp4

cv2 = pytest.importorskip("cv2", reason="ego_stats requires opencv")

pytestmark = pytest.mark.optional

LINES = [
    "LAT 34.12345 N  LON 117.54321 W",
    "ALT 1520 FT  HDG 274  SPD 85 KT",
    "TGT RNG 2.4 NM  EL -12.5",
    "2026-10-22 19:03:11Z  MISB 0601",
    "SNSR EO  FOV 4.2 DEG  ZOOM 12X",
]
SIDE = ["ALT 1520", "HDG 274", "SPD 85", "RNG 2.4", "EL -12"]


def _text(out, s, org):
    """Draw ``s`` as a 4 px black stroke under a 1 px white one, without antialiasing."""
    cv2.putText(out, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 4)
    cv2.putText(out, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)


def hud(img, dense=True):
    """The !1119 review's HUD: telemetry text top and bottom (2 lines light, 5 dense), a reticle and a box."""
    out = img.copy()
    for i, s in enumerate(LINES if dense else LINES[:2]):
        for y in (22 + 26 * i, HEIGHT - 18 - 26 * i):
            _text(out, s, (10, y))
    cv2.line(out, (320, 140), (320, 220), 255, 2)
    cv2.line(out, (280, 180), (360, 180), 255, 2)
    cv2.rectangle(out, (200, 100), (440, 260), 255, 1)
    return out


def side_hud(img):
    """Short readouts down the left and right edges, in `hud`'s strokes."""
    out = img.copy()
    for i, s in enumerate(SIDE):
        for x in (10, WIDTH - 110):
            _text(out, s, (x, 40 + 60 * i))
    return out


def drawn_mask(draw, shape):
    """Every pixel ``draw`` changes, found by drawing over two flat gray levels."""
    a, b = np.full(shape, 77, np.uint8), np.full(shape, 177, np.uint8)
    return (draw(a) != a) | (draw(b) != b)


def hud_pan(shift, n, *, dense=True, seed=0, dy=0, draw=None, channels=1):
    """``n`` frames of a texture moving ``shift`` px right and ``dy`` down a frame, under a fixed overlay."""
    base = texture(seed, blur=4)
    draw = draw or partial(hud, dense=dense)
    return [FakeFrame(draw(np.roll(np.roll(base, k * dy, axis=0), k * shift, axis=1)), k, channels) for k in range(n)]


DENSE = drawn_mask(hud, (HEIGHT, WIDTH))
CORNER = np.ones((HEIGHT, WIDTH), bool)
CORNER[-96:, -128:] = False  # a 128x96 corner, trimmed to 96x64 by the margin
BOTTOM_STRIP = np.zeros((HEIGHT, WIDTH), bool)
BOTTOM_STRIP[int(0.88 * HEIGHT) :] = True


class TestMaskFit:
    def test_nonzero_is_overlay_whatever_the_dtype(self):  # Review Focus 1
        bool_mask = np.zeros((360, 640), bool)
        bool_mask[:40] = True
        for raw in (bool_mask, bool_mask.astype(np.uint8) * 255, bool_mask.astype(np.float32)):
            np.testing.assert_array_equal(_as_mask(raw, "mask"), bool_mask)

    def test_no_mask_and_an_all_false_mask_are_none(self):
        assert _as_mask(None, "mask") is None
        assert _as_mask(np.zeros((360, 640)), "mask") is None

    @pytest.mark.parametrize("bad", [np.ones((3, 360, 640)), np.ones(640), np.ones((0, 640))])
    def test_a_mask_that_is_not_a_2d_image_raises(self, bad):
        with pytest.raises(ValueError, match=r"mask"):
            _as_mask(bad, "mask")

    def test_a_uniformly_scaled_mask_matches_a_native_one(self):
        native = np.zeros((360, 640), bool)
        native[300:, 100:300] = True
        double = np.repeat(np.repeat(native, 2, axis=0), 2, axis=1)
        fit_native = _OverlayMask(native, "mask").fit((640, 360), (360, 640))
        fit_double = _OverlayMask(double, "mask").fit((640, 360), (360, 640))
        np.testing.assert_array_equal(fit_native.overlay, fit_double.overlay)

    def test_bottom_padding_keeps_the_rows(self):
        mask = np.zeros((1080, 1920), bool)
        mask[1000:1080] = True  # a bottom HUD band
        fitted = _OverlayMask(mask, "mask").fit((1920, 1088), (1088, 1920))
        assert fitted.overlay[1000:1080].all()
        assert not fitted.overlay[1080:].any()
        assert not fitted.overlay[:1000].any()

    @pytest.mark.parametrize(
        "frame_wh",
        [(1920, 1072), (1920, 1104), (1376, 768), (1080, 1920)],
        ids=["crop", "padding over 1%", "aspect change", "transposed"],  # transposed: Review Focus 2
    )
    def test_geometry_that_is_not_a_scale_or_padding_raises(self, frame_wh):
        with pytest.raises(ValueError, match=r"1920x1080.*" + r"x".join(map(str, frame_wh))):
            _OverlayMask(np.ones((1080, 1920), bool), "mask").fit(frame_wh, (frame_wh[1], frame_wh[0]))

    def test_any_source_pixel_masks_the_processed_pixel(self):
        mask = np.zeros((720, 1280), bool)
        mask[0, 0] = True
        assert _OverlayMask(mask, "mask").fit((1280, 720), (360, 640)).overlay[0, 0]

    def test_near_reaches_the_margin_and_no_further(self):
        mask = np.zeros((360, 640), bool)
        mask[180, 320] = True
        near = _OverlayMask(mask, "mask").fit((640, 360), (360, 640)).near
        assert near[180, 320 + _MASK_MARGIN]
        assert not near[180, 320 + _MASK_MARGIN + 1]
        assert near[180 + _MASK_MARGIN, 320 + _MASK_MARGIN]  # square, as the probe measured


def _pair(frames, mask=None, flow: FlowMethod = "dis", gap: int = 1):
    est = _flow_estimator(flow)
    previous, current = (_FlowFrame(f.pixels[0], (WIDTH, HEIGHT)) for f in frames[:2])
    fitted = None if mask is None else _OverlayMask(mask, "mask").fit((WIDTH, HEIGHT), (HEIGHT, WIDTH))
    return _measure_pair(est, previous, current, 40, gap, mask=fitted)


def _about_center(*motions):
    """Frames of one texture, each rolled ``angle`` degrees and zoomed ``scale`` about the frame center."""
    big = texture(0, size=(560, 840), blur=4)
    frames = []
    for k, (angle, scale) in enumerate(motions):
        window = cv2.getRotationMatrix2D((420, 280), angle, scale)
        window[:, 2] -= 100
        frames.append(FakeFrame(cv2.warpAffine(big, window, (WIDTH, HEIGHT)), k))
    return frames


class TestMaskedPair:
    @pytest.mark.parametrize("seed", range(3))
    def test_a_dense_overlay_pan_is_refused_without_a_mask(self, seed):  # decision 5, revised by #1577
        """The slow reading a dense overlay drags a pan to is a compromise of two motions, so it is refused."""
        reading = _pair(hud_pan(8, 2, seed=seed))
        assert reading is None or not reading.trusted

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("shift", [8, -8, 24, -24])
    def test_a_masked_dense_overlay_reads_the_true_pan(self, seed, shift):
        reading = _pair(hud_pan(shift, 2, seed=seed), DENSE)
        assert reading is not None
        assert reading.trusted
        assert reading.t0x == pytest.approx(shift, abs=0.5)

    @pytest.mark.parametrize("seed", range(3))
    def test_a_masked_light_overlay_reads_a_pan_it_refuses_today(self, seed):
        light = partial(hud, dense=False)
        today = _pair(hud_pan(8, 2, seed=seed, draw=light))
        assert today is None or not today.trusted
        reading = _pair(hud_pan(8, 2, seed=seed, draw=light), drawn_mask(light, (HEIGHT, WIDTH)))
        assert reading is not None
        assert reading.trusted
        assert reading.t0x == pytest.approx(8, abs=0.5)

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("dense", [True, False])
    def test_a_masked_still_camera_reads_static(self, seed, dense):
        reading = _pair(hud_pan(0, 2, seed=seed, dense=dense), drawn_mask(partial(hud, dense=dense), (HEIGHT, WIDTH)))
        assert reading is not None
        assert reading.trusted
        assert reading.motion == "static"

    def test_a_mask_leaving_too_few_samples_refuses_the_pair(self):
        mask = np.ones((HEIGHT, WIDTH), bool)
        mask[148:212, 288:352] = False  # a 64x64 box at the center
        assert _pair(hud_pan(8, 2), mask) is None

    def test_samples_bunched_in_one_corner_refuse_the_pair(self):  # spec test 10
        frames = _about_center((0, 1.0), (0, 1.02))  # a 2% zoom
        p, c = (f.pixels[0] for f in frames)
        pts, flow = _grid_flow_dis(_flow_estimator("dis"), p, c, 40)
        near = _OverlayMask(CORNER, "mask").fit((WIDTH, HEIGHT), (HEIGHT, WIDTH)).near
        keep = ~near[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        # the corner's samples alone read the zoom as a trusted pan
        before = _pair_ego(pts[keep], flow[keep], (WIDTH, HEIGHT), (WIDTH, HEIGHT), 1, frames=(p, c, 16))
        assert before is not None
        assert before.trusted
        assert abs(before.t0x) > 0.5
        assert _pair(frames, CORNER) is None

    @pytest.mark.parametrize(("angle", "scale"), [(1.0, 1.0), (0, 1.02)], ids=["roll 1 degree", "zoom 2%"])
    def test_inliers_bunched_in_one_corner_refuse_the_pair(self, angle, scale):
        frames = _about_center((0, 1.0), (angle, scale))
        mask = CORNER.copy()
        mask[20:120, 20:120] = mask[240:340, 20:120] = False  # two windows on the left reach a third quadrant
        p, c = (f.pixels[0] for f in frames)
        fitted = _OverlayMask(mask, "mask").fit((WIDTH, HEIGHT), (HEIGHT, WIDTH))
        pts, flow = _clear_of_overlay(*_grid_flow_dis(_flow_estimator("dis"), p, c, 40), fitted, (WIDTH, HEIGHT))
        # the kept samples pass the quadrant rule, but the fit's inliers all sit in the corner and read a trusted pan
        before = _pair_ego(pts, flow, (WIDTH, HEIGHT), (WIDTH, HEIGHT), 1, frames=(p, c, 16))
        assert before is not None
        assert before.trusted
        assert np.hypot(before.t0x, before.t0y) > 0.5
        assert _pair(frames, mask) is None

    def test_the_agreement_ignores_overlay_pixels(self):
        p, c = (f.pixels[0] for f in hud_pan(8, 2))
        pts, flow = _grid_flow_dis(_flow_estimator("dis"), p, c, 40)
        fitted = _OverlayMask(DENSE, "mask").fit((WIDTH, HEIGHT), (HEIGHT, WIDTH))
        keep = ~fitted.near[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        fit = _fit_irls(pts[keep], flow[keep])
        assert fit is not None
        inlier = np.ones(len(pts), bool)
        assert _photometric_agreement(fit, pts, inlier, p, c, 16, overlay=None) <= 0.70
        assert _photometric_agreement(fit, pts, inlier, p, c, 16, overlay=fitted.overlay) >= 0.99


def _paste(img, obj, x, y):
    """Paste ``obj`` with its top-left at (x, y), clipped to the frame."""
    h, w = obj.shape
    sx, sy, dx0, dy0 = max(0, -x), max(0, -y), max(0, x), max(0, y)
    dx1, dy1 = min(WIDTH, x + w), min(HEIGHT, y + h)
    if dx1 > dx0 and dy1 > dy0:
        img[dy0:dy1, dx0:dx1] = obj[sy : sy + dy1 - dy0, sx : sx + dx1 - dx0]


def _still_mover(rect, v, *, seed=0, draw=None):
    """A still camera over texture, with a textured object at ``rect`` (x, y, w, h) moving ``v`` px."""
    x0, y0, w, h = rect
    background, obj = texture(seed, blur=4), texture(seed + 1000, size=(h, w), blur=4)
    frames = []
    for k, (dx, dy) in enumerate(((0, 0), v)):
        img = background.copy()
        _paste(img, obj, x0 + dx, y0 + dy)
        frames.append(FakeFrame(draw(img) if draw else img, k))
    return frames


def _pan_with_mover(pan, rect, v, *, seed=0):
    """A camera panning ``pan`` px over texture, with a textured object at ``rect`` (x, y, w, h) moving ``v`` px."""
    x0, y0, w, h = rect
    background, obj = texture(seed, blur=4), texture(seed + 1000, size=(h, w), blur=4)
    frames = []
    for k in range(2):
        img = np.roll(background, k * pan, axis=1)
        _paste(img, obj, x0 + k * v, y0)
        frames.append(FakeFrame(img, k))
    return frames


def _spy_two_motion(monkeypatch):
    """The arguments of every `_two_motion` call, which the ego walk makes only for a pair nothing else refused."""
    calls = []
    real = ego_stats_module._two_motion

    def spy(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(ego_stats_module, "_two_motion", spy)
    return calls


def _split(pts, flow):
    """The camera fit's residuals and outliers, as `_pair_ego` passes them, for a hand-built flow field."""
    fit = _fit_irls(pts, flow)
    assert fit is not None
    resid = fit.residuals(pts, flow)
    speed = np.hypot(*resid.T)
    return resid, _biweight_weights(speed, _robust_cutoff(speed)) <= 0.05


def _lattice_points():
    ys, xs = np.mgrid[8:HEIGHT:16, 8:WIDTH:16]
    return np.c_[xs.ravel(), ys.ravel()].astype(np.float64)


def _with_flow_noise(flow, sigma=0.2, seed=0):
    """``flow`` plus the per-sample noise real optical flow carries, so the two-motion split actually runs."""
    return flow + np.random.default_rng(seed).normal(0, sigma, flow.shape)


def _strength(pts, flow, resid, outlier):
    """`q / two` of a split: the quadratic flow's median residual over the camera-plus-one-mover split's."""
    two = _two_motion_residual(pts, flow, resid)
    assert two is not None
    return _quadratic_residual(pts, flow, ~outlier) / two


def _half_step(step, seed=0):
    """Lattice flow with a ``step`` px jump across the middle of the frame, plus flow noise."""
    pts = _lattice_points()
    return pts, _with_flow_noise(np.where((pts[:, 0] >= WIDTH / 2)[:, None], [step, 0.0], [0.0, 0.0]), seed=seed)


def _gain(pts, flow, resid, outlier):
    """`single / two` of a split: the camera fit's median inlier residual over the camera-plus-one-mover split's."""
    two = _two_motion_residual(pts, flow, resid)
    assert two is not None
    return float(np.median(np.hypot(*resid[~outlier].T))) / two


def _bowed(bow, step=0.0, seed=0):
    """Lattice flow whose x part grows with the square of the distance from the center column, plus flow noise.

    It is ``bow`` px at the left and right edges. The right 30% of the frame moves ``step`` px further, if there is one.
    """
    pts = _lattice_points()
    across = (pts[:, 0] - WIDTH / 2) / (WIDTH / 2)
    return pts, _with_flow_noise(np.c_[bow * across**2 + step * (across > 0.4), np.zeros(len(pts))], seed=seed)


def _noisy(frames, sigma, seed=0):
    """``frames`` with independent Gaussian pixel noise on each."""
    rng = np.random.default_rng(seed)
    return [
        FakeFrame(
            np.clip(f.pixels[0] + rng.normal(0, sigma, f.pixels[0].shape), 0, 255).astype(np.uint8), f.frame_index
        )
        for f in frames
    ]


def _yaw(degrees, fov_degrees=60, seed=0):
    """A camera turning ``degrees`` in place over a far textured plane: a homography the affine fit leaves out."""
    big = texture(seed, size=(2 * HEIGHT, 2 * WIDTH), blur=4)
    focal = (WIDTH / 2) / np.tan(np.radians(fov_degrees) / 2)
    k = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]])
    shift = np.array([[1, 0, -WIDTH / 2], [0, 1, -HEIGHT / 2], [0, 0, 1]], float)
    frames = []
    for i, angle in enumerate((0.0, np.radians(degrees))):
        turn = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        frames.append(FakeFrame(cv2.warpPerspective(big, k @ turn @ np.linalg.inv(k) @ shift, (WIDTH, HEIGHT)), i))
    return frames


class TestQuadraticResidual:
    """The robust quadratic flow behind the two-motion test's second refusal: what it explains and what it must not."""

    def test_an_exact_quadratic_flow_has_no_residual(self):
        pts = _lattice_points()
        x, y = pts[:, 0] - WIDTH / 2, pts[:, 1] - HEIGHT / 2
        flow = np.c_[0.001 * (x * x + x * y) + 0.5, 0.001 * (y * y - x * y) - 0.3]
        assert _quadratic_residual(pts, flow, np.ones(len(pts), bool)) < 1e-9

    def test_a_step_in_the_flow_is_not_absorbed(self):
        pts = _lattice_points()
        flow = _with_flow_noise(np.where((pts[:, 0] >= WIDTH / 2)[:, None], [4.0, 0.0], [0.0, 0.0]))
        assert _quadratic_residual(pts, flow, np.ones(len(pts), bool)) > 0.5

    @pytest.mark.parametrize("rows", [1, 2, 3])
    def test_samples_in_one_to_three_rows_do_not_raise(self, rows):
        pts = _lattice_points()
        pts = pts[pts[:, 1] < 8 + 16 * rows]
        flow = _with_flow_noise(np.random.default_rng(rows).normal(0, 1, pts.shape), 0.1)
        assert np.isfinite(_quadratic_residual(pts, flow, np.ones(len(pts), bool)))

    def test_fewer_than_six_samples_cannot_refuse(self):
        rng = np.random.default_rng(0)
        pts, flow = rng.uniform(0, [WIDTH, HEIGHT], (5, 2)), rng.normal(0, 1, (5, 2))
        assert _quadratic_residual(pts, flow, np.ones(5, bool)) == pytest.approx(0.0, abs=1e-9)

    def test_gross_outliers_are_reweighted_away(self):
        """An exact quadratic flow with 20% gross outliers: the Tukey rounds leave none, least squares leaves 1.8."""
        pts = _lattice_points()
        x, y = pts[:, 0] - WIDTH / 2, pts[:, 1] - HEIGHT / 2
        flow = np.c_[0.001 * (x * x + x * y) + 0.5, 0.001 * (y * y - x * y) - 0.3]
        flow[::5] += [6, -6]
        assert _quadratic_residual(pts, flow, np.ones(len(pts), bool)) < 1e-6

    def test_only_the_kept_samples_are_fitted(self):
        """Half the samples step away from an exact quadratic, too many for the reweighting to drop: only `keep` can."""
        pts = _lattice_points()
        x, y = pts[:, 0] - WIDTH / 2, pts[:, 1] - HEIGHT / 2
        flow = np.c_[0.001 * (x * x + x * y) + 0.5, 0.001 * (y * y - x * y) - 0.3]
        right = pts[:, 0] >= WIDTH / 2
        flow[right] += [6, -6]
        assert _quadratic_residual(pts, flow, ~right) < 1e-9
        assert _quadratic_residual(pts, flow, np.ones(len(pts), bool)) > 0.5

    def test_the_order_of_the_samples_does_not_matter(self):
        pts = _lattice_points()
        flow = _with_flow_noise(np.where((pts[:, 0] >= WIDTH / 2)[:, None], [2.0, 0.0], [0.0, 0.0]))
        keep = np.ones(len(pts), bool)
        order = np.random.default_rng(0).permutation(len(pts))
        once = _quadratic_residual(pts, flow, keep)
        assert abs(_quadratic_residual(pts[order], flow[order], keep) - once) < 1e-9 * once
        assert _quadratic_residual(pts, flow, keep) == once
        # and no decision: this split is stable, a refusal over the bar of 1.5 whatever the order
        assert _two_motion(pts, flow, *_split(pts, flow), 1)
        assert _two_motion(pts[order], flow[order], *_split(pts[order], flow[order]), 1)


class TestTwoMotions:
    """#1577: a still camera with a coherent mover holding up to half the samples is never a trusted motion."""

    @pytest.mark.parametrize(
        ("rect", "v", "mask"),
        [
            ((425, 30, 200, 207), (0, 8), DENSE),  # 18% under the dense mask; main read a trusted 1.9 px zoom
            ((320, 0, 320, 274), (8, 0), DENSE),  # 38% under the dense mask; main read a trusted 8 px pan
            ((320, 0, 320, 360), (8, 0), None),  # half the frame, no mask; main read a trusted 6.9 px zoom
        ],
    )
    def test_a_still_camera_with_a_mover_is_static_or_refused(self, rect, v, mask):
        reading = _pair(_still_mover(rect, v, draw=hud if mask is not None else None), mask)
        assert reading is None or not reading.trusted or reading.motion == "static"

    def test_two_translations_split_in_half_are_two_motions(self):
        pts = _lattice_points()
        flow = _with_flow_noise(np.where((pts[:, 0] >= WIDTH / 2)[:, None], [8.0, 0.0], [0.0, 0.0]))
        assert _two_motion(pts, flow, *_split(pts, flow), 1)

    def test_a_pure_zoom_is_one_motion(self):
        pts = _lattice_points()
        flow = _with_flow_noise(0.02 * (pts - [WIDTH / 2, HEIGHT / 2]))
        assert not _two_motion(pts, flow, *_split(pts, flow), 1)

    def test_a_pan_with_a_small_mover_is_one_motion(self):
        pts = _lattice_points()
        mover = pts[:, 0] < 0.15 * WIDTH
        flow = _with_flow_noise(np.where(mover[:, None], [-3.0, 2.0], [5.0, 0.0]))
        assert not _two_motion(pts, flow, *_split(pts, flow), 1)

    def test_a_split_is_judged_per_frame_of_gap(self):
        """A split the quadratic flow leaves 2.0 times over is two motions over one frame, but not over three.

        The quadratic bar scales with the square root of the gap, an empirical choice: 1.5 over one frame, 2.6 over
        three, and this split falls between them.
        """
        pts = _lattice_points()
        flow = _with_flow_noise(np.where((pts[:, 0] >= WIDTH / 2)[:, None], [2.1, 0.0], [0.0, 0.0]))
        split = _split(pts, flow)
        assert 1.9 < _strength(pts, flow, *split) < 2.1
        assert _two_motion(pts, flow, *split, 1)
        assert not _two_motion(pts, flow, *split, 3)

    def test_a_bowed_flow_is_one_motion_because_the_quadratic_flow_explains_it(self):
        """The split explains a flow bowed 4 px by a quadratic 2.1 times better than the affine: over the bar of 1.5.

        It is under the old bar of 3, so only the quadratic test can refuse it. The quadratic flow explains it to the
        noise, 0.5 times the split's residual, so it does not.
        """
        pts, flow = _bowed(4.0)
        split = _split(pts, flow)
        assert 1.5 < _gain(pts, flow, *split) < 3.0
        assert _strength(pts, flow, *split) < 1.5
        assert not _two_motion(pts, flow, *split, 1)

    def test_the_old_bar_is_still_judged_per_frame_of_gap(self):
        """A split the quadratic flow explains is refused over one frame by the old bar of 3, and not over three, by 9.

        A bowed flow with the right 30% of the frame moving 12 px further: the split explains it 6 times better than the
        affine, over the old bar of 3 and under 9. It is over 5.2 as well, which the old bar over three frames would be
        if it scaled with the square root of the gap. The quadratic flow sets the moving side aside and leaves 1.2 times
        the split's residual, under the quadratic bar of 1.5, 2.6 over three frames, so only the old bar can refuse it.
        """
        pts, flow = _bowed(10.0, step=12.0)
        split = _split(pts, flow)
        assert 5.4 < _gain(pts, flow, *split) < 9.0
        assert _strength(pts, flow, *split) < 1.5
        assert _two_motion(pts, flow, *split, 1)
        assert not _two_motion(pts, flow, *split, 3)

    def test_a_yaw_at_a_wide_field_of_view_is_trusted(self):
        """A correct reading the two-motion split explains only about 2.4 times better than the fit: under the bar."""
        reading = _pair(_yaw(0.5))
        assert reading is not None
        assert reading.trusted

    def test_a_slow_half_frame_mover_in_noisy_footage_is_refused(self):
        """A 5 px mover in noisy frames, whose split is about 4.5 times better than the fit: over the bar."""
        reading = _pair(_noisy(_still_mover((320, 0, 320, 360), (5, 0)), 3))
        assert reading is not None
        assert reading.ambiguous

    def test_a_mover_across_a_wider_gap_is_still_refused(self):
        """A half-frame mover at 4 px a frame over three frames: judged per frame, it is still two motions.

        Faster than this, the compromise fails the photometric agreement check first, which refuses it anyway.
        """
        reading = _pair(_still_mover((320, 0, 320, 360), (12, 0)), gap=3)
        assert reading is not None
        assert reading.ambiguous

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize(("sigma", "v"), [(1, 1), (1, 2), (1, 3), (3, 2), (3, 3)])
    def test_a_slow_half_frame_mover_in_noise_is_refused(self, sigma, v, seed):
        """#1592: a half-frame mover of 1 to 3 px in noisy frames drags the fit to a trusted zoom of a pixel or two.

        The quadratic flow leaves at least 2.0 times the split's residual in every cell, over a bar of 1.5.
        """
        reading = _pair(_noisy(_still_mover((320, 0, 320, 360), (v, 0), seed=seed), sigma, seed=seed))
        assert reading is not None
        assert reading.ambiguous

    @pytest.mark.parametrize("seed", range(5))
    def test_a_slow_half_frame_mover_over_two_frames_is_refused(self, seed):
        """A half-frame mover of 1 px a frame over two frames, in noisy footage: the quadratic bar is 2.1 here.

        The quadratic flow leaves 3.9 to 4.5 times the split's residual, under the old bar of 6, so only the quadratic
        test refuses it.
        """
        reading = _pair(_noisy(_still_mover((320, 0, 320, 360), (2, 0), seed=seed), 1, seed=seed), gap=2)
        assert reading is not None
        assert reading.ambiguous

    @pytest.mark.parametrize("seed", [0, 3, 4])
    @pytest.mark.parametrize(("step", "refused"), [(1.15, False), (1.55, True)])
    def test_the_quadratic_bar_is_pinned_from_both_sides(self, step, refused, seed):
        """A step the quadratic flow leaves 1.4 times the split's residual is not refused, and one at 1.65 times is.

        The seeds are three whose noise draws keep that ratio within 0.04 of one another, so moving the 1.5 bar by more
        than about 10% fails one side.
        """
        pts, flow = _half_step(step, seed)
        split = _split(pts, flow)
        assert _strength(pts, flow, *split) == pytest.approx(1.65 if refused else 1.4, abs=0.05)
        assert _two_motion(pts, flow, *split, 1) == refused

    @pytest.mark.parametrize("seed", range(3))
    def test_the_quadratic_bar_scales_with_the_square_root_of_the_gap(self, seed):
        """The quadratic flow leaves 2.3 times the split's residual: refused over 1 and 2 frames, not over 4.

        The bars are 1.5, 2.1 and 3.0. A bar of `1.5 * gap` would not refuse it over 2 frames, and a bar that
        ignored the gap would refuse it over 4.
        """
        pts, flow = _half_step(2.6, seed)
        split = _split(pts, flow)
        assert 2.25 < _strength(pts, flow, *split) < 2.45
        assert _two_motion(pts, flow, *split, 1)
        assert _two_motion(pts, flow, *split, 2)
        assert not _two_motion(pts, flow, *split, 4)

    @pytest.mark.parametrize("seed", range(5))
    def test_a_masked_slow_mover_is_refused(self, seed):
        reading = _pair(_noisy(_still_mover((320, 0, 320, 360), (2, 0), seed=seed), 3, seed=seed), BOTTOM_STRIP)
        assert reading is not None
        assert reading.ambiguous

    @pytest.mark.skipif(
        tuple(int(part) for part in cv2.__version__.split(".")[:2]) < (4, 8), reason="measured on OpenCV 4.8 and later"
    )
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("v", [1, 2])
    def test_a_slow_mover_under_lucas_kanade_is_refused(self, v, seed):
        reading = _pair(_noisy(_still_mover((320, 0, 320, 360), (v, 0), seed=seed), 1, seed=seed), flow="lk")
        assert reading is not None
        assert reading.ambiguous

    def test_a_pan_with_a_small_real_mover_stays_trusted(self):
        """A 5 px pan under a mover a tenth of the frame at the corner, 3 px a frame: the affine fit is inside the bar.

        The fit's residual is 1.1 times the split's, so the quadratic flow is not fitted.
        """
        reading = _pair(_noisy(_pan_with_mover(5, (0, 0, 203, 114), 3), 3))
        assert reading is not None
        assert reading.trusted

    def test_a_tiny_lattice_does_not_raise(self, monkeypatch):
        """Ten samples across the frame leave a lattice of 60: the quadratic flow still gives a reading or None."""
        kept = []
        fit = ego_stats_module._quadratic_residual

        def record(pts, flow, keep):
            kept.append(int(keep.sum()))
            return fit(pts, flow, keep)

        monkeypatch.setattr(ego_stats_module, "_quadratic_residual", record)
        frames = _noisy(_still_mover((320, 0, 320, 360), (1, 0)), 3)
        previous, current = (_FlowFrame(f.pixels[0], (WIDTH, HEIGHT)) for f in frames)
        _measure_pair(_flow_estimator("dis"), previous, current, 10, 1)
        assert kept == [60]

    def test_the_early_exit_and_the_skip_never_fit_the_quadratic(self, monkeypatch):
        """A fit at the floor and a fit within the bar are settled without the quadratic flow, which a mover reaches."""
        calls = _spy_two_motion(monkeypatch)

        def boom(*_):
            raise AssertionError("the quadratic flow was fitted")

        monkeypatch.setattr(ego_stats_module, "_quadratic_residual", boom)

        def settled(frames):
            """Read the pair, which must come out trusted, and return the affine inlier residual and the split's."""
            reading = _pair(frames)
            assert reading is not None
            assert reading.trusted
            pts, flow, resid, outlier, _ = calls[-1]
            two = _two_motion_residual(pts, flow, resid)
            assert two is not None
            return float(np.median(np.hypot(*resid[~outlier].T))), two

        single, _ = settled(hud_pan(5, 2, draw=lambda img: img))  # an exact pan: the fit is at the floor
        assert single <= _TWO_MOTION_FLOOR
        # Two motions in a nearly exact flow: either test would refuse it if the fit were not at the floor.
        pts = _lattice_points()
        flow = _with_flow_noise(np.where((pts[:, 0] >= WIDTH / 2)[:, None], [0.4, 0.0], [0.0, 0.0]), sigma=0.01)
        resid, outlier = _split(pts, flow)
        assert float(np.median(np.hypot(*resid[~outlier].T))) <= _TWO_MOTION_FLOOR
        assert not _two_motion(pts, flow, resid, outlier, 1)
        # A still camera and no mover, in noise: the fit is above the floor and within 1.5 times the split.
        single, two = settled(_noisy(_still_mover((320, 0, 320, 360), (0, 0)), 3))
        assert _TWO_MOTION_FLOOR < single <= 1.5 * two
        with pytest.raises(AssertionError, match="quadratic flow was fitted"):  # a mover does reach it
            _pair(_noisy(_still_mover((320, 0, 320, 360), (2, 0)), 3))

    def test_the_quadratic_flow_is_fitted_to_the_samples_the_fit_kept(self, monkeypatch):
        """A bowed flow with a corner mover reaches the quadratic flow, fitted to the 95% the fit kept, not all."""
        pts, flow = _bowed(4.0)
        flow[(pts[:, 0] < 0.15 * WIDTH) & (pts[:, 1] < 0.3 * HEIGHT)] += [0.0, 12.0]
        resid, outlier = _split(pts, flow)
        assert 0 < outlier.sum() < 0.1 * len(pts)
        kept = []
        fit = ego_stats_module._quadratic_residual

        def record(pts, flow, keep):
            kept.append(keep)
            return fit(pts, flow, keep)

        monkeypatch.setattr(ego_stats_module, "_quadratic_residual", record)
        assert not _two_motion(pts, flow, resid, outlier, 1)
        assert len(kept) == 1
        np.testing.assert_array_equal(kept[0], ~outlier)

    @pytest.mark.parametrize(
        ("build", "mask", "gap"),
        [
            pytest.param(lambda: hud_pan(5, 2, draw=lambda img: img), None, 1, id="pan"),
            pytest.param(lambda: _about_center((0, 1.0), (0, 1.02)), None, 1, id="zoom"),
            pytest.param(lambda: _about_center((0, 1.0), (1.0, 1.0)), None, 1, id="roll"),
            pytest.param(lambda: _noisy(_yaw(0.5), 0), None, 1, id="yaw-half-degree"),
            pytest.param(lambda: _noisy(_yaw(0.5), 3), None, 1, id="yaw-half-degree-noisy"),
            pytest.param(lambda: _noisy(_yaw(1), 0), None, 1, id="yaw-one-degree"),
            pytest.param(lambda: _noisy(_yaw(1), 3), None, 1, id="yaw-one-degree-noisy"),
            pytest.param(lambda: _noisy(_still_mover((320, 0, 320, 360), (0, 0)), 0), None, 1, id="still"),
            pytest.param(lambda: _noisy(_still_mover((320, 0, 320, 360), (0, 0)), 3), None, 1, id="still-noisy"),
            pytest.param(lambda: _noisy(_still_mover((0, 0, 203, 114), (3, 0)), 3), None, 1, id="small-mover"),
            pytest.param(
                lambda: _noisy(_still_mover((0, 0, 203, 114), (3, 0)), 3), BOTTOM_STRIP, 1, id="small-mover-masked"
            ),
            pytest.param(lambda: _noisy(_yaw(0.5), 0), None, 2, id="yaw-half-degree-gap-2"),
            pytest.param(lambda: _noisy(_still_mover((0, 0, 203, 114), (6, 0)), 3), None, 2, id="small-mover-gap-2"),
        ],
    )
    def test_a_pair_the_new_test_passes_is_unchanged_and_nothing_is_newly_trusted(self, monkeypatch, build, mask, gap):
        """With the quadratic test off, an infinite ratio, each of these pairs reads exactly as it does with it on.

        They are pairs the quadratic test passes: where the test only adds refusals to the old one, nothing is newly
        trusted, and a pair it does not refuse leaves every field of the reading alone, NaN for NaN.
        """
        frames = build()
        with monkeypatch.context() as off:
            off.setattr(ego_stats_module, "_QUADRATIC_RATIO", float("inf"))
            before = _pair(frames, mask, gap=gap)
        after = _pair(frames, mask, gap=gap)
        assert before is not None
        assert after is not None
        assert not (after.trusted and not before.trusted)
        np.testing.assert_equal(after, before)

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize(("sigma", "v"), [(1, 1), (3, 2)])
    def test_a_pair_the_new_test_refuses_differs_only_in_being_ambiguous(self, monkeypatch, sigma, v, seed):
        """With the quadratic test off, a slow mover reads trusted; with it on, the same reading is ambiguous.

        Nothing else about the reading changes, so the mover's fraction and speed, which users see on a refused pair,
        stay as they were.
        """
        frames = _noisy(_still_mover((320, 0, 320, 360), (v, 0), seed=seed), sigma, seed=seed)
        with monkeypatch.context() as off:
            off.setattr(ego_stats_module, "_QUADRATIC_RATIO", float("inf"))
            before = _pair(frames)
        after = _pair(frames)
        assert before is not None
        assert after is not None
        assert before.trusted
        assert after.ambiguous
        assert not after.trusted
        np.testing.assert_equal(after._replace(ambiguous=False), before)


def _pans(frames, **kwargs):
    """The ``pan_x`` and ``ego_trusted`` columns of frames 1 onward."""
    stats = ego_stats(frames, **kwargs)["stats"]
    return np.asarray(stats["pan_x"][1:], float), np.asarray(stats["ego_trusted"][1:], bool)


def _upscaled(frames, start=0):
    """``frames`` at twice the size, renumbered from ``start``."""
    return [
        FakeFrame(cv2.resize(f.pixels[0], (2 * WIDTH, 2 * HEIGHT), interpolation=cv2.INTER_NEAREST), start + k)
        for k, f in enumerate(frames)
    ]


class TestMaskedEgoStats:
    @pytest.mark.parametrize("shift", [8, -8, 24])
    def test_the_dense_overlay_pan_reads_true_and_trusted(self, shift):
        for seed in range(3):
            stats = ego_stats(hud_pan(shift, 6, seed=seed), mask=DENSE)["stats"]
            assert all(stats["ego_trusted"][1:])
            np.testing.assert_allclose(stats["pan_x"][1:], shift, atol=0.5)

    def test_without_a_mask_the_dense_overlay_pan_is_refused(self):  # spec test 1, decision 5, revised by #1577
        assert not any(ego_stats(hud_pan(8, 6))["stats"]["ego_trusted"][1:])

    @pytest.mark.parametrize("shift", [8, -8, 24])
    def test_lucas_kanade_reads_the_masked_pan(self, shift):  # spec test 5
        for seed in range(3):
            pan, trusted = _pans(hud_pan(shift, 6, seed=seed), mask=DENSE, flow="lk")
            if abs(shift) < 24:
                assert trusted.all()
            assert np.all(np.abs(pan[trusted] - shift) <= 0.5)

    @pytest.mark.parametrize(("shift", "dy"), [(0, 8), (0, -8), (6, 6)])
    def test_other_directions(self, shift, dy):  # spec test 6
        stats = ego_stats(hud_pan(shift, 6, dy=dy), mask=DENSE)["stats"]
        assert all(stats["ego_trusted"][1:])
        np.testing.assert_allclose(stats["pan_x"][1:], shift, atol=0.5)
        np.testing.assert_allclose(stats["pan_y"][1:], dy, atol=0.5)

    @pytest.mark.parametrize(("shift", "dy"), [(8, 0), (0, 8)])
    def test_the_side_layout(self, shift, dy):  # spec test 6
        side = drawn_mask(side_hud, (HEIGHT, WIDTH))
        stats = ego_stats(hud_pan(shift, 6, dy=dy, draw=side_hud), mask=side)["stats"]
        assert all(stats["ego_trusted"][1:])
        np.testing.assert_allclose(stats["pan_x"][1:], shift, atol=0.5)
        np.testing.assert_allclose(stats["pan_y"][1:], dy, atol=0.5)

    def test_other_processing_sizes(self):  # spec test 6
        light = partial(hud, dense=False)
        for draw, mask in ((hud, DENSE), (light, drawn_mask(light, (HEIGHT, WIDTH)))):
            big = _upscaled(hud_pan(8, 6, draw=draw))
            big_mask = cv2.resize(mask.astype(np.uint8), (2 * WIDTH, 2 * HEIGHT), interpolation=cv2.INTER_NEAREST)
            pan, trusted = _pans(big, mask=big_mask, proc_short_side=480)
            assert trusted.all()
            np.testing.assert_allclose(pan, 16, atol=1.0)
            pan, trusted = _pans(big, mask=big_mask, proc_short_side=240)
            if draw is hud:
                assert not trusted.any()
            else:
                assert trusted.all()
                np.testing.assert_allclose(pan, 16, atol=1.0)

    @pytest.mark.parametrize("make", [panning_stream, ambiguous_stream, lambda: hud_pan(8, 6)])
    def test_an_all_false_mask_reads_like_no_mask(self, make):  # spec test 7
        frames = make()
        bare = ego_stats(frames)["stats"]
        zeros = ego_stats(frames, mask=np.zeros((HEIGHT, WIDTH), bool))["stats"]
        assert set(bare) == set(zeros)
        for name in bare:
            # NaN-aware, and exact: a NaN becomes None, which compares equal to itself.
            assert [None if v != v else v for v in zeros[name]] == [None if v != v else v for v in bare[name]], name

    def test_a_mask_at_twice_the_resolution_reads_the_same(self):  # spec test 8
        frames = hud_pan(8, 6)
        double = cv2.resize(DENSE.astype(np.uint8), (2 * WIDTH, 2 * HEIGHT), interpolation=cv2.INTER_NEAREST)
        pan, trusted = _pans(frames, mask=double)
        assert trusted.all()
        np.testing.assert_allclose(pan, 8, atol=0.5)

    def test_a_bad_geometry_raises_in_the_walk(self):  # spec test 8
        with pytest.raises(ValueError, match="mask of 640x480 does not fit a 640x360 frame"):
            ego_stats(hud_pan(8, 3), mask=np.ones((480, 640), bool))

    def test_video_stats_names_ego_mask_in_a_bad_geometry(self):  # spec 5.1
        with pytest.raises(ValueError, match="ego_mask of 640x480 does not fit a 640x360 frame"):
            video_stats([hud_pan(8, 3)], stats=VideoStats.EGO, ego_mask=np.ones((480, 640), bool))

    def test_a_size_change_mid_video_fits_each_size(self):  # Review Focus 3
        frames = hud_pan(8, 4) + _upscaled(hud_pan(8, 4), start=4)
        pan, trusted = _pans(frames, mask=DENSE)
        assert trusted[:3].all()
        assert trusted[4:].all()
        np.testing.assert_allclose(pan[:3], 8, atol=0.5)
        np.testing.assert_allclose(pan[4:], 16, atol=0.5)
        assert not trusted[3]

    def test_a_gap_of_two_reads_the_pan_per_frame(self):  # Review Focus 4
        stats = ego_stats(hud_pan(8, 8), mask=DENSE, gap=2)["stats"]
        assert all(stats["ego_trusted"][2:])
        np.testing.assert_allclose(stats["pan_x"][2:], 8, atol=0.5)

    def test_rgb_frames_read_like_gray_ones(self):  # Review Focus 5
        pan, trusted = _pans(hud_pan(8, 6, channels=3), mask=DENSE)
        assert trusted.all()
        np.testing.assert_allclose(pan, 8, atol=0.5)


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg on PATH")
class TestEncodedOverlay:
    def test_a_masked_pan_survives_compression(self, tmp_path):  # spec test 11
        avi, mp4 = tmp_path / "pan.avi", tmp_path / "pan.mp4"
        writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"XVID"), 30, (WIDTH, HEIGHT))
        for f in hud_pan(8, 30):
            writer.write(cv2.cvtColor(f.pixels[0], cv2.COLOR_GRAY2BGR))
        writer.release()
        make_canonical_mp4(str(avi), str(mp4))
        capture = cv2.VideoCapture(str(mp4))
        frames = []
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                frames.append(FakeFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), len(frames)))
        finally:
            capture.release()
        assert len(frames) > 10, "fixture setup problem: expected a multi-frame clip"
        pan, trusted = _pans(frames, mask=DENSE)
        assert (trusted & (np.abs(pan - 8) <= 1)).mean() >= 0.8
        assert not (trusted & (np.abs(pan) < 6)).any()

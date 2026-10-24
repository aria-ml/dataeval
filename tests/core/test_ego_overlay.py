"""Tests for the overlay mask the ego path takes: validating it, fitting it to each frame size, reading around it."""

import shutil
from functools import partial

import numpy as np
import pytest

from dataeval.core._ego_stats import (
    _MASK_MARGIN,
    FlowMethod,
    _as_mask,
    _clear_of_overlay,
    _fit_irls,
    _flow_estimator,
    _FlowFrame,
    _grid_flow_dis,
    _measure_pair,
    _OverlayMask,
    _pair_ego,
    _photometric_agreement,
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


def _pair(frames, mask=None, flow: FlowMethod = "dis"):
    est = _flow_estimator(flow)
    previous, current = (_FlowFrame(f.pixels[0], (WIDTH, HEIGHT)) for f in frames[:2])
    fitted = None if mask is None else _OverlayMask(mask, "mask").fit((WIDTH, HEIGHT), (HEIGHT, WIDTH))
    return _measure_pair(est, previous, current, 40, 1, mask=fitted)


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
    def test_a_dense_overlay_reads_a_slow_pan_as_trusted_without_a_mask(self, seed):  # pins decision 5
        reading = _pair(hud_pan(8, 2, seed=seed))
        assert reading is not None
        assert reading.trusted
        assert abs(reading.t0x) < 6.0

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

    def test_without_a_mask_the_dense_overlay_reads_slow_and_trusted(self):  # spec test 1, decision 5
        stats = ego_stats(hud_pan(8, 6))["stats"]
        trusted = [x for x, ok in zip(stats["pan_x"][1:], stats["ego_trusted"][1:], strict=True) if ok]
        assert trusted
        assert all(abs(x) < 6.0 for x in trusted)

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

"""Tests for the core ego-motion module (``_ego_stats.py``).

Ground truth is constructed rather than measured wherever possible. Most tests build a
flow field from known kinematics and check the fit recovers them, which isolates the
estimator from optical flow; the tests that do exercise flow warp a textured image by a
known transform. The distinction matters because a failure in the first kind is a bug in
this module and a failure in the second might be OpenCV.
"""

from __future__ import annotations

import gc
import weakref
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval.core import ego_stats
from dataeval.core._ego_stats import (
    EGO_AGGREGATIONS,
    EgoFactors,
    _biweight_weights,
    _classify_motion,
    _fit_irls,
    _is_ambiguous,
    _pair_ego,
    _seed_weights,
    _stream_pairs,
    _temporal_outliers,
)
from dataeval.protocols import MultiobjectTrackingDataset, VideoStream

cv2 = pytest.importorskip("cv2", reason="ego_stats requires opencv")

pytestmark = pytest.mark.optional

WIDTH, HEIGHT = 640, 360
FULL_WH = (WIDTH, HEIGHT)


# ============================== construction helpers ==============================


def lattice(step: int = 16) -> NDArray[np.float64]:
    """A regular sampling grid over the frame, matching what the flow layer produces."""
    gx, gy = np.meshgrid(np.arange(step // 2, WIDTH, step), np.arange(step // 2, HEIGHT, step))
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(float)


def affine_flow(
    pts: NDArray[np.float64],
    div: float = 0.0,
    curl: float = 0.0,
    t0: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Build the exact flow field a camera with these kinematics would produce.

    The matrix is assembled so that ``a00 + a11 == div`` and ``a10 - a01 == curl``, which
    is the decomposition the module claims to invert.
    """
    centered = pts - np.array([WIDTH / 2.0, HEIGHT / 2.0])
    matrix = np.array([[div / 2.0, -curl / 2.0], [curl / 2.0, div / 2.0]])
    return (matrix @ centered.T).T + np.array(t0, float)


def radial_flow(pts: NDArray[np.float64], focus: tuple[float, float], rate: float) -> NDArray[np.float64]:
    """Pure expansion away from a point offset from the image center."""
    return rate * (pts - (np.array([WIDTH / 2.0, HEIGHT / 2.0]) + np.array(focus, float)))


def texture(seed: int = 0, size: tuple[int, int] = (HEIGHT, WIDTH), blur: float = 1.5) -> NDArray[np.uint8]:
    """A blurred noise field — trackable by optical flow, unlike per-pixel noise."""
    rng = np.random.default_rng(seed)
    return cast("NDArray[np.uint8]", cv2.GaussianBlur(rng.integers(0, 255, size, dtype=np.uint8), (0, 0), blur))


class FakeFrame:
    """A decoded frame carrying ``(C, H, W)`` pixels, per the MAITE video protocol."""

    def __init__(self, gray: NDArray[np.uint8], index: int, channels: int = 1) -> None:
        self.pixels = np.repeat(gray[np.newaxis, :, :], channels, axis=0)
        self.frame_index = index
        self.time_s = index / 30.0
        self.pts = index * 1001


def panning_stream(shift: int = 3, n_frames: int = 10, seed: int = 0) -> list[FakeFrame]:
    """A sequence whose content translates by a known number of pixels each frame."""
    base = texture(seed)
    return [FakeFrame(np.roll(base, k * shift, axis=1), k) for k in range(n_frames)]


def ambiguous_stream(n_frames: int = 8, seed: int = 0) -> list[FakeFrame]:
    """A still camera with a large coherent mover across the lower frame.

    At this mover fraction the pair has two self-consistent readings and no way to choose,
    which is what the ambiguity gate refuses.
    """
    base = texture(seed)
    frames = []
    for k in range(n_frames):
        canvas = base.copy()
        band = slice(int(0.60 * HEIGHT), HEIGHT)
        canvas[band] = np.roll(base[band], k * 12, axis=1)
        frames.append(FakeFrame(canvas, k))
    return frames


class FakeDataset:
    """A tracking dataset of video streams, indexable as the real protocol requires."""

    def __init__(self, streams: Sequence[Sequence[FakeFrame]]) -> None:
        self._streams = streams

    def __len__(self) -> int:
        return len(self._streams)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        return self._streams[index], None, {"id": index}


def measure(flow: NDArray[np.float64], pts: NDArray[np.float64] | None = None):
    """Fit one frame pair from a constructed flow field."""
    grid = lattice() if pts is None else pts
    pair = _pair_ego(grid, flow, FULL_WH, FULL_WH, 1)
    assert pair is not None
    return pair


# ============================== kinematic recovery ==============================


class TestKinematicRecovery:
    """The affine decomposition must invert exactly on fields built from known kinematics."""

    def test_pure_zoom_recovers_divergence(self):
        pair = measure(affine_flow(lattice(), div=0.02))
        assert pair.div == pytest.approx(0.02, abs=1e-9)
        assert pair.curl == pytest.approx(0.0, abs=1e-9)
        assert pair.motion == "zoom_dolly"

    def test_pure_roll_recovers_curl(self):
        # A roll of phi produces curl = 2 phi, which is the factor of two the model names.
        pair = measure(affine_flow(lattice(), curl=2 * 0.01))
        assert pair.curl == pytest.approx(0.02, abs=1e-9)
        assert pair.div == pytest.approx(0.0, abs=1e-9)
        assert pair.motion == "roll"

    def test_pure_pan_recovers_translation(self):
        pair = measure(affine_flow(lattice(), t0=(5.0, 3.0)))
        assert (pair.t0x, pair.t0y) == (pytest.approx(5.0, abs=1e-9), pytest.approx(3.0, abs=1e-9))
        assert pair.motion == "pan_tilt"

    def test_shear_is_picked_up_where_div_and_curl_are_blind(self):
        # A traceless symmetric field: both div and curl are zero by construction, so a
        # non-zero reading here is the only evidence the motion happened at all.
        pts = lattice()
        centered = pts - np.array([WIDTH / 2.0, HEIGHT / 2.0])
        pair = measure((np.array([[0.01, 0.0], [0.0, -0.01]]) @ centered.T).T, pts)
        assert pair.div == pytest.approx(0.0, abs=1e-9)
        assert pair.curl == pytest.approx(0.0, abs=1e-9)
        assert pair.shear == pytest.approx(0.02, abs=1e-9)

    def test_focus_of_expansion_locates_the_heading(self):
        pair = measure(radial_flow(lattice(), focus=(100.0, -50.0), rate=0.008))
        assert pair.foe_in_frame
        assert (pair.foe_x, pair.foe_y) == (pytest.approx(100.0, abs=1e-3), pytest.approx(-50.0, abs=1e-3))

    def test_pure_pan_has_no_focus_of_expansion(self):
        # A translating field never reaches zero, so there is nowhere the camera is heading.
        assert not measure(affine_flow(lattice(), t0=(5.0, 3.0))).foe_in_frame

    def test_pure_roll_reports_no_focus_of_expansion(self):
        """A roll pivots about a point but is not heading toward it.

        Regression test: gating on the whole gradient admits a roll, whose singular values
        are both large, and reports its pivot as a focus. The gate belongs on the trace,
        which a roll leaves at zero.
        """
        pair = measure(affine_flow(lattice(), curl=0.02))
        assert pair.motion == "roll"
        assert not pair.foe_in_frame
        assert np.isnan(pair.foe_x)
        assert np.isnan(pair.foe_y)

    def test_a_dolly_toward_frame_center_still_has_one(self):
        # The counterpart to the roll case: divergence present, so the focus is real.
        assert measure(affine_flow(lattice(), div=0.02)).foe_in_frame


class TestResolutionInvariance:
    """Rates must not change with resolution; speeds must come back in full-resolution pixels."""

    @pytest.mark.parametrize("proc_short_side", [720, 360, 180])
    def test_pan_speed_is_reported_in_full_resolution_pixels(self, proc_short_side: int):
        frames = [FakeFrame(f, i) for i, f in enumerate(_scaled_pan(shift=8, n_frames=6))]
        result = ego_stats(cast("VideoStream", frames), proc_short_side=proc_short_side)
        # Frame 0 is the leading, NaN frame; frame 1 is the first frame with a reading.
        assert result["stats"]["pan_speed"][1] == pytest.approx(8.0, abs=0.2)

    @pytest.mark.parametrize("proc_short_side", [720, 360])
    def test_zoom_rate_does_not_change_with_resolution(self, proc_short_side: int):
        base = texture(0, size=(720, 1280), blur=2.0)
        warp = cv2.getRotationMatrix2D((1280 / 2, 720 / 2), 0.0, 1.03)
        zoomed = cv2.warpAffine(base, warp, (1280, 720), borderMode=cv2.BORDER_REFLECT)
        frames = [FakeFrame(base, 0), FakeFrame(zoomed, 1)]
        result = ego_stats(cast("VideoStream", frames), proc_short_side=proc_short_side)
        # A 3% scale-up over one frame is a divergence of about 0.058 per frame.
        assert result["stats"]["zoom_rate"][1] == pytest.approx(0.058, abs=0.01)


def _scaled_pan(shift: int, n_frames: int) -> list[NDArray[np.uint8]]:
    """Frames at 1280x720 translating by `shift` full-resolution pixels each frame."""
    base = texture(0, size=(720, 1280), blur=2.0)
    return [np.roll(base, k * shift, axis=1) for k in range(n_frames)]


# ============================== mover rejection ==============================


class TestMoverRejection:
    """Independent motion must be excluded from the camera estimate, not averaged into it."""

    def test_central_mover_does_not_leak_into_pan(self):
        pts = lattice()
        flow = affine_flow(pts, t0=(4.0, 0.0))
        blob = np.linalg.norm(pts - np.array([WIDTH / 2.0, HEIGHT / 2.0]), axis=1) < 0.28 * min(WIDTH, HEIGHT)
        flow[blob] = np.array([-10.0, 8.0])
        pair = measure(flow, pts)
        assert pair.t0x == pytest.approx(4.0, abs=0.3)
        assert pair.t0y == pytest.approx(0.0, abs=0.3)

    def test_static_camera_with_large_mover_does_not_read_as_pan(self):
        """The fit must not adopt a coherent mover filling the lower frame.

        This is the near-lane vehicle case: a large, perfectly coherent patch of motion
        against a still background, which an unseeded least-squares fit adopts outright.
        """
        pts = lattice()
        flow = affine_flow(pts)
        flow[pts[:, 1] > 0.60 * HEIGHT] = np.array([12.0, 0.0])
        pair = measure(flow, pts)
        assert abs(pair.t0x) < 0.5
        assert abs(pair.t0y) < 0.5

    def test_seed_weights_fall_back_when_a_mover_holds_the_majority(self):
        # Median flips onto the mover; the seed must not starve the fit of samples.
        flow = np.tile(np.array([12.0, 0.0]), (100, 1))
        flow[:20] = np.array([0.0, 0.0])
        assert _seed_weights(flow).sum() >= 3


class TestContaminationIsEgoInvariant:
    """The independent-motion proxy must not move when only the camera speed changes."""

    @staticmethod
    def _with_mover(**kinematics: Any):
        pts = lattice()
        flow = affine_flow(pts, **kinematics)
        blob = np.linalg.norm(pts - np.array([WIDTH / 2.0, HEIGHT / 2.0]), axis=1) < 0.28 * min(WIDTH, HEIGHT)
        flow[blob] += np.array([-10.0, 8.0])
        return measure(flow, pts), blob.mean()

    @pytest.mark.parametrize(
        "kinematics",
        [{}, {"t0": (4.0, 0.0)}, {"t0": (40.0, 0.0)}, {"div": 0.02}, {"t0": (40.0, 0.0), "div": 0.02}],
    )
    def test_mover_frac_and_speed_track_the_mover_not_the_camera(self, kinematics: dict[str, Any]):
        """Regression test for a residual-energy share standing in for contamination.

        A ratio against total flow energy reads 1.0 for a static camera and near zero for
        a fast pan carrying the identical mover — a 65x swing driven entirely by the
        camera. Both replacements are absolute quantities and hold steady.
        """
        pair, mover_fraction = self._with_mover(**kinematics)
        assert pair.mover_frac == pytest.approx(mover_fraction, abs=0.02)
        assert pair.mover_speed == pytest.approx(float(np.hypot(-10.0, 8.0)), abs=0.5)

    def test_mover_frac_is_a_fraction(self):
        """Regression test for a mean biweight weight being reported as a fraction.

        A weight is continuous in [0, 1], so its mean is not the share of samples kept and
        does not complement the outlier share. Under noise the two differ outright.
        """
        pts = lattice()
        rng = np.random.default_rng(0)
        flow = affine_flow(pts, t0=(4.0, 0.0)) + rng.normal(0, 0.5, pts.shape)
        blob = np.linalg.norm(pts - np.array([WIDTH / 2.0, HEIGHT / 2.0]), axis=1) < 0.28 * min(WIDTH, HEIGHT)
        flow[blob] = np.array([-10.0, 8.0])

        pair = measure(flow, pts)
        assert pair.mover_frac == pytest.approx(float(blob.mean()), abs=0.02)


# ============================== refusal ==============================


class TestAmbiguityGate:
    """A pair where camera and mover cannot be told apart must be refused, not guessed."""

    @staticmethod
    def _banded(mover_frac: float, t0: tuple[float, float] = (0.0, 0.0)):
        pts = lattice()
        flow = affine_flow(pts, t0=t0)
        flow[pts[:, 1] > (1.0 - mover_frac) * HEIGHT] = np.array([12.0, 0.0])
        return measure(flow, pts)

    def test_balanced_mover_is_refused(self):
        # 40% coherent mover: rejected samples form their own consensus, so there are two
        # readings and no way to choose between them.
        pair = self._banded(0.40)
        assert pair.ambiguous
        assert abs(pair.t0x) < 0.5

    def test_dominant_mover_is_refused(self):
        # 65%: the mover wins the fit outright, so the reported ego IS the mover.
        pair = self._banded(0.65)
        assert pair.ambiguous
        assert pair.t0x > 5.0

    def test_minor_mover_is_kept(self):
        assert not self._banded(0.15).ambiguous

    def test_clean_pan_is_kept(self):
        assert not measure(affine_flow(lattice(), t0=(5.0, 3.0))).ambiguous

    def test_too_few_outliers_is_not_ambiguous(self):
        pts = lattice()
        assert not _is_ambiguous(pts, affine_flow(pts), np.zeros(len(pts), bool))


class TestTemporalGate:
    """A burst departing from the sequence's dominant camera state must be flagged whole."""

    @staticmethod
    def _track(t0x: float, div: float = 0.0):
        pair = object.__new__(_PairStub)
        pair.t0x, pair.t0y, pair.div, pair.curl = t0x, 0.0, div, 0.0
        return pair

    def test_wide_burst_is_flagged_including_its_core(self):
        readings = [self._track(0.0) for _ in range(40)]
        for i in range(15, 33):
            readings[i] = self._track(75.0, div=0.3)
        flags = _temporal_outliers(cast("Any", readings), half_diagonal=400.0)
        assert flags[15:33].all(), "a rolling baseline would track the burst and miss its core"
        assert not flags[:15].any()
        assert not flags[33:].any()

    def test_steady_real_motion_is_not_flagged(self):
        readings = [self._track(5.0) for _ in range(40)]
        assert not _temporal_outliers(cast("Any", readings), half_diagonal=400.0).any()

    def test_too_few_measured_pairs_flags_nothing(self):
        assert not _temporal_outliers(cast("Any", [None] * 40), half_diagonal=400.0).any()


class _PairStub:
    """Minimal stand-in carrying only the four fields the temporal gate reads."""

    t0x: float
    t0y: float
    div: float
    curl: float


class TestDegeneracy:
    """Sample geometry that cannot determine an affine field must be reported, not fitted."""

    def test_collinear_samples_are_degenerate(self):
        """Regression test for an absolute threshold on the normal matrix.

        The normal matrix scales with both sample count and coordinate magnitude — a
        healthy fit here has a smallest singular value around 880 — so a fixed cutoff is
        not a measure of conditioning. The ratio to the largest is.
        """
        pts = np.column_stack([
            np.linspace(0, WIDTH, 200),
            np.full(200, HEIGHT / 2) + 1e-6 * np.arange(200),
        ]).astype(np.float64)
        fit = _fit_irls(pts, affine_flow(pts, div=0.02))
        assert fit is not None
        assert fit.condition < 1e-12

    def test_a_healthy_fit_is_well_conditioned(self):
        fit = _fit_irls(lattice(), affine_flow(lattice(), t0=(5.0, 3.0)))
        assert fit is not None
        assert fit.condition > 1e-6


class TestClassifyMotion:
    def test_below_the_static_floor_is_static(self):
        assert _classify_motion(0.0, 0.0, 0.0, 400.0, 0.2) == "static"

    def test_a_clear_leader_wins(self):
        assert _classify_motion(0.0, 0.0, 10.0, 400.0, 0.2) == "pan_tilt"

    def test_a_tie_is_complex(self):
        # Two components producing the same corner speed: neither leads, so neither is named.
        assert _classify_motion(10.0 / 400.0, 0.0, 10.0, 400.0, 0.2) == "complex"


def test_biweight_weights_are_zero_beyond_the_cutoff():
    weights = _biweight_weights(np.array([0.0, 0.5, 1.0, 2.0]), 1.0)
    assert weights[0] == 1.0
    assert 0.0 < weights[1] < 1.0
    assert weights[2] == 0.0
    assert weights[3] == 0.0


# ============================== end to end ==============================


class TestPerFrameOutput:
    """One row per frame, aligned to the frame the motion arrived at."""

    def test_one_row_per_frame(self):
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=8)))
        assert len(result["stats"]["pan_speed"]) == 8
        assert list(result["stats"]["unit_index"]) == list(range(8))

    def test_the_reading_lands_on_the_later_frame(self):
        # A pair spans (k, k+gap); its motion describes arriving at k+gap, so the leading
        # `gap` frames have nothing to report.
        result = ego_stats(cast("VideoStream", panning_stream(shift=3, n_frames=8)))
        speeds = list(result["stats"]["pan_speed"])
        assert np.isnan(speeds[0])
        assert speeds[1] == pytest.approx(3.0, abs=0.5)

    @pytest.mark.parametrize("gap", [1, 2, 3])
    def test_the_leading_gap_frames_are_nan(self, gap: int):
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=10)), gap=gap)
        speeds = np.asarray(result["stats"]["pan_speed"], dtype=float)
        assert np.isnan(speeds[:gap]).all()
        assert not np.isnan(speeds[gap:]).any()

    def test_a_dataset_concatenates_in_item_order(self):
        dataset = FakeDataset([panning_stream(n_frames=4), panning_stream(n_frames=6, seed=1)])
        result = ego_stats(cast("MultiobjectTrackingDataset", dataset))
        assert list(result["stats"]["item_index"]) == [0] * 4 + [1] * 6
        assert list(result["stats"]["unit_index"]) == list(range(4)) + list(range(6))

    def test_signed_components_are_reported(self):
        # Direction, not just magnitude: a leftward pan must be distinguishable.
        right = ego_stats(cast("VideoStream", panning_stream(shift=3, n_frames=6)))
        left = ego_stats(cast("VideoStream", panning_stream(shift=-3, n_frames=6)))
        assert np.nanmedian(np.asarray(right["stats"]["pan_x"], float)) > 0
        assert np.nanmedian(np.asarray(left["stats"]["pan_x"], float)) < 0

    def test_an_empty_sequence_contributes_no_rows(self):
        result = ego_stats(cast("MultiobjectTrackingDataset", FakeDataset([[], panning_stream(n_frames=4)])))
        assert list(result["stats"]["item_index"]) == [1] * 4

    def test_stride_is_gone(self):
        with pytest.raises(TypeError):
            ego_stats(cast("VideoStream", panning_stream()), stride=2)  # type: ignore[call-arg]

    def test_max_frames_limits_the_walk(self):
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=20)), max_frames=5)
        assert len(result["stats"]["pan_speed"]) == 5

    def test_lk_flow_agrees_with_dis_on_a_clean_pan(self):
        frames = panning_stream(shift=3, n_frames=8)
        lk = ego_stats(cast("VideoStream", frames), flow="lk")
        assert lk["stats"]["pan_speed"][1] == pytest.approx(3.0, abs=0.5)

    def test_rgb_frames_are_accepted(self):
        base = texture(3)
        frames = [FakeFrame(np.roll(base, k * 3, axis=1), k, channels=3) for k in range(6)]
        result = ego_stats(cast("VideoStream", frames))
        assert result["stats"]["pan_speed"][1] == pytest.approx(3.0, abs=0.5)


class TestTrustNulling:
    """An untrusted frame abstains rather than contributing a wrong number."""

    def test_kinematics_are_null_where_untrusted(self):
        # A 40% coherent mover makes the pair ambiguous, which is a refusal.
        frames = ambiguous_stream(n_frames=8)
        result = ego_stats(cast("VideoStream", frames))
        trusted = np.asarray(result["stats"]["ego_trusted"], dtype=bool)
        speeds = np.asarray(result["stats"]["pan_speed"], dtype=float)
        assert not trusted.all(), "fixture problem: expected some refused pairs"
        assert np.isnan(speeds[~trusted]).all()

    def test_contamination_survives_a_refusal(self):
        # mover_frac is most informative exactly on the pairs the ego fit refuses.
        result = ego_stats(cast("VideoStream", ambiguous_stream(n_frames=8)))
        trusted = np.asarray(result["stats"]["ego_trusted"], dtype=bool)
        fracs = np.asarray(result["stats"]["mover_frac"], dtype=float)
        measured = ~np.isnan(fracs)
        assert (measured & ~trusted).any()

    def test_ego_trusted_is_never_null(self):
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=6)))
        assert all(v is not None for v in result["stats"]["ego_trusted"])


class TestParameterValidation:
    """Bad parameters must fail loudly rather than silently changing the measurement."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gap": 0},
            {"gap": -1},
            {"proc_short_side": 0},
            {"grid_target": 0},
            {"max_frames": 0},
        ],
    )
    def test_non_positive_parameters_are_rejected(self, kwargs: dict[str, Any]):
        with pytest.raises(ValueError, match="must be positive"):
            ego_stats(cast("VideoStream", panning_stream()), **kwargs)

    def test_unknown_flow_estimator_is_rejected(self):
        """Regression test for an unrecognized estimator falling back to Lucas-Kanade.

        Silently substituting an estimator makes the reported ``flow`` a lie and the
        measurement irreproducible.
        """
        with pytest.raises(ValueError, match="Unknown flow estimator"):
            ego_stats(cast("VideoStream", panning_stream()), flow=cast("Any", "totally-bogus"))


# ============================== streaming ==============================


class TestStreamPairs:
    """Pairing must be correct and must not hold the sequence in memory to do it."""

    @staticmethod
    def _numbered(n: int) -> Iterator[NDArray[np.uint8]]:
        for k in range(n):
            frame = np.zeros((8, 8), np.uint8)
            frame[0, 0] = k
            yield frame

    @pytest.mark.parametrize(
        ("gap", "expected"),
        [
            (1, [(0, 1), (1, 2), (2, 3), (3, 4)]),
            (2, [(0, 2), (1, 3), (2, 4), (3, 5)]),
            (3, [(0, 3), (1, 4), (2, 5), (3, 6)]),
        ],
    )
    def test_every_pair_is_gap_apart(self, gap: int, expected: list[tuple[int, int]]):
        """Every consecutive `gap`-apart pair is yielded, none skipped -- `stride` is gone."""
        pairs = [(int(a[0, 0]), int(b[0, 0])) for a, b in _stream_pairs(self._numbered(12), gap)]
        assert pairs[: len(expected)] == expected

    @pytest.mark.parametrize("gap", [1, 2, 3])
    def test_at_most_gap_plus_one_frames_are_held_at_once(self, gap: int):
        """A metadata pass runs over whole datasets of video.

        Materializing a sequence is the difference between a measurement that runs and one
        that exhausts memory, so the window has to stay bounded by `gap` however long the
        sequence is.
        """
        alive: list[weakref.ref[Any]] = []

        def frames() -> Iterator[NDArray[np.uint8]]:
            for _ in range(12):
                frame = np.zeros((8, 8), np.uint8)
                alive.append(weakref.ref(frame))
                yield frame

        peak = 0
        for _pair in _stream_pairs(frames(), gap):
            gc.collect()
            peak = max(peak, sum(ref() is not None for ref in alive))
        assert peak == gap + 1

    def test_a_sequence_shorter_than_the_gap_yields_nothing(self):
        assert list(_stream_pairs(self._numbered(2), gap=5)) == []


# ============================== metadata landing ==============================


class TestMetadataIntegration:
    """The result must be attachable to metadata at the unit level, unreshaped."""

    @staticmethod
    def _dataset(shifts: Sequence[int]):
        """A tracking dataset whose sequences pan by a known amount each frame."""
        from tests.embeddings.test_embeddings import MockDataset
        from tests.metadata.test_structurers import _FrameTracks, _MOTTarget

        streams = [panning_stream(shift=shift, n_frames=8, seed=i) for i, shift in enumerate(shifts)]
        targets = [_MOTTarget([_FrameTracks([0]) for _ in stream]) for stream in streams]
        return MockDataset(streams, targets, None)

    def test_every_field_is_a_flat_column_of_scalars(self):
        result = ego_stats(cast("MultiobjectTrackingDataset", FakeDataset([panning_stream(), panning_stream()])))
        for name, values in result["stats"].items():
            for value in cast("Sequence[Any]", values):
                if name == "motion":
                    # A string column's None is a real, bin-able null; a float column's
                    # None would instead silently become an unbinnable object column, which
                    # is what the rest of this loop guards against.
                    assert value is None or isinstance(value, str)
                    continue
                assert np.isscalar(value) or isinstance(value, (str, bool, np.bool_)), (
                    f"{name} holds {type(value)}, which has no single-column representation"
                )

    def test_missing_measurements_are_nan_not_none(self):
        """A float column holds NaN for a missing measurement, never ``None``.

        ``None`` in a float column makes an object-dtype column that will not bin. The
        leading `gap` frame of every sequence has nothing to report, which exercises this
        directly. ``motion`` is excepted: it is a string column, where ``None`` is a real
        null rather than an unbinnable object dtype, which is why `_frame_rows` uses it
        there on purpose.
        """
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=1)))
        for name, values in result["stats"].items():
            if name == "motion":
                continue
            assert all(value is not None for value in cast("Sequence[Any]", values))
        assert np.isnan(cast("Any", result["stats"]["pan_speed"])[0])

    def test_motion_is_a_bare_categorical(self):
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=6)))
        assert isinstance(result["stats"]["motion"][1], str)

    def test_no_factor_is_dropped_as_multidimensional(self):
        """Vector-valued factors are skipped with a warning; none of these should be."""
        from dataeval import Metadata

        dataset = self._dataset([3, 0])
        metadata = Metadata(dataset)
        metadata.add_factors(
            cast("Any", ego_stats(cast("MultiobjectTrackingDataset", dataset))),
            level="unit",
            key="unit_index",
        )
        assert not metadata.dropped_factors

    def test_attaching_the_whole_result_adds_no_stray_column(self):
        """The documented one-liner must not leave a reserved-name factor behind.

        Columns the landing contract reserves -- ``item_index`` and ``unit_index`` among
        them -- are kept but renamed with a ``metadata_`` prefix rather than rejected, so
        emitting either as a factor would quietly add a junk factor to every dataset this
        is run on.
        """
        from dataeval import Metadata

        dataset = self._dataset([3, 0])
        metadata = Metadata(dataset)
        before = set(metadata.factor_names)
        metadata.add_factors(
            cast("Any", ego_stats(cast("MultiobjectTrackingDataset", dataset))),
            level="unit",
            key="unit_index",
        )
        added = set(metadata.factor_names) - before
        # The declared roll-ups land too, now that add_factors applies them by default.
        rollups = {a.name_for(f) for a in EGO_AGGREGATIONS for f in a.factors}
        assert added == set(EgoFactors.__annotations__) - {"item_index", "unit_index"} | rollups


# ============================== declared roll-ups ==============================


@pytest.mark.optional
class TestDeclaredRollUps:
    """The result must declare how its per-frame factors summarize to one row per sequence."""

    EXPECTED = {
        "pan_speed_median",
        "pan_x_variability",
        "pan_y_variability",
        "zoom_rate_excursion",
        "roll_rate_excursion",
        "zoom_rate_median",
        "roll_rate_median",
        "shear_rate_median",
        "motion_mode",
        "mover_frac_median",
        "mover_speed_median",
    }

    _dataset = staticmethod(TestMetadataIntegration._dataset)

    def test_the_result_declares_its_roll_ups(self):
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=6)))
        names = {a.name_for(f) for a in result.get("aggregations", ()) for f in a.factors}
        assert names == self.EXPECTED

    def test_every_declaration_uses_the_trust_threshold(self):
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=6)))
        assert {a.min_coverage for a in result.get("aggregations", ())} == {0.25}

    def test_every_declaration_rolls_unit_to_sequence(self):
        result = ego_stats(cast("VideoStream", panning_stream(n_frames=6)))
        assert {(a.source, a.target) for a in result.get("aggregations", ())} == {("unit", "sequence")}

    def test_they_land_on_a_real_metadata(self):
        from dataeval import Metadata

        dataset = self._dataset([3, 0])
        md = Metadata(dataset)
        md.add_factors(cast("Any", ego_stats(dataset)), level="unit", key="unit_index")
        assert set(md.factor_names) >= self.EXPECTED
        assert md.rows_at("sequence")["pan_speed_median"].to_list()[0] == pytest.approx(3.0, abs=0.5)
        assert md.rows_at("sequence")["motion_mode"].to_list() == ["pan_tilt", "static"]

    def test_each_roll_up_matches_the_reduction_applied_to_its_own_column(self):
        """A declaration is a claim about which reduction produced which column.

        Asserting only that the column exists would pass with a swapped reduction or a
        column of nulls, so each one is recomputed here from the per-frame values it
        summarizes.
        """
        from dataeval import Metadata

        dataset = self._dataset([3, 0])
        result = ego_stats(cast("MultiobjectTrackingDataset", dataset))
        metadata = Metadata(dataset)
        metadata.add_factors(cast("Any", result), level="unit", key="unit_index")

        stats = result["stats"]
        items = np.asarray(stats["item_index"])
        rows = metadata.rows_at("sequence")
        sequences = sorted(set(items.tolist()))

        def per_sequence(column: str, reduce):
            """Apply `reduce` to one factor's values within each sequence, skipping NaN."""
            out = []
            for item in sequences:
                values = np.asarray(stats[column], dtype=float)[items == item]
                values = values[~np.isnan(values)]
                out.append(reduce(values) if values.size else None)
            return out

        for column, how, reduce in [
            ("pan_speed", "median", np.median),
            ("zoom_rate", "median", np.median),
            ("roll_rate", "median", np.median),
            ("shear_rate", "median", np.median),
            ("mover_frac", "median", np.median),
            ("mover_speed", "median", np.median),
        ]:
            expected = per_sequence(column, reduce)
            got = rows[f"{column}_{how}"].to_list()
            assert got == pytest.approx(expected, nan_ok=True), column

        for column in ("zoom_rate", "roll_rate"):
            expected = per_sequence(column, lambda v: float(np.abs(v).sum()))
            got = rows[f"{column}_excursion"].to_list()
            assert got == pytest.approx(expected, nan_ok=True), column

        # `variability` is mean absolute change per unit of the ordering key (`time_s` on
        # this schema), not a plain reduction over the values -- recomputed faithfully,
        # mirroring `_variability` in `_metadata/_reductions.py`, rather than approximated:
        # a swapped reduction (e.g. `median` or `sum`) would slip through a looser check
        # since it would still produce a small number for a near-constant pan.
        unit_rows = metadata.rows_at("unit")
        unit_item = np.asarray(unit_rows["item_index"])
        unit_index = np.asarray(unit_rows["unit_index"])
        unit_time = np.asarray(unit_rows["time_s"], dtype=float)

        def expected_variability(column: str) -> list[float | None]:
            out = []
            for item in sequences:
                values = np.asarray(stats[column], dtype=float)[items == item]
                order = unit_time[unit_item == item][np.argsort(unit_index[unit_item == item])]
                diffs = np.diff(values)
                steps = np.diff(order)
                valid = ~np.isnan(diffs) & (steps != 0)
                rate = np.abs(diffs[valid] / steps[valid])
                out.append(float(rate.mean()) if rate.size else None)
            return out

        for column in ("pan_x", "pan_y"):
            expected = expected_variability(column)
            got = rows[f"{column}_variability"].to_list()
            assert got == pytest.approx(expected, nan_ok=True), column


# ============================== real decoded video ==============================


@pytest.mark.ffmpeg
class TestRealVideo:
    """Against ffmpeg-produced clips, where compression artifacts and codec noise are real.

    The synthetic fixtures are all filmed from a locked-off camera, so what they pin is the
    static reading and the refusal machinery — not motion accuracy, which the constructed
    flow fields above cover far more precisely.
    """

    @staticmethod
    def _stream(path: Any) -> list[FakeFrame]:
        """Decode a video file into frames carrying ``(C, H, W)`` pixels."""
        capture = cv2.VideoCapture(str(path))
        try:
            frames = []
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                frames.append(FakeFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), len(frames)))
            return frames
        finally:
            capture.release()

    def test_a_locked_off_camera_reads_as_static(self, video_path: Any, control_clean_clip: Any):
        frames = self._stream(video_path(control_clean_clip))
        assert len(frames) > 10, "fixture setup problem: expected a multi-frame clip"
        result = ego_stats(cast("VideoStream", frames))
        trusted = np.asarray(result["stats"]["ego_trusted"], dtype=bool)
        assert trusted.any(), "fixture problem: expected at least one trusted frame"
        motions = [m for m, t in zip(result["stats"]["motion"], trusted, strict=True) if t]
        assert Counter(motions).most_common(1)[0][0] == "static"
        speeds = np.asarray(result["stats"]["pan_speed"], dtype=float)
        assert np.nanmedian(speeds) == pytest.approx(0.0, abs=0.5)

    def test_a_decoded_clip_produces_a_complete_row(self, video_path: Any, control_clean_clip: Any):
        frames = self._stream(video_path(control_clean_clip))
        result = ego_stats(cast("VideoStream", frames))
        assert set(result["stats"]) == set(EgoFactors.__annotations__)
        assert {len(cast("Sequence[Any]", values)) for values in result["stats"].values()} == {len(frames)}

    def test_moving_shapes_surface_as_contamination(self, video_path: Any, control_clean_clip: Any):
        """The clips hold moving shapes against a still background — that is what movers are."""
        result = ego_stats(cast("VideoStream", self._stream(video_path(control_clean_clip))))
        fracs = np.asarray(result["stats"]["mover_frac"], dtype=float)
        assert np.nanmax(fracs) > 0.0

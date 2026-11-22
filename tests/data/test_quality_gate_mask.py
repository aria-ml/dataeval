"""`QualityGate(mask=)`: a gate on the pixels a mask leaves, measured as `image_stats(mask=)` measures them.

The fixtures are synthetic. A 48x64 frame carries a hood (the bottom 12 rows) and a HUD block near the top-left corner,
and a scene is random texture, so every statistic has a value. The masked pixels are painted with values that move every
pixel statistic. A gate measures through `QualityGate._measure`, one frame at a time, as it does in a video.
"""

import sys
import warnings
from typing import Any

import numpy as np
import pytest

import dataeval.core._ego_stats as ego_module
import dataeval.core._image_stats as image_stats_module
from dataeval.core import image_stats
from dataeval.core._ego_stats import _OverlayMask
from dataeval.data import QualityGate, SequenceFrames
from dataeval.data._quality_gate import _numeric_scalar_stats, _single_statistics
from dataeval.exceptions import ExperimentalWarning
from tests.data.test_frames import image_dataset

pytestmark = pytest.mark.required

H, W = 48, 64
MASK = np.zeros((H, W), dtype=bool)
MASK[36:] = True  # a hood across the bottom
MASK[2:8, 2:20] = True  # a HUD readout

# What the masked view does not measure: the 17 dimension statistics, and directionality, which reads NaN under a mask.
REFUSED = {
    *("aspect_ratio", "channels", "depth", "distance_center", "distance_edge", "height", "offset_x", "offset_y"),
    *(
        "rel_distance_center",
        "rel_distance_edge",
        "rel_height",
        "rel_offset_x",
        "rel_offset_y",
        "rel_size",
        "rel_width",
    ),
    *("size", "width", "blur_edge_directionality"),
}
ACCEPTED = sorted(set(_numeric_scalar_stats()) - REFUSED)


def scene(seed: int = 0, low: int = 0, high: int = 256, dtype: Any = np.uint8) -> np.ndarray:
    """A textured (3, H, W) scene with values in ``[low, high)``."""
    return np.random.default_rng(seed).integers(low, high, (3, H, W)).astype(dtype)


def painted(image: np.ndarray, fill: float, where: np.ndarray = MASK) -> np.ndarray:
    """The image with only the pixels `where` marks set to `fill`."""
    out = image.copy()
    out[:, where] = fill
    return out


def gate(stat: str = "mean", *, mask: Any = MASK, **kwargs: Any) -> QualityGate:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)  # the blur statistics warn once, at construction
        return QualityGate(stat, mask=mask, **kwargs) if mask is not None else QualityGate(stat, **kwargs)


def measure(selector: QualityGate, frame: np.ndarray) -> float:
    """One frame's value, with every warning the measurement raises an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return selector._measure(frame)


def entries(selector: QualityGate, frames: list[np.ndarray]) -> list[dict[str, Any]]:
    """The frames the gate keeps, from a one-sequence video of `frames`."""
    return list(SequenceFrames(image_dataset(frames), selector)._record_of(0).entries)


def reported(frame: np.ndarray, stat: str, **kwargs: Any) -> float:
    """The value `image_stats` reports for the frame, as a gate reads it, rounded to float32."""
    flag = _single_statistics()[stat][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        result = image_stats([frame], stats=flag, per_target=False, **kwargs)
    return float(result["stats"][stat][0])


class TestTheBound:
    def test_an_absolute_bound_keeps_and_drops_as_the_masked_value_says(self):
        # Dark scene, bright hood; bright scene, dark hood. Unmasked, the dark scene is the brighter frame.
        frames = [painted(scene(0, 0, 60), 255), painted(scene(1, 100, 140), 0)]
        masked = gate("mean", min=90.0)
        assert [measure(masked, f) < 90.0 for f in frames] == [True, False]
        assert [e["position"] for e in entries(masked, frames)] == [1]

        loose = gate("mean", min=90.0, mask=None)
        assert [e["position"] for e in entries(loose, frames)] == [0]

    def test_a_masked_quantile_bound_reads_the_masked_values(self):
        frames = [painted(scene(0, 0, 60), 255), painted(scene(1, 100, 140), 0), painted(scene(2, 150, 200), 0)]
        kept = [e["position"] for e in entries(gate("mean", min_quantile=0.34), frames)]
        assert kept == [1, 2]

    @pytest.mark.optional
    def test_a_frame_the_mask_covers_entirely_reads_nan_and_is_kept_unmeasured(self):
        pytest.importorskip("cv2", reason="rescaling a mask needs opencv")
        nearly = np.ones((2 * H, 2 * W), dtype=bool)
        nearly[0, 0] = False  # a gap the downscale fills
        assert image_stats_module._fit_mask(nearly, (H, W)).all()
        selector = gate("mean", min=0.0, mask=nearly)
        frames = [scene(0), scene(1)]
        assert np.isnan(measure(selector, frames[0]))
        assert [(e["position"], e["unmeasured"]) for e in entries(selector, frames)] == [(0, True), (1, True)]


class TestRefusals:
    def test_every_statistic_the_masked_view_does_not_measure_is_refused_naming_the_ones_it_does(self):
        assert set(_numeric_scalar_stats()) >= REFUSED
        for stat in sorted(REFUSED):
            with pytest.raises(ValueError, match="QualityGate") as raised:
                QualityGate(stat, min=0.0, mask=MASK)
            message = str(raised.value)
            assert repr(stat) in message
            for accepted in ACCEPTED:
                assert accepted in message, (stat, accepted)

    @pytest.mark.parametrize("stat", ["blur_edge_width", *ACCEPTED])
    def test_every_other_statistic_is_accepted_under_a_mask(self, stat):
        assert gate(stat, min=0.0).mask is not None

    def test_the_refused_statistics_are_accepted_without_a_mask(self):
        for stat in sorted(REFUSED):
            assert gate(stat, min=0.0, mask=None).mask is None

    def test_an_all_true_mask_raises_at_construction(self):
        with pytest.raises(ValueError, match="mask leaves out every pixel"):
            QualityGate("mean", min=0.0, mask=np.ones((H, W)))

    def test_a_mask_that_is_not_2d_raises_at_construction(self):
        with pytest.raises(ValueError, match=r"2D .* got shape \(3, 48, 64\)"):
            QualityGate("mean", min=0.0, mask=np.ones((3, H, W)))

    def test_an_all_false_mask_is_no_mask(self):
        assert gate("mean", min=0.0, mask=np.zeros((H, W))).mask is None


class TestTheValue:
    @pytest.mark.parametrize("stat", ACCEPTED)
    @pytest.mark.parametrize("fill", [0, 255])
    def test_changing_only_the_masked_pixels_does_not_move_the_value(self, stat, fill):
        base = scene(0)
        selector = gate(stat, min=0.0)
        value = measure(selector, painted(base, fill))
        assert value == measure(selector, painted(base, 255 - fill))
        # Not vacuous: without the mask the same two frames read differently.
        loose = gate(stat, min=0.0, mask=None)
        one, two = measure(loose, painted(base, fill)), measure(loose, painted(base, 255 - fill))
        assert np.isfinite([value, one, two]).all()
        assert stat == "missing" or one != two  # nothing is missing from an integer frame

    def test_a_part_written_as_nan_is_not_missing_data(self):
        base = scene(0, 0, 256, np.float64)
        selector = gate("missing", min=0.0)
        assert measure(selector, painted(base, np.nan)) == measure(selector, painted(base, 0.0)) == 0.0
        assert measure(gate("missing", min=0.0, mask=None), painted(base, np.nan)) > 0.0

    @pytest.mark.parametrize("stat", ["sharpness", "blur_edge_width"])
    def test_a_uint16_frame_reads_its_scene_not_the_part(self, stat):
        base = scene(0, 0, 4096, np.uint16)
        selector = gate(stat, min=0.0)
        assert measure(selector, painted(base, 65535)) == measure(selector, painted(base, 0))

    @pytest.mark.parametrize("stat", ACCEPTED)
    def test_the_value_is_the_one_image_stats_reports_under_the_same_mask(self, stat):
        frame = painted(scene(3), 255)
        assert measure(gate(stat, min=0.0), frame) == reported(frame, stat, mask=MASK)

    def test_the_value_is_the_one_image_stats_reports_for_a_16_bit_frame(self):
        frame = painted(scene(3, 0, 4096, np.uint16), 65535)
        for stat in ("mean", "brightness", "sharpness"):
            assert measure(gate(stat, min=0.0), frame) == reported(frame, stat, mask=MASK)

    @pytest.mark.parametrize("stat", ["mean", "sharpness", "blur_edge_width"])
    def test_without_a_mask_the_value_is_unchanged(self, stat):
        frame = painted(scene(3), 255)
        assert measure(gate(stat, min=0.0, mask=None), frame) == reported(frame, stat)
        assert measure(gate(stat, min=0.0, mask=np.zeros((H, W))), frame) == reported(frame, stat)


class TestTheFit:
    def test_a_mask_of_the_frames_size_needs_no_opencv(self, monkeypatch):
        def refuse(*_: Any) -> Any:
            raise AssertionError("OpenCV was asked for")

        monkeypatch.setitem(sys.modules, "cv2", None)
        monkeypatch.setattr(ego_module, "_require_cv2", refuse)
        monkeypatch.setattr(image_stats_module, "_require_cv2", refuse)
        frame = scene(0)
        assert measure(gate("mean", min=0.0), painted(frame, 0)) == measure(gate("mean", min=0.0), painted(frame, 255))

    @pytest.mark.optional
    def test_a_rescaled_mask_without_opencv_names_the_extra(self, monkeypatch):
        def refuse() -> Any:
            raise ImportError("no cv2")

        monkeypatch.setattr(image_stats_module, "_require_cv2", refuse)
        with pytest.raises(ImportError, match=r"QualityGate need OpenCV.*dataeval\[opencv\]"):
            measure(gate("mean", min=0.0, mask=MASK.repeat(2, 0).repeat(2, 1)), scene())

    @pytest.mark.optional
    @pytest.mark.parametrize(("source", "frame"), [((1080, 1920), (1088, 1920)), ((360, 640), (721, 1280))])
    def test_the_padding_rows_are_fitted_as_image_stats_fits_them(self, source, frame):
        pytest.importorskip("cv2", reason="rescaling a mask needs opencv")
        mask = np.zeros(source, dtype=bool)
        mask[int(0.9 * source[0]) :] = True
        height, width = frame
        rows = ~_OverlayMask(np.ones_like(mask), "mask").fit((width, height), frame).overlay
        assert rows.any()
        assert not _OverlayMask(mask, "mask").fit((width, height), frame).overlay[rows].any()  # not masked but padding

        base = np.random.default_rng(0).integers(0, 256, (1, *frame)).astype(np.uint8)
        selector = gate("mean", min=0.0, mask=mask)
        value = measure(selector, base)
        assert value == measure(selector, painted(base, 255, rows))  # the padding rows are not scene
        assert value == reported(base, "mean", mask=mask)

    @pytest.mark.optional
    def test_the_padding_is_fitted_once_per_frame_size(self, monkeypatch):
        pytest.importorskip("cv2", reason="rescaling a mask needs opencv")
        builds: list[tuple[int, int]] = []
        original = _OverlayMask._build

        def counting(self: _OverlayMask, full_wh: tuple[int, int], proc_shape: tuple[int, int]) -> Any:
            builds.append(full_wh)
            return original(self, full_wh, proc_shape)

        image_stats_module._padding_rows.cache_clear()
        monkeypatch.setattr(_OverlayMask, "_build", counting)
        mask = np.zeros((360, 640), dtype=bool)
        mask[300:] = True
        selector = gate("mean", min=0.0, mask=mask)
        frames = [np.random.default_rng(i).integers(0, 256, (1, 721, 1280)).astype(np.uint8) for i in range(3)]
        for frame in frames:
            measure(selector, frame)
        # The mask once, by the gate's own `_OverlayMask`, and the padding once, however many frames of the size.
        assert builds == [(1280, 721), (1280, 721)]
        measure(selector, np.zeros((1, 360, 640), dtype=np.uint8))  # a frame of the mask's own size fits nothing
        measure(selector, np.zeros((1, 180, 320), dtype=np.uint8))
        assert len(builds) == 4
        measure(selector, np.zeros((1, 180, 320), dtype=np.uint8))
        assert len(builds) == 4

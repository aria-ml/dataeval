"""`image_stats(mask=)`: a mask that leaves a vehicle part or a burned-in HUD out of the pixel statistics (#1593).

The fixtures are synthetic. A 48x64 frame carries a hood (the bottom 12 rows) and a HUD block near the top-left corner,
and four boxes: one clear of the mask, one straddling the hood, one wholly under it and one that leaves the image over
the HUD. A scene is random texture, so every statistic has a value, and the masked pixels are painted with values that
move every pixel statistic and the decoded range of an unmasked call. The context statistics are left out of every
request here but the changing-only-the-masked-pixels test, which asks for them too: their own cases are in
`test_context_mask.py`, and blur's are in `test_blur_mask.py`.
"""

import logging
import re
import sys
import warnings
from typing import Any

import numpy as np
import pytest

import dataeval.core._ego_stats as ego_module
import dataeval.core._image_stats as image_stats_module
from dataeval.config import use_max_processes
from dataeval.core import combine_stats_results, image_stats
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._ego_stats import _as_mask, _OverlayMask
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import ImageStats
from dataeval.quality import Outliers
from dataeval.utils.preprocessing import BoundingBox, _decode_range, boxes_to_mask, get_value_range

pytestmark = pytest.mark.required

H, W = 48, 64
MASK = np.zeros((H, W), dtype=bool)
MASK[36:] = True  # a hood across the bottom
MASK[2:8, 2:20] = True  # a HUD readout

CLEAR = (30.0, 10.0, 50.0, 30.0)
STRADDLE = (10.0, 30.0, 40.0, 44.0)
UNDER = (40.0, 38.0, 60.0, 46.0)
PAST = (-5.0, -4.0, 10.0, 6.0)
BOXES = [CLEAR, STRADDLE, UNDER, PAST]
RGB = (0, 1, 2)

# The masked families of task 1, over every view that measures them.
PIXEL_REQUEST = ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.COLOR | ImageStats.NOISE
# With blur, whose threshold, widths and directionality have their own rule under a mask.
BLUR_REQUEST = PIXEL_REQUEST | ImageStats.BLUR
# With the context statistics, which read a box's surround and target and leave a masked pixel unreadable.
CONTEXT_REQUEST = BLUR_REQUEST | ImageStats.CONTEXT
# The families that keep the whole frame.
WHOLE_REQUEST = ImageStats.HASH | ImageStats.DIMENSION | ImageStats.SPATIAL

COVERS = "The mask, alone or with the boxes, covers the entire datum"
NO_RANGE = "no value range could be established"


def scene(kind: str, seed: int = 0) -> np.ndarray:
    """A textured (3, H, W) scene: uint8 over its full range, uint16 below 4096, float in [0, 1]."""
    rng = np.random.default_rng(seed)
    if kind == "uint8":
        return rng.integers(0, 256, (3, H, W), dtype=np.uint8)
    if kind == "uint16":
        return rng.integers(0, 4096, (3, H, W), dtype=np.uint16)
    return rng.random((3, H, W))


def painted(image: np.ndarray, fill: float, where: np.ndarray = MASK) -> np.ndarray:
    """The image with only the pixels `where` marks set to `fill`."""
    out = image.copy()
    out[:, where] = fill
    return out


def run(images: list[np.ndarray], boxes: list[list[Any]] | None = None, **kwargs: Any) -> Any:
    kwargs.setdefault("stats", PIXEL_REQUEST)
    if not isinstance(kwargs["stats"], dict) and kwargs["stats"] & ImageStats.COLOR:
        kwargs.setdefault("color_rgb_channels", RGB)
    if boxes is not None:
        kwargs.setdefault("labels", [[0] * len(b) for b in boxes])
    with warnings.catch_warnings():
        # Only what these calls say on purpose: the experimental flags and the user-facing coverage and range
        # warnings. A RuntimeWarning from numpy is a defect, so it stays visible (and fails under -W error).
        warnings.simplefilter("ignore", ExperimentalWarning)
        warnings.simplefilter("ignore", UserWarning)
        return image_stats(images, boxes=boxes, **kwargs)


def assert_same(left: Any, right: Any, *, skip: tuple[str, ...] = ()) -> None:
    """Every statistic and detail of two results is the same, in the same order, NaN equal to NaN."""
    for group in ("stats", "details"):
        names = [n for n in left.get(group, {}) if n not in skip]
        assert names == [n for n in right.get(group, {}) if n not in skip], group
        for name in names:
            a, b = left[group][name], right[group][name]
            assert a.dtype == b.dtype, f"{group}.{name}"
            np.testing.assert_array_equal(a, b, err_msg=f"{group}.{name}")


def column(result: Any, name: str) -> list[Any]:
    values = result["stats"][name] if name in result["stats"] else result["details"][name]
    return [v.item() if hasattr(v, "item") else v for v in values]


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


class TestMaskedPixelsDoNotMatter:
    """AT1, task 1 part: the image, a box, the background, the surround and a channel group."""

    @pytest.mark.parametrize(
        ("kind", "first", "second", "declared"),
        [
            ("uint8", 0, 255, None),
            ("uint16", 0, 65535, None),  # a HUD at full scale over a 12-bit scene
            ("uint16", 0, 65535, (0.0, 4095.0)),  # the same, with the range stated, so the context statistics read
            ("float", 0.0, 255.0, None),  # a part that would read the frame as 8-bit
            ("float", 0.0, np.nan, None),  # a part written as NaN
        ],
    )
    @pytest.mark.parametrize("normalize", [False, True])
    def test_changing_only_the_masked_pixels_moves_nothing(self, kind, first, second, declared, normalize):
        base = scene(kind)
        kwargs: dict[str, Any] = {
            "stats": CONTEXT_REQUEST,
            "per_background": True,
            "per_surround": True,
            "channels": {"rg": [0, 1]},
            "normalize_pixel_values": normalize,
        }
        if declared is not None:
            kwargs["value_range"] = declared
        one = run([painted(base, first)], [BOXES], mask=MASK, **kwargs)
        two = run([painted(base, second)], [BOXES], mask=MASK, **kwargs)
        assert_same(one, two)

        # Not vacuous: without the mask, the same two images differ.
        loose_one = run([painted(base, first)], [BOXES], **kwargs)
        loose_two = run([painted(base, second)], [BOXES], **kwargs)
        assert not np.array_equal(loose_one["stats"]["mean"], loose_two["stats"]["mean"], equal_nan=True)
        # The context statistics too, where the data kind has a range to read them against: a 16-bit frame has none
        # unless one is declared, so they are NaN in both, and the equality above holds of them as of any column.
        context = [n for n in ("surround_intensity_mean", "target_surround_contrast") if n in one["stats"]]
        assert len(context) == 2
        measured = kind != "uint16" or declared is not None
        for name in context:
            assert np.isfinite(one["stats"][name]).any() == measured, name
            moved = not np.array_equal(loose_one["stats"][name], loose_two["stats"][name], equal_nan=True)
            assert moved == measured, name

    def test_a_uint16_hud_does_not_set_the_range_the_scene_decodes_against(self):
        hud = painted(scene("uint16"), 65535)
        assert get_value_range(hud).depth == 16
        assert image_stats_module._masked_range(hud, MASK, None).depth == 12
        # The 12-bit scene reads as bright as itself, not as a scene squeezed into the bottom of 16 bits.
        masked = run([hud], mask=MASK, stats=ImageStats.VISUAL_BRIGHTNESS)
        loose = run([hud], stats=ImageStats.VISUAL_BRIGHTNESS)
        assert masked["stats"]["brightness"][0] > 8 * loose["stats"]["brightness"][0]


class TestTheWholeFrameClasses:
    """AT2, task 1 part: hash, dimension and spatial values do not see the mask."""

    def test_hash_dimension_and_spatial_values_are_unchanged(self):
        hud = painted(scene("uint16"), 65535)
        near_black = painted(np.random.default_rng(1).integers(0, 3, (3, H, W), dtype=np.uint8), 255)
        kwargs: dict[str, Any] = {"stats": WHOLE_REQUEST, "channels": {"rgb": [0, 1, 2]}}
        masked = run([hud, near_black], [BOXES, BOXES], mask=MASK, **kwargs)
        loose = run([hud, near_black], [BOXES, BOXES], **kwargs)
        assert_same(masked, loose, skip=("masked_fraction",))
        # The box wholly under the mask still hashes its pixels, and the depth is the whole frame's encoding.
        under = [i for i, s in enumerate(masked["source_index"]) if s.item == 0 and s.key == 2][0]
        assert masked["stats"]["xxhash"][under] != ""
        assert column(masked, "depth")[0] == 16
        assert column(masked, "rgb_depth")[0] == 16

    def test_a_mapping_of_band_groups_reads_each_groups_masked_range_and_its_whole_depth(self):
        base = scene("uint16")
        kwargs: dict[str, Any] = {
            "stats": {"rgb": ImageStats.PIXEL_HISTOGRAM | ImageStats.DIMENSION_DEPTH},
            "channels": {"rgb": [0, 1, 2]},
        }
        hud = run([painted(base, 65535)], mask=MASK, **kwargs)
        dark = run([painted(base, 0)], mask=MASK, **kwargs)
        np.testing.assert_array_equal(hud["stats"]["rgb_histogram"], dark["stats"]["rgb_histogram"])
        assert column(hud, "rgb_depth") == [16]
        assert column(dark, "rgb_depth") == [12]
        assert column(hud, "rgb_depth") == column(run([painted(base, 65535)], **kwargs), "rgb_depth")


class TestCoverage:
    """AT4: a region the mask covers, `masked_fraction` and `background_fraction`, and the coverage warning."""

    def test_a_box_under_the_mask_reads_as_a_region_exclude_removed_today(self):
        image = scene("uint8")
        masked = run([image], [BOXES], mask=MASK, per_image=False)
        under = column(masked, "masked_fraction").index(1.0)
        # Today's wholly removed region: a background its box covers.
        removed = run([image], [[(0, 0, W, H)]], per_image=False, per_target=False, per_background=True)
        for name in masked["stats"]:
            np.testing.assert_array_equal(
                masked["stats"][name][under], removed["stats"][f"background_{name}"][0], err_msg=name
            )
        statuses = [name for name in masked["details"] if name.endswith("_status")]
        assert statuses
        for name in statuses:
            assert (
                masked["details"][name][under] == removed["details"][f"background_{name}"][0] == "insufficient_support"
            )

    def test_masked_fraction_is_the_share_of_each_rows_in_image_region(self):
        outside = (100.0, 100.0, 120.0, 120.0)
        result = run([scene("uint8")], [[*BOXES, outside]], mask=MASK, stats=ImageStats.PIXEL_MEAN)
        expected = [
            MASK.mean(),
            0.0,
            MASK[30:44, 10:40].mean(),
            1.0,
            MASK[0:6, 0:10].mean(),  # the part of PAST inside the image
            np.nan,  # not a valid box
        ]
        np.testing.assert_allclose(column(result, "masked_fraction"), expected)

    def test_background_fraction_is_the_share_of_the_unmasked_pixels_the_boxes_leave(self):
        image = scene("uint8")
        result = run([image, image], [BOXES, []], mask=MASK, stats=ImageStats.PIXEL_MEAN, per_background=True)
        rows = [i for i, s in enumerate(result["source_index"]) if s.key is None]
        boxes = boxes_to_mask(image.shape, BOXES)
        expected = np.count_nonzero(~(boxes | MASK)) / np.count_nonzero(~MASK)
        assert result["stats"]["background_fraction"][rows[0]] == pytest.approx(expected)
        assert result["stats"]["background_fraction"][rows[1]] == 1.0

    @pytest.mark.optional
    def test_a_frame_the_fitted_mask_covers_warns_once_and_reads_no_range_warning(self, caplog):
        pytest.importorskip("cv2", reason="rescaling a mask needs opencv")
        nearly = np.ones((2 * H, 2 * W), dtype=bool)
        nearly[0, 0] = False  # a gap the downscale fills
        assert image_stats_module._fit_mask(nearly, (H, W)).all()
        kwargs: dict[str, Any] = {
            "stats": ImageStats.PIXEL | ImageStats.VISUAL,
            "channels": {"rg": [0, 1]},
            "per_background": True,
        }
        with caplog.at_level(logging.WARNING, logger="dataeval.core"):
            result = run([scene("uint8")], [[CLEAR]], mask=nearly, **kwargs)
            declared = run([scene("float")], [[CLEAR]], mask=nearly, value_range=(0.0, 1.0), **kwargs)
        logged = messages(caplog)
        assert sum(COVERS in m for m in logged) == 2
        assert not any(NO_RANGE in m for m in logged)
        assert not any("Boxes cover the entire datum" in m for m in logged)
        for found in (result, declared):
            for name in found["details"]:
                if name.endswith("_status"):  # box rows hold padding for the background's statuses
                    assert set(column(found, name)) - {"not_applicable"} == {"insufficient_support"}, name
            assert np.isnan(found["stats"]["mean"]).all()
            assert np.isnan(found["stats"]["background_fraction"]).all()

    def test_the_boxes_and_the_mask_together_covering_the_frame_warns(self, caplog):
        lower = np.zeros((H, W), dtype=bool)
        lower[H // 2 :] = True
        with caplog.at_level(logging.WARNING, logger="dataeval.core"):
            run([scene("uint8")], [[(0, 0, W, H // 2)]], mask=lower, stats=ImageStats.PIXEL_MEAN, per_background=True)
        logged = messages(caplog)
        assert any(COVERS in m for m in logged)
        assert not any("Boxes cover the entire datum" in m for m in logged)

    def test_boxes_that_cover_the_frame_alone_give_only_todays_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="dataeval.core"):
            run([scene("uint8")], [[(0, 0, W, H)]], mask=MASK, stats=ImageStats.PIXEL_MEAN, per_background=True)
        logged = messages(caplog)
        assert any("Boxes cover the entire datum" in m for m in logged)
        assert not any(COVERS in m for m in logged)

    def test_a_box_with_one_unmasked_row_reads_sharpness_nan_and_warns_nothing(self):
        # Rows 36 and below are the hood, so this box keeps only row 35: too little to filter, and numpy would
        # warn "Degrees of freedom <= 0" over the all-NaN window if the spread were taken blind.
        box = (10.0, 35.0, 40.0, 46.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = image_stats(
                [scene("uint8")], boxes=[[box]], labels=[[0]], stats=ImageStats.VISUAL_SHARPNESS, mask=MASK
            )
        assert np.isfinite(result["stats"]["sharpness"][0])
        assert np.isnan(result["stats"]["sharpness"][1])


def _refuse_opencv(*_: Any) -> Any:
    raise AssertionError("OpenCV was asked for")


class TestFitting:
    """AT5: the fit, by `ego_stats`' rule, with the padding rows masked."""

    def test_a_mask_of_the_images_size_needs_no_opencv(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        monkeypatch.setattr(ego_module, "_require_cv2", _refuse_opencv)
        monkeypatch.setattr(image_stats_module, "_require_cv2", _refuse_opencv)
        result = run([scene("uint8")], mask=MASK, stats=ImageStats.PIXEL_MEAN)
        assert column(result, "masked_fraction") == [pytest.approx(MASK.mean())]

    @pytest.mark.optional
    @pytest.mark.parametrize(("source", "frame"), [((1080, 1920), (1088, 1920)), ((360, 640), (721, 1280))])
    def test_a_rescaled_mask_is_egos_fit_with_the_padding_rows_masked(self, source, frame):
        pytest.importorskip("cv2", reason="rescaling a mask needs opencv")
        mask = np.zeros(source, dtype=bool)
        mask[int(0.9 * source[0]) :] = True
        height, width = frame
        fit = _OverlayMask(mask, "mask").fit((width, height), frame).overlay
        padding = ~_OverlayMask(np.ones_like(mask), "mask").fit((width, height), frame).overlay
        fitted = image_stats_module._fit_mask(mask, frame)
        np.testing.assert_array_equal(fitted, fit | padding)
        assert padding.any()
        assert fitted[padding].all()
        assert not padding[: frame[0] - 8].any()

    @pytest.mark.optional
    def test_a_rescaled_mask_reaches_the_rows(self):
        pytest.importorskip("cv2", reason="rescaling a mask needs opencv")
        mask = np.zeros((360, 640), dtype=bool)
        mask[300:] = True
        image = np.zeros((1, 721, 1280), dtype=np.uint8)
        result = run([image], mask=mask, stats=ImageStats.PIXEL_MEAN)
        assert column(result, "masked_fraction") == [
            pytest.approx(image_stats_module._fit_mask(mask, (721, 1280)).mean())
        ]

    @pytest.mark.optional
    def test_each_image_of_a_mixed_dataset_gets_its_own_fit(self):
        pytest.importorskip("cv2", reason="rescaling a mask needs opencv")
        small, large = scene("uint8"), np.repeat(np.repeat(scene("uint8", 1), 2, axis=1), 2, axis=2)
        result = run([small, large], mask=MASK, stats=ImageStats.PIXEL_MEAN)
        expected = [MASK.mean(), image_stats_module._fit_mask(MASK, (2 * H, 2 * W)).mean()]
        np.testing.assert_allclose(column(result, "masked_fraction"), expected)

    @pytest.mark.optional
    def test_a_mask_that_fits_no_frame_raises_naming_the_image(self):
        pytest.importorskip("cv2", reason="rescaling a mask needs opencv")
        with pytest.raises(ValueError, match=r"image 1: mask of 64x48 does not fit a 48x48 frame"):
            run([scene("uint8"), scene("uint8")[:, :, :H]], mask=MASK, stats=ImageStats.PIXEL_MEAN)


class TestRefusals:
    """AT6: the refusals and the messages."""

    def test_an_all_true_mask_raises_before_any_image_is_read(self):
        def images():
            raise AssertionError("an image was read")
            yield  # pragma: no cover

        with pytest.raises(ValueError, match="mask leaves out every pixel"):
            image_stats(images(), mask=np.ones((H, W)), stats=ImageStats.PIXEL_MEAN)

    def test_a_mask_on_data_without_an_image_plane_raises_naming_it(self):
        with pytest.raises(ValueError, match=r"image 1: .*shape \(8,\)"):
            run([scene("uint8"), np.zeros(8)], mask=MASK, stats=ImageStats.PIXEL_MEAN)

    @pytest.mark.parametrize(
        ("shape", "fix"),
        [((H, W, 3), "mask.any(axis=-1)"), ((3, H, W), "mask.any(axis=0)"), ((1, H, W), "np.squeeze(mask)")],
    )
    def test_a_mask_that_is_not_2d_is_told_how_to_make_it_so(self, shape, fix):
        with pytest.raises(ValueError, match=re.escape(str(shape))) as raised:
            _as_mask(np.ones(shape), "mask")
        assert fix in str(raised.value)

    def test_a_missing_opencv_names_the_extra_and_image_stats(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        with pytest.raises(ImportError, match=r"image_stats.*dataeval\[opencv\]"):
            run([scene("uint8")], mask=MASK.repeat(2, axis=0).repeat(2, axis=1))


class TestNoMask:
    """AT7: no mask, and an all-false mask, are today's call."""

    @staticmethod
    def observe(caplog: pytest.LogCaptureFixture, **kwargs: Any) -> tuple[Any, list[str], list[str]]:
        caplog.clear()
        with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING, logger="dataeval"):
            warnings.simplefilter("always")
            result = image_stats(
                [painted(scene("uint16"), 65535), scene("float", 1), scene("uint8", 2)],
                boxes=[BOXES, [CLEAR], []],
                labels=[[0, 1, 0, 1], [0], []],
                stats=ImageStats.ALL,
                per_background=True,
                per_surround=True,
                channels={"rg": [0, 1]},
                color_rgb_channels=RGB,
                **kwargs,
            )
        return result, [f"{w.category.__name__}: {w.message}" for w in caught], messages(caplog)

    def test_none_and_an_all_false_mask_equal_the_call_without_the_keyword(self, caplog):
        self.observe(caplog)  # the once-per-process notices, said before the calls compared
        base, base_warned, base_logged = self.observe(caplog)
        for kwargs in ({"mask": None}, {"mask": np.zeros((H, W), dtype=np.uint8)}):
            result, warned, logged = self.observe(caplog, **kwargs)
            assert result.keys() == base.keys()
            assert_same(result, base)
            assert result["source_index"] == base["source_index"]
            assert result["measurement_specs"] == base["measurement_specs"]
            assert warned == base_warned
            assert logged == base_logged


class TestCombine:
    """AT8: results measured with a mask and without one combine, and `masked_fraction` is never scored."""

    def test_a_masked_and_an_unmasked_result_combine(self):
        images = [painted(scene("float", seed), 1.0) for seed in range(12)]
        masked = run(images, mask=MASK, stats=ImageStats.PIXEL | ImageStats.VISUAL)
        loose = run(images, stats=ImageStats.PIXEL | ImageStats.VISUAL)
        stats, _, steps = combine_stats_results([masked, loose])
        assert "masked_fraction" not in stats
        assert steps == [12, 24]
        flagged = Outliers().from_stats([masked, loose]).data()
        assert "masked_fraction" not in set(flagged["metric_name"].to_list())


class TestConsistency:
    """AT13, task 1 parts, and the process pool."""

    def test_a_masked_row_lists_its_columns_in_the_same_order(self):
        kwargs: dict[str, Any] = {
            "stats": WHOLE_REQUEST | BLUR_REQUEST,
            "per_background": True,
            "channels": {"rg": [0, 1]},
        }
        masked = run([scene("uint8")], [BOXES], mask=MASK, **kwargs)
        loose = run([scene("uint8")], [BOXES], **kwargs)
        assert list(masked["stats"]) == list(loose["stats"])
        assert [n for n in masked["details"] if n != "masked_fraction"] == list(loose["details"])

    def test_the_image_row_of_a_background_only_call_carries_masked_fraction(self):
        result = run([scene("uint8")], [BOXES], mask=MASK, per_image=False, per_background=True)
        image_row = [i for i, s in enumerate(result["source_index"]) if s.key is None]
        assert column(result, "masked_fraction")[image_row[0]] == pytest.approx(MASK.mean())

    def test_a_box_past_the_image_edge_is_measured_over_its_part_inside(self):
        result = run([scene("uint8")], [[PAST]], mask=MASK, per_image=False)
        assert column(result, "masked_fraction") == [pytest.approx(MASK[:6, :10].mean())]
        assert np.isfinite(result["stats"]["mean"]).all()

    def test_several_workers_equal_the_serial_run(self):
        images = [painted(scene("uint16", seed), 65535) for seed in range(4)]
        kwargs: dict[str, Any] = {
            "stats": BLUR_REQUEST,
            "mask": MASK,
            "per_background": True,
            "channels": {"rg": [0, 1]},
        }
        with use_max_processes(1):
            serial = run(images, [BOXES] * 4, **kwargs)
        with use_max_processes(2):
            pooled = run(images, [BOXES] * 4, **kwargs)
        assert_same(serial, pooled)


class TestCacheMask:
    """The fitted mask in the cache, and its crop to a row's window."""

    def test_the_mask_is_cropped_as_exclude_is(self):
        image = scene("uint8")
        for box in (None, BoundingBox(*PAST, image_shape=image.shape), BoundingBox(*STRADDLE, image_shape=image.shape)):
            cache = CalculatorCache(image, box, exclude=MASK, mask=MASK)
            np.testing.assert_array_equal(cache.mask_window, cache.window_mask)
        assert CalculatorCache(image, None).mask_window is None

    def test_the_range_of_a_frame_with_no_unmasked_pixel_comes_from_its_dtype(self):
        everything = np.ones((H, W), dtype=bool)
        assert image_stats_module._masked_range(scene("uint16"), everything, None) == _decode_range(np.uint16, 0.0)
        assert image_stats_module._masked_range(scene("float"), everything, None) == _decode_range(np.float64, 0.0)
        declared = image_stats_module._masked_range(scene("float"), everything, (0.0, 2.0))
        assert (declared.pmin, declared.pmax) == (0.0, 2.0)

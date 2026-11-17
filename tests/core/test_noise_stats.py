"""The noise statistics in `image_stats`: `noise_std` and `noise_var` from `noise_immerkaer`, on every view.

Every fixture is synthetic and seeded: white Gaussian noise on a flat level, so a reading is near the noise added and
the exact value is what `noise_immerkaer` gives on the same pixels.
"""

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

import dataeval._experimental as experimental_module
from dataeval.core import image_stats, noise_immerkaer
from dataeval.core._calculators._contextstats import surround_regions
from dataeval.data import Crop, DetectionCrops, Resize, SelectChannels, TorchvisionTransform
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import boxes_to_mask

pytestmark = pytest.mark.required

NOISE = ImageStats.NOISE
COLUMNS = {"noise_std", "noise_var"}
DETAILS = {"noise_status", "noise_neighborhoods"}

# Recorded from main at b9c7a666b, before the noise statistics, from `pinned()` below: what each request returned, in
# order. A request that does not name NOISE must return exactly these.
STANDARD_STATS = """
    offset_x offset_y width height channels size aspect_ratio depth center distance_center distance_edge invalid_box
    rel_offset_x rel_offset_y rel_width rel_height rel_size rel_distance_center rel_distance_edge
    mean std var skew kurtosis entropy missing zeros histogram saturated_fraction
    brightness contrast darkness sharpness percentiles
"""
SATURATED_DETAILS = """
    saturated_status saturated_above_range_fraction saturated_upper
"""
BASIC_STATS = "width height channels aspect_ratio mean std var brightness contrast sharpness"
STANDARD_HASH_STATS = """
    offset_x offset_y width height channels size aspect_ratio depth center distance_center distance_edge invalid_box
    rel_offset_x rel_offset_y rel_width rel_height rel_size rel_distance_center rel_distance_edge
    xxhash phash dhash phash_d4 dhash_d4
    mean std var skew kurtosis entropy missing zeros histogram saturated_fraction
    brightness contrast darkness sharpness percentiles
"""
ALL_STATS = """
    blur_edge_width blur_edge_directionality
    color_lightness_mean color_chroma_mean color_achromatic_fraction color_hue_cos_mean color_hue_sin_mean
    surround_intensity_mean surround_intensity_std target_surround_contrast signal_clutter_ratio
    offset_x offset_y width height channels size aspect_ratio depth center distance_center distance_edge invalid_box
    rel_offset_x rel_offset_y rel_width rel_height rel_size rel_distance_center rel_distance_edge
    xxhash phash dhash phash_d4 dhash_d4
    mean std var skew kurtosis entropy missing zeros histogram saturated_fraction
    nn_distance_norm box_in_image_fraction covered_fraction max_iou_same_class max_iou_other_class
    brightness contrast darkness sharpness percentiles
"""
ALL_DETAILS = """
    blur_status blur_edge_directionality_reason
    color_status color_valid_fraction color_valid_pixels color_excluded_pixels color_hue_reason
    context_status context_incomplete_geometry surround_candidate_pixels surround_retained_pixels
    surround_peer_excluded_fraction surround_measurable_pixels surround_valid_fraction target_valid_fraction
    context_range_low context_range_high context_ratio_reason target_surround_contrast_floored
    signal_clutter_ratio_floored
    saturated_status saturated_above_range_fraction saturated_upper
    spatial_status spatial_incomplete_geometry nn_distance_norm_target_index nn_distance_norm_reason
    spatial_peer_count spatial_skipped_count covered_fraction_reason max_iou_same_class_target_index
    max_iou_same_class_reason max_iou_other_class_target_index max_iou_other_class_label max_iou_other_class_reason
"""


@pytest.fixture
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start the test with no experimental warning given yet, as in a new process."""
    monkeypatch.setattr(experimental_module, "_WARNED", set())


def noisy(shape: tuple[int, ...], sigma: float = 4.0, seed: int = 0, level: float = 100.0) -> np.ndarray:
    """White Gaussian noise of `sigma` on a flat `level`, in float64."""
    return level + np.random.default_rng(seed).normal(0.0, sigma, size=shape)


def noisy_uint8(shape: tuple[int, ...] = (3, 48, 64), sigma: float = 6.0, seed: int = 0) -> np.ndarray:
    return np.clip(np.round(noisy(shape, sigma, seed, 128.0)), 0, 255).astype(np.uint8)


def run(data: Any, stats: Any = NOISE, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return image_stats(data, stats=stats, **kwargs)


def col(result: Any, name: str) -> list[Any]:
    values = result["stats"][name] if name in result["stats"] else result["details"][name]
    return [v.item() if hasattr(v, "item") else v for v in values]


def joint(pixels: np.ndarray, scale: float = 1.0) -> tuple[float, float]:
    """The joint reading `image_stats` reports, from `noise_immerkaer`: the channels' mean variance and its root."""
    std = noise_immerkaer(pixels)["noise_std"] / scale
    var = float(np.mean(std**2))
    return float(np.sqrt(var)), var


def assert_reads(result: Any, row: int, expected: tuple[float, float], prefix: str = "") -> None:
    # The statistics are stored as float32.
    assert col(result, f"{prefix}noise_std")[row] == pytest.approx(expected[0], rel=1e-6)
    assert col(result, f"{prefix}noise_var")[row] == pytest.approx(expected[1], rel=1e-6)


class Sensors:
    """An image classification dataset whose items carry their own metadata."""

    metadata = {"id": "sensors"}

    def __init__(self, images: list[np.ndarray], metadata: Sequence[Mapping[str, Any]]) -> None:
        self.images, self.meta = images, metadata

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return self.images[i], np.array([1.0, 0.0]), {"id": i, **self.meta[i]}


class TestFloor:
    def test_ten_by_ten_is_measured_and_nine_by_nine_is_insufficient_support(self):
        ten, nine = noisy((10, 10)), noisy((9, 9))
        result = run([ten, nine])
        assert col(result, "noise_status") == ["ok", "insufficient_support"]
        assert col(result, "noise_neighborhoods") == [64, 49]
        assert_reads(result, 0, joint(ten))
        assert np.isnan(col(result, "noise_std")[1])
        assert np.isnan(col(result, "noise_var")[1])

    def test_the_variance_is_the_square_of_the_standard_deviation(self):
        result = run([noisy((3, 32, 32))])
        std, var = col(result, "noise_std")[0], col(result, "noise_var")[0]
        assert var == pytest.approx(std**2, rel=1e-6)

    @pytest.mark.parametrize("datum", [np.arange(50, dtype=np.float64), np.zeros((0, 16, 16), np.uint8)])
    def test_a_datum_with_no_image_plane_or_no_channel_has_no_neighborhood(self, datum):
        result = run([datum])
        assert col(result, "noise_status") == ["insufficient_support"]
        assert col(result, "noise_neighborhoods") == [0]
        assert np.isnan(col(result, "noise_std")[0])

    def test_an_hw_image_reads_as_its_one_channel_image(self):
        image = noisy_uint8((1, 32, 32))
        flat, stacked = run([image[0]]), run([image])
        for name in (*COLUMNS, *DETAILS):
            assert col(flat, name) == col(stacked, name), name


class TestJointView:
    def test_three_scaled_channels_read_the_mean_of_their_variances(self):
        base = noisy((40, 40))
        scaled = np.stack([base, 2 * base, 3 * base])
        alone = run([base])
        result = run([scaled, scaled[::-1]])
        assert col(result, "noise_var")[0] == pytest.approx(col(alone, "noise_var")[0] * 14 / 3, rel=1e-6)
        assert col(result, "noise_var")[1] == pytest.approx(col(result, "noise_var")[0], rel=1e-6)
        assert col(result, "noise_std")[0] == pytest.approx(np.sqrt(col(result, "noise_var")[0]), rel=1e-6)
        assert_reads(result, 0, joint(scaled))

    def test_the_least_channel_count_is_reported(self):
        image = noisy((2, 16, 16))
        image[1, :4] = np.nan  # four rows gone from the second channel: 10 x 14 responses
        result = run([image])
        assert col(result, "noise_neighborhoods") == [10 * 14]
        assert col(result, "noise_status") == ["ok"]


class TestUnits:
    def test_uint8_normalized_reads_the_raw_reading_over_255(self):
        image = noisy_uint8()
        raw = run([image])
        normalized = run([image], normalize_pixel_values=True)
        assert_reads(raw, 0, joint(image))
        assert_reads(normalized, 0, joint(image, 255.0))
        assert col(normalized, "noise_neighborhoods") == col(raw, "noise_neighborhoods")

    def test_float_conventions_follow_each_images_own_values(self):
        # One sensor's noise, sigma 0.03, on 0-1 data at three levels, not clipped: the docs tell float users to declare
        # value_range because the convention, and so the scale, follows each image's own values.
        rng = np.random.default_rng(7)
        mid, high, low = (level + rng.normal(0, 0.03, (1, 32, 32)) for level in (0.5, 0.97, 0.03))
        assert mid.min() >= 0
        assert mid.max() <= 1
        assert high.max() > 1
        assert high.min() >= 0
        assert low.min() < 0
        result = run([mid, high, low], normalize_pixel_values=True)
        raw = [joint(image)[0] for image in (mid, high, low)]
        assert col(result, "noise_status") == ["ok", "ok", "unknown_range"]
        assert col(result, "noise_std")[0] == pytest.approx(raw[0], rel=1e-6)
        assert col(result, "noise_std")[1] == pytest.approx(joint(high, 255.0)[0], rel=1e-6)
        declared = run([mid, high, low], normalize_pixel_values=True, value_range=(0.0, 1.0))
        assert col(declared, "noise_status") == ["ok"] * 3
        assert col(declared, "noise_std") == pytest.approx(raw, rel=1e-6)

    def test_floats_with_no_conventional_range_normalized_are_unknown_range(self, caplog: pytest.LogCaptureFixture):
        image = np.random.default_rng(1).normal(0.5, 0.6, (3, 16, 16))
        with caplog.at_level(logging.WARNING, logger="dataeval"):
            result = run([image], normalize_pixel_values=True)
        assert col(result, "noise_status") == ["unknown_range"]
        assert col(result, "noise_neighborhoods") == [-1]
        assert np.isnan(col(result, "noise_std")[0])
        assert np.isnan(col(result, "noise_var")[0])
        said = [r.getMessage() for r in caplog.records if "noise" in r.getMessage()]
        assert len(said) == 1
        assert "value_range=" in said[0]
        # Read in its own units, the same image needs no range.
        assert_reads(run([image]), 0, joint(image))

    def test_uint16_normalized_needs_a_declared_range(self):
        image = (noisy_uint8() * 16).astype(np.uint16)
        assert col(run([image], normalize_pixel_values=True), "noise_status") == ["unknown_range"]
        declared = run([image], normalize_pixel_values=True, value_range=(0, 4095))
        assert col(declared, "noise_status") == ["ok"]
        assert_reads(declared, 0, joint(image, 4095.0))

    def test_a_declared_range_is_used_from_the_call_and_from_the_datum_metadata(self):
        image = np.random.default_rng(1).normal(0.5, 0.6, (3, 16, 16))
        from_call = run([image], normalize_pixel_values=True, value_range=(-3.0, 4.0))
        from_metadata = run(Sensors([image], [{"value_range": (-3.0, 4.0)}]), normalize_pixel_values=True)
        assert_reads(from_call, 0, joint(image, 7.0))
        assert_reads(from_metadata, 0, joint(image, 7.0))
        assert col(from_metadata, "noise_status") == ["ok"]

    def test_a_channel_group_reads_against_its_own_declared_range(self):
        from dataeval.utils.preprocessing import ChannelGroup

        image = (noisy_uint8() * 16).astype(np.uint16)
        result = run(
            [image], {None: ImageStats.NONE, "g": NOISE}, normalize_pixel_values=True,
            channels={"g": ChannelGroup([1], value_range=(0, 4095))},
        )  # fmt: skip
        assert_reads(result, 0, joint(image[[1]], 4095.0), prefix="g_")


class TestViews:
    def test_a_box_row_reads_its_own_window(self):
        image = noisy_uint8()
        result = run([image], boxes=[[[2, 3, 20, 30]]], labels=[[0]])
        assert col(result, "noise_status") == ["ok", "ok"]
        assert_reads(result, 0, joint(image))
        assert_reads(result, 1, joint(image[:, 3:30, 2:20]))

    def test_the_background_counts_only_neighborhoods_clear_of_the_boxes(self):
        image = noisy_uint8()
        result = run([image], boxes=[[[10, 10, 30, 30]]], labels=[[0]], per_background=True, per_target=False)
        # A response's 3x3 window touches rows and columns 10-29 of the box from 22 positions on each axis.
        assert col(result, "background_noise_neighborhoods") == [46 * 62 - 22 * 22]
        masked = image.astype(np.float64)
        masked[:, boxes_to_mask(image.shape, [[10, 10, 30, 30]])] = np.nan
        assert_reads(result, 0, joint(masked), prefix="background_")
        assert col(result, "background_noise_status") == ["ok"]

    def test_each_band_group_reads_its_channel_alone_beside_the_joint_view(self):
        image = noisy_uint8()
        result = run([image], channels={"r": 0, "g": 1, "b": 2})
        assert_reads(result, 0, joint(image))
        for c, name in enumerate("rgb"):
            assert_reads(result, 0, joint(image[[c]]), prefix=f"{name}_")
            alone = run([image[[c]]])
            assert col(result, f"{name}_noise_std") == col(alone, "noise_std")
            assert col(result, f"{name}_noise_neighborhoods") == col(alone, "noise_neighborhoods")

    def test_the_surround_view_reads_the_ring_with_every_box_masked(self, fresh_process):
        image = noisy_uint8((3, 96, 128))
        boxes = [[40.0, 30.0, 60.0, 50.0], [62.0, 30.0, 70.0, 40.0]]
        with pytest.warns(ExperimentalWarning, match="'per_surround' is experimental") as record:
            result = image_stats([image], boxes=[boxes], labels=[[0, 0]], stats=NOISE, per_surround=True)
        assert len([w for w in record if issubclass(w.category, ExperimentalWarning)]) == 1
        assert record[0].filename == __file__, "the warning points at the caller of image_stats"
        x0, y0, x1, y1 = surround_regions(np.asarray(boxes[0]), (96, 128), 0.25).window
        ring = image.astype(np.float64)
        ring[:, boxes_to_mask(image.shape, boxes)] = np.nan
        assert_reads(result, 1, joint(ring[:, y0:y1, x0:x1]), prefix="surround_")
        assert col(result, "surround_noise_status")[:2] == ["not_applicable", "ok"]
        assert col(result, "surround_noise_neighborhoods")[0] == -1
        assert np.isnan(col(result, "surround_noise_std")[0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            image_stats([image], boxes=[boxes], labels=[[0, 0]], stats=NOISE, per_surround=True)

    def test_a_band_the_datum_lacks_is_missing_channel(self):
        result = run([noisy_uint8()], {None: ImageStats.PIXEL_MEAN, "far": NOISE}, channels={"far": [7]})
        assert col(result, "far_noise_status") == ["missing_channel"]
        assert col(result, "far_noise_neighborhoods") == [-1]
        assert np.isnan(col(result, "far_noise_std")[0])

    def test_an_invalid_box_is_invalid_box_and_a_reversed_one_reads_its_twin(self):
        image = noisy_uint8()
        boxes = [[300.0, 300.0, 320.0, 320.0], [20.0, 20.0, 20.0, 30.0], [30.0, 40.0, 10.0, 5.0], [10.0, 5.0, 30.0, 40]]
        result = run([image], boxes=[boxes], labels=[[0] * 4], per_image=False)
        assert col(result, "noise_status") == ["invalid_box", "invalid_box", "ok", "ok"]
        assert col(result, "noise_neighborhoods")[:2] == [-1, -1]
        assert col(result, "noise_std")[2] == col(result, "noise_std")[3]
        assert col(result, "noise_neighborhoods")[2] == col(result, "noise_neighborhoods")[3]

    def test_the_statuses_apply_in_the_shared_order(self):
        # An invalid box, then a band the datum lacks, then no range, then too few neighborhoods.
        image = np.random.default_rng(1).normal(0.5, 0.6, (3, 4, 4))
        far = {"stats": {None: NOISE, "far": NOISE}, "channels": {"far": [7]}, "normalize_pixel_values": True}
        boxed = run([image], boxes=[[[300, 300, 310, 310]]], per_image=False, **far)
        assert col(boxed, "far_noise_status") == col(boxed, "noise_status") == ["invalid_box"]
        unboxed = run([image], **far)
        assert col(unboxed, "far_noise_status") == ["missing_channel"]
        assert col(unboxed, "noise_status") == ["unknown_range"]
        assert col(run([image]), "noise_status") == ["insufficient_support"]


class TestRows:
    def test_ok_if_and_only_if_both_values_are_finite(self):
        images = [noisy_uint8(), noisy_uint8((3, 9, 9)), np.random.default_rng(1).normal(0.5, 0.6, (3, 16, 16))]
        boxes = [[[2, 3, 20, 30], [300, 300, 310, 310]], [[0, 0, 4, 4]], [[1, 1, 12, 12]]]
        result = run(images, boxes=boxes, labels=[[0] * len(b) for b in boxes], per_background=True)
        for prefix, other in (("", "invalid_box"), ("background_", "not_applicable")):
            statuses = col(result, f"{prefix}noise_status")
            assert {"ok", "insufficient_support", other} <= set(statuses)
            for status, std, var in zip(
                statuses, col(result, f"{prefix}noise_std"), col(result, f"{prefix}noise_var"), strict=True
            ):
                assert (status == "ok") == (np.isfinite(std) and np.isfinite(var)), (prefix, status, std, var)

    def test_reordering_the_images_permutes_the_rows(self):
        images = [noisy_uint8(seed=s, sigma=2.0 + 3 * s) for s in range(3)]
        forward, backward = run(images), run(images[::-1])
        for name in (*COLUMNS, *DETAILS):
            assert col(backward, name) == col(forward, name)[::-1], name


class TestNothingElseMoves:
    @staticmethod
    def pinned(stats: ImageStats) -> Any:
        image = np.random.default_rng(0).integers(0, 256, (3, 32, 32), dtype=np.uint8)
        if stats & ImageStats.SPATIAL:
            return run([image], stats, boxes=[[[2, 3, 20, 30]]], labels=[[0]], color_rgb_channels=(0, 1, 2))
        return run([image], stats)

    @pytest.mark.parametrize(
        ("stats", "columns", "details"),
        [
            (ImageStats.STANDARD, STANDARD_STATS, SATURATED_DETAILS),
            (ImageStats.BASIC, BASIC_STATS, ""),
            (ImageStats.STANDARD | ImageStats.HASH, STANDARD_HASH_STATS, SATURATED_DETAILS),
            (ImageStats.ALL & ~ImageStats.NOISE, ALL_STATS, ALL_DETAILS),
        ],
    )
    def test_a_request_without_noise_returns_the_columns_it_returned_before(self, stats, columns, details):
        result = self.pinned(stats)
        assert list(result["stats"]) == columns.split()
        assert list(result.get("details", {})) == details.split()

    def test_all_gains_exactly_the_two_statistics_and_the_two_details(self):
        everything, without = self.pinned(ImageStats.ALL), self.pinned(ImageStats.ALL & ~ImageStats.NOISE)
        assert set(everything["stats"]) - set(without["stats"]) == COLUMNS
        assert set(everything["details"]) - set(without["details"]) == DETAILS
        assert set(without["stats"]) <= set(everything["stats"])
        for part in ("stats", "details"):
            for name in without[part]:
                np.testing.assert_array_equal(everything[part][name], without[part][name], err_msg=name)

    def test_noise_is_in_all_and_in_no_smaller_group(self):
        assert ImageStats.PIXEL_NOISE_STD | ImageStats.PIXEL_NOISE_VAR == NOISE
        assert NOISE in ImageStats.ALL
        for group in ("STANDARD", "BASIC", "PIXEL", "VISUAL", "DIMENSION", "HASH", "BLUR", "SPATIAL", "CONTEXT"):
            assert not NOISE & getattr(ImageStats, group), group
        assert ImageStats.PIXEL_NOISE_STD.value == 2 * ImageStats.VISUAL_COLOR_HUE_SIN_MEAN.value
        assert [(m.name or "").split("_", 1)[1].lower() for m in NOISE] == ["noise_std", "noise_var"]

    def test_no_experimental_warning_names_noise(self, fresh_process):
        image = noisy_uint8()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            image_stats([image], boxes=[[[2, 3, 20, 30]]], labels=[[0]], stats=ImageStats.ALL, per_background=True)
        experimental = [str(w.message) for w in caught if issubclass(w.category, ExperimentalWarning)]
        assert experimental
        assert not [m for m in experimental if "noise" in m.lower()]

    def test_all_without_blur_and_color_warns_nothing_experimental(self, fresh_process):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            result = image_stats(
                [noisy_uint8()],
                boxes=[[[2, 3, 20, 30]]],
                labels=[[0]],
                stats=ImageStats.ALL & ~ImageStats.BLUR & ~ImageStats.COLOR,
                per_background=True,
                channels={"g": 1},
            )
        assert set(result["stats"]) >= COLUMNS


class TestInvalidation:
    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    def test_resize_declares_it_in_every_mode(self, mode):
        assert Resize((8, 8), mode=mode).invalidates & NOISE == NOISE  # pyright: ignore[reportArgumentType]

    def test_a_gray_mix_and_detection_crops_declare_it(self):
        assert SelectChannels("gray").invalidates & NOISE == NOISE
        assert DetectionCrops.invalidates & NOISE == NOISE
        assert TorchvisionTransform(object()).invalidates & NOISE == NOISE

    @pytest.mark.parametrize("operation", [SelectChannels([0, 1]), SelectChannels("rgb"), Crop((0, 0, 8, 8))])
    def test_operations_that_keep_the_pixels_do_not(self, operation):
        assert not operation.invalidates & NOISE

"""The names of the v1.3 statistic families: flags, groups, columns, details and keywords."""

import warnings

import numpy as np
import pytest

from dataeval.core import image_stats
from dataeval.flags import ImageStats

pytestmark = pytest.mark.required


@pytest.fixture
def images() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (3, 96, 128), dtype=np.uint8) for _ in range(2)]


@pytest.fixture
def boxes() -> list[list[list[float]]]:
    return [[[8, 8, 72, 72], [40, 30, 110, 90]], [[16, 16, 100, 80]]]


def _quiet_stats(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return image_stats(*args, **kwargs)


class TestBlurNames:
    def test_flags_and_group(self):
        assert set(ImageStats.BLUR) == {ImageStats.VISUAL_BLUR_EDGE_WIDTH, ImageStats.VISUAL_BLUR_EDGE_DIRECTIONALITY}
        assert not ImageStats.BLUR & ImageStats.VISUAL
        assert not hasattr(ImageStats, "VISUAL_BLUR")
        assert not hasattr(ImageStats, "VISUAL_BLUR_DIRECTIONALITY")

    def test_columns(self, images, boxes):
        result = _quiet_stats(images, boxes=boxes, stats=ImageStats.BLUR, per_background=True)
        columns = set(result["stats"])
        assert {"blur_edge_width", "blur_edge_directionality", "background_blur_edge_width"} <= columns
        assert not {"blur", "blur_directionality", "background_blur"} & columns


class TestHueNames:
    def test_flags_and_group(self):
        assert ImageStats.VISUAL_COLOR_HUE_COS_MEAN in ImageStats.COLOR
        assert ImageStats.VISUAL_COLOR_HUE_SIN_MEAN in ImageStats.COLOR
        assert len(list(ImageStats.COLOR)) == 5
        assert all((m.name or "").startswith("VISUAL_COLOR_") for m in ImageStats.COLOR)
        assert not hasattr(ImageStats, "VISUAL_COLOR_HUE_COS")

    def test_columns(self, images):
        result = _quiet_stats(images, stats=ImageStats.COLOR, color_rgb_channels=(0, 1, 2))
        assert {"color_hue_cos_mean", "color_hue_sin_mean"} <= set(result["stats"])
        assert not {"color_hue_cos", "color_hue_sin"} & set(result["stats"])
        assert "color_hue_reason" in result.get("details", {})


class TestSaturatedFractionNames:
    DETAILS = {"saturated_status", "saturated_above_range_fraction", "saturated_upper"}

    def test_flag(self):
        assert ImageStats.PIXEL_SATURATED_FRACTION in ImageStats.PIXEL
        assert not hasattr(ImageStats, "PIXEL_SATURATION")

    def test_column_details_and_keyword(self, images):
        result = _quiet_stats(images, stats=ImageStats.PIXEL_SATURATED_FRACTION, saturated_atol=0.5)
        assert set(result["stats"]) == {"saturated_fraction"}
        assert set(result.get("details", {})) == self.DETAILS
        assert "saturated_fraction" in result.get("measurement_specs", {})

    def test_the_old_keyword_is_gone(self, images):
        with pytest.raises(TypeError, match="saturation_atol"):
            image_stats(images, stats=ImageStats.PIXEL_SATURATED_FRACTION, saturation_atol=0.5)  # type: ignore[call-arg]

    def test_results_combine_by_the_new_spec(self, images):
        from dataeval.core import combine_stats_results

        one = _quiet_stats(images, stats=ImageStats.PIXEL_SATURATED_FRACTION, saturated_atol=0.5)
        two = _quiet_stats(images, stats=ImageStats.PIXEL_SATURATED_FRACTION, saturated_atol=0.5)
        other = _quiet_stats(images, stats=ImageStats.PIXEL_SATURATED_FRACTION, saturated_atol=1.0)
        stats, _, _ = combine_stats_results([one, two])
        assert len(stats["saturated_fraction"]) == 4
        with pytest.raises(ValueError, match="saturated_fraction"):
            combine_stats_results([one, other])

    def test_quality_gate_refusal_names_the_column(self):
        from dataeval.data import QualityGate

        with pytest.raises(ValueError, match="Nor is .saturated_fraction., because data other than uint8 or float"):
            QualityGate("saturated_fraction", min=0.0)


class TestContextNames:
    def test_flags(self):
        assert set(ImageStats.CONTEXT) == {
            ImageStats.CONTEXT_SURROUND_INTENSITY_MEAN,
            ImageStats.CONTEXT_SURROUND_INTENSITY_STD,
            ImageStats.CONTEXT_TARGET_SURROUND_CONTRAST,
            ImageStats.CONTEXT_SIGNAL_CLUTTER_RATIO,
        }
        assert not hasattr(ImageStats, "CONTEXT_SURROUND_MEAN")

    def test_columns_details_and_keyword(self, images, boxes):
        result = _quiet_stats(images, boxes=boxes, stats=ImageStats.CONTEXT, context_ratio_floor=0.01)
        assert {"surround_intensity_mean", "surround_intensity_std"} <= set(result["stats"])
        assert {"surround_peer_excluded_fraction", "context_ratio_reason"} <= set(result.get("details", {}))
        assert not {"surround_mean", "surround_std"} & set(result["stats"])
        assert not {"ring_peer_excluded_fraction", "contrast_reason"} & set(result.get("details", {}))

    def test_the_old_keyword_is_gone(self, images, boxes):
        with pytest.raises(TypeError, match="surround_intensity_floor"):
            image_stats(images, boxes=boxes, stats=ImageStats.CONTEXT, surround_intensity_floor=0.01)  # type: ignore[call-arg]

    def test_the_surround_columns_go_view_first_on_a_channel_group(self, images, boxes):
        result = _quiet_stats(
            images, boxes=boxes, stats={None: ImageStats.CONTEXT, "nir": ImageStats.CONTEXT}, channels={"nir": [2]}
        )
        stats, details = set(result["stats"]), set(result.get("details", {}))
        assert {
            "surround_nir_intensity_mean",
            "surround_nir_intensity_std",
            "nir_target_surround_contrast",
            "nir_signal_clutter_ratio",
        } <= stats
        assert {
            "surround_nir_measurable_pixels",
            "surround_nir_valid_fraction",
            "nir_context_status",
            "nir_target_valid_fraction",
            "nir_context_range_low",
            "nir_context_range_high",
            "nir_context_ratio_reason",
            "nir_target_surround_contrast_floored",
            "nir_signal_clutter_ratio_floored",
        } <= details
        assert not {name for name in stats | details if name.startswith("nir_surround_")}
        assert {"surround_nir_intensity_mean", "surround_nir_intensity_std"} <= set(result.get("measurement_specs", {}))

    def test_background_and_band_names_are_unchanged(self, images, boxes):
        result = _quiet_stats(
            images,
            boxes=boxes,
            stats={None: ImageStats.PIXEL_MEAN, "nir": ImageStats.PIXEL_MEAN},
            channels={"nir": [2]},
            per_background=True,
        )
        assert {"mean", "nir_mean", "background_mean", "background_nir_mean"} <= set(result["stats"])


class TestVideoAndFrameNames:
    def test_video_blur(self):
        from dataeval.flags import VideoStats

        assert VideoStats.BLUR is VideoStats.BLUR_EDGE_WIDTH
        assert not hasattr(VideoStats, "BLUR_LEVEL")

    def test_video_blur_column(self):
        from dataeval.core._calculators._videoblurstats import VideoBlurStatCalculator
        from dataeval.flags import VideoStats

        names = {handler.name for handler in VideoBlurStatCalculator.get_handlers(VideoBlurStatCalculator).values()}
        assert names == {"blur_edge_width_median"}
        assert VideoStats.BLUR_EDGE_WIDTH in VideoStats.ALL

    def test_frame_flags(self):
        from dataeval.flags import FrameStats

        assert FrameStats.TIMING is FrameStats.TIMING_ELAPSED
        assert FrameStats.REPEAT is FrameStats.REPEAT_EXACT
        assert FrameStats.ALL == FrameStats.TIMING_ELAPSED | FrameStats.REPEAT_EXACT
        assert not hasattr(FrameStats, "EXACT_REPEAT")

    def test_frame_stats_message_names_the_flags(self):
        from dataeval.core import frame_stats
        from dataeval.flags import FrameStats

        with pytest.raises(ValueError, match="TIMING_ELAPSED") as raised:
            frame_stats([], stats=FrameStats.NONE)
        assert "REPEAT_EXACT" in str(raised.value)


class TestParseRule:
    """A column is its flag's name after the first underscore, lowercased: dataeval-flow derives columns this way."""

    def test_every_member_parses_to_its_column(self):
        from dataeval.core._calculators._registry import CalculatorRegistry

        for member in ImageStats:
            if member.value == 0 or member.value & (member.value - 1):
                continue
            produced = {
                name
                for calculator, flags in CalculatorRegistry.get_calculators(member)
                for name in calculator.stat_names(flags)
            }
            parsed = (member.name or "").split("_", 1)[1].lower()
            assert parsed in produced, (member.name, produced)

    def test_parsed_names_are_unique_and_not_reserved(self):
        from typing import get_args

        from dataeval.types._factors import FactorLevel

        singles = [m for m in ImageStats if m.value and not m.value & (m.value - 1)]
        parsed = [(m.name or "").split("_", 1)[1].lower() for m in singles]
        assert len(parsed) == len(set(parsed))
        assert not {"background", *get_args(FactorLevel)} & set(parsed)

"""The color statistics record what they were measured with, so results measured differently are not combined."""

import json
from typing import Any

import numpy as np
import pytest

from dataeval.core import combine_stats_results, image_stats
from dataeval.flags import ImageStats
from dataeval.types import MeasurementSpec

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

IMAGES = [np.random.default_rng(i).integers(0, 256, (3, 12, 12)).astype(np.uint8) for i in range(2)]
UNITS = {
    "color_lightness_mean": "L*",
    "color_chroma_mean": "C*",
    "color_achromatic_fraction": "fraction",
    "color_hue_cos": "unit vector component",
    "color_hue_sin": "unit vector component",
}


def run(**kw: Any) -> Any:
    kw = {"color_rgb_channels": (0, 1, 2)} | kw
    return image_stats(IMAGES, stats=ImageStats.COLOR, **kw)


def specs_of(**kw: Any) -> dict[str, MeasurementSpec]:
    return dict(run(**kw).get("measurement_specs", {}))


def test_every_statistic_records_the_algorithm_the_channels_and_the_settings():
    specs = specs_of(value_range=(0, 255))
    assert set(specs) == set(UNITS)
    for name, spec in specs.items():
        assert spec.algorithm == "cielab_srgb_d65"
        assert spec.version == "1"
        assert spec.units == UNITS[name]
        assert spec.region == "pixels"
        assert spec.rgb_channels == (0, 1, 2)
        assert dict(spec.config) == {
            "encoding": "srgb",
            "min_pixels": 32,
            "min_valid_fraction": 0.9,
            "chroma_threshold": 12.0,
        }
        assert (spec.range_policy, spec.declared_range) == ("declared", (0.0, 255.0))


def test_an_undeclared_range_is_recorded_as_inferred():
    assert {spec.range_policy for spec in specs_of().values()} == {"inferred"}


def test_the_thresholds_and_the_channel_order_are_recorded():
    spec = specs_of(color_rgb_channels=[2, 1, 0], color_min_pixels=10, color_chroma_threshold=5.0)["color_hue_cos"]
    assert spec.rgb_channels == (2, 1, 0)
    assert (dict(spec.config)["min_pixels"], dict(spec.config)["chroma_threshold"]) == (10, 5.0)


def test_the_background_column_has_its_own_spec_and_region():
    specs = specs_of(boxes=[[(0, 0, 6, 6)], [(0, 0, 6, 6)]], per_background=True)
    assert specs["color_chroma_mean"].region == "pixels"
    assert specs["background_color_chroma_mean"].region == "background"


def test_a_call_that_asks_for_no_color_records_no_color_spec():
    result = image_stats(IMAGES, stats=ImageStats.PIXEL_MEAN)
    assert not [name for name in result.get("measurement_specs", {}) if "color" in name]


def test_a_spec_serializes_and_rebuilds_equal():
    spec = specs_of(value_range=(0, 255))["color_chroma_mean"]
    assert MeasurementSpec(**json.loads(json.dumps(spec.to_dict()))) == spec


class TestCombining:
    def test_results_measured_with_the_same_settings_combine(self):
        stats, _, _ = combine_stats_results([run(), run()])
        assert len(stats["color_chroma_mean"]) == 4

    def test_results_with_different_channels_are_refused(self):
        with pytest.raises(ValueError, match="rgb_channels"):
            combine_stats_results([run(), run(color_rgb_channels=(2, 1, 0))])

    def test_results_with_different_thresholds_are_refused(self):
        with pytest.raises(ValueError, match="chroma_threshold"):
            combine_stats_results([run(), run(color_chroma_threshold=5.0)])

    def test_results_with_different_floors_are_refused(self):
        with pytest.raises(ValueError, match="min_pixels"):
            combine_stats_results([run(), run(color_min_pixels=10)])

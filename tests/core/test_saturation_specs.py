"""SATURATION records the settings it was measured with, so results measured differently are not combined."""

import json
import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import combine_stats_results, image_stats
from dataeval.flags import ImageStats
from dataeval.types import MeasurementSpec
from dataeval.utils.preprocessing import ChannelGroup

pytestmark = pytest.mark.required


def specs_of(images: list[np.ndarray], **kw: Any) -> dict[str, MeasurementSpec]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = image_stats(images, stats=kw.pop("stats", ImageStats.PIXEL_SATURATION), **kw)
    return dict(result.get("measurement_specs", {}))


def run(images: Any, **kw: Any) -> Any:
    return image_stats(images, stats=ImageStats.PIXEL_SATURATION, **kw)


U8 = [np.full((3, 6, 6), 255, dtype=np.uint8)]
U16 = [np.full((3, 6, 6), 4095, dtype=np.uint16)]


class TestRecorded:
    def test_a_call_without_saturation_records_no_spec(self):
        assert "measurement_specs" not in image_stats(U8, stats=ImageStats.PIXEL_MEAN)

    def test_a_default_call_records_it_because_the_pixel_family_holds_saturation(self):
        assert set(image_stats(U8).get("measurement_specs", {})) == {"saturation"}

    def test_the_defaults_are_recorded_with_the_range_inferred_from_each_image(self):
        assert specs_of(U8) == {
            "saturation": MeasurementSpec(
                algorithm="saturation",
                version="1",
                units="fraction",
                region="pixels",
                config={"atol": 0.0},
                range_policy="inferred",
            )
        }

    def test_a_declared_value_range_is_recorded(self):
        spec = specs_of(U16, value_range=(0, 4095))["saturation"]
        assert (spec.range_policy, spec.declared_range) == ("declared", (0.0, 4095.0))

    def test_the_tolerance_is_recorded(self):
        assert dict(specs_of(U8, saturation_atol=0.5)["saturation"].config) == {"atol": 0.5}

    def test_the_background_column_has_its_own_spec_and_region(self):
        specs = specs_of(U8, boxes=[[(0, 0, 3, 3)]], per_background=True)
        assert set(specs) == {"saturation", "background_saturation"}
        assert (specs["saturation"].region, specs["background_saturation"].region) == ("pixels", "background")

    def test_a_call_that_measures_only_the_background_records_only_the_background_spec(self):
        specs = specs_of(U8, boxes=[[(0, 0, 3, 3)]], per_background=True, per_image=False, per_target=False)
        assert set(specs) == {"background_saturation"}

    def test_a_band_group_records_its_own_declared_range_under_its_own_column(self):
        image = [np.zeros((4, 6, 6), dtype=np.uint16)]
        specs = specs_of(
            image,
            boxes=[[(0, 0, 3, 3)]],
            per_background=True,
            value_range=(0, 65535),
            channels={"nir": ChannelGroup(3, value_range=(0, 4095))},
        )
        assert set(specs) == {"saturation", "background_saturation", "nir_saturation", "background_nir_saturation"}
        assert specs["saturation"].declared_range == (0.0, 65535.0)
        assert specs["nir_saturation"].declared_range == (0.0, 4095.0)
        assert specs["background_nir_saturation"].declared_range == (0.0, 4095.0)
        assert specs["background_nir_saturation"].region == "background"

    def test_only_the_columns_the_result_holds_have_a_spec(self):
        specs = specs_of(
            U8, stats={None: ImageStats.PIXEL_MEAN, "nir": ImageStats.PIXEL_SATURATION}, channels={"nir": 1}
        )
        assert set(specs) == {"nir_saturation"}

    def test_a_spec_serializes_and_rebuilds_equal(self):
        spec = specs_of(U16, value_range=(0, 4095), saturation_atol=1.5)["saturation"]
        assert MeasurementSpec(**json.loads(json.dumps(spec.to_dict()))) == spec

    def test_a_request_for_all_records_it_beside_the_statistics_that_record_none(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = image_stats(U8, stats=ImageStats.ALL)
        assert set(result.get("measurement_specs", {})) == {"saturation"}


class TestCombining:
    def test_results_measured_with_the_same_settings_combine(self):
        stats, _, _ = combine_stats_results([run(U8), run(U8)])
        assert stats["saturation"].tolist() == [1.0, 1.0]

    def test_results_with_different_tolerances_are_refused(self):
        with pytest.raises(ValueError, match="atol"):
            combine_stats_results([run(U8, saturation_atol=1), run(U8)])

    def test_results_with_different_declared_ranges_are_refused(self):
        with pytest.raises(ValueError, match="declared_range"):
            combine_stats_results([run(U16, value_range=(0, 4095)), run(U16, value_range=(0, 65535))])

    def test_a_declared_range_in_one_result_and_none_in_the_other_is_refused(self):
        with pytest.raises(ValueError, match="range_policy"):
            combine_stats_results([run(U16, value_range=(0, 4095)), run(U16)])

    def test_two_dtypes_under_the_dtype_rule_still_combine(self):
        # The stated limit: both specs read as undeclared, and the endpoint each row used is in the details, which
        # are not compared. Pinned so that closing the gap is a decision and not an accident.
        stats, _, _ = combine_stats_results([run(U8), run(U16)])
        assert stats["saturation"].tolist() == [1.0, 0.0]


class Declared:
    """A dataset whose one image declares its range in its own metadata."""

    metadata = {"id": "declared"}

    def __init__(self, **declaration: Any) -> None:
        self.declaration = declaration

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return U16[0], np.array([1.0]), {"id": i, **self.declaration}


class TestPerImageDeclarations:
    def test_a_range_an_image_declares_is_not_in_the_spec_so_the_results_still_combine(self):
        # The stated limit: the endpoint each row used is in the details, which are not compared.
        declared, undeclared = run(Declared(value_range=(0, 4095))), run(Declared())
        assert declared["details"]["saturation_upper"].tolist() == [4095.0]
        assert undeclared["details"]["saturation_upper"].tolist() == [65535.0]
        combine_stats_results([declared, undeclared])

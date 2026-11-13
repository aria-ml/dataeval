"""The surround's two settings are plain keywords, checked before any image is read, and recorded with the result."""

import inspect
import json
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from dataeval.core import combine_stats_results, image_stats
from dataeval.flags import ImageStats
from dataeval.types import MeasurementSpec
from dataeval.types._stats_config import SurroundConfig
from dataeval.utils.preprocessing import ChannelGroup

pytestmark = [
    pytest.mark.required,
    pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning"),
]

CONTEXT = ImageStats.CONTEXT
BOX = (10, 10, 40, 40)
F2_BOXES = [(3, 3, 6, 6), (5, 5, 9, 9)]


def details_of(result: Any) -> Any:
    """The result's details, which a result carries only when a requested statistic reports some."""
    assert "details" in result
    return result["details"]


def picture() -> np.ndarray:
    return np.random.default_rng(1605).random((3, 60, 80))


def run(*, stats: ImageStats = CONTEXT, image: Any = None, boxes: Any = None, **kw: Any) -> Any:
    image = picture() if image is None else image
    return image_stats([image], boxes=[boxes or [BOX]], stats=stats, per_image=False, **kw)


def specs_of(**kw: Any) -> dict[str, MeasurementSpec]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return dict(run(**kw).get("measurement_specs", {}))


BAD = {
    "surround_padding": [True, np.True_, 0, -1, float("nan"), float("inf"), "1", None],
    "surround_min_pixels": [True, np.True_, 0, -1, 0.5, 32.0, float("nan"), "1", None],
}


class TestKeywords:
    @pytest.mark.parametrize(
        ("name", "value"), [(name, value) for name, values in BAD.items() for value in values], ids=repr
    )
    @pytest.mark.parametrize("stats", [CONTEXT, ImageStats.PIXEL_MEAN])
    def test_a_bad_value_raises_naming_the_keyword_whether_or_not_a_context_flag_is_requested(self, name, value, stats):
        with pytest.raises(ValueError, match=name):
            run(stats=stats, **{name: value})

    def test_a_whole_number_padding_and_numpy_scalars_are_accepted(self):
        assert run(surround_padding=1)["details"]["context_status"].tolist() == ["ok"]
        numpy = run(surround_padding=np.float32(0.5), surround_min_pixels=np.int64(5))
        assert details_of(numpy)["context_status"].tolist() == ["ok"]

    def test_the_defaults_are_the_sections_defaults(self):
        parameters = inspect.signature(image_stats).parameters
        defaults = SurroundConfig()
        assert parameters["surround_padding"].default == defaults.padding == 0.25
        assert parameters["surround_min_pixels"].default == defaults.min_pixels == 32

    def test_stating_the_defaults_changes_nothing(self):
        plain, stated = run(), run(surround_padding=0.25, surround_min_pixels=32)
        for group in ("stats", "details"):
            assert plain[group].keys() == stated[group].keys()
            for name in plain[group]:
                np.testing.assert_array_equal(plain[group][name], stated[group][name], err_msg=name)

    def test_the_settings_change_the_values(self):
        # F2 zero-padded into a 40 x 40 picture: 13 measurable pixels, under the default of 32.
        image = np.zeros((1, 40, 40))
        image[0, :10, :10] = np.arange(100).reshape(10, 10) / 99
        by_default = run(image=image, boxes=F2_BOXES)["details"]
        assert by_default["context_status"][0] == "insufficient_support"
        assert by_default["surround_candidate_pixels"][0] == 16
        relaxed = run(image=image, boxes=F2_BOXES, surround_min_pixels=13)
        assert details_of(relaxed)["context_status"][0] == "ok"
        ring = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 2), (3, 6), (4, 2), (4, 6), (5, 2), (6, 2), (6, 3), (6, 4)]
        expected = np.mean([(10 * y + x) / 99 for y, x in ring])
        assert relaxed["stats"]["surround_mean"][0] == pytest.approx(expected, rel=1e-6)
        # Padding 0.5: the window is (1, 1, 8, 8), 49 pixels, less the 9 of the box.
        wider = run(image=image, boxes=F2_BOXES, surround_padding=0.5, surround_min_pixels=13)
        assert details_of(wider)["surround_candidate_pixels"][0] == 40


class TestTheWarning:
    @pytest.mark.parametrize(("name", "value"), [("surround_padding", 0.5), ("surround_min_pixels", 10)])
    def test_a_setting_that_differs_warns_once_when_no_context_flag_is_requested(self, name, value):
        with pytest.warns(UserWarning, match=rf"{name} is ignored.*ImageStats\.CONTEXT") as caught:
            run(stats=ImageStats.PIXEL_MEAN, **{name: value})
        assert len([w for w in caught if name in str(w.message)]) == 1

    def test_both_settings_are_named_together(self):
        with pytest.warns(UserWarning, match=r"surround_padding and surround_min_pixels are ignored"):
            run(stats=ImageStats.PIXEL_MEAN, surround_padding=0.5, surround_min_pixels=10)

    @pytest.mark.parametrize(
        "kw",
        [
            {"stats": CONTEXT, "surround_padding": 0.5},
            {"stats": ImageStats.CONTEXT_SURROUND_STD, "surround_min_pixels": 10},
            {"stats": ImageStats.PIXEL_MEAN, "surround_padding": 0.25, "surround_min_pixels": 32},
        ],
        ids=["requested", "one_of_the_two", "stated_default"],
    )
    def test_no_warning_when_a_flag_is_requested_or_the_value_is_the_default(self, kw):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run(**kw)


class TestTheSpec:
    SPEC = MeasurementSpec(
        algorithm="surround_intensity",
        version="1",
        units="normalized intensity",
        region="surround",
        normalization="[0, 1]",
        config={"padding": 0.25, "min_pixels": 32},
        range_policy="inferred",
    )

    def test_a_default_call_records_both_columns_with_the_defaults_and_the_inferred_range(self):
        assert specs_of() == {"surround_mean": self.SPEC, "surround_std": self.SPEC}

    def test_only_the_selected_statistic_has_a_spec(self):
        assert set(specs_of(stats=ImageStats.CONTEXT_SURROUND_STD)) == {"surround_std"}

    def test_the_settings_are_recorded(self):
        spec = specs_of(surround_padding=0.5, surround_min_pixels=10)["surround_mean"]
        assert dict(spec.config) == {"padding": 0.5, "min_pixels": 10}

    def test_a_declared_range_is_recorded(self):
        spec = specs_of(image=picture().astype(np.float32), value_range=(0, 4095))["surround_std"]
        assert (spec.range_policy, spec.declared_range) == ("declared", (0.0, 4095.0))

    def test_a_band_group_records_its_own_declared_range_under_its_own_columns(self):
        cube = np.full((4, 60, 80), 1000, dtype=np.uint16)
        specs = specs_of(image=cube, value_range=(0, 65535), channels={"nir": ChannelGroup(3, value_range=(0, 4095))})
        assert set(specs) == {"surround_mean", "surround_std", "nir_surround_mean", "nir_surround_std"}
        assert specs["surround_mean"].declared_range == (0.0, 65535.0)
        assert specs["nir_surround_std"].declared_range == (0.0, 4095.0)

    def test_per_background_adds_no_spec_for_these_columns(self):
        assert set(specs_of(stats=ImageStats.PIXEL_MEAN | CONTEXT, per_background=True)) == {
            "surround_mean",
            "surround_std",
        }

    def test_a_call_with_no_context_flag_records_none(self):
        assert specs_of(stats=ImageStats.PIXEL_MEAN) == {}

    def test_a_spec_serializes_and_rebuilds_equal(self):
        spec = specs_of(surround_padding=0.5)["surround_mean"]
        assert MeasurementSpec(**json.loads(json.dumps(spec.to_dict()))) == spec


class Declared:
    """A dataset whose one image may declare its range in its own metadata."""

    metadata = {"id": "declared"}

    def __init__(self, declaration: Mapping[str, Any]) -> None:
        self.declaration = declaration

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return np.full((3, 60, 80), 1000, dtype=np.uint16), np.array([1.0]), {"id": i, **self.declaration}


class TestCombining:
    def test_results_measured_the_same_way_combine(self):
        stats, _, _ = combine_stats_results([run(), run()])
        assert len(stats["surround_mean"]) == 2

    @pytest.mark.parametrize("name", ["surround_padding", "surround_min_pixels"])
    def test_results_measured_with_different_settings_are_refused(self, name):
        other = {"surround_padding": 0.5, "surround_min_pixels": 10}[name]
        with pytest.raises(ValueError, match=name.removeprefix("surround_")):
            combine_stats_results([run(**{name: other}), run()])

    def test_results_that_differ_only_in_a_range_an_image_declares_still_combine(self):
        # The stated limit, as for saturation: the interval each row used is in the details, which are not compared.
        declared = image_stats(Declared({"value_range": (0, 65535)}), boxes=[[BOX]], stats=CONTEXT, per_image=False)
        undeclared = image_stats(Declared({}), boxes=[[BOX]], stats=CONTEXT, per_image=False)
        assert details_of(declared)["context_range_high"].tolist() == [65535.0]
        assert details_of(undeclared)["context_range_high"].tolist() == [4095.0]
        combine_stats_results([declared, undeclared])

"""Details and measurement descriptors: what a statistic reports beside its values."""

import json
from typing import Any, get_type_hints

import numpy as np
import pytest

from dataeval.core import FactorResult, image_stats
from dataeval.core._calculators._base import ALL_VIEWS, Calculator, Handler
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._image_stats import _sort
from dataeval.flags import ImageStats
from dataeval.types import MeasurementSpec, SourceIndex
from tests.metadata.test_add_factors_collisions import _images

pytestmark = pytest.mark.required

# Not representable in float64, let alone float32, so any cast through a float changes it.
ID_BASE = 2**53 + 1


class _ProbeCalculator(Calculator[ImageStats]):
    """Reports each row's box offset as a statistic and as an identifier detail."""

    def __init__(self, datum: Any, cache: Any) -> None:
        self.cache = cache

    def get_applicable_flags(self) -> ImageStats:
        return ImageStats.DIMENSION_OFFSET_X

    def get_handlers(self) -> dict[ImageStats, Handler]:
        return {ImageStats.DIMENSION_OFFSET_X: Handler("probe_x0", lambda: [float(self.cache.box.x0)], ALL_VIEWS)}

    def details(self, flags: ImageStats, view: Any = None) -> dict[str, list[Any]]:
        return {"probe_target": [ID_BASE + int(self.cache.box.x0)]}

    def get_empty_values(self) -> dict[str, Any]:
        return {"probe_target": -1}


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the probe the only image calculator for the test."""
    monkeypatch.setitem(CalculatorRegistry._registry, ImageStats, [_ProbeCalculator])


IMAGES = [np.zeros((3, 16, 16)), np.zeros((3, 16, 16))]
BOXES = [[[1, 0, 4, 4], [3, 0, 9, 9]], [[5, 2, 8, 8]]]


def _spec(**overrides: Any) -> MeasurementSpec:
    fields: dict[str, Any] = {"algorithm": "probe", "version": "1", "units": "pixels", "region": "box"} | overrides
    return MeasurementSpec(**fields)


class TestMeasurementSpec:
    def test_survives_a_json_round_trip(self):
        spec = _spec(
            normalization="[0, 1]",
            config={"padding": 0.25, "sizes": (1, 2)},
            range_policy="declared",
            declared_range=(0, 4095),
            rgb_channels=[2, 1, 0],
        )
        assert MeasurementSpec(**json.loads(json.dumps(spec.to_dict()))) == spec

    def test_is_hashable(self):
        assert len({_spec(config={"a": [1]}), _spec(config={"a": [1]})}) == 1

    def test_config_is_read_only(self):
        with pytest.raises(TypeError):
            _spec(config={"a": 1}).config["a"] = 2  # type: ignore[index]

    @pytest.mark.parametrize("value", [len, np.zeros(2), float("nan")])
    def test_refuses_config_json_cannot_hold(self, value):
        with pytest.raises(TypeError, match="JSON values"):
            _spec(config={"value": value})

    @pytest.mark.parametrize(
        ("policy", "declared"),
        [("declared", None), ("inferred", (0.0, 1.0)), (None, (0.0, 1.0)), ("guessed", None)],
    )
    def test_declared_range_goes_with_the_declared_policy(self, policy, declared):
        with pytest.raises(ValueError, match="range_policy"):
            _spec(range_policy=policy, declared_range=declared)

    def test_rgb_channels_names_three_bands(self):
        with pytest.raises(ValueError, match="red, green and blue"):
            _spec(rgb_channels=(0, 1))


class TestFactorResultFields:
    def test_both_fields_are_optional(self):
        hints = get_type_hints(FactorResult, include_extras=True)
        assert {"measurement_specs", "details"} <= hints.keys()
        assert FactorResult.__required_keys__ == frozenset({"stats"})


class TestSort:
    def test_details_take_the_stats_permutation_without_the_float_cast(self):
        source_index = [SourceIndex(1, 0), SourceIndex(0, None), SourceIndex(1, None), SourceIndex(0, 0)]
        ids = [ID_BASE + 3, ID_BASE + 0, ID_BASE + 2, ID_BASE + 1]
        stats = {"value": [3.0, 0.0, 2.0, 1.0]}
        details = {"target": ids, "status": ["c", "a", "b", "b"], "floored": [True, False, False, True]}

        ordered, sorted_stats, sorted_details = _sort(source_index, stats, details)

        assert ordered == [SourceIndex(0, None), SourceIndex(0, 0), SourceIndex(1, None), SourceIndex(1, 0)]
        assert sorted_stats["value"].dtype == np.float32
        assert sorted_details["target"].dtype == np.int64
        assert sorted_details["target"].tolist() == [ID_BASE + i for i in range(4)]
        assert sorted_details["status"].tolist() == ["a", "b", "b", "c"]
        assert sorted_details["floored"].tolist() == [False, True, False, True]

    def test_small_integers_are_widened_to_int64(self):
        _, _, details = _sort([SourceIndex(0, None)], {}, {"count": [3]})
        assert details["count"].dtype == np.int64


class TestImageStatsDetails:
    @pytest.mark.usefixtures("probe")
    def test_details_stay_aligned_with_source_index(self):
        result = image_stats(IMAGES, boxes=BOXES, stats=ImageStats.DIMENSION_OFFSET_X)

        target = result.get("details", {})["probe_target"]
        assert target.dtype == np.int64
        assert (target - ID_BASE).tolist() == result["stats"]["probe_x0"].astype(int).tolist()
        assert (target - ID_BASE).tolist() == [0, 1, 3, 0, 5]
        assert "probe_target" not in result["stats"]

    @pytest.mark.usefixtures("probe")
    def test_background_details_are_padded_on_box_rows(self):
        result = image_stats(IMAGES, boxes=BOXES, stats=ImageStats.DIMENSION_OFFSET_X, per_background=True)

        on_item_row = [s.key is None for s in result["source_index"]]
        background = result.get("details", {})["background_probe_target"]
        assert background.dtype == np.int64
        assert background.tolist() == [ID_BASE if item else -1 for item in on_item_row]

    @pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")  # ALL holds the blur flags
    def test_legacy_calls_produce_neither_field(self):
        # `ALL` also holds the SPATIAL and SATURATION flags, which report details; a legacy call is `ALL` without them
        legacy = ImageStats.ALL & ~ImageStats.SPATIAL & ~ImageStats.SATURATION
        result = image_stats(IMAGES, boxes=BOXES, stats=legacy, per_background=True)
        assert "details" not in result
        assert "measurement_specs" not in result


class TestAddFactors:
    def test_ignores_details_and_specs(self):
        md = _images(w=[1.0, 2.0, 3.0, 4.0])
        result: Any = {
            "stats": {"probe": np.arange(4.0)},
            "source_index": [SourceIndex(i, None) for i in range(4)],
            "details": {"probe_status": np.array(["ok", "ok", "invalid_box", "ok"])},
            "measurement_specs": {"probe": _spec()},
        }

        md.add_factors(result)

        assert "probe" in md.factor_names
        assert not any(name.startswith("probe_status") for name in md.factor_names)

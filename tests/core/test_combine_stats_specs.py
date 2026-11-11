"""`combine_stats_results` refuses to concatenate statistics that were measured differently."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from dataeval.core import AddressedResult, combine_stats_results, image_stats
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers
from dataeval.types import MeasurementSpec, SourceIndex


def _spec(**changes: Any) -> MeasurementSpec:
    values: dict[str, Any] = {
        "algorithm": "edge_width",
        "version": "1",
        "units": "pixels",
        "region": "box",
        "config": {"pad": 0.25},
    }
    return MeasurementSpec(**{**values, **changes})


def _result(values: list[float], specs: dict[str, MeasurementSpec] | None = None) -> AddressedResult[Any]:
    result: AddressedResult[Any] = {
        "stats": {"blur": np.array(values), "other": np.array(values)},
        "source_index": [SourceIndex(i) for i in range(len(values))],
    }
    if specs is not None:
        result["measurement_specs"] = specs
    return result


@pytest.mark.required
class TestCombineStatsSpecs:
    def test_agreeing_specs_combine_and_the_return_is_unchanged(self):
        one = _result([1.0, 2.0], {"blur": _spec()})
        one["details"] = {"status": np.array([0, 0])}
        two = _result([3.0], {"blur": _spec()})
        two["details"] = {"status": np.array([1])}

        combined = combine_stats_results([one, two])

        stats, source_index, steps = combined
        assert len(combined) == 3
        assert stats["blur"].tolist() == [1.0, 2.0, 3.0]
        assert [s.item for s in source_index] == [0, 1, 2]
        assert steps == [2, 3]
        # Details stay with the results that made them.
        assert set(stats) == {"blur", "other"}

    def test_results_with_no_specs_combine_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stats, _, _ = combine_stats_results([_result([1.0]), _result([2.0])])
        assert stats["blur"].tolist() == [1.0, 2.0]

    @pytest.mark.parametrize(
        ("changes", "named"),
        [
            ({"algorithm": "gradient"}, "algorithm: 'edge_width' vs 'gradient'"),
            ({"version": "2"}, "version: '1' vs '2'"),
            ({"units": "fraction"}, "units: 'pixels' vs 'fraction'"),
            ({"region": "surround"}, "region: 'box' vs 'surround'"),
            ({"normalization": "[0, 1]"}, r"normalization: None vs '\[0, 1\]'"),
            ({"config": {"pad": 0.5}}, r"config\['pad'\]: 0.25 vs 0.5"),
            ({"config": {"pad": 0.25, "sigma": 2}}, r"config\['sigma'\]: unset vs 2"),
            ({"config": {}}, r"config\['pad'\]: 0.25 vs unset"),
            ({"config": {"pad": 0.25, "mode": "unset"}}, r"config\['mode'\]: unset vs 'unset'"),
            ({"range_policy": "inferred"}, "range_policy: None vs 'inferred'"),
            (
                {"range_policy": "declared", "declared_range": (0, 255)},
                r"declared_range: None vs \(0\.0, 255\.0\)",
            ),
            ({"rgb_channels": (2, 1, 0)}, r"rgb_channels: None vs \(2, 1, 0\)"),
        ],
    )
    def test_a_differing_setting_raises_naming_factor_and_setting(self, changes: dict[str, Any], named: str):
        with pytest.raises(ValueError, match=f"'blur'.*{named}"):
            combine_stats_results([_result([1.0], {"blur": _spec()}), _result([2.0], {"blur": _spec(**changes)})])

    def test_only_the_conflicting_factor_is_named_and_the_result_is_located(self):
        agreed = {"blur": _spec(), "other": _spec(algorithm="gradient")}
        conflict = {"blur": _spec(), "other": _spec(algorithm="gradient", version="2")}
        results = [_result([1.0], agreed), _result([2.0], agreed), _result([3.0], conflict)]

        with pytest.raises(ValueError, match="'other'.*result 2") as raised:
            combine_stats_results(results)
        assert "'blur'" not in str(raised.value)

    @pytest.mark.parametrize(
        ("spec_first", "named"),
        [
            (True, "has a measurement spec in the results so far but result 1 records none"),
            (False, "has no measurement spec in the results so far but result 1 records one"),
        ],
    )
    def test_a_spec_in_some_results_and_not_others_raises(self, spec_first: bool, named: str):
        with_spec = _result([1.0], {"blur": _spec()})
        without = _result([2.0])
        results = [with_spec, without] if spec_first else [without, with_spec]

        with pytest.raises(ValueError, match=f"'blur' {named}"):
            combine_stats_results(results)

    def test_a_result_with_no_rows_causes_no_conflict(self):
        empty: AddressedResult[Any] = {"stats": {}, "source_index": [], "measurement_specs": {"blur": _spec(units="x")}}
        results = [empty, _result([1.0], {"blur": _spec()}), empty, _result([2.0], {"blur": _spec()})]

        stats, _, steps = combine_stats_results(results)

        assert stats["blur"].tolist() == [1.0, 2.0]
        assert steps == [0, 1, 1, 2]

    def test_a_single_result_is_returned_as_before(self):
        one = _result([1.0], {"blur": _spec()})
        stats, _, steps = combine_stats_results([one])
        assert stats is one["stats"]
        assert steps == []


def _with_specs(result: Any, factor: str, spec: MeasurementSpec) -> Any:
    result["measurement_specs"] = {factor: spec}
    return result


@pytest.mark.required
class TestEvaluatorsCombineWithSpecs:
    """The evaluators reach `combine_stats_results` for every multi-dataset call."""

    def test_outliers_from_stats_raises_on_conflicting_specs(self):
        images = np.random.default_rng(0).random((10, 3, 8, 8))
        one = _with_specs(image_stats(images, stats=ImageStats.PIXEL), "mean", _spec())
        two = _with_specs(image_stats(images, stats=ImageStats.PIXEL), "mean", _spec(config={"pad": 0.5}))

        with pytest.raises(ValueError, match=r"'mean'.*config\['pad'\]"):
            Outliers().from_stats([one, two])

    def test_outliers_from_stats_runs_on_agreeing_specs(self):
        images = np.random.default_rng(0).random((10, 3, 8, 8))
        one = _with_specs(image_stats(images, stats=ImageStats.PIXEL), "mean", _spec())
        two = _with_specs(image_stats(images, stats=ImageStats.PIXEL), "mean", _spec())

        assert "dataset_index" in Outliers().from_stats([one, two]).data().columns

    def test_duplicates_from_stats_raises_on_conflicting_specs(self):
        data = np.concatenate((np.ones((1, 16, 16)), np.zeros((1, 16, 16))))
        one = _with_specs(image_stats(data, stats=ImageStats.HASH, per_image=True, per_target=False), "phash", _spec())
        two = image_stats(data, stats=ImageStats.HASH, per_image=True, per_target=False)

        with pytest.raises(ValueError, match="'phash'"):
            Duplicates().from_stats([one, two])

    def test_duplicates_from_stats_runs_on_agreeing_specs(self):
        data = np.concatenate((np.ones((1, 16, 16)), np.zeros((1, 16, 16))))
        one = _with_specs(image_stats(data, stats=ImageStats.HASH, per_image=True, per_target=False), "phash", _spec())
        two = _with_specs(image_stats(data, stats=ImageStats.HASH, per_image=True, per_target=False), "phash", _spec())

        assert "dataset_indices" in Duplicates().from_stats([one, two]).data().columns

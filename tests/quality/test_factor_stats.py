import datetime as dt
import enum

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.core import track_stats
from dataeval.quality._factor_stats import eligible_factors, factor_kind, factor_levels, factor_stats
from dataeval.types import SourceIndex
from tests.metadata.test_structurers import _mot_dataset


def _kinds(md):
    return {name: factor_kind(md, name) for name in factor_levels(md)}


class _Camera(enum.Enum):
    """A plain Python enum: polars infers an Enum column from a list of its members."""

    A = "a"
    B = "b"


@pytest.fixture
def mixed():
    """One unit-level factor of each dtype the rule has to place."""
    n = 30
    return Metadata.from_factors(
        {
            "temp": np.linspace(10, 30, n),
            "count": np.arange(n) % 4,
            "when": [dt.datetime(2024, 1, 1) + dt.timedelta(hours=i) for i in range(n)],
            "when_utc": [dt.datetime(2024, 1, 1, tzinfo=dt.UTC) + dt.timedelta(hours=i) for i in range(n)],
            "elapsed": [dt.timedelta(seconds=i) for i in range(n)],
            "camera": ["a", "b"] * (n // 2),
            "ok": [True, False] * (n // 2),
            "serial": np.array([101, 202] * (n // 2)),
        },
        factor_levels={"serial": [101, 202]},
    )


@pytest.fixture
def tracked():
    ds = _mot_dataset([[[5, 9], [5], [5, 9]], [[7], [3, 7], [3]]])
    md = Metadata(ds)
    md.add_factors(track_stats(ds), level="track", key="track_id")
    return md


@pytest.mark.required
class TestFactorKind:
    def test_numbers_and_times_are_ordered(self, mixed):
        kinds = _kinds(mixed)
        assert {kinds[name] for name in ("temp", "count", "when", "when_utc", "elapsed")} == {"ordered"}

    def test_strings_and_booleans_are_categorical(self, mixed):
        kinds = _kinds(mixed)
        assert (kinds["camera"], kinds["ok"]) == ("categorical", "categorical")

    def test_a_declared_integer_vocabulary_is_categorical(self, mixed):
        assert _kinds(mixed)["serial"] == "categorical"

    @pytest.mark.skipif(
        not isinstance(pl.Series([_Camera.A]).dtype, pl.Enum),
        reason="this polars infers Python enum members as object, not pl.Enum",
    )
    def test_an_enum_is_categorical(self):
        md = Metadata.from_factors({"model": [_Camera.A, _Camera.B] * 15})
        assert _kinds(md)["model"] == "categorical"

    def test_the_binning_classifier_is_not_consulted(self, tracked):
        """n_gaps is classified discrete for binning; it is still a quantity."""
        assert tracked.factor_info["n_gaps"].factor_type == "discrete"
        assert _kinds(tracked)["n_gaps"] == "ordered"

    def test_track_labels_are_categorical(self, tracked):
        assert _kinds(tracked)["labels"] == "categorical"

    def test_reserved_columns_are_not_factors(self, tracked):
        assert not {"track_id", "item_index", "unit_index", "target_index"} & set(factor_levels(tracked))

    def test_each_factor_is_read_at_its_own_level(self, tracked):
        levels = factor_levels(tracked)
        assert (levels["n_gaps"], levels["time_s"]) == ("track", "unit")


@pytest.mark.required
class TestEligibleFactors:
    def test_by_default_every_ordered_factor(self, mixed):
        assert set(eligible_factors(mixed, None)) == {"temp", "count", "when", "when_utc", "elapsed"}

    def test_named_factors_only(self, mixed):
        assert set(eligible_factors(mixed, ["temp"])) == {"temp"}

    def test_an_unknown_name_is_refused_with_the_names_there_are(self, mixed):
        with pytest.raises(KeyError, match="temp"):
            eligible_factors(mixed, ["tmep"])

    def test_a_categorical_name_is_refused_with_the_reason(self, mixed):
        with pytest.raises(ValueError, match="categorical"):
            eligible_factors(mixed, ["camera"])

    def test_a_bare_string_is_refused(self, mixed):
        with pytest.raises(TypeError, match=r"\['temp'\]"):
            eligible_factors(mixed, "temp")

    def test_an_empty_list_is_refused(self, mixed):
        with pytest.raises(ValueError, match="None"):
            eligible_factors(mixed, [])

    def test_nothing_ordered_is_refused_rather_than_reported_clean(self):
        md = Metadata.from_factors({"camera": ["a", "b"] * 15})
        with pytest.raises(ValueError, match="camera"):
            eligible_factors(md, None)


@pytest.mark.required
class TestFactorStats:
    def test_raw_values_not_codes(self, mixed):
        stats = factor_stats(mixed, eligible_factors(mixed, ["temp"]))
        np.testing.assert_allclose(stats["stats"]["temp"], np.linspace(10, 30, 30))

    def test_times_are_their_epoch(self, mixed):
        stats = factor_stats(mixed, eligible_factors(mixed, ["when", "when_utc"]))
        step = np.diff(stats["stats"]["when"])
        assert np.all(step == step[0])
        assert step[0] > 0
        np.testing.assert_allclose(np.diff(stats["stats"]["when_utc"]), step)

    def test_a_null_is_nan(self):
        md = Metadata.from_factors({"x": np.r_[np.arange(29.0), np.nan]})
        assert np.isnan(factor_stats(md, eligible_factors(md, None))["stats"]["x"][-1])

    def test_rows_are_addressed_in_the_minimal_spelling(self, tracked):
        stats = factor_stats(tracked, eligible_factors(tracked, ["n_gaps", "time_s"]))
        addresses = set(stats["source_index"])
        assert SourceIndex(0, 5, "track") in addresses
        assert SourceIndex(0, 1, "unit") in addresses

    def test_a_factor_is_nan_on_every_other_level(self, tracked):
        stats = factor_stats(tracked, eligible_factors(tracked, ["n_gaps", "time_s"]))
        on_tracks = np.array([si.level == "track" for si in stats["source_index"]])
        assert np.all(np.isnan(stats["stats"]["time_s"][on_tracks]))
        assert not np.any(np.isnan(stats["stats"]["n_gaps"][on_tracks]))

    def test_rebinning_does_not_move_a_value(self):
        values = np.random.default_rng(0).normal(20, 1, 40)
        plain = Metadata.from_factors({"x": values})
        cut = Metadata.from_factors({"x": values}, continuous_factor_bins={"x": 3})
        np.testing.assert_array_equal(
            factor_stats(plain, eligible_factors(plain, None))["stats"]["x"],
            factor_stats(cut, eligible_factors(cut, None))["stats"]["x"],
        )

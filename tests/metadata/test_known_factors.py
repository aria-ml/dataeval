import datetime
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._metadata._aggregate import aggregate
from dataeval._metadata._known import known_violation, reject_invalid_known
from dataeval.types import KNOWN_FACTORS, ParseDateTime, Remap, Rescale
from tests.metadata.test_structurers import _mot_dataset


@pytest.mark.required
class TestRegistry:
    def test_registry_has_the_five_names_in_order(self):
        assert list(KNOWN_FACTORS) == ["timestamp", "frame_rate", "video_duration", "latitude", "longitude"]

    def test_registry_is_read_only(self):
        with pytest.raises(TypeError):
            KNOWN_FACTORS["x"] = KNOWN_FACTORS["latitude"]  # type: ignore[index]


@pytest.mark.required
class TestKnownViolation:
    def test_unknown_name_is_never_a_violation(self):
        assert known_violation("altitude", "instance", ["N"]) is None

    @pytest.mark.parametrize(
        ("name", "level", "values"),
        [
            ("latitude", "unit", [-90, 0.5, 90, None]),
            ("latitude", "unit", [10.0, float("nan")]),
            ("frame_rate", "sequence", [29.97]),
            ("video_duration", "sequence", [0]),
            ("timestamp", "unit", np.array([1704067200, 1704067201])),
        ],
    )
    def test_valid_values_pass(self, name, level, values):
        assert known_violation(name, level, values) is None

    @pytest.mark.parametrize(
        ("name", "level", "values"),
        [
            ("latitude", "unit", [91.0]),
            ("longitude", "unit", [-180.5]),
            ("frame_rate", "sequence", [0.0]),
            ("video_duration", "sequence", [-1]),
            ("timestamp", "unit", ["2024-01-01T00:00:00"]),
            ("timestamp", "unit", [True]),
            ("latitude", "unit", [-float("inf")]),
            ("timestamp", "unit", [float("inf")]),
        ],
    )
    def test_invalid_values_fail(self, name, level, values):
        assert name in (known_violation(name, level, values) or "")

    def test_violation_counts_failures_and_shows_the_first(self):
        msg = known_violation("latitude", "unit", [1.0, 200.0, 3.0, -95.0, 4.0])
        assert msg == (
            "'latitude' must be a finite number in [-90, 90] (decimal degrees); 2 of 5 values are not, e.g. 200.0"
        )

    def test_wrong_level_fails(self):
        assert "'sequence'" in (known_violation("frame_rate", "unit", [30.0]) or "")

    def test_level_is_checked_before_values(self):
        assert "level" in (known_violation("frame_rate", "unit", [-1.0]) or "")

    def test_reject_raises_with_rename_hint(self):
        columns = [("altitude", "unit", ["N"]), ("latitude", "unit", [91.0])]
        with pytest.raises(ValueError, match="Rename the factor"):
            reject_invalid_known(columns)

    def test_reject_tells_a_timestamp_to_pass_epoch_seconds(self):
        with pytest.raises(ValueError, match="seconds since the Unix epoch"):
            reject_invalid_known([("timestamp", "unit", ["2024-01-01T00:00:00"])])

    def test_reject_passes_valid_columns(self):
        reject_invalid_known([("altitude", "unit", ["N"]), ("latitude", "unit", [1.0])])


@pytest.mark.required
class TestExplicitCallsRaise:
    def test_from_factors_rejects_out_of_range_latitude(self):
        with pytest.raises(ValueError, match="'latitude'"):
            Metadata.from_factors({"latitude": [10.0, 200.0]})

    def test_from_factors_rejects_datetime_timestamp_with_a_conversion_hint(self):
        stamps = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[s]")
        with pytest.raises(ValueError, match="seconds since the Unix epoch"):
            Metadata.from_factors({"timestamp": stamps})

    def test_from_factors_accepts_valid_known(self):
        assert "latitude" in Metadata.from_factors({"latitude": [10.0, 20.0]}).factor_names

    def test_add_factors_rejects_wrong_level(self):
        md = Metadata(_mot_dataset([[1, 1]] * 2))
        with pytest.raises(ValueError, match="'sequence'"):
            md.add_factors({"frame_rate": [30.0] * 4}, level="unit")

    def test_add_factors_is_all_or_nothing(self):
        md = Metadata.from_factors({"x": [1, 2]})
        columns = list(md.dataframe.columns)
        with pytest.raises(ValueError, match="'latitude'"):
            md.add_factors({"y": [1, 2], "latitude": [0.0, 999.0]}, level="unit")
        assert "y" not in md.factor_names
        assert list(md.dataframe.columns) == columns

    def test_add_factors_renamed_known_is_not_checked(self):
        md = Metadata.from_factors({"latitude": [10.0, 20.0]})
        md.add_factors({"latitude": [999.0, 999.0]}, level="unit")
        assert "latitude_added" in md.factor_names

    def test_agg_named_known_raises(self):
        md = Metadata(_mot_dataset([[1, 1]] * 2))
        with pytest.raises(ValueError, match="'video_duration'"):
            md.agg("unit", "sequence", pl.lit(-5.0).alias("video_duration"))

    def test_unknown_names_unaffected(self):
        assert "altitude" in Metadata.from_factors({"altitude": ["N", "S"]}).factor_names

    def test_replay_skips_and_records_invalid_known(self, caplog):
        md = Metadata(_mot_dataset([[1, 1]] * 2))
        md._structure()
        rolled = aggregate(md._store, "unit", "sequence", [pl.lit(-5.0).alias("video_duration")], None)
        for _ in range(2):
            md._write_rolled([("unit", "sequence", None, None, rolled, None)], target=md)
        assert "video_duration" not in md.factor_names
        assert md.dropped_factors["video_duration"] == ["invalid_known_factor"]
        assert "'video_duration'" in caplog.text


def _md_latitude(values):
    # Whole-number floats are stored as integers, and a Remap key matches by type as well
    # as value, so the repair tests use fractional degrees to key on the stored value.
    return Metadata(_mot_dataset([[1, 1]] * len(values), [{"latitude": v} for v in values]))


@pytest.mark.required
class TestReadingPathsHoldBack:
    def test_walk_holds_back_out_of_range(self, caplog):
        md = _md_latitude([47.0, 200.0, 48.0])
        assert "latitude" not in md.at("sequence").factor_names
        assert md.dropped_factors["latitude"] == ["invalid_known_factor"]
        assert md.unusable["latitude"].reasons == ("invalid_known_factor",)
        assert md.unusable["latitude"].repairable
        assert "'latitude'" in caplog.text

    def test_walk_holds_back_iso_timestamp(self):
        stamps = ["2024-01-01T00:00:00", "2024-01-01T00:00:01"]
        md = Metadata(_mot_dataset([[1, 1]] * 3, [{"timestamp": stamps}] * 3))
        assert "timestamp" not in md.at("unit").factor_names
        assert md.unusable["timestamp"].reasons == ("invalid_known_factor",)

        md = md.repair([ParseDateTime("timestamp")])
        assert "timestamp" in md.at("unit").factor_names
        assert set(md.rows_at("unit")["timestamp"].to_list()) == {1704067200, 1704067201}

    def test_walk_holds_back_datetime_timestamp(self):
        stamps = [datetime.datetime(2024, 1, 1, 0, 0, s) for s in range(2)]
        md = Metadata(_mot_dataset([[1, 1]] * 3, [{"timestamp": stamps}] * 3))
        assert md.unusable["timestamp"].reasons == ("invalid_known_factor",)

        md = md.repair([ParseDateTime("timestamp")])
        assert set(md.rows_at("unit")["timestamp"].to_list()) == {1704067200, 1704067201}

    def test_repair_that_leaves_it_invalid_keeps_it_held(self):
        md = _md_latitude([47.0, 200.0, 48.0]).repair([Rescale("latitude", multiply=2.0)])
        assert md.unusable["latitude"].reasons == ("invalid_known_factor",)
        assert "latitude" not in md.factor_names

    def test_derived_copies_keep_it_held(self):
        md = _md_latitude([47.0, 200.0, 48.0])
        for copy in (
            md.at("unit"),
            md.where(pl.col("pts") >= 0, level="unit"),
            md.having(pl.col("pts") >= 0, level="unit"),
            md.reencode(),
            md.classed_by("width"),
        ):
            assert "latitude" in copy.unusable
            assert "latitude" not in copy.factor_names

    def test_repair_that_fixes_it_promotes(self):
        md = _md_latitude([47.5, 200.5, 48.5]).repair([Remap("latitude", {200.5: 49.5})])
        assert "latitude" in md.at("sequence").factor_names
        assert md.rows_at("sequence")["latitude"].to_list() == [47.5, 49.5, 48.5]
        assert md.unusable == {}
        assert md.dropped_factors == {}

    def test_unrepair_holds_it_back_again(self):
        md = _md_latitude([47.5, 200.5, 48.5]).repair([Remap("latitude", {200.5: 49.5})]).unrepair()
        assert md.unusable["latitude"].reasons == ("invalid_known_factor",)
        assert "latitude" not in md.factor_names

    def test_repair_that_breaks_a_valid_factor_is_not_applied(self, caplog):
        md = _md_latitude([47.5, 46.5, 48.5]).repair([Remap("latitude", {47.5: 500.5})])
        assert md.rows_at("sequence")["latitude"].to_list() == [47.5, 46.5, 48.5]
        assert "latitude" in md.factor_names
        assert "'latitude'" in caplog.text

    def test_mixed_and_known_stays_mixed_types(self):
        md = _md_latitude([47.0, "N", 48.0])
        assert md.unusable["latitude"].reasons == ("mixed_types",)
        md = md.repair([Remap("latitude", {"N": -1.0})])
        assert "latitude" in md.at("sequence").factor_names

    def test_mixed_repaired_to_invalid_stays_held_with_both_reasons(self):
        md = _md_latitude([47.0, "N", 48.0])
        repaired = md.repair([Remap("latitude", {"N": 500.0})])
        assert repaired.unusable["latitude"].reasons == ("mixed_types", "invalid_known_factor")
        assert "latitude" not in repaired.factor_names
        assert md.unusable["latitude"].reasons == ("mixed_types",)

    def test_wrong_level_is_not_repairable(self):
        md = Metadata(_mot_dataset([[1, 1]] * 3, [{"frame_rate": [30.0, 30.0]}] * 3))
        assert md.unusable["frame_rate"].reasons == ("invalid_known_factor",)
        assert md.unusable["frame_rate"].repairable is False

    def test_valid_known_from_dataset_is_a_factor(self):
        md = _md_latitude([47.0, 46.0, 48.0])
        assert "latitude" in md.at("sequence").factor_names
        assert md.dropped_factors == {}


@pytest.mark.required
class TestLoad:
    def test_round_trip_keeps_held_back(self, tmp_path):
        _md_latitude([47.0, 200.0, 48.0]).save(tmp_path / "m")
        loaded = Metadata.load(tmp_path / "m")
        assert "latitude" not in loaded.at("sequence").factor_names
        assert loaded.unusable["latitude"].reasons == ("invalid_known_factor",)

    def test_archive_from_before_the_check_is_rechecked(self, tmp_path, monkeypatch, caplog):
        # The check is imported by name, so it is switched off wherever it is read from.
        for module in ("dataeval._metadata._known", "dataeval._metadata._metadata"):
            monkeypatch.setattr(f"{module}.known_violation", lambda *a: None)
        old = _md_latitude([47.0, 200.0, 48.0])
        assert "latitude" in old.at("sequence").factor_names
        # An encoding derived from the invalid values, which the archive keeps.
        assert old.factor_info["latitude"].encoding is not None
        old.save(tmp_path / "m")
        monkeypatch.undo()

        loaded = Metadata.load(tmp_path / "m")
        with warnings.catch_warnings(action="error", category=UserWarning):
            assert "latitude" not in loaded.at("sequence").factor_names
            assert "latitude" not in loaded.factor_names
        assert loaded.unusable["latitude"].reasons == ("invalid_known_factor",)
        assert loaded.dropped_factors["latitude"] == ["invalid_known_factor"]
        assert loaded.unusable["latitude"].repairable
        assert "'latitude'" in caplog.text

        fixed = loaded.repair([Remap("latitude", {47: 47.5, 200: 49.5, 48: 48.5})])
        assert "latitude" in fixed.at("sequence").factor_names
        fresh = _md_latitude([47.5, 49.5, 48.5])
        info = fixed.at("sequence").factor_info["latitude"]
        assert info.encoding == fresh.at("sequence").factor_info["latitude"].encoding

    @staticmethod
    def _archive_without_check(monkeypatch, md_factory, path):
        for module in ("dataeval._metadata._known", "dataeval._metadata._metadata"):
            monkeypatch.setattr(f"{module}.known_violation", lambda *a: None)
        md_factory().save(path)
        monkeypatch.undo()

    def test_archived_repaired_mixed_column_keeps_its_raw_values(self, tmp_path, monkeypatch):
        def repaired():
            md = _md_latitude([47.0, "N", 48.0]).repair([Remap("latitude", {"N": 500.0})])
            assert "latitude" in md.at("sequence").factor_names
            assert md.factor_info["latitude"].encoding is not None
            return md

        self._archive_without_check(monkeypatch, repaired, tmp_path / "m")
        loaded = Metadata.load(tmp_path / "m")
        with warnings.catch_warnings(action="error", category=UserWarning):
            assert "latitude" not in loaded.at("sequence").factor_names
        assert "N" in loaded._unusable_values["sequence"]["latitude"]

        fixed = loaded.repair([Remap("latitude", {"N": 45.0})])
        assert "latitude" in fixed.at("sequence").factor_names
        fresh = _md_latitude([47.0, 45.0, 48.0])
        info = fixed.at("sequence").factor_info["latitude"]
        assert info.encoding == fresh.at("sequence").factor_info["latitude"].encoding

    def test_archived_corrected_factor_is_held_as_the_original_values(self, tmp_path, monkeypatch):
        def repaired():
            return _md_latitude([47.5, 46.5, 48.5]).repair([Remap("latitude", {47.5: 500.5})])

        self._archive_without_check(monkeypatch, repaired, tmp_path / "m")
        loaded = Metadata.load(tmp_path / "m")
        assert loaded.unusable["latitude"].reasons == ("invalid_known_factor",)
        assert loaded._unusable_values["sequence"]["latitude"] == [47.5, 46.5, 48.5]


@pytest.mark.required
class TestConceptPage:
    def test_every_known_factor_is_on_the_concept_page(self):
        page = Path(__file__).parents[2] / "docs" / "source" / "concepts" / "ExtrinsicFactors.md"
        _, heading, rest = page.read_text().partition("## Known factor names")
        assert heading
        section = rest.split("\n## ")[0]
        for name in KNOWN_FACTORS:
            assert f"`{name}`" in section

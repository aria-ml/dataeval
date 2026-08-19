"""Factors are binned at their own level, not at the target level.

The behaviour these pin down: a factor's bin edges, its bin count and its
continuous/discrete verdict are read off the level where it holds one value per
entity. Binning at the target level instead reads each factor's distribution
through however many descendants an entity happens to have.
"""

import warnings

import numpy as np
import pytest

from dataeval import Metadata
from dataeval._helpers import resolve_label_axis
from dataeval._metadata._columns import to_col
from dataeval.core._bin import bin_data
from dataeval.types import BinSpec, LevelSpec
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget


def _target(count: int) -> ObjectDetectionTarget:
    if not count:
        return ObjectDetectionTarget(np.empty((0, 4)), np.empty(0), np.empty(0))
    return ObjectDetectionTarget(
        np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (count, 1)),
        np.arange(count) % 2,
        np.full(count, 0.5),
    )


def _od(counts, factors, **kwargs) -> Metadata:
    """Object detection over ``len(counts)`` images with the given detection counts."""
    dataset = MockDataset(
        np.zeros((len(counts), 3, 16, 16)),
        [_target(count) for count in counts],
        [{name: values[i] for name, values in factors.items()} for i in range(len(counts))],
    )
    return Metadata(dataset, **kwargs)


def _companion(md: Metadata, name: str) -> str:
    """The binned or digitized column backing a factor."""
    return to_col(name, md.factor_info[name])


@pytest.mark.required
class TestNativeLevelBinning:
    def test_binned_column_is_populated_at_the_factors_own_level(self):
        """An image-level factor carries its bin on image rows, not only on instances."""
        md = _od([2, 1, 2], {"brightness": [0.1, 0.5, 0.9]})
        column = _companion(md, "brightness")

        assert md.factor_info["brightness"].level == "unit"
        assert md.rows_at("unit")[column].to_list() == [0, 1, 2]

    def test_bin_assignment_is_the_same_from_either_level(self):
        """The invariant: a level projection must not change what a bin means."""
        md = _od([3, 1, 2, 4], {"brightness": [0.1, 0.5, 0.9, 0.3]})
        column = _companion(md, "brightness")

        at_image = md.rows_at("unit")[column].to_list()
        instances = md.target_data
        gathered = [instances.filter(instances["item_index"] == i)[column][0] for i in range(4)]

        assert at_image == gathered

    def test_item_without_targets_still_reaches_the_binner(self):
        """The regression: a childless item contributes no target row at all.

        Binning over target rows never saw its value, so it was absent from the
        edges and had no bin of its own.
        """
        md = _od([2, 1, 2, 0], {"brightness": [0.1, 0.5, 0.9, 99.0]})
        column = _companion(md, "brightness")

        assert md.rows_at("unit")["brightness"].to_list() == [0.1, 0.5, 0.9, 99.0]
        assert md.rows_at("unit")[column].to_list() == [0, 1, 2, 3]

    def test_edges_come_from_the_native_distribution(self):
        """uniform_count quantiles are density-weighted when taken over target rows."""
        rng = np.random.default_rng(0)
        counts = [1] * 50 + [10] * 50
        values = np.sort(rng.uniform(0.0, 1.0, len(counts)))

        md = _od(counts, {"b": values.tolist()}, auto_bin_method="uniform_count")
        column = _companion(md, "b")

        assert md.factor_info["b"].factor_type == "continuous"
        np.testing.assert_array_equal(md.rows_at("unit")[column].to_numpy(), bin_data(values, "uniform_count")[0])

    def test_target_level_factors_are_unchanged(self):
        """The compatibility pin: nothing moves for a factor already at the target level."""
        md = _od([2, 1, 2], {"brightness": [0.1, 0.5, 0.9]})
        md.add_factors({"iou": np.array([0.1, 0.2, 0.3, 0.4, 0.5])}, level="instance")
        column = _companion(md, "iou")

        assert md.factor_info["iou"].level == "instance"
        assert md.target_data[column].to_list() == md.rows_at("instance")[column].to_list()


@pytest.mark.required
class TestImageClassificationUnaffected:
    """IC factors sit at image level over a fully labelled dataset, so nothing moves."""

    def _ic(self, labels, factors) -> Metadata:
        """``labels`` is a sequence of one-hot rows; an empty row means unlabelled."""
        return Metadata(
            MockDataset(
                np.zeros((len(labels), 3, 16, 16)),
                [np.asarray(row, dtype=float) for row in labels],
                [{name: values[i] for name, values in factors.items()} for i in range(len(labels))],
            ),
        )

    def test_factor_data_shape_and_alignment_hold(self):
        md = self._ic(np.eye(4, 2)[[0, 1, 0, 1]], {"brightness": [0.1, 0.5, 0.9, 0.3]})

        assert md.factor_data.shape == (4, 1)
        assert len(md.class_labels) == 4

    def test_unlabeled_image_keeps_its_factor_binned(self):
        """An unlabelled image has an image row but no instance row.

        Its factor value was previously invisible to the binner, which read only
        target rows; it now has a bin like every other image.
        """
        one_hot = np.eye(2)
        labels = [one_hot[0], one_hot[1], np.empty(0), one_hot[1]]  # image 2 carries no label
        md = self._ic(labels, {"brightness": [0.1, 0.5, 99.0, 0.3]})
        column = _companion(md, "brightness")

        assert md.level_counts["unit"] == 4
        assert md.level_counts["instance"] == 3
        assert None not in md.rows_at("unit")[column].to_list()


@pytest.mark.required
class TestEncodingReachesFactorInfo:
    """The map that produced each factor's codes survives structuring."""

    def test_a_declared_cutoff_is_recorded_as_declared(self):
        """The distinction the whole policy argument rests on, visible on the factor.

        Same edges could have been derived; what a reviewer needs to know is that a person
        chose them.
        """
        md = _od([1, 1, 1], {"temp_c": [-5.0, 5.0, 15.0]}, continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})
        spec = md.factor_info["temp_c"].encoding
        assert isinstance(spec, BinSpec)
        assert spec.provenance == "edges"
        assert spec.edges == (-np.inf, 0.0, np.inf)

    def test_an_auto_binned_factor_says_nobody_chose_it(self):
        rng = np.random.default_rng(5)
        md = _od([1] * 60, {"brightness": rng.normal(size=60).tolist()})
        spec = md.factor_info["brightness"].encoding
        assert isinstance(spec, BinSpec)
        assert spec.provenance == "derived"
        assert spec.method == "uniform_width"

    def test_a_categorical_factor_records_its_vocabulary(self):
        """A bare code cannot say what it stands for; the level list can."""
        md = _od([1] * 4, {"weather": ["sun", "rain", "fog", "sun"]})
        spec = md.factor_info["weather"].encoding
        assert isinstance(spec, LevelSpec)
        assert spec.provenance == "derived"
        assert spec.levels == ("fog", "rain", "sun")

    def test_levels_are_sorted_so_codes_match_what_sorting_would_give(self):
        """Sorted at first structuring, which is what keeps existing codes unchanged.

        Extension appends out of sort order instead of renumbering; that is the property
        the record exists for, and it costs nothing here because nothing is extended yet.
        """
        md = _od([1] * 5, {"n_people": [3, 1, 7, 1, 3]})
        spec = md.factor_info["n_people"].encoding
        assert isinstance(spec, LevelSpec)
        assert spec.levels == (1, 3, 7)
        codes = md.rows_at("unit")[_companion(md, "n_people")].to_numpy()
        # Code i means levels[i], so reading the codes back through the record recovers
        # the values exactly.
        assert [spec.levels[c] for c in codes] == [3, 1, 7, 1, 3]

    def test_every_factor_carries_one_map_or_the_other(self):
        rng = np.random.default_rng(6)
        md = _od(
            [1] * 60,
            {
                "brightness": rng.normal(size=60).tolist(),
                "weather": ["sun", "rain"] * 30,
                "n_people": (np.arange(60) % 4).tolist(),
            },
        )
        for name, info in md.factor_info.items():
            expected = BinSpec if info.is_binned else LevelSpec
            assert isinstance(info.encoding, expected), f"{name} recorded {type(info.encoding)}"


@pytest.mark.required
class TestIdentifierColumnsAreDropped:
    """A column that names its rows is not a factor, and cannot be turned into one.

    The line is near-uniqueness, not the level budget. The budget answers "how many cells
    can this sample fill", which is the right question for choosing a bin count and the
    wrong one for deciding whether a column is a factor at all.
    """

    def test_a_level_per_row_is_dropped_and_says_why(self):
        names = [f"img_{i:04d}.png" for i in range(60)]
        md = _od([1] * 60, {"filename": names})

        assert "filename" not in md.factor_names
        assert "cardinality_over_budget" in md.dropped_factors["filename"]

    def test_an_ordinary_vocabulary_is_untouched(self):
        """A handful of repeated values groups rows, so nothing is in question."""
        md = _od([1] * 60, {"weather": ["sun", "rain", "fog"] * 20})

        assert "weather" in md.factor_names
        assert "weather" not in md.dropped_factors
        spec = md.factor_info["weather"].encoding
        assert isinstance(spec, LevelSpec)
        assert spec.levels == ("fog", "rain", "sun")

    def test_a_numeric_factor_at_the_same_cardinality_is_binned_instead(self):
        """The asymmetry is the point: an ordered column can be coarsened, a set of labels cannot.

        Both factors carry 60 distinct values over 60 entities. The numeric one is cut into
        bins and kept; the text one has no order to cut along.
        """
        values = np.arange(60.0) * 3.0
        # Prefixed so the structurer cannot coerce these back to numbers, which is what
        # makes this the non-numeric path rather than a second copy of the numeric one.
        md = _od([1] * 60, {"numeric": values.tolist(), "text": [f"id_{v}" for v in values]})

        assert "numeric" in md.factor_names
        assert md.factor_info["numeric"].is_binned
        assert "text" not in md.factor_names
        assert "cardinality_over_budget" in md.dropped_factors["text"]

    def test_the_surviving_factors_are_still_analysable(self):
        """Dropping one factor must not disturb the others' codes or their records."""
        names = [f"img_{i:04d}.png" for i in range(60)]
        md = _od([1] * 60, {"filename": names, "weather": ["sun", "rain"] * 30})

        assert md.factor_names == ["weather"]
        spec = md.factor_info["weather"].encoding
        assert isinstance(spec, LevelSpec)
        assert spec.levels == ("rain", "sun")
        codes = md.rows_at("unit")[_companion(md, "weather")].to_numpy()
        assert set(codes.tolist()) == {0, 1}


@pytest.mark.required
class TestStructuringAnnouncesItself:
    """The advice existed on both auto-binning paths and reached nobody.

    Both were ``_logger.warning``, and ``dataeval`` attaches a ``NullHandler`` to its root
    logger, which suppresses Python's last-resort stderr handler. The forcing function was
    written; it did not fire.
    """

    def _factors(self, n=80):
        rng = np.random.default_rng(0)
        return {
            "illum_lux": rng.normal(50, 20, n).tolist(),
            "exposure_ms": rng.normal(10, 3, n).tolist(),
            "box_area_px": rng.normal(500, 100, n).tolist(),
        }

    def test_auto_binning_warns_once_naming_every_factor(self):
        """One warning, not one per factor: twelve near-identical warnings teach filtering."""
        with pytest.warns(UserWarning, match="binned automatically") as caught:
            _ = _od([1] * 80, self._factors()).factor_data

        binning = [w for w in caught if "binned automatically" in str(w.message)]
        assert len(binning) == 1
        message = str(binning[0].message)
        for name in ("illum_lux", "exposure_ms", "box_area_px"):
            assert name in message
        assert "uniform_width" in message
        assert "continuous_factor_bins" in message

    def test_a_declared_factor_is_not_named(self):
        """The warning is about what nobody chose, so a declared cutoff is not in it."""
        factors = self._factors()
        factors["temp_c"] = np.random.default_rng(1).normal(0, 15, 80).tolist()
        with pytest.warns(UserWarning, match="binned automatically") as caught:
            _ = _od([1] * 80, factors, continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]}).factor_data

        message = next(str(w.message) for w in caught if "binned automatically" in str(w.message))
        assert "temp_c" not in message

    def test_declaring_every_factor_warns_about_none(self):
        """The exit condition: nothing was auto-binned, so nothing is announced as such.

        A declared cut can still draw the *fitness* report — these factors are all
        positive, so the below-zero bin stays empty — which is a different message about a
        different thing and is asserted separately.
        """
        edges = {name: [-np.inf, 0.0, np.inf] for name in self._factors()}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = _od([1] * 80, self._factors(), continuous_factor_bins=edges).factor_data

        assert not [w for w in caught if "binned automatically" in str(w.message)]

    def test_a_dropped_factor_says_so_out_loud(self):
        """Removing a factor outright is at least as worth hearing as binning one."""
        factors = self._factors()
        factors["filename"] = [f"img_{i:04d}.png" for i in range(80)]
        with pytest.warns(UserWarning, match="was dropped") as caught:
            _ = _od([1] * 80, factors).factor_data

        message = next(str(w.message) for w in caught if "was dropped" in str(w.message))
        assert "filename" in message
        assert "dropped_factors" in message

    def test_the_repr_reports_the_silent_default(self):
        """Visible on inspection too, for a caller who filtered the warning."""
        md = _od([1] * 80, self._factors())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _ = md.factor_data
        assert "auto_encoded=3" in repr(md)

    def test_the_repr_does_not_force_binning(self):
        """A repr must not trigger the expensive pass just to count what it did."""
        md = _od([1] * 80, self._factors())
        assert "auto_encoded" not in repr(md)
        assert md._is_binned is False

    def test_a_wide_but_real_vocabulary_survives(self):
        """Twenty-five cities over a hundred images is a factor, not an identifier.

        The level budget would drop this — ``max(20, sqrt(100))`` is 20 — and dropping it
        forecloses the mechanism that exists to report thin levels:
        :attr:`~dataeval.bias.ParityOutput.insufficient_data` says a cell is under-sampled,
        which is the honest answer for a factor this wide on a sample this size.
        """
        cities = [f"city_{i % 25}" for i in range(100)]
        md = _od([1] * 100, {"location": cities})

        assert md.factor_names == ["location"]
        assert "location" not in md.dropped_factors
        spec = md.factor_info["location"].encoding
        assert isinstance(spec, LevelSpec)
        assert len(spec.levels) == 25

    def test_a_timestamp_is_cut_not_dropped(self):
        """A capture time is totally ordered, so it bins like a number.

        One distinct value per row would otherwise read as an identifier, and a timestamp
        is one of the most common per-row fields there is.
        """
        # Millisecond resolution: polars rejects minute-resolution datetime64 outright.
        stamps = np.datetime64("2026-01-01", "ms") + np.arange(60).astype("timedelta64[ms]") * 60_000
        # Through from_factors, where the dtype survives into the frame. A per-image
        # metadata dict stringifies its values before they reach structuring, and a column
        # of 60 distinct strings is an identifier whatever it once meant.
        md = Metadata.from_factors({"captured_at": stamps}, class_labels=np.zeros(60, dtype=int))

        assert "captured_at" in md.factor_names
        assert "captured_at" not in md.dropped_factors
        assert md.factor_info["captured_at"].is_binned

    def test_an_unset_timestamp_is_missing_not_an_extreme_magnitude(self):
        """``NaT`` is a capture time nobody recorded, not one nine quintillion years early.

        Casting a temporal column straight to ``int64`` renders ``NaT`` as ``INT64_MIN``,
        which the edge placement then reads as an *observed* value it has to span: every
        real timestamp collapses into a single bin against it, and the missing code the
        record reserves is never used. Five unset capture times would destroy the factor.
        """
        stamps = (np.datetime64("2026-01-01", "ms") + np.arange(60).astype("timedelta64[ms]") * 60_000).astype(
            "datetime64[ms]",
        )
        stamps[::12] = np.datetime64("NaT")
        md = Metadata.from_factors({"captured_at": stamps}, class_labels=np.zeros(60, dtype=int))

        spec = md.factor_info["captured_at"].encoding
        assert isinstance(spec, BinSpec)
        codes = md.factor_data[:, 0]
        assert int((codes == spec.missing_code).sum()) == 5
        # The 55 real timestamps still spread out; nothing was flattened to make room.
        assert len(np.unique(codes[codes != spec.missing_code])) > 1
        assert all(np.isfinite(edge) for edge in spec.edges[1:-1])

    def test_every_accessor_agrees_without_binning_first(self):
        """The drop is decided while the factor set is built, so nothing depends on access order.

        Deciding it during binning made the answer turn on which property was touched
        first: ``dropped_factors`` read ``{}`` on a fresh instance while the warning that
        announced the drop pointed the reader at it.
        """
        factors = {"filename": [f"img_{i:04d}.png" for i in range(60)], "weather": ["a", "b"] * 30}

        assert dict(_od([1] * 60, factors).dropped_factors) == {"filename": ["cardinality_over_budget"]}
        assert _od([1] * 60, factors).factor_names == ["weather"]
        assert _od([1] * 60, factors).shape[1] == 1
        assert "filename" not in str(_od([1] * 60, factors))
        assert _od([1] * 60, factors).factor_data.shape[1] == 1

    def test_factor_names_does_not_force_the_binning_pass(self):
        """Reading the names is a lookup, not the digitize-every-factor pass."""
        md = _od([1] * 60, {"weather": ["a", "b"] * 30})

        assert md.factor_names == ["weather"]
        assert md._is_binned is False

    def test_the_drop_is_announced_once_across_views(self):
        """A derived view inherits the reason; it must not re-announce it."""
        factors = {"filename": [f"img_{i:04d}.png" for i in range(60)], "weather": ["a", "b"] * 30}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md = _od([1] * 60, factors)
            _ = md.factor_data
            _ = md.at("unit").factor_data
            md.exclude = ["weather"]
            _ = md.factor_names

        assert len([w for w in caught if "dropped" in str(w.message)]) == 1


@pytest.mark.required
class TestTheIdentifierVerdictIsDecidedOnce:
    """Whether a column names its rows is a question about the column, not about which rows
    are in view. Re-asking it on a derived instance made a filter delete a factor its source
    had kept -- and reported it as "nearly every row holds a different value", which was
    never true of the data."""

    def _cities(self, n=60):
        """Twenty-five cities over sixty rows: an ordinary factor, and near-unique over the
        thirty rows a filter leaves."""
        return {
            "city": [f"c{i % 25}" for i in range(n)],
            "flag": [float(i % 2) for i in range(n)],
        }

    def test_a_filter_does_not_delete_a_factor_the_source_kept(self):
        import polars as pl

        md = _od([1] * 60, self._cities())
        assert "city" in md.factor_names

        kept = md.where(pl.col("flag") == 1.0)
        assert kept.shape[0] == 30
        assert "city" in kept.factor_names
        assert "city" not in kept.dropped_factors

    def test_a_view_move_does_not_delete_one_either(self):
        md = _od([1] * 60, self._cities())

        assert "city" in md.at("unit").factor_names

    def test_a_warning_filter_turned_into_an_error_leaves_the_factors_in_place(self):
        """Structuring marks itself complete before announcing, so a raise inside the
        announcement used to leave the instance permanently claiming it has no factors."""
        factors = {"filename": [f"img_{i:04d}.png" for i in range(60)], "weather": ["a", "b"] * 30}
        md = _od([1] * 60, factors)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning):
                _ = md.factor_names

        assert md.factor_names == ["weather"]


@pytest.mark.required
def test_an_empty_edge_list_names_its_codes_rather_than_raising():
    """`continuous_factor_bins={"x": []}` places no cut, so there is no interval to name a
    code after -- every naming path used to dereference `edges[0]`."""
    md = _od([1] * 60, {"temp": list(np.linspace(0, 1, 60)), "w": ["a", "b"] * 30}, continuous_factor_bins={"temp": []})

    assert set(resolve_label_axis(md, "temp").names.values()) == {"0"}

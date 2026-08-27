"""Evaluators report at the level they detected at — FE-D.

An evaluator holds statistics, not metadata, so it cannot resolve an unstated level. What
it can do is not *lose* a stated one: read the level off each address, keep readings of one
kind of row apart from readings of another, and hand back the addresses that went in.

There is no video statistics producer yet — ``compute_stats`` treats a dataset item as one
image — so these tests relevel an image-task result, exactly as
``tests/metadata/test_video_stats_landing.py`` stands in for the same missing producer on
the metadata side. What is being pinned is that a level survives the round trip, not any
particular video number.

The other half of every assertion here is that **image-task output is unchanged**. Under
the minimal-spelling rule a producer states a level only where an unstated one would mean
something else, so nothing an image dataset produces carries one, and neither column
appears.
"""

import numpy as np
import pytest

from dataeval.core import StatsResult, compute_stats
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers
from dataeval.types import SourceIndex

_LEVELS = ["sequence", "unit", "track", "instance"]


def _outlier_stats():
    """Twelve readings with one obvious outlier, addressed however a test likes."""
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (3, 16, 16), dtype=np.uint8) for _ in range(12)]
    images[3][:] = 255
    return compute_stats(
        images,
        stats=ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS,
        normalize_pixel_values=False,
    )


def _duplicate_stats():
    """Four readings where three are the same picture."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (3, 32, 32), dtype=np.uint8)
    images = [base, base.copy(), rng.integers(0, 256, (3, 32, 32), dtype=np.uint8), base.copy()]
    return compute_stats(images, stats=ImageStats.HASH_DUPLICATES_BASIC, normalize_pixel_values=False)


def _relevelled(stats: StatsResult, address) -> StatsResult:
    """Return the same statistics, readdressed by `address(position)`."""
    return {**stats, "source_index": [address(i) for i in range(len(stats["source_index"]))]}


@pytest.mark.required
class TestOutliersReportAtTheLevelDetected:
    def test_an_image_result_states_no_level(self):
        """The column is absent, not null: nothing an image dataset produces states one."""
        result = Outliers().from_stats(_outlier_stats())

        assert "level" not in result.data().columns
        assert result.outliers == {3: ["brightness", "mean"]}

    @pytest.mark.parametrize("level", ["unit", "track"])
    def test_a_level_between_the_ends_survives_the_round_trip(self, level):
        result = Outliers().from_stats(_relevelled(_outlier_stats(), lambda i: SourceIndex(i // 6, i % 6, level)))

        assert list(result.outliers) == [SourceIndex(0, 3, level)]

    def test_the_label_end_is_reported_canonically(self):
        """`instance` is the label end, and a key already says a row is one.

        The address comes back in its canonical spelling rather than the one it went in as,
        which is what makes two runs over differently-spelled statistics compare equal.
        """
        stats = _relevelled(_outlier_stats(), lambda i: SourceIndex(i // 6, i % 6, "instance"))
        result = Outliers().from_stats(stats, per_target=True)

        assert list(result.outliers) == [SourceIndex(0, 3)]
        assert "level" not in result.data().columns

    def test_an_unkeyed_address_reports_as_its_item(self):
        """An item has one own-row, so its item names it — whatever level that row is at.

        Echoing the caller's `SourceIndex(3, None, "sequence")` back would key the result on
        a spelling that is not equal to the `3` the same finding has always been reported
        as, and that `Duplicates` still reports it as.
        """
        result = Outliers().from_stats(_relevelled(_outlier_stats(), lambda i: SourceIndex(i, None, "sequence")))

        assert list(result.outliers) == [3]
        assert "level" not in result.data().columns

    def test_the_level_is_a_column_of_the_frame(self):
        result = Outliers().from_stats(_relevelled(_outlier_stats(), lambda i: SourceIndex(i // 6, i % 6, "unit")))

        assert result.data()["level"].to_list() == ["unit", "unit"]

    def test_a_level_between_the_ends_is_included_whatever_the_flags_say(self):
        """`per_image`/`per_target` name the two ends; a frame is neither, so neither gates it."""
        stats = _relevelled(_outlier_stats(), lambda i: SourceIndex(i // 6, i % 6, "unit"))

        assert Outliers().from_stats(stats, per_target=False).outliers
        assert Outliers().from_stats(stats, per_image=False).outliers

    def test_levels_are_thresholded_apart(self) -> None:
        """A per-frame reading is not compared against a spread that includes sequences.

        Every value is duplicated across two levels and one level's copy is shifted far
        out. Thresholded together the shifted level would drag the bounds and the untouched
        one would stop flagging its own outlier; thresholded apart, both still flag.
        """
        stats = _outlier_stats()
        doubled = {name: np.concatenate([values, values + 1000.0]) for name, values in stats["stats"].items()}
        addresses = [SourceIndex(0, i, "unit") for i in range(12)]
        addresses += [SourceIndex(0, i, "track") for i in range(12)]

        result = Outliers().from_stats({**stats, "stats": doubled, "source_index": addresses})
        # Every address here states a level, so every key is a SourceIndex rather than an int.
        flagged = {address.level for address in result.outliers if isinstance(address, SourceIndex)}

        assert flagged == {"unit", "track"}, "one level's spread swallowed the other's outlier"


@pytest.mark.required
class TestDuplicatesReportAtTheLevelDetected:
    def test_an_image_result_states_no_level(self):
        result = Duplicates().from_stats(_duplicate_stats())

        assert "address_levels" not in result.data().columns
        assert result.exact == [[0, 1, 3]]

    @pytest.mark.parametrize("level", ["unit", "track"])
    def test_a_level_between_the_ends_survives_the_round_trip(self, level):
        result = Duplicates().from_stats(_relevelled(_duplicate_stats(), lambda i: SourceIndex(0, i, level)))

        assert result.exact == [[SourceIndex(0, 0, level), SourceIndex(0, 1, level), SourceIndex(0, 3, level)]]

    def test_the_label_end_is_reported_canonically(self):
        """As `Outliers` does: a key already says the row is one of its item's labels."""
        stats = _relevelled(_duplicate_stats(), lambda i: SourceIndex(0, i, "instance"))
        result = Duplicates().from_stats(stats, per_target=True)

        assert result.exact == [[SourceIndex(0, 0), SourceIndex(0, 1), SourceIndex(0, 3)]]
        assert "address_levels" not in result.data().columns

    def test_an_unkeyed_address_reports_as_its_item(self):
        """As `Outliers` does, on the same input: an item's own row is named by its item."""
        result = Duplicates().from_stats(_relevelled(_duplicate_stats(), lambda i: SourceIndex(i, None, "sequence")))

        assert result.exact == [[0, 1, 3]]
        assert "address_levels" not in result.data().columns

    def test_a_level_between_the_ends_is_included_whatever_per_target_says(self):
        stats = _relevelled(_duplicate_stats(), lambda i: SourceIndex(0, i, "unit"))

        assert Duplicates().from_stats(stats, per_target=False).exact

    def test_readings_of_different_kinds_are_not_compared(self):
        """A frame and a detection are both keyed addresses; pooling them invents duplicates.

        Positions 0 and 3 are the same picture and are addressed at different levels, so a
        detector that hashed every keyed row together would call them duplicates of each
        other. Only positions 1 and 3, which share a level, are.
        """
        addresses = [SourceIndex(0, 0, "unit"), SourceIndex(0, 0, "instance")]
        addresses += [SourceIndex(0, 1, "unit"), SourceIndex(0, 1, "instance")]
        result = Duplicates().from_stats(_relevelled(_duplicate_stats(), addresses.__getitem__), per_target=True)

        assert result.exact == [[SourceIndex(0, 0), SourceIndex(0, 1)]]


@pytest.mark.required
class TestFindingsFeedBackAsFactors:
    """The sentence FE-D exists to make writable: findings out, factors in.

    An evaluator's keys are addresses and ``add_factors(source_index=)`` takes addresses, so
    a finding needs no translation to become a factor. This is the round trip end to end.
    """

    @staticmethod
    def _tracking_metadata():
        from dataeval import Metadata
        from tests.metadata.test_structurers import _mot_dataset

        metadata = Metadata(_mot_dataset([[[5, 9], [], [9, -1]], [[7], [7, 3]]]))
        metadata._structure()
        return metadata

    def test_frame_findings_become_frame_factors(self):
        metadata = self._tracking_metadata()
        # One reading per frame of the fixture, addressed out of row order.
        addresses = [SourceIndex(item, key, "unit") for item, key in [(0, 2), (1, 0), (0, 0), (1, 1), (0, 1)]]
        flagged = dict.fromkeys(addresses[:2], 1.0)

        metadata.add_factors(
            {"is_outlier": [1.0 if address in flagged else 0.0 for address in addresses]},
            source_index=addresses,
        )

        assert metadata._store.frame("unit")["is_outlier"].to_list() == [0.0, 0.0, 1.0, 1.0, 0.0]

    def test_track_findings_become_track_factors(self):
        metadata = self._tracking_metadata()
        addresses = [SourceIndex(0, 9, "track"), SourceIndex(1, 3, "track")]
        addresses += [SourceIndex(0, 5, "track"), SourceIndex(1, 7, "track")]

        metadata.add_factors({"is_duplicate": [1.0, 0.0, 0.0, 1.0]}, source_index=addresses)

        assert metadata._store.frame("track").select("track_id", "is_duplicate").rows() == [
            (5, 0.0),
            (9, 1.0),
            (7, 1.0),
            (3, 0.0),
        ]

    def test_an_outliers_result_keys_straight_back_in(self):
        """`.outliers` keys are addresses, and that is exactly what `source_index=` takes."""
        metadata = self._tracking_metadata()
        readings = np.array([0.0, 0.1, 0.2, 9.9, 0.1])
        addresses = [SourceIndex(item, key, "unit") for item, key in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]]
        stats: StatsResult = {
            "stats": {"blur": readings},
            "source_index": addresses,
            "object_count": [],
            "invalid_box_count": [],
            "image_count": 2,
        }
        result = Outliers().from_stats(stats)

        assert list(result.outliers) == [SourceIndex(1, 0, "unit")]

        metadata.add_factors(
            {"is_outlier": [1.0 if address in result.outliers else 0.0 for address in addresses]},
            source_index=addresses,
        )
        assert metadata._store.frame("unit")["is_outlier"].to_list() == [0.0, 0.0, 0.0, 1.0, 0.0]


@pytest.mark.required
class TestTheTwoSpellingsBehaveIdentically:
    """The minimal-spelling rule is a convention; nothing may *depend* on which was used.

    An evaluator groups and gates on :attr:`~dataeval.types.SourceIndex.kind`, which reads
    two spellings of one row as one kind — so the fully explicit spelling of a result, which
    the migration guide blesses, is gated exactly as the minimal one is and buckets with it.
    Reading `level` directly instead made the flags no-ops and split one level's readings
    into two populations.
    """

    @pytest.mark.parametrize(("per_image", "per_target"), [(True, False), (False, True), (True, True)])
    def test_outliers_gate_both_spellings_alike(self, per_image, per_target):
        minimal = _relevelled(_outlier_stats(), lambda i: SourceIndex(0, i))
        explicit = _relevelled(_outlier_stats(), lambda i: SourceIndex(0, i, "instance"))

        assert len(Outliers().from_stats(minimal, per_image=per_image, per_target=per_target).outliers) == len(
            Outliers().from_stats(explicit, per_image=per_image, per_target=per_target).outliers
        )

    @pytest.mark.parametrize(("per_image", "per_target"), [(True, False), (False, True), (True, True)])
    def test_duplicates_gate_both_spellings_alike(self, per_image, per_target):
        minimal = _relevelled(_duplicate_stats(), lambda i: SourceIndex(0, i))
        explicit = _relevelled(_duplicate_stats(), lambda i: SourceIndex(0, i, "instance"))

        assert len(Duplicates().from_stats(minimal, per_image=per_image, per_target=per_target).exact) == len(
            Duplicates().from_stats(explicit, per_image=per_image, per_target=per_target).exact
        )

    def test_the_two_spellings_are_compared_against_each_other(self):
        """Positions 0, 1 and 3 are one picture, addressed two ways. They are one group.

        Bucketing on the stated level rather than the kind put them in two populations, so
        genuine duplicates spelled differently were never hashed against each other.
        """
        addresses = [SourceIndex(0, 0, "instance"), SourceIndex(0, 1), SourceIndex(0, 2, "instance"), SourceIndex(0, 3)]
        result = Duplicates().from_stats(_relevelled(_duplicate_stats(), addresses.__getitem__), per_target=True)

        assert result.exact == [[SourceIndex(0, 0), SourceIndex(0, 1), SourceIndex(0, 3)]]

    def test_both_spellings_produce_the_same_frame(self):
        """Not just the same groups — the same table, column for column."""
        minimal = Duplicates().from_stats(_relevelled(_duplicate_stats(), lambda i: SourceIndex(0, i)), per_target=True)
        explicit = Duplicates().from_stats(
            _relevelled(_duplicate_stats(), lambda i: SourceIndex(0, i, "instance")), per_target=True
        )

        assert minimal.data().equals(explicit.data())

    def test_an_unkeyed_address_is_one_kind_whatever_level_it_states(self):
        """An item has exactly one own-row, so `sequence`, `unit` and unstated all name it."""
        addresses = [SourceIndex(0, None, "sequence"), SourceIndex(1), SourceIndex(2, None, "unit"), SourceIndex(3)]
        result = Duplicates().from_stats(_relevelled(_duplicate_stats(), addresses.__getitem__))

        assert result.exact == [[0, 1, 3]]

    @pytest.mark.parametrize(
        ("address", "kind"),
        [
            (SourceIndex(3), None),
            (SourceIndex(3, None, "sequence"), None),
            (SourceIndex(3, None, "unit"), None),
            (SourceIndex(3, 7), "instance"),
            (SourceIndex(3, 7, "instance"), "instance"),
            (SourceIndex(3, 12, "unit"), "unit"),
            (SourceIndex(3, 5, "track"), "track"),
        ],
    )
    def test_kind_is_what_the_evaluators_group_on(self, address, kind):
        assert address.kind == kind

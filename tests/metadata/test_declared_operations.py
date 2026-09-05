"""Corrections and roll-ups as constructor arguments, not only as method calls.

``repair`` and ``aggregate`` are post-construction, so anything building a Metadata from a
configuration had to construct it, then call two methods in the right order, and reproduce
that order again on every path that rebuilds one — including a cache restoring from an
archive. Getting the order wrong is silent: the numbers are computed against
differently-read columns and nothing says so.

Declaring them up front makes the order the library's rather than the caller's.
"""

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.types import Aggregator, ParseValue, Rescale
from tests.metadata.test_structurers import _mot_dataset

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

RAW = ["1,000", "2,000", "3,000", "4,000", "5,000"]


def _md(**kwargs) -> Metadata:
    return Metadata.from_factors({"count": list(RAW)}, class_labels=np.zeros(5, dtype=int), **kwargs)


@pytest.mark.required
class TestCorrectionsAsAConstructorArgument:
    def test_they_are_applied_when_the_walk_happens(self):
        md = _md(corrections=[ParseValue("count", drop=[","])])
        assert md.dataframe["count"].to_list() == [1000, 2000, 3000, 4000, 5000]

    def test_they_are_reported_as_declared(self):
        md = _md(corrections=[ParseValue("count", drop=[","])])
        assert md.repairs == (ParseValue("count", drop=[","]),)

    def test_they_apply_in_the_order_given(self):
        md = _md(corrections=[ParseValue("count", drop=[","]), Rescale("count", multiply=2.0)])
        assert md.dataframe["count"].to_list() == [2000, 4000, 6000, 8000, 10000]

    def test_declaring_none_leaves_the_values_as_written(self):
        assert _md().dataframe["count"].to_list() == RAW

    def test_it_agrees_with_calling_repair(self):
        """The argument is the same operation, moved earlier — not a second one."""
        declared = _md(corrections=[ParseValue("count", drop=[","])])
        called = _md()
        called.repair([ParseValue("count", drop=[","])])
        assert declared.dataframe["count"].to_list() == called.dataframe["count"].to_list()
        assert declared.repairs == called.repairs


@pytest.mark.required
class TestAggregationsAsAConstructorArgument:
    """A tracking dataset, so there is a level above the one the factor is measured at."""

    def test_a_declared_roll_up_runs_on_the_walk(self):
        dataset = _mot_dataset([[2, 1], [1]])
        md = Metadata(
            dataset,
            aggregations=[Aggregator(how="mean", source="unit", target="sequence", factors=("width",))],
        )
        assert "width_mean" in md.factor_names

    def test_it_agrees_with_calling_aggregate(self):
        declared = Metadata(
            _mot_dataset([[2, 1], [1]]),
            aggregations=[Aggregator(how="mean", source="unit", target="sequence", factors=("width",))],
        )
        called = Metadata(_mot_dataset([[2, 1], [1]])).aggregate("width", level="sequence", how="mean")
        assert declared.dataframe["width_mean"].to_list() == called.dataframe["width_mean"].to_list()

    def test_declaring_none_rolls_nothing_up(self):
        assert "width_mean" not in Metadata(_mot_dataset([[2, 1], [1]])).factor_names

    def test_a_later_roll_up_reads_what_an_earlier_one_wrote(self):
        """The reason these are a sequence rather than a set: a roll-up onto a level may
        read a column an earlier one has just written there."""
        md = Metadata(
            _mot_dataset([[2, 1], [1]]),
            aggregations=[
                Aggregator(how="mean", source="track", target="sequence", factors=("track_length",)),
                Aggregator(how="mean", source="unit", target="sequence", factors=("width",)),
            ],
        )
        assert {"track_length_mean", "width_mean"} <= set(md.factor_names)

    def test_a_roll_up_this_dataset_cannot_answer_is_not_fatal(self):
        """Carried declarations are resolved against *this* dataset, and one it cannot
        answer leaves the metadata usable rather than refusing to build it."""
        md = Metadata(
            _mot_dataset([[2, 1], [1]]),
            aggregations=[Aggregator(how="mean", source="unit", target="sequence", factors=("absent",))],
        )
        assert "width" in md.factor_names


@pytest.mark.required
class TestPartialFactorsOnLoad:
    """`load` was the only one of the three constructors without it, so a caller building
    one kwargs mapping for both paths got a TypeError a cache reads as a permanent miss."""

    def test_load_accepts_it(self, tmp_path):
        md = _md()
        md.save(tmp_path / "md.dem")
        assert Metadata.load(tmp_path / "md.dem", partial_factors=True).partial_factors is True

    def test_it_sits_underneath_the_archives(self, tmp_path):
        """Like `strict`: passing True closes what the archive left open, and passing
        nothing keeps what was written."""
        md = _md(partial_factors=True)
        md.save(tmp_path / "md.dem")
        assert Metadata.load(tmp_path / "md.dem").partial_factors is True

    def test_the_three_constructors_agree_on_the_argument(self):
        import inspect

        for build in (Metadata.__init__, Metadata.from_factors, Metadata.load):
            assert "partial_factors" in inspect.signature(build).parameters


@pytest.mark.required
class TestADescriptorsCorrectionsMeetALoad:
    """`load` restores the archive's corrections, so a descriptor's would be read and then
    dropped. Refused rather than silently ignored, and `corrections=` is the way to say it."""

    def test_a_descriptor_carrying_corrections_is_refused(self, tmp_path):
        seed = _md()
        seed.repair([ParseValue("count", drop=[","])])
        seed.export_encoding(tmp_path / "enc.json")
        _md().save(tmp_path / "md.dem")

        with pytest.raises(ValueError, match="corrections"):
            Metadata.load(tmp_path / "md.dem", encoding=tmp_path / "enc.json")

    def test_a_descriptor_with_no_corrections_is_fine(self, tmp_path):
        _md().export_encoding(tmp_path / "enc.json")
        _md().save(tmp_path / "md.dem")

        assert Metadata.load(tmp_path / "md.dem", encoding=tmp_path / "enc.json") is not None

    def test_the_constructor_still_reads_them(self, tmp_path):
        """Only `load` refuses; the path a descriptor exists for is unaffected."""
        seed = _md()
        seed.repair([ParseValue("count", drop=[","])])
        seed.export_encoding(tmp_path / "enc.json")

        assert _md(encoding=tmp_path / "enc.json").dataframe["count"].to_list() == [1000, 2000, 3000, 4000, 5000]

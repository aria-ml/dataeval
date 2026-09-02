"""Rolling rows up into a level above them — FE-5 issue 5.3.

The oracle here is a hand-written polars ``group_by`` over the *flat* frame, keyed by the
reserved columns rather than by positions. That is the implementation ``agg`` exists to
avoid — it hashes a key per row — so agreeing with it is the statement that the positional
grouping loses nothing.
"""

import logging

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._metadata._aggregate import aggregate, validate
from dataeval.types import Aggregator
from tests.metadata.test_structurers import _mot_dataset

# How to name the ancestor a row belongs to, from the row's own reserved columns.
GROUP_KEY = {
    ("unit", "sequence"): ("item_index",),
    ("track", "sequence"): ("item_index",),
    ("instance", "sequence"): ("item_index",),
    ("instance", "unit"): ("item_index", "unit_index"),
    ("instance", "track"): ("item_index", "track_index"),
}
ROW_KEY = {
    "sequence": ("item_index",),
    "unit": ("item_index", "unit_index"),
    "track": ("item_index", "track_index"),
    "instance": ("item_index", "unit_index", "instance_index"),
}


def _oracle(metadata, from_level, to_level, expr, unique_by=None):
    """One value per ``to_level`` row, by keyed group_by on the flat frame."""
    rows = metadata.rows_at(from_level)
    group = GROUP_KEY[(from_level, to_level)]
    marker = "track_index" if to_level == "track" else None
    frame = rows.filter(pl.col(marker) >= 0) if marker else rows
    if unique_by is not None and unique_by != from_level:
        keys = [*group, *GROUP_KEY[(from_level, unique_by)]]
        frame = frame.unique(subset=keys, keep="first", maintain_order=True)
    rolled = frame.group_by(group, maintain_order=True).agg(expr)
    name = expr.meta.output_name()
    answer = dict(zip(rolled.select(group).rows(), rolled[name].to_list(), strict=True))
    parents = metadata.rows_at(to_level).select(ROW_KEY[to_level]).rows()
    return [answer.get(key) for key in parents]


FIXTURES = {
    "diamond": [[2, 0, [1, -1]], [[0, 2], [1]]],
    "dense": [[3, 3], [2]],
    "untracked": [[[-1, -1], [0]], [[-1]]],
    "empty_sequence": [[0, 0], [2]],
}


@pytest.fixture(params=sorted(FIXTURES))
def tracking(request):
    metadata = Metadata(_mot_dataset(FIXTURES[request.param]))
    metadata._structure()
    return request.param, metadata


@pytest.mark.required
class TestAggMatchesAKeyedGroupBy:
    """The four acceptance cases from the spec, plus their generalisation."""

    @pytest.mark.parametrize(
        ("from_level", "to_level", "unique_by"),
        [
            ("instance", "unit", None),
            ("instance", "sequence", None),
            ("track", "sequence", None),
            ("unit", "sequence", None),
            ("instance", "track", "unit"),
            ("instance", "track", "instance"),
        ],
    )
    def test_counting_matches(self, tracking, from_level, to_level, unique_by):
        name, metadata = tracking
        expr = pl.len().alias("n")
        rolled = metadata.agg(from_level, to_level, expr, unique_by=unique_by)
        got = rolled.rows_at(to_level)["n"].to_list()
        assert got == _oracle(metadata, from_level, to_level, expr, unique_by), f"{name}: {from_level}->{to_level}"

    def test_a_mean_over_an_ancestor_column_matches(self, tracking):
        """Case 6: the fan-out question, answered by counting each frame once."""
        name, metadata = tracking
        expr = pl.col("time_s").mean().alias("mean_t")
        rolled = metadata.agg("instance", "track", expr, unique_by="unit")
        got = rolled.rows_at("track")["mean_t"].to_list()
        assert got == _oracle(metadata, "instance", "track", expr, "unit"), name

    def test_a_mean_over_the_levels_own_column_matches(self, tracking):
        name, metadata = tracking
        expr = pl.col("class_label").mean().alias("mean_label")
        rolled = metadata.agg("instance", "unit", expr)
        got = rolled.rows_at("unit")["mean_label"].to_list()
        assert got == _oracle(metadata, "instance", "unit", expr), name

    def test_several_expressions_in_one_call(self, tracking):
        name, metadata = tracking
        rolled = metadata.agg("instance", "unit", pl.len().alias("n"), pl.col("class_label").max().alias("worst"))
        assert rolled.rows_at("unit")["n"].to_list() == _oracle(metadata, "instance", "unit", pl.len().alias("n"))
        assert "worst" in rolled._store.frame("unit").columns, name


@pytest.mark.required
class TestAggSemantics:
    def test_a_group_with_no_rows_beneath_it_is_null_not_zero(self):
        """Nothing was measured there, which is not the same as measuring zero."""
        metadata = Metadata(_mot_dataset([[2, 0, 1]]))
        metadata._structure()
        counted = metadata.agg("instance", "unit", pl.len().alias("n"))
        assert counted.rows_at("unit")["n"].to_list() == [2, None, 1]

    def test_rows_with_no_ancestor_take_no_part(self):
        """An untracked detection belongs to no track, so no track counts it."""
        metadata = Metadata(_mot_dataset([[[0, -1, -1], [0]]]))
        metadata._structure()
        counted = metadata.agg("instance", "track", pl.len().alias("n"))
        assert counted.rows_at("track")["n"].to_list() == [2], "only the two tracked detections"

    def test_unique_by_counts_each_entity_once(self):
        """Two detections of one track in one frame are one frame, not two."""
        metadata = Metadata(_mot_dataset([[[7, 7], [7]]]))
        metadata._structure()
        assert metadata.agg("instance", "track", pl.len().alias("n")).rows_at("track")["n"].to_list() == [3]
        deduped = metadata.agg("instance", "track", pl.len().alias("n"), unique_by="unit")
        assert deduped.rows_at("track")["n"].to_list() == [2], "two frames, three detections"

    def test_an_untracked_row_is_not_deduplicated_against_another(self):
        """Rows sharing the no-ancestor marker are not the same entity."""
        metadata = Metadata(_mot_dataset([[[-1, -1]]]))
        metadata._structure()
        counted = metadata.agg("instance", "unit", pl.len().alias("n"), unique_by="track")
        assert counted.rows_at("unit")["n"].to_list() == [2], "both untracked detections counted"

    def test_the_source_is_not_mutated(self, tracking):
        name, metadata = tracking
        before = set(metadata._store.columns)
        metadata.agg("instance", "unit", pl.len().alias("n"))
        assert set(metadata._store.columns) == before, name

    def test_the_factor_records_where_it_was_rolled_up_from(self, tracking):
        name, metadata = tracking
        rolled = metadata.agg("instance", "unit", pl.len().alias("n")).at("unit")
        info = rolled.factor_info["n"]
        assert info.aggregated_from == "instance", name
        assert info.level == "unit", name

    def test_a_directly_measured_factor_records_no_source(self, tracking):
        name, metadata = tracking
        info = metadata.at("unit").factor_info
        assert all(entry.aggregated_from is None for entry in info.values()), name

    def test_a_name_collision_is_renamed_rather_than_overwritten(self, tracking):
        name, metadata = tracking
        once = metadata.agg("instance", "unit", pl.len().alias("n"))
        twice = once.agg("instance", "unit", pl.len().alias("n"))
        assert "n" in twice._store.columns, name
        assert "n_agg" in twice._store.columns, name


@pytest.mark.required
class TestAggRejections:
    def test_a_sibling_target_is_rejected(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="does not sit above"):
            metadata.agg("unit", "track", pl.len())

    def test_a_descendant_target_is_rejected(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="does not sit above"):
            metadata.agg("unit", "instance", pl.len())

    def test_an_ancestor_column_without_unique_by_names_both_options(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="unique_by") as excinfo:
            metadata.agg("instance", "track", pl.col("time_s").mean())
        assert "agg('unit'" in str(excinfo.value), "does not name aggregating at the defining level"

    def test_a_count_never_needs_unique_by(self, tracking):
        """pl.len() reads no column, so it is always a question about from_level."""
        _, metadata = tracking
        metadata.agg("instance", "track", pl.len().alias("n"))

    def test_unique_by_below_from_level_is_rejected(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="unique_by"):
            metadata.agg("unit", "sequence", pl.len(), unique_by="instance")

    def test_no_expression_is_rejected(self, tracking):
        _, metadata = tracking
        with pytest.raises(ValueError, match="at least one expression"):
            metadata.agg("instance", "unit")


@pytest.mark.required
def test_a_level_the_schema_declares_but_that_has_no_rows_is_rejected(tracking):
    """Aggregating into a level with nowhere to land would be a silent no-op."""
    import dataclasses

    from dataeval._metadata._aggregate import validate

    _, metadata = tracking
    metadata._structure()
    store = metadata._store
    gone = list(store.frames)[-1]
    thinned = dataclasses.replace(store, frames={k: v for k, v in store.frames.items() if k != gone})

    with pytest.raises(ValueError, match="agg needs rows at both levels"):
        validate(thinned, gone, list(store.frames)[0], [pl.len()], None)


@pytest.mark.required
class TestViaChangesWhichRowsTakePart:
    """A roll-up and the same roll-up through a branch are different questions.

    ``instance -> sequence`` reaches every detection through its frame; the same roll-up
    ``via="track"`` reaches only the ones a tracker linked. Both are correct, so the
    library states which is being asked rather than letting the graph decide silently.
    """

    @staticmethod
    def _store():
        """Seven detections over two sequences; detection 3 is untracked."""
        metadata = Metadata(_mot_dataset([[2, 0, [1, -1]], [[0, 2], [1]]]))
        metadata._structure()
        return metadata._store

    def test_the_two_routes_differ_by_exactly_the_untracked_count(self):
        store = self._store()
        untracked = int((store.link("instance", "track").positions() < 0).sum())
        assert untracked == 1, "fixture no longer has an untracked detection"
        every = aggregate(store, "instance", "sequence", [pl.len().alias("n")], None)
        tracked = aggregate(store, "instance", "sequence", [pl.len().alias("n")], None, via="track")
        assert sum(every.columns[0].to_list()) - sum(tracked.columns[0].to_list()) == untracked

    def test_rolling_up_in_two_hops_through_a_partial_branch_matches_going_via_it(self):
        """Non-associativity, stated as an equality: the two-hop answer is the ``via`` answer."""
        store = self._store()
        two_hop = aggregate(store, "instance", "track", [pl.len().alias("n")], None)
        per_track = aggregate(
            store.with_column("track", two_hop.columns[0]), "track", "sequence", [pl.col("n").sum().alias("n")], None
        )
        one_hop = aggregate(store, "instance", "sequence", [pl.len().alias("n")], None, via="track")
        assert per_track.columns[0].to_list() == one_hop.columns[0].to_list()

    def test_a_total_route_reaches_every_row(self):
        store = self._store()
        every = aggregate(store, "instance", "sequence", [pl.len().alias("n")], None)
        assert sum(every.columns[0].to_list()) == store.height("instance")

    def test_the_rows_that_took_no_part_are_reported(self, caplog):
        caplog.set_level(logging.INFO, logger="dataeval.metadata")
        aggregate(self._store(), "instance", "sequence", [pl.len().alias("n")], None, via="track")
        assert "1 of 7 'instance' row(s) took no part, having no 'sequence' ancestor" in caplog.text

    def test_a_full_roll_up_reports_nothing(self, caplog):
        caplog.set_level(logging.INFO, logger="dataeval.metadata")
        aggregate(self._store(), "instance", "sequence", [pl.len().alias("n")], None)
        assert "took no part" not in caplog.text

    def test_destinations_with_nothing_beneath_them_are_reported(self, caplog):
        caplog.set_level(logging.INFO, logger="dataeval.metadata")
        metadata = Metadata(_mot_dataset(FIXTURES["empty_sequence"]))
        metadata._structure()
        aggregate(metadata._store, "instance", "unit", [pl.len().alias("n")], None)
        assert "have nothing beneath them and answer null" in caplog.text

    def test_a_route_that_does_not_exist_is_rejected(self):
        metadata = Metadata(_mot_dataset([[2, 0, [1, -1]], [[0, 2], [1]]]))
        metadata._structure()
        with pytest.raises(ValueError, match="No route from 'instance' to 'unit' passes through 'track'"):
            validate(metadata._store, "instance", "unit", [pl.len()], None, via="track")


@pytest.mark.required
class TestAnExpressionNamesOneOutput:
    """Results are read back one per expression, which is what pairs one with its coverage."""

    @pytest.mark.parametrize("expr", ["all", "regex"])
    def test_a_multi_output_selector_is_refused_by_name(self, expr):
        """It reached polars' own `output_name()` and came back as a ComputeError about root
        column names, mentioning neither agg nor the level graph."""
        metadata = Metadata(_mot_dataset([[2, 1], [1]]))
        selector = pl.all().mean() if expr == "all" else pl.col("^time.*$").mean()
        with pytest.raises(ValueError, match="has to name one output column"):
            metadata.agg("instance", "unit", selector)

    def test_a_named_expression_still_works(self):
        metadata = Metadata(_mot_dataset([[2, 1], [1]]))
        rolled = metadata.agg("instance", "unit", pl.len().alias("n_det"))
        assert rolled.at("unit").rows_at("unit")["n_det"].to_list() == [2, 1, 1]


@pytest.mark.required
class TestARollUpIsRecordedProvenance:
    """A roll-up is a declaration, not just the column it produced. Recording it is what
    lets a pipeline be rebuilt over the next dataset rather than described after the fact."""

    @staticmethod
    def _two_levels():
        dataset = _mot_dataset([[2, 1], [1]])
        metadata = Metadata(dataset)
        metadata._structure()
        metadata.add_factors({"area": np.arange(float(metadata.level_counts["instance"]))}, level="instance")
        rolled = metadata.aggregate("area", level="unit", how="mean")
        return dataset, rolled.aggregate("area_mean", level="sequence", how="max")

    def test_each_output_is_keyed_on_the_column_it_produced(self):
        _, two = self._two_levels()
        assert list(two._aggregations) == ["area_mean", "area_mean_max"]

    def test_the_order_is_the_order_they_must_replay_in(self):
        """The second reads the column the first wrote, so it is only answerable after it."""
        _, two = self._two_levels()
        assert [a.how for a in two._aggregations.values()] == ["mean", "max"]

    def test_every_column_a_roll_up_produced_has_an_entry(self):
        """Keyed on the *output* name rather than the input, because that is what a replay
        has to rebuild. Running the same roll-up twice writes a second column under the
        collision name `aggregate` already gives it, and that column gets its own entry —
        so a replay reproduces exactly the columns the metadata carries, duplicates and all.
        """
        dataset = _mot_dataset([[2, 1], [1]])
        once = Metadata(dataset).aggregate("width", level="sequence", how="mean")
        twice = once.aggregate("width", level="sequence", how="mean")

        assert list(twice._aggregations) == ["width_mean", "width_mean_agg"]
        assert set(twice._aggregations) <= set(twice.factor_names)

    def test_both_levels_survive_the_archive(self, tmp_path):
        dataset, two = self._two_levels()
        two.save(tmp_path / "m.dem")

        back = Metadata.load(tmp_path / "m.dem", dataset)
        assert list(back._aggregations) == ["area_mean", "area_mean_max"]
        assert back.at("sequence").rows_at("sequence")["area_mean_max"].to_list() == [2.0, 3.0]

    def test_the_declaration_comes_back_whole(self, tmp_path):
        """Not just `how` and `via`, which the generated name already carries."""
        dataset = _mot_dataset([[2, 1], [1]])
        declared = Aggregator("mean", "unit", "sequence", ("width",), min_coverage=0.5)
        Metadata(dataset).aggregate(declared).save(tmp_path / "m.dem")

        (back,) = Metadata.load(tmp_path / "m.dem", dataset)._aggregations.values()
        assert back.min_coverage == 0.5
        assert (back.how, back.source, back.target, back.factors) == ("mean", "unit", "sequence", ("width",))

    def test_a_fitted_tolerance_is_reapplied_rather_than_refitted(self, tmp_path):
        """The number was measured off *this* dataset, which is why it is worth recording."""
        dataset = _mot_dataset([[1] * 4])
        metadata = Metadata(dataset)
        metadata._structure()
        metadata.add_factors({"b": np.array([0.10, 0.11, 0.12, 0.90])}, level="unit")
        declared = Aggregator("longest_run", "unit", "sequence", ("b",), options={"tolerance": ("iqr", (None, 1.5))})
        rolled = metadata.aggregate(declared)
        (fitted,) = rolled._aggregations.values()
        rolled.save(tmp_path / "m.dem")

        (back,) = Metadata.load(tmp_path / "m.dem", dataset)._aggregations.values()
        assert isinstance(back.options["tolerance"], float)
        assert back == fitted

    def test_it_rebuilds_over_the_next_dataset(self):
        rolled = Metadata(_mot_dataset([[2, 1], [1]])).aggregate("width", level="sequence", how="mean")
        following = rolled.new(_mot_dataset([[3, 2], [2], [1]]))

        assert following.at("sequence").rows_at("sequence")["width_mean"].to_list() == [4.0, 4.0, 4.0]

    def test_a_dataset_that_cannot_answer_one_says_so(self, caplog):
        """Rather than refusing to build, which would make `new()` unusable wherever the
        factor was one the caller added rather than one the walk found."""
        dataset, two = self._two_levels()
        with caplog.at_level(logging.WARNING, logger="dataeval.metadata"):
            following = two.new(_mot_dataset([[2], [1]]))
            following._structure()

        assert "Not replaying" in caplog.text
        assert "area_mean" not in following.factor_names

    def test_an_expression_roll_up_records_nothing(self):
        """`agg` takes arbitrary polars expressions, which have no serializable form."""
        rolled = Metadata(_mot_dataset([[2, 1], [1]])).agg("instance", "unit", pl.len().alias("n_det"))
        assert rolled._aggregations == {}

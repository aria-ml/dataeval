"""Rolling rows up into a level above them — FE-5 issue 5.3.

The oracle here is a hand-written polars ``group_by`` over the *flat* frame, keyed by the
reserved columns rather than by positions. That is the implementation ``agg`` exists to
avoid — it hashes a key per row — so agreeing with it is the statement that the positional
grouping loses nothing.
"""

import polars as pl
import pytest

from dataeval import Metadata
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

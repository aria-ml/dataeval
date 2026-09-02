"""Saving a Metadata to one file and reading it back — FE-7 issue 7.1.

The hazard a serialization test has to chase is not "does it crash". It is that a file
restores into an instance that is *shaped* right and *answers* wrong: an edge renumbered
onto the wrong parent, a level whose frame came back in a different position in the
mapping, a no-ancestor marker read as a real row. All of those produce a working object.

So the central assertion here is not on any one field but on the whole instance:
:class:`TestRoundTripIsFaithful` compares a restored instance against a freshly structured
one attribute by attribute, over every task, and :class:`TestFieldCoverage` is what makes
that comparison stay honest as fields are added.
"""

import io
import json
import zipfile

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval._metadata._links import GatherLink, RunLengthLink
from dataeval._metadata._serialize import FORMAT_VERSION
from dataeval._metadata._structurers import FactorsStructurer
from dataeval.exceptions import MetadataFormatError, NotFittedError
from dataeval.types import SourceIndex
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget
from tests.metadata.test_structurers import _mot_dataset


def _ic_dataset(n=6):
    """Image classification: one label per image, one metadata factor per image."""
    return MockDataset(
        np.zeros((n, 3, 4, 4)),
        np.eye(n, 3)[np.arange(n) % 3],
        [{"id": i, "weather": ["clear", "rain"][i % 2], "temp": float(i)} for i in range(n)],
    )


def _od_dataset(n=5):
    """Object detection: a varying number of boxes per image, so the edge is a real fan-out."""
    targets = [
        ObjectDetectionTarget(
            np.tile(np.array([[1.0, 2.0, 3.0, 4.0]]), (1 + i % 3, 1)),
            np.arange(1 + i % 3) % 3,
            np.full(1 + i % 3, 0.5),
        )
        for i in range(n)
    ]
    return MockDataset(
        np.zeros((n, 3, 4, 4)),
        targets,
        [{"id": i, "weather": ["clear", "rain"][i % 2], "temp": float(i)} for i in range(n)],
    )


def _mot_with_untracked():
    """Tracking, including a detection no tracker linked.

    The ``-1`` in that row's ``instance -> track`` edge is the marker the whole gather
    discipline is built around, so a round trip that loses it is the failure this dataset
    exists to catch.
    """
    return _mot_dataset([[[0, 1], [0, -1]], [[2]]], [{"id": 0, "site": "north"}, {"id": 1, "site": "south"}])


def _factors_metadata():
    """A factors-only instance: one level, no dataset, nothing to bind."""
    return Metadata.from_factors(
        {"brightness": np.array([0.1, 0.4, 0.9, 0.2]), "kind": np.array(["a", "b", "a", "b"])},
        np.array([0, 1, 0, 1]),
        index2label={0: "cat", 1: "dog"},
    )


def _source_index_metadata():
    """A factors-only instance spanning two levels, built from a source index."""
    return Metadata.from_factors(
        {"score": np.array([1.0, 2.0, 3.0, 4.0, 5.0])},
        np.array([0, 1, 0]),
        source_index=[
            SourceIndex(0, None),
            SourceIndex(0, 0),
            SourceIndex(0, 1),
            SourceIndex(1, None),
            SourceIndex(1, 0),
        ],
    )


# Every task, as (name, build the metadata, the dataset to rebind on load or None).
CASES = {
    "ic": lambda: (lambda ds: (Metadata(ds), ds))(_ic_dataset()),
    "od": lambda: (lambda ds: (Metadata(ds), ds))(_od_dataset()),
    "mot": lambda: (lambda ds: (Metadata(ds), ds))(_mot_with_untracked()),
    "factors": lambda: (_factors_metadata(), None),
    "source_index": lambda: (_source_index_metadata(), None),
}


@pytest.fixture(params=sorted(CASES))
def case(request):
    """One built Metadata plus the dataset (or None) that reloading it should bind."""
    return CASES[request.param]()


def _round_trip(md, dataset, tmp_path, name="m.dem", **config):
    """Save and reload, returning the reloaded instance."""
    path = tmp_path / name
    md.save(path)
    return Metadata.load(path, dataset, **config)


def _values_only(frame):
    """The frame without its binned/digitized companions.

    Binning is configuration, so a companion column is not something a file promises to
    reproduce. Comparing the values a companion is derived from is the strongest claim
    that survives that, and the one worth making.
    """
    return frame.drop([name for name in frame.columns if name.endswith(("↕", "#"))])


# Fields a restored instance is *not* expected to match a freshly structured one on, and
# why. Anything not named here has to agree, which is what makes a newly added field fail
# this comparison rather than pass it unnoticed.
EXPECTED_DIFFERENCES = {
    # Structuring produces this; nothing reads it, and FE-7 retires it. The store holds
    # every question it used to answer.
    "_layout",
    # Deliberately not written — unbounded in size, arbitrary in type.
    "_raw",
    "_raw_omitted",
    # Identity differs (a distinct LevelStore, a distinct Structurer instance); compared
    # by value in TestRoundTripIsFaithful instead.
    "_level_store",
    "_structurer",
    # Memoized derivations, rebuilt on demand from the fields above.
    "_flat",
    # Restored *fuller* than it was written, deliberately. The writer's ``_encoding`` holds
    # only what a caller declared; the archive carries the encodings that were actually
    # applied, so that a restored instance reproduces its codes rather than re-deriving
    # them and loses neither an accepted placement nor a grown vocabulary. Reproduction is
    # asserted by comparing factor_data, which TestRoundTripIsFaithful already does.
    "_encoding",
}


@pytest.mark.required
class TestRoundTripIsFaithful:
    """A restored instance answers every question the original does."""

    def test_public_answers_match(self, case, tmp_path):
        original, dataset = case
        # Read everything first, so the original is fully structured *and* binned — a
        # saved file must reproduce a worked-on instance, not only a fresh one.
        _ = original.factor_data
        back = _round_trip(original, dataset, tmp_path)

        assert back.levels == original.levels
        assert back.level_counts == original.level_counts
        assert sorted(back.factor_names) == sorted(original.factor_names)
        assert back.index2label == original.index2label
        assert back.item_count == original.item_count
        assert back.dropped_factors == original.dropped_factors
        assert np.array_equal(back.class_labels, original.class_labels)
        assert np.array_equal(back.item_indices, original.item_indices)
        assert np.array_equal(back.factor_data, original.factor_data)
        assert list(back.is_discrete) == list(original.is_discrete)

    def test_the_flat_frame_matches_column_for_column(self, case, tmp_path):
        """Including dtypes and column order, which a parquet round trip could quietly change."""
        original, dataset = case
        back = _round_trip(original, dataset, tmp_path)

        assert back.dataframe.columns == original.dataframe.columns
        assert back.dataframe.schema == original.dataframe.schema
        assert back.dataframe.equals(original.dataframe)

    def test_every_level_reads_the_same_rows(self, case, tmp_path):
        """Each level on its own, not only the flattening of all of them.

        The flat frame is a concatenation, so two levels whose frames came back swapped
        would still flatten to the same rows in the same order for many datasets. Reading
        each level separately is what tells them apart.
        """
        original, dataset = case
        back = _round_trip(original, dataset, tmp_path)

        assert list(back._store.frames) == list(original._store.frames)
        for level in original.levels:
            assert back.rows_at(level).equals(original.rows_at(level)), level

    def test_every_edge_points_where_it_did(self, case, tmp_path):
        """Positions, not shapes: an edge renumbered onto the wrong parent has the right shape."""
        original, dataset = case
        back = _round_trip(original, dataset, tmp_path)

        schema = original._levels
        for level in original.levels:
            for ancestor in schema.ancestors(level):
                assert np.array_equal(
                    back._store.link(level, ancestor).positions(),
                    original._store.link(level, ancestor).positions(),
                ), (level, ancestor)

    def test_propagated_factors_still_gather_correctly(self, case, tmp_path):
        """A per-item factor read from the finest level, which is a gather along the edges."""
        original, dataset = case
        finest = original.levels[-1]
        back = _round_trip(original, dataset, tmp_path)

        for name in original.factor_names:
            assert back._store.column(finest, name).equals(original._store.column(finest, name)), name


@pytest.mark.required
class TestFieldCoverage:
    """The comparison above stays honest only if nothing new slips past it."""

    def test_round_trip_restores_every_field_adopt_sets(self, tmp_path):
        """Compare a restored instance to a fresh one field by field.

        ``_serialize._adopt_manifest`` mirrors ``Metadata._adopt`` by hand rather than
        sharing its field list, so a field added to one and not the other would restore as
        its unstructured default — an empty set, a False flag — and every shape assertion
        would still pass. This is the test that fails instead. A genuinely new field
        belongs either in ``_adopt_manifest`` or in ``EXPECTED_DIFFERENCES`` with a reason.

        Both sides are binned before comparing, since binning is configuration rather than
        data and is deliberately absent from the file. That makes the assertion stronger
        rather than weaker: re-binning a restored instance has to reach the same
        :class:`FactorInfo` for every factor, which is the statement that the values the
        bin edges were derived from came back unchanged.
        """
        dataset = _od_dataset()
        original = Metadata(dataset)
        _ = original.factor_data
        back = _round_trip(original, dataset, tmp_path)
        _ = back.factor_data

        assert set(vars(back)) == set(vars(original)), "restoring produced a different set of fields"
        for name in sorted(set(vars(original)) - EXPECTED_DIFFERENCES):
            mine, theirs = getattr(back, name), getattr(original, name)
            if isinstance(theirs, pl.DataFrame):
                assert mine.equals(theirs), name
            elif isinstance(theirs, np.ndarray):
                assert np.array_equal(mine, theirs), name
            else:
                assert mine == theirs, name

    def test_the_store_matches_field_by_field(self, tmp_path):
        """Same, one level down: the store is compared by value, so its fields must line up."""
        dataset = _od_dataset()
        original = Metadata(dataset)
        back = _round_trip(original, dataset, tmp_path)

        assert back._store.column_order == original._store.column_order
        assert back._store.propagating == original._store.propagating
        assert list(back._store.schema) == list(original._store.schema)
        assert back._store.links.keys() == original._store.links.keys()
        assert back._store.counts == original._store.counts


@pytest.mark.required
class TestStructureThatIsEasyToLoseQuietly:
    """The specific shapes whose loss changes an answer without changing a shape."""

    def test_the_no_ancestor_marker_survives(self, tmp_path):
        """An untracked detection must come back with no track, not with the last one."""
        dataset = _mot_with_untracked()
        original = Metadata(dataset)
        # ``_store`` is a plain field, so reading it does not structure. Every test that
        # reaches into the store has to ask a public question first.
        assert original.levels == ("sequence", "unit", "track", "instance")
        positions = original._store.link("instance", "track").positions()
        assert (positions < 0).any(), "fixture no longer holds an untracked detection"

        back = _round_trip(original, dataset, tmp_path)
        assert np.array_equal(back._store.link("instance", "track").positions(), positions)
        assert back._store.partial_ancestry("track", "instance")

    def test_the_link_representation_is_re_picked_not_stored(self, tmp_path):
        """A grouped edge comes back grouped, and a marked one comes back general.

        The file stores what an edge *means* — one parent position per child — and
        ``LinkIndex.of`` re-derives the representation. Asserting the classes is how this
        pins that the file is not quietly recording an internal detail it would then have
        to keep supporting.
        """
        dataset = _mot_with_untracked()
        back = _round_trip(Metadata(dataset), dataset, tmp_path)

        assert isinstance(back._store.links[("unit", "sequence")], RunLengthLink)
        assert isinstance(back._store.links[("instance", "track")], GatherLink)

    def test_a_fixed_width_box_column_keeps_its_dtype(self, tmp_path):
        """``box`` is an ``Array(_, 4)``; parquet could return it as a ``List``."""
        dataset = _od_dataset()
        original = Metadata(dataset)
        written = original.rows_at("instance").schema["box"]
        assert isinstance(written, pl.Array)
        assert written.size == 4

        back = _round_trip(original, dataset, tmp_path)
        assert back.rows_at("instance").schema["box"] == written

    def test_a_level_with_no_rows_comes_back_with_no_rows(self, tmp_path):
        """Not merely absent: a level with an empty frame is a different thing from one with none."""
        dataset = _mot_dataset([[0], [0]])
        original = Metadata(dataset)
        assert original.level_counts["instance"] == 0
        assert "instance" in original._store.frames

        back = _round_trip(original, dataset, tmp_path)
        assert "instance" in back._store.frames
        assert back._store.height("instance") == 0
        assert back.rows_at("instance").schema == original.rows_at("instance").schema

    def test_an_empty_dataset_round_trips(self, tmp_path):
        """Its item level has no columns at all, which no parquet file can carry."""
        original = Metadata(MockDataset(np.zeros((0, 3, 4, 4)), np.zeros((0, 3))), task="IC")
        assert original.dataframe.height == 0
        assert original._store.frame("unit").width == 0

        back = _round_trip(original, None, tmp_path)
        assert back.level_counts == original.level_counts
        assert back._store.frame("unit").width == 0
        assert back.dataframe.equals(original.dataframe)

    def test_a_columnless_frame_is_described_rather_than_written(self, tmp_path):
        """No member for it, so a reader never asks parquet to hold nothing."""
        path = tmp_path / "empty.dem"
        Metadata(MockDataset(np.zeros((0, 3, 4, 4)), np.zeros((0, 3))), task="IC").save(path)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()

        assert "frames/unit.parquet" not in names
        assert "frames/instance.parquet" in names

    def test_dropped_factors_survive(self, tmp_path):
        """What structuring refused to keep, and why, is part of what the metadata says."""
        md = Metadata.from_factors({"ok": np.array([1.0, 2.0]), "vector": np.zeros((2, 3))})
        assert md.dropped_factors == {"vector": ["multi_dimensional"]}

        back = _round_trip(md, None, tmp_path)
        assert back.dropped_factors == {"vector": ["multi_dimensional"]}


@pytest.mark.required
class TestDerivedStateSurvives:
    """Rows produced by the library rather than read off the dataset."""

    def test_a_filtered_instance_reloads_as_filtered(self, tmp_path):
        """Losing the flag would silently re-arm evaluators that refuse filtered metadata."""
        dataset = _od_dataset()
        filtered = Metadata(dataset).where(pl.col("weather") == "clear", level="unit")
        assert filtered.is_filtered

        back = _round_trip(filtered, dataset, tmp_path)
        assert back.is_filtered
        assert back.level_counts == filtered.level_counts
        assert _values_only(back.dataframe).equals(_values_only(filtered.dataframe))

    def test_a_reloaded_filter_bins_over_the_rows_that_survived(self, tmp_path):
        """The bin edges are the reader's, and after a filter the two sides can differ.

        A filtered instance carries the ``FactorInfo`` computed before the cut, so its
        edges describe the whole dataset. A reloaded one has no such history and bins over
        the rows in front of it. Neither is wrong, and the file cannot hold both, which is
        the concrete meaning of binning being configuration: what round-trips is the
        values, and the edges are recomputed from whatever is asked for at load.
        """
        dataset = _od_dataset(12)
        filtered = Metadata(dataset, continuous_factor_bins={"temp": 3}).where(
            pl.col("weather") == "clear", level="unit"
        )
        back = _round_trip(filtered, dataset, tmp_path, continuous_factor_bins={"temp": 3})

        column = filtered.factor_names.index("temp")
        assert np.array_equal(
            _values_only(back.dataframe)["temp"].to_numpy(),
            _values_only(filtered.dataframe)["temp"].to_numpy(),
        )
        assert back.factor_data[:, back.factor_names.index("temp")].shape == filtered.factor_data[:, column].shape

    def test_a_filtered_instance_is_still_refused_by_the_evaluators(self, tmp_path):
        from dataeval._helpers import reject_filtered_metadata

        dataset = _od_dataset()
        filtered = Metadata(dataset).where(pl.col("weather") == "clear", level="unit")
        back = _round_trip(filtered, dataset, tmp_path)

        with pytest.raises(ValueError, match="filtered"):
            reject_filtered_metadata(back, "Coverage")

    def test_the_item_count_of_a_filtered_instance_is_its_own(self, tmp_path):
        """A filter rewrites ``_count`` to the surviving items; a reload must not restore the dataset's."""
        dataset = _od_dataset()
        filtered = Metadata(dataset).where(pl.col("weather") == "clear", level="unit")
        assert filtered.item_count < len(dataset)

        path = tmp_path / "filtered.dem"
        filtered.save(path)
        # Bound without the length check, which compares against the whole dataset and is
        # right to: these rows are no longer one per dataset item.
        back = Metadata.load(path)
        assert back.item_count == filtered.item_count

    def test_aggregated_factors_and_their_source_level_survive(self, tmp_path):
        dataset = _mot_with_untracked()
        rolled = Metadata(dataset).agg("instance", "track", pl.len().alias("n_detections"))
        assert rolled._aggregated_from == {"n_detections": "instance"}

        back = _round_trip(rolled, dataset, tmp_path)
        assert back._aggregated_from == {"n_detections": "instance"}
        assert back._store.column("track", "n_detections").equals(rolled._store.column("track", "n_detections"))

    def test_keyed_factors_survive(self, tmp_path):
        """Values placed by matching a key column are ordinary columns once placed."""
        dataset = _mot_dataset([[[5, 3], [5]]])
        md = Metadata(dataset)
        ids = md.rows_at("track")["track_id"].to_list()
        md.add_factors({"track_id": ids[::-1], "speed": [9.0, 1.0][: len(ids)][::-1]}, level="track", key="track_id")

        back = _round_trip(md, dataset, tmp_path)
        assert back._store.column("track", "speed").equals(md._store.column("track", "speed"))

    def test_factors_added_after_structuring_survive(self, tmp_path):
        dataset = _od_dataset()
        md = Metadata(dataset)
        md.add_factors({"sharpness": np.arange(md.level_counts["instance"], dtype=np.float64)}, level="instance")

        back = _round_trip(md, dataset, tmp_path)
        assert "sharpness" in back.factor_names
        assert back._store.column("instance", "sharpness").equals(md._store.column("instance", "sharpness"))


@pytest.mark.required
class TestBinningIsConfigurationNotData:
    """One file, read back with whatever bins the reader wants."""

    def test_companion_columns_are_not_written(self, tmp_path):
        dataset = _od_dataset()
        md = Metadata(dataset)
        _ = md.factor_data
        assert any(name.endswith(("↕", "#")) for name in md._store.columns)

        path = tmp_path / "m.dem"
        md.save(path)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        assert not [name for name in manifest["column_order"] if name.endswith(("↕", "#"))]

    def test_the_reader_s_bins_are_applied_not_the_writer_s(self, tmp_path):
        dataset = _od_dataset(12)
        md = Metadata(dataset, continuous_factor_bins={"temp": 2})
        _ = md.factor_data
        column = md.factor_names.index("temp")
        assert len(np.unique(md.factor_data[:, column])) <= 2

        back = _round_trip(md, dataset, tmp_path, continuous_factor_bins={"temp": 4})
        assert len(np.unique(back.factor_data[:, back.factor_names.index("temp")])) > 2

    def test_saving_a_binned_instance_does_not_disturb_it(self, tmp_path):
        """Stripping happens on a copy of the store, so the live instance keeps its columns."""
        dataset = _od_dataset()
        md = Metadata(dataset)
        before = md.factor_data.copy()
        md.save(tmp_path / "m.dem")

        assert np.array_equal(md.factor_data, before)
        assert any(name.endswith(("↕", "#")) for name in md._store.columns)


@pytest.mark.required
class TestReadingConfiguration:
    """What the caller asks of the rows is theirs, not the file's."""

    def test_exclude_view_and_inherited_are_the_reader_s(self, tmp_path):
        dataset = _od_dataset()
        md = Metadata(dataset, exclude="temp")
        back = _round_trip(md, dataset, tmp_path, view="unit")

        assert back.view == "unit"
        assert back.exclude == set()
        assert "temp" in back.factor_names

    def test_loading_without_a_dataset_gives_an_unbound_instance(self, tmp_path):
        dataset = _od_dataset()
        md = Metadata(dataset)
        path = tmp_path / "m.dem"
        md.save(path)

        back = Metadata.load(path)
        assert back.dataframe.equals(md.dataframe)
        assert not back.is_bound

    def test_an_unbound_loaded_instance_can_be_rebound_and_restructures(self, tmp_path):
        """``bind`` re-structures from scratch, so it must clear the loaded rows rather than mix them."""
        dataset = _od_dataset()
        path = tmp_path / "m.dem"
        Metadata(dataset).save(path)

        back = Metadata.load(path)
        back.bind(_od_dataset(9))
        assert back.item_count == 9
        assert back.level_counts["unit"] == 9


@pytest.mark.required
class TestRawIsNotWritten:
    def test_raw_raises_rather_than_answering_empty(self, tmp_path):
        dataset = _od_dataset()
        back = _round_trip(Metadata(dataset), dataset, tmp_path)

        with pytest.raises(ValueError, match="does not hold the per-item metadata"):
            _ = back.raw

    def test_a_view_of_a_loaded_instance_says_the_same(self, tmp_path):
        dataset = _od_dataset()
        back = _round_trip(Metadata(dataset), dataset, tmp_path)

        with pytest.raises(ValueError, match="does not hold the per-item metadata"):
            _ = back.at("unit").raw

    def test_a_freshly_built_instance_still_answers(self):
        md = Metadata(_od_dataset())
        assert len(md.raw) == 5


@pytest.mark.required
class TestRefusals:
    """Every way a file can be wrong, and the one error a caching caller catches."""

    def test_a_file_that_is_not_an_archive(self, tmp_path):
        path = tmp_path / "junk.dem"
        path.write_bytes(b"not a zip file at all")

        with pytest.raises(MetadataFormatError, match="Not a dataeval metadata file"):
            Metadata.load(path)

    def test_an_archive_with_no_manifest(self, tmp_path):
        path = tmp_path / "empty.dem"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("nothing.txt", "hello")

        with pytest.raises(MetadataFormatError, match="Not a dataeval metadata file"):
            Metadata.load(path)

    def test_a_missing_file_is_not_a_format_error(self, tmp_path):
        """A path that is not there is the caller's mistake, not a stale cache."""
        with pytest.raises(FileNotFoundError):
            Metadata.load(tmp_path / "absent.dem")

    def test_another_format_version(self, tmp_path):
        path = _retagged(tmp_path, _od_dataset(), {"format_version": FORMAT_VERSION + 1})

        with pytest.raises(MetadataFormatError, match=f"reads format {FORMAT_VERSION}"):
            Metadata.load(path)

    def test_a_task_this_version_does_not_structure(self, tmp_path):
        path = _retagged(tmp_path, _od_dataset(), {"task": "PANOPTIC"})

        with pytest.raises(MetadataFormatError, match="does not structure"):
            Metadata.load(path)

    def test_a_level_graph_this_version_no_longer_declares(self, tmp_path):
        """The check with teeth: same task, different levels, and every read would succeed."""
        path = _retagged(tmp_path, _od_dataset(), {"levels": ["unit", "track", "instance"]})

        with pytest.raises(MetadataFormatError, match="was written against levels"):
            Metadata.load(path)

    def test_a_different_set_of_edges(self, tmp_path):
        path = _retagged(tmp_path, _od_dataset(), {"edges": []})

        with pytest.raises(MetadataFormatError, match="was written against levels"):
            Metadata.load(path)

    def test_a_truncated_frame(self, tmp_path):
        path = _retagged(tmp_path, _od_dataset(), {"frames": [{"level": "unit", "height": 999, "columns": 4}]})

        with pytest.raises(MetadataFormatError, match="the file is incomplete"):
            Metadata.load(path)

    def test_rows_recorded_in_a_frame_with_no_columns(self, tmp_path):
        """A columnless frame is written as an entry and no member, so the entry must cohere."""
        path = _retagged(tmp_path, _od_dataset(), {"frames": [{"level": "unit", "height": 3, "columns": 0}]})

        with pytest.raises(MetadataFormatError, match="no columns"):
            Metadata.load(path)

    def test_a_missing_frame_member(self, tmp_path):
        source = tmp_path / "m.dem"
        Metadata(_od_dataset()).save(source)
        path = tmp_path / "gutted.dem"
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(path, "w") as new:
            for name in old.namelist():
                if name != "frames/instance.parquet":
                    new.writestr(name, old.read(name))

        with pytest.raises(MetadataFormatError, match="missing or cannot read"):
            Metadata.load(path)

    def test_an_edge_naming_a_row_its_parent_does_not_have(self, tmp_path):
        """Corruption that is in range for the file and out of range for the level."""
        source = tmp_path / "m.dem"
        Metadata(_od_dataset()).save(source)
        path = tmp_path / "bad_edge.dem"
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(path, "w") as new:
            for name in old.namelist():
                if name == "links/instance.parquet":
                    shifted = pl.read_parquet(old.read(name)).with_columns(pl.col("unit") + 100)
                    buffer = io.BytesIO()
                    shifted.write_parquet(buffer)
                    new.writestr(name, buffer.getvalue())
                else:
                    new.writestr(name, old.read(name))

        with pytest.raises(MetadataFormatError, match="does not fit its levels"):
            Metadata.load(path)

    def test_a_dataset_of_the_wrong_length(self, tmp_path):
        path = tmp_path / "m.dem"
        Metadata(_od_dataset(5)).save(path)

        with pytest.raises(ValueError, match="saved for a dataset of 5 item"):
            Metadata.load(path, _od_dataset(9))

    def test_saving_with_nothing_to_save(self, tmp_path):
        with pytest.raises(NotFittedError):
            Metadata().save(tmp_path / "m.dem")


def _retagged(tmp_path, dataset, changes):
    """Save ``dataset``'s metadata, then rewrite its manifest with ``changes`` applied."""
    source = tmp_path / "source.dem"
    Metadata(dataset).save(source)
    path = tmp_path / "retagged.dem"
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(path, "w") as new:
        for name in old.namelist():
            if name == "manifest.json":
                new.writestr(name, json.dumps(json.loads(old.read(name)) | changes))
            else:
                new.writestr(name, old.read(name))
    return path


@pytest.mark.required
class TestWriting:
    def test_the_previous_file_survives_a_failed_write(self, tmp_path, monkeypatch):
        """A cache that can be left half-written is a cache that is permanently broken."""
        dataset = _od_dataset()
        path = tmp_path / "m.dem"
        Metadata(dataset).save(path)
        good = path.read_bytes()

        import dataeval._metadata._serialize as serialize

        def explode(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(serialize, "_write_members", explode)
        with pytest.raises(OSError, match="disk full"):
            Metadata(dataset).save(path)

        assert path.read_bytes() == good
        assert not [child for child in tmp_path.iterdir() if child.name.startswith(".")]

    def test_missing_parent_directories_are_created(self, tmp_path):
        path = tmp_path / "a" / "b" / "m.dem"
        Metadata(_od_dataset()).save(path)
        assert path.exists()

    def test_saving_over_an_existing_file_replaces_it(self, tmp_path):
        path = tmp_path / "m.dem"
        Metadata(_od_dataset(5)).save(path)
        Metadata(_od_dataset(9)).save(path)

        assert Metadata.load(path).level_counts["unit"] == 9

    def test_saving_structures_an_unstructured_instance(self, tmp_path):
        """Like ``Embeddings.save``, which computes before it writes."""
        dataset = _od_dataset()
        md = Metadata(dataset)
        assert not md._is_structured

        md.save(tmp_path / "m.dem")
        assert Metadata.load(tmp_path / "m.dem", dataset).level_counts["unit"] == 5

    def test_a_path_may_be_a_string(self, tmp_path):
        dataset = _od_dataset()
        Metadata(dataset).save(str(tmp_path / "m.dem"))
        assert Metadata.load(str(tmp_path / "m.dem"), dataset).item_count == 5

    def test_the_archive_holds_only_data(self, tmp_path):
        """No pickles, no code: one JSON manifest and parquet frames."""
        Metadata(_od_dataset()).save(tmp_path / "m.dem")
        with zipfile.ZipFile(tmp_path / "m.dem") as archive:
            names = archive.namelist()

        assert names[0] == "manifest.json"
        assert all(name == "manifest.json" or name.endswith(".parquet") for name in names)


@pytest.mark.required
class TestFactorsStructurerForShape:
    """Rebuilding the dataset-free declaration from the shape it produced."""

    def test_a_single_level_shape(self):
        structurer = FactorsStructurer.for_shape("unit", "unit")
        assert list(structurer.levels) == ["unit"]
        assert structurer.item_level == structurer.label_level == "unit"

    def test_the_two_level_shape_matches_what_a_source_index_builds(self):
        built = _source_index_metadata()._structurer
        restored = FactorsStructurer.for_shape("unit", "instance")

        assert list(restored.levels) == list(built.levels)
        assert restored.item_level == built.item_level
        assert restored.label_level == built.label_level
        assert restored.multi_target == built.multi_target

    def test_a_shape_it_never_produces(self):
        with pytest.raises(ValueError, match="is neither"):
            FactorsStructurer.for_shape("sequence", "instance")


@pytest.mark.required
class TestAStructuringPolicySurvivesTheRoundTrip:
    """``partial_factors`` decides what the rows *are*, so it is written like ``strict``."""

    @staticmethod
    def _partly_declared():
        from tests.metadata.test_structurers import _TRACKED, _undeclared

        return _undeclared(_mot_dataset(_TRACKED), 0, 1, "time_s")

    def test_a_restored_instance_reports_the_policy_it_was_structured_under(self, tmp_path):
        """Reporting the default described a walk that did not happen."""
        dataset = self._partly_declared()
        back = _round_trip(Metadata(dataset, partial_factors=True), dataset, tmp_path)
        assert back.partial_factors is True

    def test_a_new_dataset_from_a_restored_one_is_structured_the_same_way(self, tmp_path):
        """Otherwise the next dataset silently drops the factors this one kept."""
        dataset = self._partly_declared()
        back = _round_trip(Metadata(dataset, partial_factors=True), dataset, tmp_path)
        assert back.new(dataset).partial_factors is True
        assert "time_s" in back.new(dataset).dataframe.columns

    def test_the_default_is_still_the_default(self, tmp_path):
        dataset = _mot_dataset([[1], [1]])
        assert _round_trip(Metadata(dataset), dataset, tmp_path).partial_factors is False

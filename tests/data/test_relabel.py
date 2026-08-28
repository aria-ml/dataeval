import warnings
from typing import cast

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata, Ontology
from dataeval.core import label_alignment
from dataeval.data import Operation, Relabel, View, merge_datasets
from dataeval.data._relabel import _label_remap, _own_class_scores
from dataeval.exceptions import DeprecatedWarning, OntologyError
from dataeval.protocols import ObjectDetectionTarget
from dataeval.types import OntologyConcept


def target_ontology() -> Ontology:
    return Ontology.from_hierarchy({"vehicle": {"car": {"sedan": None, "suv": None}, "truck": None}})


@pytest.fixture
def vehicle_target() -> Ontology:
    return target_ontology()


@pytest.mark.required
class TestLabelRemap:
    def test_index2label_covers_whole_target_in_order(self):
        target = target_ontology()
        _, index2label, _ = _label_remap({0: "sedan"}, label_alignment(["sedan"], target)["class_remap"], target)
        # enumerate over target.ids (insertion order)
        assert index2label == dict(enumerate([target.concept(c).label for c in target.ids]))

    def test_equivalence_mapping(self):
        target = target_ontology()
        i2l = {0: "sedan", 1: "truck"}
        mapping, index2label, dropped = _label_remap(i2l, label_alignment(i2l.values(), target)["class_remap"], target)
        names = {idx: index2label[mapping[idx]] for idx in mapping}
        assert names == {0: "sedan", 1: "truck"}
        assert dropped == {}

    def test_coarsening_is_non_injective(self):
        # source carries hierarchy: sedan is-a car; target only has car -> both collapse to car
        target = Ontology.from_hierarchy({"vehicle": {"car": None, "truck": None}})
        source = Ontology.from_hierarchy({"car": {"sedan": None}})
        alignment = label_alignment(source, target)
        mapping, index2label, _ = _label_remap({0: "car", 1: "sedan"}, alignment["class_remap"], target)
        car_index = next(i for i, name in index2label.items() if name == "car")
        assert mapping == {0: car_index, 1: car_index}  # collapsed

    def test_dropped_out_of_vocabulary(self):
        target = target_ontology()
        i2l = {0: "sedan", 1: "spaceship"}
        mapping, _, dropped = _label_remap(i2l, label_alignment(i2l.values(), target)["class_remap"], target)
        assert 0 in mapping
        assert dropped == {1: "spaceship"}

    def test_indexing_is_shared_across_sources(self):
        # two different source vocabularies aligned to one target share the indexing
        target = target_ontology()
        _, a, _ = _label_remap({0: "sedan"}, label_alignment(["sedan"], target)["class_remap"], target)
        _, b, _ = _label_remap(
            {0: "truck", 1: "sedan"}, label_alignment(["truck", "sedan"], target)["class_remap"], target
        )
        assert a == b

    def test_target_as_sequence_no_ontology(self):
        # fully manual: a name->name remap + a plain list target, no Ontology
        mapping, index2label, dropped = _label_remap(
            {0: "car", 1: "van", 2: "boat"},
            {"car": "vehicle", "van": "vehicle", "boat": "watercraft"},
            ["vehicle", "watercraft"],
        )
        assert index2label == {0: "vehicle", 1: "watercraft"}
        assert mapping == {0: 0, 1: 0, 2: 1}  # car, van -> vehicle; boat -> watercraft
        assert dropped == {}

    def test_target_as_index2label_mapping(self):
        mapping, index2label, dropped = _label_remap(
            {0: "car", 1: "boat", 2: "ufo"},
            {"car": "vehicle", "boat": "vessel"},
            {0: "vessel", 1: "vehicle"},
        )
        assert index2label == {0: "vessel", 1: "vehicle"}
        assert mapping == {0: 1, 1: 0}  # car -> vehicle(1), boat -> vessel(0)
        assert dropped == {2: "ufo"}  # not in class_remap

    def test_target_string_raises_type_error(self):
        # a bare string is not a valid target vocabulary
        with pytest.raises(TypeError, match="Ontology"):
            _label_remap({0: "car"}, {"car": "vehicle"}, "vehicle")

    def test_target_none_derives_vocab_from_class_map(self):
        # no target: vocabulary = distinct class_map values, first-seen order
        mapping, index2label, dropped = _label_remap(
            {0: "car", 1: "van", 2: "boat", 3: "ufo"},
            {"car": "vehicle", "van": "vehicle", "boat": "watercraft"},
        )
        assert index2label == {0: "vehicle", 1: "watercraft"}
        assert mapping == {0: 0, 1: 0, 2: 1}  # car, van -> vehicle; boat -> watercraft
        assert dropped == {3: "ufo"}  # not in class_map


def _argmax(datum) -> int:
    return int(np.argmax(np.asarray(datum[1])))


@pytest.mark.required
class TestRelabel:
    def test_is_an_operation(self, vehicle_target):
        assert isinstance(Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target), Operation)

    def test_repr(self):
        # ReprMixin introspects __init__ annotations; ensure that does not blow up
        assert "Relabel(" in repr(Relabel({"car": "vehicle"}))

    def test_ic_remap_and_metadata(self, ic_dataset, vehicle_target):
        i2l = {0: "sedan", 1: "truck"}
        ds = ic_dataset([0, 1, 0], i2l)
        conformed = View(ds, [Relabel(label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target)])
        # metadata is now the full target vocabulary
        target_i2l = dict(enumerate(vehicle_target.concept(c).label for c in vehicle_target.ids))
        assert "index2label" in conformed.metadata
        assert dict(conformed.metadata["index2label"]) == target_i2l
        # each datum's winning class is the target index for its source name
        labels = [conformed.metadata["index2label"][_argmax(d)] for d in conformed]
        assert labels == ["sedan", "truck", "sedan"]
        # one-hot is resized to the target vocabulary
        assert len(conformed[0][1]) == len(target_i2l)

    def test_ic_drops_out_of_vocabulary_image(self, ic_dataset, vehicle_target):
        i2l = {0: "sedan", 1: "truck", 2: "spaceship"}
        ds = ic_dataset([0, 1, 2, 0], i2l)  # 4 images, one is spaceship
        conformed = View(ds, [Relabel(label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target)])
        assert len(conformed) == 3  # spaceship image dropped

    def test_ic_coarsening_collapses_classes(self, ic_dataset):
        # source: sedan is-a car; target only has car -> sedan and car collapse to car
        target = Ontology.from_hierarchy({"vehicle": {"car": None, "truck": None}})
        source = Ontology.from_hierarchy({"car": {"sedan": None}})
        ds = ic_dataset([0, 1], {0: "car", 1: "sedan"})
        conformed = View(ds, [Relabel(label_alignment(source, target)["class_remap"], target)])
        assert _argmax(conformed[0]) == _argmax(conformed[1])  # both -> car

    def test_od_remaps_and_drops_detections(self, od_dataset, vehicle_target):
        i2l = {0: "sedan", 1: "truck", 2: "spaceship"}
        ds = od_dataset([[0, 1], [0, 2], [2]], i2l)
        conformed = View(ds, [Relabel(label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target)])
        assert len(conformed) == 2  # image with only spaceship is dropped
        assert "index2label" in conformed.metadata
        names = conformed.metadata["index2label"]
        first = conformed[0][1]
        assert [names[int(label)] for label in np.asarray(first.labels)] == ["sedan", "truck"]
        second = conformed[1][1]
        assert [names[int(label)] for label in np.asarray(second.labels)] == ["sedan"]  # spaceship detection dropped
        assert np.asarray(second.boxes).shape[0] == 1  # boxes masked to surviving detection
        assert len(np.asarray(second.scores)) == 1

    def test_on_unmatched_raise(self, ic_dataset, vehicle_target):
        i2l = {0: "sedan", 1: "spaceship"}
        ds = ic_dataset([0, 1], i2l)
        relabel = Relabel(
            label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target, on_unmatched="raise"
        )
        with pytest.raises(OntologyError, match="spaceship"):
            View(ds, [relabel])

    def test_missing_index2label_raises(self, ic_dataset, vehicle_target):
        ds = ic_dataset([0], {0: "sedan"})
        ds.metadata = {"id": "no-vocab"}  # strip index2label
        with pytest.raises(OntologyError, match="index2label"):
            View(ds, [Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)])

    def test_mapping_and_dropped_properties(self, ic_dataset, vehicle_target):
        i2l = {0: "sedan", 1: "spaceship"}
        relabel = Relabel(label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target)
        View(ic_dataset([0, 1], i2l), [relabel])
        assert 0 in relabel.mapping
        assert relabel.dropped == {1: "spaceship"}

    def test_unapplied_relabel_raises(self, vehicle_target):
        relabel = Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)
        with pytest.raises(OntologyError, match="View"):
            _ = relabel.mapping

    def test_unapplied_relabel_dropped_raises(self, vehicle_target):
        relabel = Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)
        with pytest.raises(OntologyError, match="View"):
            _ = relabel.dropped

    def test_unapplied_relabel_index2label_raises(self, vehicle_target):
        relabel = Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)
        with pytest.raises(OntologyError, match="View"):
            _ = relabel.index2label

    def test_keep_unsupported_target_type_raises(self, vehicle_target):
        relabel = Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)
        # a plain string is neither an ObjectDetectionTarget nor an Array
        with pytest.raises(TypeError, match="does not support targets of type"):
            relabel._keep(("image", "not-a-target"))

    def test_remap_unsupported_target_type_raises(self, ic_dataset, vehicle_target):
        relabel = Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)
        # apply through View so the internal mapping is populated
        View(ic_dataset([0], {0: "sedan"}), [relabel])
        with pytest.raises(TypeError, match="does not support targets of type"):
            relabel._remap(("image", "not-a-target", {}))

    def test_manual_remap_without_ontology(self, ic_dataset):
        # no Ontology, no alignment — just a hand-written remap + a plain target vocab
        ds = ic_dataset([0, 1, 2], {0: "car", 1: "van", 2: "boat"})
        relabel = Relabel({"car": "vehicle", "van": "vehicle", "boat": "watercraft"}, ["vehicle", "watercraft"])
        conformed = View(ds, [relabel])
        assert "index2label" in conformed.metadata
        assert dict(conformed.metadata["index2label"]) == {0: "vehicle", 1: "watercraft"}
        labels = [conformed.metadata["index2label"][_argmax(d)] for d in conformed]
        assert labels == ["vehicle", "vehicle", "watercraft"]

    def test_manual_remap_default_target(self, ic_dataset):
        # convenience fallback: omit target entirely, vocab derived from the class_map
        ds = ic_dataset([0, 1, 2], {0: "car", 1: "van", 2: "boat"})
        conformed = View(ds, [Relabel({"car": "vehicle", "van": "vehicle", "boat": "watercraft"})])
        assert "index2label" in conformed.metadata
        assert dict(conformed.metadata["index2label"]) == {0: "vehicle", 1: "watercraft"}

    def test_shared_vocabulary_across_datasets(self, ic_dataset, vehicle_target):
        # two datasets with different source vocabularies, same target -> same index2label
        a = View(
            ic_dataset([0], {0: "sedan"}),
            [Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)],
        )
        b = View(
            ic_dataset([0, 1], {0: "truck", 1: "sedan"}),
            [Relabel(label_alignment(["truck", "sedan"], vehicle_target)["class_remap"], vehicle_target)],
        )
        assert "index2label" in a.metadata
        assert "index2label" in b.metadata
        assert dict(a.metadata["index2label"]) == dict(b.metadata["index2label"])


@pytest.mark.required
class TestAliasTargets:
    def test_class_remap_naming_an_alias_id_resolves_to_its_canonical(self):
        # aliases are transparent everywhere else on Ontology; keying the target
        # off ids alone made an alias look out-of-vocabulary and silently
        # dropped every datum mapped to it
        target = Ontology([
            OntologyConcept(id="ex:Car", label="Car", equivalent_to=("ex:Auto",)),
            OntologyConcept(id="ex:Auto", label="Auto"),
        ])
        assert target.canonical("ex:Car") == "ex:Auto"
        mapping, index2label, dropped = _label_remap({0: "sedan"}, {"sedan": "ex:Car"}, target)
        assert dropped == {}
        assert mapping == {0: 0}
        assert index2label == {0: "Auto"}


@pytest.mark.required
class TestRelabelScores:
    """Per-class scores are indexed by a vocabulary, so conforming labels must conform them.

    Masking alone — all ``MaskedTarget`` does — leaves source-indexed columns beside
    target-indexed labels, so a detection's score is read from whichever class landed at
    its new index, or from off the end of the array. Two datasets conformed to one
    vocabulary also keep the two widths they arrived with, which is what stopped their
    metadata being stacked into one frame at all.
    """

    def test_od_per_class_scores_are_folded_into_the_target_vocabulary(self, od_dataset):
        ds = od_dataset([[0, 1]], {0: "car", 1: "boat"}, per_class=True, confidence=0.75)
        remap = {"car": "vehicle", "boat": "watercraft"}
        relabel = Relabel(remap, ["watercraft", "vehicle"], reduce_detection_scores=False)
        target = View(ds, [relabel])[0][1]
        assert list(np.asarray(target.labels)) == [1, 0]
        scores = np.asarray(target.scores)
        assert scores.shape == (2, 2)
        # each detection's confidence moved with it, into the column its new label names
        np.testing.assert_allclose(scores, [[0.0, 0.75], [0.75, 0.0]])

    def test_od_dropped_detections_take_their_scores_with_them(self, od_dataset):
        ds = od_dataset([[0, 1]], {0: "car", 1: "spaceship"}, per_class=True, confidence=0.75)
        conformed = View(ds, [Relabel({"car": "vehicle"}, ["vehicle"], reduce_detection_scores=False)])
        target = conformed[0][1]
        assert list(np.asarray(target.labels)) == [0]
        np.testing.assert_allclose(np.asarray(target.scores), [[0.75]])

    def test_od_per_box_scores_are_masked_not_folded(self, od_dataset):
        """A score that is one per box carries no vocabulary and needs only the masking."""
        ds = od_dataset([[0, 1]], {0: "car", 1: "spaceship"})
        conformed = View(ds, [Relabel({"car": "vehicle"}, ["vehicle"], reduce_detection_scores=False)])
        scores = np.asarray(conformed[0][1].scores)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(1.0)

    def test_a_score_survives_relabeling_into_a_wider_vocabulary(self, od_dataset):
        """The label is a target index by then, so a source-width array cannot be read."""
        ds = od_dataset([[0, 1]], {0: "car", 1: "bike"}, per_class=True, confidence=0.75)
        remap = {"car": "vehicle", "bike": "cycle"}
        relabel = Relabel(remap, ["a", "b", "c", "cycle", "vehicle"], reduce_detection_scores=False)
        target = View(ds, [relabel])[0][1]
        scores = np.asarray(target.scores)
        assert scores.shape == (2, 5)
        for index, label in enumerate(np.asarray(target.labels)):
            assert scores[index, label] == pytest.approx(0.75)

    def test_a_non_contiguous_target_vocabulary_is_addressed_by_its_own_indices(self, od_dataset):
        """A column position is the label value it scores, so gaps in the vocabulary are gaps."""
        ds = od_dataset([[0, 1]], {0: "car", 1: "bike"}, per_class=True, confidence=0.75)
        remap = {"car": "vehicle", "bike": "cycle"}
        conformed = View(ds, [Relabel(remap, {2: "cycle", 7: "vehicle"}, reduce_detection_scores=False)])
        target = conformed[0][1]
        assert list(np.asarray(target.labels)) == [7, 2]
        scores = np.asarray(target.scores)
        assert scores.shape == (2, 8)
        assert scores[0, 7] == pytest.approx(0.75)
        assert scores[1, 2] == pytest.approx(0.75)

    def test_a_non_contiguous_classification_vocabulary_no_longer_raises(self, ic_dataset):
        """It sized the fold by the class count, so a gap indexed past the end of it."""
        relabel = Relabel({"car": "v", "bike": "c"}, {0: "v", 7: "c"})
        conformed = View(ic_dataset([0, 1], {0: "car", 1: "bike"}), [relabel])
        np.testing.assert_allclose(np.asarray(conformed[1][1]), [0, 0, 0, 0, 0, 0, 0, 1])

    def test_a_score_array_disagreeing_on_the_count_reads_unknown_not_wrong(self):
        """Declining to fold left the masking to decline too, on the same length test."""

        class _Mismatched:
            labels = np.asarray([0, 1], dtype=np.intp)
            boxes = np.zeros((2, 4), dtype=np.float32)
            # three score rows for two detections
            scores = np.asarray([[0.9, 0.0], [0.0, 0.8], [0.5, 0.5]], dtype=np.float32)

        # a remap reversing the vocabulary, so a stale column would return the other score
        target, _ = Relabel._conform_detections(cast(ObjectDetectionTarget, _Mismatched()), {0: 1, 1: 0}, 2)
        scores = np.asarray(target.scores)
        assert scores[0, 1] == pytest.approx(0.9)
        assert scores[1, 0] == pytest.approx(0.8)

    def test_a_score_row_the_target_never_supplied_reads_unknown(self):
        """Padded with nan rather than dropped, so the count stays the labels' to set."""

        class _Short:
            labels = np.asarray([0, 1], dtype=np.intp)
            boxes = np.zeros((2, 4), dtype=np.float32)
            scores = np.asarray([[0.9, 0.0]], dtype=np.float32)

        target, _ = Relabel._conform_detections(cast(ObjectDetectionTarget, _Short()), {0: 0, 1: 1}, 2)
        scores = np.asarray(target.scores)
        assert scores[0, 0] == pytest.approx(0.9)
        assert np.isnan(scores[1]).all()

    def test_a_negative_target_index_is_refused(self, od_dataset):
        """It has no column to be, and would wrap onto another class's confidence."""
        ds = od_dataset([[0]], {0: "car"}, per_class=True)
        with pytest.raises(OntologyError, match="non-negative"):
            View(ds, [Relabel({"car": "vehicle"}, {-1: "vehicle"})])

    def test_coarsening_sums_the_collapsed_columns(self):
        folded = Relabel._conform_scores(np.array([[0.3, 0.5, 0.2]]), {0: 0, 1: 0, 2: 1}, 2)
        np.testing.assert_allclose(folded, [[0.8, 0.2]])

    def test_a_source_class_the_scores_have_no_column_for_contributes_nothing(self):
        folded = Relabel._conform_scores(np.array([0.6]), {0: 0, 3: 1}, 2)
        np.testing.assert_allclose(folded, [0.6, 0.0])

    def test_datasets_conformed_to_one_vocabulary_merge_and_structure(self, od_dataset):
        """The reported failure: np.concatenate rejected (N, 5) beside (N, 6)."""
        target = ["car", "truck", "van", "bus", "cycle", "rail"]
        a = View(
            od_dataset(
                [[0, 1], [2], [3, 4]],
                {0: "sedan", 1: "lorry", 2: "minivan", 3: "coach", 4: "bike"},
                per_class=True,
                confidence=0.75,
            ),
            [
                Relabel(
                    {"sedan": "car", "lorry": "truck", "minivan": "van", "coach": "bus", "bike": "cycle"},
                    target,
                    reduce_detection_scores=False,
                )
            ],
        )
        b = View(
            od_dataset(
                [[0, 5], [1, 2], [3, 4]],
                {0: "auto", 1: "hgv", 2: "mpv", 3: "omnibus", 4: "cycle", 5: "tram"},
                per_class=True,
                confidence=0.75,
                dataset_id="mock-od-b",
            ),
            [
                Relabel(
                    {"auto": "car", "hgv": "truck", "mpv": "van", "omnibus": "bus", "cycle": "cycle", "tram": "rail"},
                    target,
                    reduce_detection_scores=False,
                )
            ],
        )
        md = Metadata(merge_datasets(a, b))
        assert md.level_counts["instance"] == 11
        assert set(md.rows_at("instance")["class_label"].to_list()) == {0, 1, 2, 3, 4, 5}


@pytest.mark.required
class TestReduceDetectionScores:
    """The opt-in for v1.2's score handling, and the warning where the two differ."""

    def test_reducing_gives_one_confidence_per_detection(self, od_dataset):
        ds = od_dataset([[0, 1]], {0: "car", 1: "boat"}, per_class=True, confidence=0.75)
        remap = {"car": "vehicle", "boat": "watercraft"}
        relabel = Relabel(remap, ["watercraft", "vehicle"], reduce_detection_scores=True)
        target = View(ds, [relabel])[0][1]
        assert list(np.asarray(target.labels)) == [1, 0]
        # each detection kept its own number, and it carries no vocabulary for the new
        # labels to disagree with
        np.testing.assert_allclose(np.asarray(target.scores), [0.75, 0.75])

    def test_reducing_leaves_per_box_scores_alone(self, od_dataset):
        ds = od_dataset([[0, 1]], {0: "car", 1: "spaceship"})
        target = View(ds, [Relabel({"car": "vehicle"}, ["vehicle"], reduce_detection_scores=True)])[0][1]
        np.testing.assert_allclose(np.asarray(target.scores), [1.0])

    def test_reducing_reads_unknown_where_a_label_has_no_column(self, od_dataset):
        """The narrow-source case: not another class's mass, and not a confident 0.0."""

        class _Narrow:
            labels = np.asarray([0, 1], dtype=np.intp)
            boxes = np.zeros((2, 4), dtype=np.float32)
            scores = np.asarray([[0.9], [0.8]], dtype=np.float32)

        target, _ = Relabel._conform_detections(cast(ObjectDetectionTarget, _Narrow()), {0: 0, 1: 1}, 2, True)
        scores = np.asarray(target.scores)
        assert scores[0] == pytest.approx(0.9)
        assert np.isnan(scores[1])

    def test_reducing_lets_datasets_that_scored_differently_merge(self, od_dataset):
        """The gap folding cannot close: a per-class array and a per-box one in one frame."""
        shared = ["car", "truck", "cycle"]
        per_class = View(
            od_dataset([[0, 1]], {0: "sedan", 1: "lorry"}, per_class=True, confidence=0.75),
            [Relabel({"sedan": "car", "lorry": "truck"}, shared, reduce_detection_scores=True)],
        )
        per_box = View(
            od_dataset([[0, 1]], {0: "auto", 1: "bike"}, dataset_id="mock-od-b"),
            [Relabel({"auto": "car", "bike": "cycle"}, shared, reduce_detection_scores=True)],
        )
        md = Metadata(merge_datasets(per_class, per_box))
        assert md.dataframe.schema["score"] == pl.Float32
        assert md.rows_at("instance")["score"].to_list() == pytest.approx([0.75, 0.75, 1.0, 1.0])

    def test_folding_leaves_those_two_unmergeable(self, od_dataset):
        """Why the parameter exists: this is the state v1.1 cannot fix without changing shape."""
        shared = ["car", "truck", "cycle"]
        per_class = View(
            od_dataset([[0, 1]], {0: "sedan", 1: "lorry"}, per_class=True, confidence=0.75),
            [Relabel({"sedan": "car", "lorry": "truck"}, shared, reduce_detection_scores=False)],
        )
        per_box = View(
            od_dataset([[0, 1]], {0: "auto", 1: "bike"}, dataset_id="mock-od-b"),
            [Relabel({"auto": "car", "bike": "cycle"}, shared, reduce_detection_scores=False)],
        )
        with pytest.raises(ValueError, match="same number of dimensions"):
            _ = Metadata(merge_datasets(per_class, per_box)).dataframe

    def test_leaving_it_unset_folds_and_warns_once(self, od_dataset):
        ds = od_dataset([[0, 1], [0]], {0: "car", 1: "boat"}, per_class=True, confidence=0.75)
        conformed = View(ds, [Relabel({"car": "vehicle", "boat": "watercraft"}, ["watercraft", "vehicle"])])
        with pytest.warns(DeprecatedWarning, match="reduce_detection_scores"):
            first = conformed[0][1]
        assert np.asarray(first.scores).shape == (2, 2)  # unset still folds
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # said once per operation, not once per datum
            assert np.asarray(conformed[1][1].scores).shape == (1, 2)

    @pytest.mark.parametrize("choice", [True, False])
    def test_choosing_either_way_silences_the_warning(self, od_dataset, choice):
        ds = od_dataset([[0]], {0: "car"}, per_class=True)
        conformed = View(ds, [Relabel({"car": "vehicle"}, ["vehicle"], reduce_detection_scores=choice)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = conformed[0][1]

    def test_a_per_box_dataset_is_never_warned(self, od_dataset):
        """Nothing changes for it in v1.2, so telling it otherwise would be noise."""
        conformed = View(od_dataset([[0]], {0: "car"}), [Relabel({"car": "vehicle"}, ["vehicle"])])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = conformed[0][1]

    def test_a_classification_dataset_is_never_warned(self, ic_dataset):
        """Classification keeps the fold in every version; the parameter does not reach it."""
        conformed = View(ic_dataset([0, 1], {0: "car", 1: "bike"}), [Relabel({"car": "v", "bike": "c"}, ["v", "c"])])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = conformed[0][1]


@pytest.mark.required
class TestOwnClassScores:
    """The reduction itself. v1.2 moves it to ``dataeval.types._target``, shared by every reader."""

    def test_a_target_carrying_no_scores_reads_unknown(self):
        assert np.isnan(_own_class_scores(None, np.asarray([0, 1], dtype=np.intp))).all()

    def test_an_unrecognized_layout_reads_unknown(self):
        read = _own_class_scores(np.zeros((1, 2, 2), dtype=np.float32), np.asarray([0], dtype=np.intp))
        assert np.isnan(read).all()

    def test_labels_are_authoritative_on_the_count(self):
        short = _own_class_scores(np.asarray([0.5], dtype=np.float32), np.asarray([0, 1, 2], dtype=np.intp))
        assert len(short) == 3
        assert short[0] == pytest.approx(0.5)
        assert np.isnan(short[1:]).all()

    def test_the_own_class_column_is_read_not_the_highest(self):
        values = np.asarray([[0.2, 0.7], [0.9, 0.1]], dtype=np.float32)
        np.testing.assert_allclose(_own_class_scores(values, np.asarray([0, 1], dtype=np.intp)), [0.2, 0.1])

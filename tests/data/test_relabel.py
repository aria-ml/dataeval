from typing import cast

import numpy as np
import pytest

from dataeval import Metadata, Ontology
from dataeval.core import label_alignment
from dataeval.data import Operation, Relabel, View, merge_datasets
from dataeval.data._relabel import _label_remap
from dataeval.exceptions import OntologyError
from dataeval.protocols import ObjectDetectionTarget
from dataeval.types import OntologyConcept
from dataeval.types._target import detection_score


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
    its new index, or from off the end of the array. Reading the score down to the
    detection's own class against its *source* label settles it before the vocabulary
    changes, and leaves a number no vocabulary indexes.
    """

    def test_od_scores_are_read_down_to_one_per_detection(self, od_dataset):
        ds = od_dataset([[0, 1]], {0: "car", 1: "boat"}, per_class=True, confidence=0.75)
        conformed = View(ds, [Relabel({"car": "vehicle", "boat": "watercraft"}, ["watercraft", "vehicle"])])
        target = conformed[0][1]
        assert list(np.asarray(target.labels)) == [1, 0]
        # one confidence per detection, carrying no vocabulary for the new labels to
        # disagree with — and each detection kept its own number, not its neighbour's
        np.testing.assert_allclose(np.asarray(target.scores), [0.75, 0.75])
        for index, label in enumerate(np.asarray(target.labels)):
            assert detection_score(target, index, int(label)) == pytest.approx(0.75)

    def test_od_dropped_detections_take_their_scores_with_them(self, od_dataset):
        ds = od_dataset([[0, 1]], {0: "car", 1: "spaceship"}, per_class=True, confidence=0.75)
        conformed = View(ds, [Relabel({"car": "vehicle"}, ["vehicle"])])
        target = conformed[0][1]
        assert list(np.asarray(target.labels)) == [0]
        np.testing.assert_allclose(np.asarray(target.scores), [0.75])

    def test_od_per_box_scores_are_masked_not_folded(self, od_dataset):
        """A score that is one per box already carries no vocabulary."""
        ds = od_dataset([[0, 1]], {0: "car", 1: "spaceship"})
        conformed = View(ds, [Relabel({"car": "vehicle"}, ["vehicle"])])
        np.testing.assert_allclose(np.asarray(conformed[0][1].scores), [1.0])

    def test_a_score_survives_relabeling_into_a_wider_vocabulary(self, od_dataset):
        """The label is a target index by then, so a source-width array cannot be read."""
        ds = od_dataset([[0, 1]], {0: "car", 1: "bike"}, per_class=True, confidence=0.75)
        conformed = View(ds, [Relabel({"car": "vehicle", "bike": "cycle"}, ["a", "b", "c", "cycle", "vehicle"])])
        target = conformed[0][1]
        assert list(np.asarray(target.labels)) == [4, 3]
        np.testing.assert_allclose(np.asarray(target.scores), [0.75, 0.75])

    def test_a_non_contiguous_target_vocabulary_costs_nothing(self, od_dataset):
        """A confidence is a property of the detection, so a sparse vocabulary sizes nothing."""
        ds = od_dataset([[0, 1]], {0: "car", 1: "bike"}, per_class=True, confidence=0.75)
        conformed = View(ds, [Relabel({"car": "vehicle", "bike": "cycle"}, {2: "cycle", 50000: "vehicle"})])
        target = conformed[0][1]
        assert list(np.asarray(target.labels)) == [50000, 2]
        assert np.asarray(target.scores).shape == (2,)

    def test_a_score_array_disagreeing_on_the_count_reads_unknown_not_wrong(self):
        """The regression: the fold *and* the masking both declined, passing raw columns through."""

        class _Mismatched:
            labels = np.asarray([0, 1], dtype=np.intp)
            boxes = np.zeros((2, 4), dtype=np.float32)
            # three score rows for two detections, the disagreement own_class_scores tolerates
            scores = np.asarray([[0.9, 0.0], [0.0, 0.8], [0.5, 0.5]], dtype=np.float32)

        # a remap that reverses the vocabulary, so reading a score at the new label
        # rather than the old one would return the other detection's number
        target, mask = Relabel._conform_detections(_Mismatched(), {0: 1, 1: 0})
        assert list(np.asarray(target.labels)) == [1, 0]
        np.testing.assert_allclose(np.asarray(target.scores), [0.9, 0.8])
        assert mask.tolist() == [True, True]

    def test_a_detection_whose_score_cannot_be_read_conforms_to_unknown(self):
        """Not to 0.0: the fold's zeros made an unreadable score look like a confident one."""

        class _Narrow:
            labels = np.asarray([0, 1], dtype=np.intp)
            boxes = np.zeros((2, 4), dtype=np.float32)
            # a per-box score stored as a column vector: only class 0 has a column
            scores = np.asarray([[0.9], [0.8]], dtype=np.float32)

        target, _ = Relabel._conform_detections(_Narrow(), {0: 0, 1: 1})
        scores = np.asarray(target.scores)
        assert scores[0] == pytest.approx(0.9)
        assert np.isnan(scores[1])

    def test_a_target_carrying_no_scores_is_left_without_any(self):
        """Nothing to conform, and inventing a NaN column would claim the target had one."""

        class _Unscored:
            labels = np.asarray([0, 1], dtype=np.intp)
            boxes = np.zeros((2, 4), dtype=np.float32)

        target, _ = Relabel._conform_detections(cast(ObjectDetectionTarget, _Unscored()), {0: 1, 1: 0})
        assert list(np.asarray(target.labels)) == [1, 0]
        assert not hasattr(target, "scores")

    def test_a_negative_target_index_is_refused(self, od_dataset):
        """It has no column to be, and would wrap onto another class's confidence."""
        ds = od_dataset([[0]], {0: "car"}, per_class=True)
        with pytest.raises(OntologyError, match="non-negative"):
            View(ds, [Relabel({"car": "vehicle"}, {-1: "vehicle"})])

    def test_ic_coarsening_sums_the_collapsed_columns(self):
        # two source classes folding into one target class keep their combined mass, so
        # the argmax still lands on the coarsened class
        folded = Relabel._conform_scores(np.array([[0.3, 0.5, 0.2]]), {0: 0, 1: 0, 2: 1}, 2)
        np.testing.assert_allclose(folded, [[0.8, 0.2]])

    def test_ic_a_source_class_the_scores_have_no_column_for_contributes_nothing(self):
        folded = Relabel._conform_scores(np.array([0.6]), {0: 0, 3: 1}, 2)
        np.testing.assert_allclose(folded, [0.6, 0.0])

    def test_datasets_conformed_to_one_vocabulary_merge_and_structure(self, od_dataset):
        """End to end: the reported failure was np.concatenate on (N, 5) beside (N, 6)."""
        target = ["car", "truck", "van", "bus", "cycle", "rail"]
        a = View(
            od_dataset(
                [[0, 1], [2], [3, 4]],
                {0: "sedan", 1: "lorry", 2: "minivan", 3: "coach", 4: "bike"},
                per_class=True,
                confidence=0.75,
            ),
            [Relabel({"sedan": "car", "lorry": "truck", "minivan": "van", "coach": "bus", "bike": "cycle"}, target)],
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
                )
            ],
        )
        md = Metadata(merge_datasets(a, b))
        assert md.level_counts["instance"] == 11
        assert md.rows_at("instance")["score"].to_list() == pytest.approx([0.75] * 11)
        assert set(md.rows_at("instance")["class_label"].to_list()) == {0, 1, 2, 3, 4, 5}

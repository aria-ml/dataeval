import numpy as np
import pytest

from dataeval import Ontology
from dataeval.core import label_alignment
from dataeval.data import Conform, Conformer, Relabel
from dataeval.data._relabel import _label_remap
from dataeval.exceptions import OntologyError


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
    def test_is_a_conformer(self, vehicle_target):
        assert isinstance(Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target), Conformer)

    def test_repr(self):
        # ReprMixin introspects __init__ annotations; ensure that does not blow up
        assert "Relabel(" in repr(Relabel({"car": "vehicle"}))

    def test_ic_remap_and_metadata(self, ic_dataset, vehicle_target):
        i2l = {0: "sedan", 1: "truck"}
        ds = ic_dataset([0, 1, 0], i2l)
        conformed = Conform(ds, [Relabel(label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target)])
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
        conformed = Conform(ds, [Relabel(label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target)])
        assert len(conformed) == 3  # spaceship image dropped

    def test_ic_coarsening_collapses_classes(self, ic_dataset):
        # source: sedan is-a car; target only has car -> sedan and car collapse to car
        target = Ontology.from_hierarchy({"vehicle": {"car": None, "truck": None}})
        source = Ontology.from_hierarchy({"car": {"sedan": None}})
        ds = ic_dataset([0, 1], {0: "car", 1: "sedan"})
        conformed = Conform(ds, [Relabel(label_alignment(source, target)["class_remap"], target)])
        assert _argmax(conformed[0]) == _argmax(conformed[1])  # both -> car

    def test_od_remaps_and_drops_detections(self, od_dataset, vehicle_target):
        i2l = {0: "sedan", 1: "truck", 2: "spaceship"}
        ds = od_dataset([[0, 1], [0, 2], [2]], i2l)
        conformed = Conform(ds, [Relabel(label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target)])
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
            Conform(ds, [relabel])

    def test_missing_index2label_raises(self, ic_dataset, vehicle_target):
        ds = ic_dataset([0], {0: "sedan"})
        ds.metadata = {"id": "no-vocab"}  # strip index2label
        with pytest.raises(OntologyError, match="index2label"):
            Conform(ds, [Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)])

    def test_mapping_and_dropped_properties(self, ic_dataset, vehicle_target):
        i2l = {0: "sedan", 1: "spaceship"}
        relabel = Relabel(label_alignment(i2l.values(), vehicle_target)["class_remap"], vehicle_target)
        Conform(ic_dataset([0, 1], i2l), [relabel])
        assert 0 in relabel.mapping
        assert relabel.dropped == {1: "spaceship"}

    def test_unapplied_relabel_raises(self, vehicle_target):
        relabel = Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)
        with pytest.raises(OntologyError, match="Conform"):
            _ = relabel.mapping

    def test_manual_remap_without_ontology(self, ic_dataset):
        # no Ontology, no alignment — just a hand-written remap + a plain target vocab
        ds = ic_dataset([0, 1, 2], {0: "car", 1: "van", 2: "boat"})
        relabel = Relabel({"car": "vehicle", "van": "vehicle", "boat": "watercraft"}, ["vehicle", "watercraft"])
        conformed = Conform(ds, [relabel])
        assert "index2label" in conformed.metadata
        assert dict(conformed.metadata["index2label"]) == {0: "vehicle", 1: "watercraft"}
        labels = [conformed.metadata["index2label"][_argmax(d)] for d in conformed]
        assert labels == ["vehicle", "vehicle", "watercraft"]

    def test_manual_remap_default_target(self, ic_dataset):
        # convenience fallback: omit target entirely, vocab derived from the class_map
        ds = ic_dataset([0, 1, 2], {0: "car", 1: "van", 2: "boat"})
        conformed = Conform(ds, [Relabel({"car": "vehicle", "van": "vehicle", "boat": "watercraft"})])
        assert "index2label" in conformed.metadata
        assert dict(conformed.metadata["index2label"]) == {0: "vehicle", 1: "watercraft"}

    def test_shared_vocabulary_across_datasets(self, ic_dataset, vehicle_target):
        # two datasets with different source vocabularies, same target -> same index2label
        a = Conform(
            ic_dataset([0], {0: "sedan"}),
            [Relabel(label_alignment(["sedan"], vehicle_target)["class_remap"], vehicle_target)],
        )
        b = Conform(
            ic_dataset([0, 1], {0: "truck", 1: "sedan"}),
            [Relabel(label_alignment(["truck", "sedan"], vehicle_target)["class_remap"], vehicle_target)],
        )
        assert "index2label" in a.metadata
        assert "index2label" in b.metadata
        assert dict(a.metadata["index2label"]) == dict(b.metadata["index2label"])

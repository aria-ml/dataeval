"""Representation.evaluate accepts a raw label sequence or count mapping, no dataset needed."""

import numpy as np
import pytest

from dataeval import Ontology
from dataeval.scope._representation import Representation, RepresentationOutput

ANIMAL_LEAVES = ["cat", "dog", "owl", "hawk"]


def animal_ontology() -> Ontology:
    """animal -> {mammal -> [cat, dog], bird -> [owl, hawk]}."""
    return Ontology.from_hierarchy({"animal": {"mammal": ["cat", "dog"], "bird": ["owl", "hawk"]}})


@pytest.mark.required
class TestRepresentationLabelsOnly:
    def test_raw_int_labels_only(self):
        """The minimal call: a raw integer label sequence and nothing else.

        The ontology's leaves are named to match the stringified label indices,
        so labels resolve without an explicit index2label mapping.
        """
        onto = Ontology.from_hierarchy({"root": ["0", "1", "2"]})
        labels = [0, 0, 0, 1, 1, 2]  # counts: 0->3, 1->2, 2->1

        result = Representation(onto).evaluate(labels)

        assert isinstance(result, RepresentationOutput)
        assert result.leaf_coverage == 1.0  # all three leaves populated

    def test_raw_int_labels_with_index2label(self):
        """Raw integer labels named through an explicit index2label mapping."""
        labels = np.array([0, 0, 1, 1, 1, 2])  # cat->2, dog->3, owl->1
        index2label = {0: "cat", 1: "dog", 2: "owl"}

        result = Representation(animal_ontology()).evaluate(labels, index2label=index2label)

        assert isinstance(result, RepresentationOutput)
        # hawk is never labeled, so 3 of 4 leaves are covered.
        assert result.leaf_coverage == 0.75

    def test_label_name_count_mapping(self):
        """A {label_name: count} mapping is the exact form the core consumes."""
        counts = {"cat": 8, "dog": 2, "owl": 1}  # hawk absent

        result = Representation(animal_ontology()).evaluate(counts)

        assert isinstance(result, RepresentationOutput)
        assert result.leaf_coverage == 0.75
        # total 11, uniform target round(11/4) = 3; sorted by -deficit
        assert [r["concept"] for r in result.data().to_dicts()] == ["hawk", "owl", "dog"]

    def test_label_index_count_mapping(self):
        """A {label_index: count} mapping is named via index2label."""
        counts = {0: 8, 1: 2, 2: 1}
        index2label = {0: "cat", 1: "dog", 2: "owl"}

        result = Representation(animal_ontology()).evaluate(counts, index2label=index2label)

        assert isinstance(result, RepresentationOutput)
        assert result.leaf_coverage == 0.75

    def test_labels_only_matches_name_mapping(self):
        """Raw labels and the equivalent name mapping produce the same worklist."""
        index2label = {0: "cat", 1: "dog", 2: "owl"}
        labels = [0] * 8 + [1] * 2 + [2] * 1
        onto = animal_ontology()

        from_labels = Representation(onto).evaluate(labels, index2label=index2label)
        from_counts = Representation(onto).evaluate({"cat": 8, "dog": 2, "owl": 1})

        assert from_labels.data().to_dicts() == from_counts.data().to_dicts()
        assert from_labels.total_deficit == from_counts.total_deficit
